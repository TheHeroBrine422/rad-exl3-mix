#!/usr/bin/env python3
"""vtsim.py DUMPDIR HEADDIR COUNTS.npy CKPT OUT.json -- vocab-tiered head simulation (O1, research-quant-sota.md).

Head-only KL of tiered heads against the bf16 head, on real head inputs:
  - X: the head's inputs from the E1b dump (DUMPDIR/336.x, [rows][5120] bf16/f16; EXL3_AQ_DUMP_ONLY=336)
  - W_K: our head decoded at each K (HEADDIR/head-K<k>.f16, mix/headdec), W: the checkpoint's lm_head (bf16)
  - vocab order: COUNTS.npy (token counts on pools disjoint from the dump's slice), descending
A plan is [(rows, K), ...] over the vocab in that order; its logits are the per-id choice among the per-K logits
(computed once per position block), so plans cost nothing extra. Per plan: mean KL(p_bf16 || q), top-1 agreement,
the head's real bpw, and the guard: mean KL over positions whose bf16 argmax lies outside the plan's first tier.

Runs on one card (ROCR_VISIBLE_DEVICES), ~20 GB VRAM. env VT_KS (default 1,2,3,4,5,6,8), VT_BLOCK (256)."""
import itertools, json, os, sys, time
import numpy as np, torch
from safetensors import safe_open

DUMP, HD, CNT, CK, OUT = sys.argv[1:6]
KS = [int(k) for k in os.environ.get("VT_KS", "1,2,3,4,5,6,8").split(",")]
BLK = int(os.environ.get("VT_BLOCK", 256))
V, H, PARAMS = 248320, 5120, 26895998464
ROWB = H / 8 * 1.003                                   # bytes per vocab row per bit (EXL3 + scales, as vocabtier.py)
dev = torch.device("cuda")

K_, bf, _ = open(f"{DUMP}/336.meta").read().split()
assert int(K_) == H
raw = np.fromfile(f"{DUMP}/336.x", np.uint16).reshape(-1, H)
X = torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16 if bf == "1" else torch.float16)
N = X.shape[0]
print(f"{N} head inputs ({'bf16' if bf == '1' else 'f16'})", file=sys.stderr, flush=True)

idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: Wref = f.get_tensor("lm_head.weight").to(dev)          # bf16 [V][H]
W = {}
for k in KS:
    W[k] = torch.from_numpy(np.fromfile(f"{HD}/head-K{k}.f16", np.float16).reshape(V, H)).to(dev)
print(f"weights on card: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", file=sys.stderr, flush=True)

cnt = np.load(CNT)
order = np.argsort(-cnt, kind="stable")                # rank -> id
rank = np.empty(V, np.int64); rank[order] = np.arange(V)
rank_t = torch.from_numpy(rank).to(dev)

plans = {f"uniform K{k}": [(V, k)] for k in KS}
for n in (2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000):
    for hi, lo in itertools.product(KS, KS):
        if lo < hi and hi >= 4: plans[f"top{n}@K{hi}+rest@K{lo}"] = [(n, hi), (V - n, lo)]
for n1, n2 in ((5000, 30000), (10000, 50000), (20000, 75000), (10000, 100000)):
    for k1, k2, k3 in ((6, 4, 2), (6, 4, 3), (5, 3, 2), (6, 3, 2), (8, 5, 3), (5, 4, 2), (6, 5, 3), (5, 3, 1), (4, 3, 2)):
        plans[f"top{n1}@K{k1}+{n2}@K{k2}+rest@K{k3}"] = [(n1, k1), (n2 - n1, k2), (V - n2, k3)]


def plan_k(p):                                         # per-id K (by id, via rank)
    edges = np.cumsum([r for r, _ in p]); ks = np.array([k for _, k in p])
    return torch.from_numpy(ks[np.searchsorted(edges, rank, side="right")]).to(dev)


PK = {name: plan_k(p) for name, p in plans.items()}
first = {name: p[0][0] if len(p) > 1 else V for name, p in plans.items()}
acc = {name: [0.0, 0, 0.0, 0] for name in plans}       # sum KL, top-1 hits, guard KL sum, guard count
def mm(x, w): return torch.mm(x, w.T, out_dtype=torch.float32)


t0 = time.time()
with torch.no_grad():
    for b0 in range(0, N, BLK):
        x = X[b0:b0 + BLK].to(dev)
        zr = mm(x.to(torch.bfloat16), Wref)                # f32 logits: f16/bf16 outputs round ~0.02 at |z|~30 (KL floor)
        lp = torch.log_softmax(zr, -1); p = lp.exp(); top = zr.argmax(-1); trank = rank_t[top]
        z = {k: mm(x.to(torch.float16), W[k]) for k in KS}
        for name, kv in PK.items():
            zq = torch.empty_like(zr)
            for k in KS:
                m = kv == k
                if m.any(): zq[:, m] = z[k][:, m]
            kl = (p * (lp - torch.log_softmax(zq, -1))).sum(-1)
            a = acc[name]; a[0] += kl.sum().item(); a[1] += (zq.argmax(-1) == top).sum().item()
            g = trank >= first[name]
            a[2] += kl[g].sum().item(); a[3] += int(g.sum().item())
        if b0 // BLK % 16 == 0: print(f"{b0 + BLK}/{N} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

res = []
for name, pl in plans.items():
    s, t, gs, gc = acc[name]
    res.append({"plan": name, "tiers": pl, "bpw": sum(r * k * ROWB for r, k in pl) * 8 / PARAMS, "kl": s / N,
                "top1": t / N, "guard_kl": gs / gc if gc else None, "guard_n": gc})
json.dump({"n": N, "ks": KS, "plans": res}, open(OUT, "w"), indent=1)
for r in sorted(res, key=lambda r: r["bpw"]):
    print(f"{r['plan']:34} {r['bpw']:.4f} bpw  KL {r['kl']:.6f}  top1 {r['top1']:.4f}  guard {r['guard_kl'] or 0:.5f} ({r['guard_n']})")
