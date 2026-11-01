#!/usr/bin/env python3
"""headsim.py DUMPDIR QDIR PERM.npy PLAN.json CKPT OUT.json -- exact head-only KL of row-slab head plans (vtsim.py's
method, for headslab.py's plans).

  - X: real head inputs (DUMPDIR/336.x, E1b hook EXL3_AQ_DUMP_ONLY=336; the calib-A dump = fit pool, disjoint from the
    qad-only Fisher ranking that made the plans)
  - W_K: the head quantized at K in sensitivity order (QDIR/K<k>/wq/lm_head.safetensors, (V, H) f16, permuted rows)
  - W: the checkpoint's lm_head (bf16); logits in f32 (f16 logits put a ~0.0002 floor under KL, see vtsim)
Every plan picks each permuted row's K from per-K logits computed once per position block, so plans cost nothing
extra. Per plan: mean KL(p_bf16 || q), top-1 agreement, real head bytes, a rare-token guard (KL where the bf16 argmax
is outside the 20k most sensitive rows), and the paired difference vs the uniform plan of the same budget with a
bootstrap over 512-position blocks (dump rows are consecutive positions, so blocks stand in for documents).
One card, ~22 GB VRAM. env HS_BLOCK (256), HS_GUARD (20000)."""
import json, os, sys, time
import numpy as np, torch
from safetensors import safe_open

DUMP, QD, PERMF, PLANF, CK, OUT = sys.argv[1:7]
BLK = int(os.environ.get("HS_BLOCK", 256)); GUARD = int(os.environ.get("HS_GUARD", 20000))
V, H, RB = 248320, 5120, 128
dev = torch.device("cuda")
K_, bf, _ = open(f"{DUMP}/336.meta").read().split(); assert int(K_) == H
X = torch.from_numpy(np.fromfile(f"{DUMP}/336.x", np.uint16).reshape(-1, H).astype(np.int16))
X = X.view(torch.bfloat16 if bf == "1" else torch.float16); N = X.shape[0]
perm = torch.from_numpy(np.load(PERMF)).to(dev)              # permuted row i = vocab id perm[i]
P = json.load(open(PLANF)); KS = P["Ks"]
plans = {n: torch.from_numpy(np.repeat(np.array(p["K_blocks"]), RB)).to(dev) for n, p in P["plans"].items()}
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: Wref = f.get_tensor("lm_head.weight").to(dev)[perm]   # bf16, permuted
W = {}
for k in KS:
    with safe_open(f"{QD}/K{k}/wq/lm_head.safetensors", "pt") as f: W[k] = f.get_tensor("weight").to(dev)
print(f"{N} head inputs, {len(plans)} plans, weights on card {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
      file=sys.stderr, flush=True)
def mm(x, w): return torch.mm(x, w.T, out_dtype=torch.float32)
kl_pos = {n: torch.zeros(N) for n in plans}; hit = {n: 0 for n in plans}; guard = torch.zeros(N, dtype=torch.bool)
t0 = time.time()
with torch.no_grad():
    for b0 in range(0, N, BLK):
        x = X[b0:b0 + BLK].to(dev); b1 = b0 + x.shape[0]
        zr = mm(x.to(torch.bfloat16), Wref); lp = torch.log_softmax(zr, -1); p = lp.exp(); top = zr.argmax(-1)
        guard[b0:b1] = (top >= GUARD).cpu()                    # permuted index = sensitivity rank
        z = {k: mm(x.to(torch.float16), W[k]) for k in KS}
        for n, kv in plans.items():
            zq = torch.empty_like(zr)
            for k in KS:
                m = kv == k
                if m.any(): zq[:, m] = z[k][:, m]
            kl_pos[n][b0:b1] = (p * (lp - torch.log_softmax(zq, -1))).sum(-1).cpu()
            hit[n] += int((zq.argmax(-1) == top).sum())
        if b0 // BLK % 16 == 0: print(f"{b1}/{N} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

nb = N // 512; rng = np.random.default_rng(0); boot = rng.integers(0, nb, (2000, nb))
def blocks(v): return v[:nb * 512].numpy().reshape(nb, 512).mean(1)
res = {}
for n, pl in P["plans"].items():
    B = n.lstrip("usb").split("-")[0]; ref = f"u{B}"
    r = {"avg_bits": pl["avg_bits"], "kl": float(kl_pos[n].mean()), "top1": hit[n] / N,
         "guard_kl": float(kl_pos[n][guard].mean()) if guard.any() else None, "guard_n": int(guard.sum()),
         "proxy_train_vs_uniform": pl.get("train_vs_uniform")}
    if ref in kl_pos and ref != n:
        d = blocks(kl_pos[n] - kl_pos[ref]); bs = d[boot].mean(1)
        r.update(vs=ref, diff=float(d.mean()), rel=float(d.mean() / kl_pos[ref].mean()), se=float(bs.std()),
                 ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    res[n] = r
json.dump({"n": N, "guard_rows": GUARD, "plans": res}, open(OUT, "w"), indent=1)
for n, r in res.items():
    v = f"  vs {r['vs']} {r['rel']:+.1%} (95% CI {r['ci95'][0] / (r['kl'] - r['diff']):+.1%}..{r['ci95'][1] / (r['kl'] - r['diff']):+.1%})" if "vs" in r else ""
    print(f"{n:12s} {r['avg_bits']:.3f} b  KL {r['kl']:.6f}  top1 {r['top1']:.4f}  guard {r['guard_kl'] or 0:.6f}{v}  "
          f"(proxy {r['proxy_train_vs_uniform']:+.1%})")
