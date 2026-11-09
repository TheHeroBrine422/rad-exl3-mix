#!/usr/bin/env python3
"""headsim94.py DUMPDIR QDIR PERM.npy OUT.npz PLAN.json [PLAN.json ...] -- headsim.py's exact head-only KL, per dump row,
on headx94.py's dumps (#124), plus a logit-temperature grid (A6's tau) and what the CPU proxy (hq94.py) needs.

Same arithmetic as v3/yaqa/headsim.py (bf16 reference logits in f32; donor heads QDIR/K<k>/wq/lm_head.safetensors, f16
rows in sensitivity order; a plan takes each 128-row block from its donor). Differences: every dump row is scored and
kept (the gen / all and document split is done by report94.py, which bootstraps by document), several plan files,
and per row: KL per plan, top-1 hit per plan, the reference logsumexp (lse), argmax and its sensitivity rank.
tau: for the plans in HS_TAU_PLANS (default u3,u4,u5,u6,u7,s5-3,s6-3) also KL(p || softmax(tau * z_q)) for tau in
HS_TAUS (default 0.96,0.98,0.99,1.01,1.02,1.04). One card, ~20 GB (5 donors)."""
import json, os, sys, time
import numpy as np, torch
from safetensors import safe_open

DUMP, QD, PERMF, OUT = sys.argv[1:5]; PLANFS = sys.argv[5:]
BLK = int(os.environ.get("HS_BLOCK", 256)); RB = 128
TAUS = [float(t) for t in os.environ.get("HS_TAUS", "0.96,0.98,0.99,1.01,1.02,1.04").split(",")]
TAUP = os.environ.get("HS_TAU_PLANS", "u3,u4,u5,u6,u7,s5-3,s6-3").split(",")
CK = os.environ.get("HS_CKPT", "/data/workspace/models/qwen38-27b-bf16")
dev = torch.device("cuda")
H, bf, N = open(f"{DUMP}/336.meta").read().split(); H, N = int(H), int(N); assert bf == "1"
X = torch.from_numpy(np.fromfile(f"{DUMP}/336.x", np.int16).reshape(N, H)).view(torch.bfloat16)
perm = torch.from_numpy(np.load(PERMF)).to(dev)
plans, meta = {}, {}
for pf in PLANFS:
    P = json.load(open(pf))
    for n, p in P["plans"].items():
        if n in plans: assert meta[n]["K_blocks"] == p["K_blocks"], f"{n} differs between plan files"; continue
        plans[n] = torch.from_numpy(np.repeat(np.array(p["K_blocks"]), RB)).to(dev); meta[n] = p
KS = sorted({k for p in meta.values() for k in p["K_blocks"]})
TAUP = [n for n in TAUP if n in plans]
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: Wref = f.get_tensor("lm_head.weight").to(dev)[perm]   # bf16, permuted
W = {}
for k in KS:
    with safe_open(f"{QD}/K{k:g}/wq/lm_head.safetensors", "pt") as f: W[k] = f.get_tensor("weight").to(dev)
print(f"{N} rows, {len(plans)} plans, donors {KS}, tau plans {TAUP}, on card {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
      file=sys.stderr, flush=True)
def mm(x, w): return torch.mm(x, w.T, out_dtype=torch.float32)
names = sorted(plans)
kl = np.zeros((len(names), N), np.float32); hit = np.zeros((len(names), N), np.uint8)
klt = np.zeros((len(TAUP), len(TAUS), N), np.float32)
lse = np.zeros(N, np.float32); top = np.zeros(N, np.int64)
t0 = time.time()
with torch.no_grad():
    for b0 in range(0, N, BLK):
        x = X[b0:b0 + BLK].to(dev); b1 = b0 + x.shape[0]
        zr = mm(x, Wref); ls = torch.logsumexp(zr, -1); lp = zr - ls[:, None]; p = lp.exp(); tp = zr.argmax(-1)
        lse[b0:b1] = ls.cpu().numpy(); top[b0:b1] = tp.cpu().numpy()       # top: permuted index = sensitivity rank
        z = {k: mm(x.to(torch.float16), W[k]) for k in KS}
        for i, n in enumerate(names):
            kv = plans[n]; zq = torch.empty_like(zr)
            for k in KS:
                m = kv == k
                if m.any(): zq[:, m] = z[k][:, m]
            kl[i, b0:b1] = (p * (lp - torch.log_softmax(zq, -1))).sum(-1).cpu().numpy()
            hit[i, b0:b1] = (zq.argmax(-1) == tp).cpu().numpy()
            if n in TAUP:
                for j, t in enumerate(TAUS):
                    klt[TAUP.index(n), j, b0:b1] = (p * (lp - torch.log_softmax(zq * t, -1))).sum(-1).cpu().numpy()
        if b0 // BLK % 16 == 0: print(f"{b1}/{N} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
np.savez(OUT + ".part.npz", names=np.array(names), kl=kl, hit=hit, tau_plans=np.array(TAUP), taus=np.array(TAUS),
         kl_tau=klt, lse=lse, top=top, avg_bits=np.array([meta[n]["avg_bits"] for n in names]))
os.replace(OUT + ".part.npz", OUT)
json.dump({n: meta[n] for n in names}, open(OUT[:-4] + "-plans.json", "w"))
print(f"done {N} rows in {time.time() - t0:.0f}s", file=sys.stderr)
