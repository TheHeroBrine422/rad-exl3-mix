#!/usr/bin/env python3
"""headpred.py CKPT HEAD_HESS META POOLS OUT.json [--sens HEADSENS.npz] [--emulate N] -- CPU estimate of what two-sided
LDLQ with a diagonal output factor buys on the 27B LM head (E2), before any GPU quant.

Output weight per vocabulary row: d_v = F_v * sv_v^2 (F = output-side Fisher diagonal, sv = the row's RMS / mean
after mean-row subtraction: the output scale regularize gives it). In the rotated basis the factor is block-diagonal
in 128-row blocks; one-sided LDLQ spreads error evenly over a block (cost ~ AM of d over the block), two-sided
shapes it (cost ~ GM). Predictor R(order, sigma_out) = sum_b GM_b(d~) / sum_b AM_b(d~), d~ = d / mean(d) + sigma_out
(the damped factor prepare_H_out builds), for the vocabulary order, the sensitivity (slab) order and the strided order (block b holds
sensitivity ranks b, b + nb, ...: every block spans the whole range; an --head-perm ourq can take as is).

F: --sens: headsens.py's exact softmax Fisher (fisher_qad_gen, iceman); else a PROXY: next-token counts on gen rows of
the qad pool (META = the calib golden's kld.json / tokens.i32 / gen.u8), + 0.01 (counts track sum p, a rough stand-in).
--emulate N: also quantize N rows (64 evenly spaced runs of N/64 rows; and the N most sensitive rows sorted; the first N of the strided order) with
exllamav3's own code on the CPU and a scalar stand-in for the trellis (cpuq.py), one- vs two-sided, K2 / K3,
sigma_out 0.01 / 0.1, and report sum_v F_v (w - q)_v^T H (w - q)_v (the head-only KL proxy, Fisher x row error)."""
import argparse, json, os, sys, time
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("hess"); ap.add_argument("meta"); ap.add_argument("pools"); ap.add_argument("out")
ap.add_argument("--sens"); ap.add_argument("--emulate", type=int, default=0); ap.add_argument("--quant", default=None)
ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
torch.set_num_threads(a.threads)
t0 = time.time()
idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{a.ckpt}/{idx}", "pt") as f: W = f.get_tensor("lm_head.weight").float()       # (V, 5120)
V = W.shape[0]
W -= W.mean(0, keepdim=True)                                   # mean-row subtraction (adopted for every head donor)
sv = W.pow(2).mean(1).sqrt(); sv /= sv.mean()

if a.sens:
    F = np.load(a.sens)["fisher_qad_gen"].astype(np.float64); src = "exact fisher_qad_gen"
else:
    m = json.load(open(f"{a.meta}/kld.json")); toks = np.fromfile(f"{a.meta}/tokens.i32", np.int32)
    gen = np.fromfile(f"{a.meta}/gen.u8", np.uint8).astype(bool); pools = json.load(open(a.pools))["pool"]
    F = np.zeros(m["n_vocab"]); off = 0
    for i, d in enumerate(m["docs"]):
        n = d["tokens"] - 1 - d["score_from"]
        if pools[i] == "qad":
            nxt = toks[off + d["score_from"] + 1: off + d["tokens"]]; g = gen[d["row0"]:d["row0"] + n]
            F += np.bincount(nxt[g], minlength=m["n_vocab"])
        off += d["tokens"]
    F += 0.01; src = "PROXY: qad gen next-token counts + 0.01"
F = F[:V]; Ft = torch.from_numpy(F).float()
d = Ft * sv.square()
perm = torch.from_numpy(np.lexsort((-F,)).copy())               # most sensitive first (headslab perm, ties ignored)
res = {"fisher": src, "V": V, "top_block_share": float(Ft[perm[:128]].sum() / Ft.sum()),
       "rows_50pct": int(np.searchsorted(np.cumsum(np.sort(F)[::-1]) / F.sum(), 0.5) + 1), "pred": {}}
nb = V // 128
strided = perm[:nb * 128].view(128, nb).T.reshape(-1)          # block b = sensitivity ranks b, b + nb, b + 2 nb, ...
ORD = {"vocab": torch.arange(V), "sorted": perm, "strided": strided}
for order, p in ORD.items():
    db = d[p[:nb * 128]].view(nb, 128).double()
    for so in (0.001, 0.01, 0.1, 1.0):
        dd = db / db.mean() + so
        R = float(torch.exp(torch.log(dd).mean(1)).sum() / dd.mean(1).sum())
        res["pred"][f"{order}/sigma_out={so:g}"] = R
        print(f"predictor {order:6s} sigma_out {so:<6g}: two-sided / one-sided damped-metric cost {R:.3f}", flush=True)
print(f"F: {src}; top block {res['top_block_share']:.1%} of F; 50% of F in {res['rows_50pct']} rows; {time.time() - t0:.0f}s", flush=True)

if a.emulate:
    import cpuq
    qm = cpuq.load(a.quant)
    with safe_open(a.hess, "pt") as f: up = f.get_tensor("hessian_upper").double(); meta = f.metadata()
    k = int(meta["k"]); H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up
    H = (H + H.triu(1).T).float() / int(meta["rows"])
    N = a.emulate; run = N // 64
    sets = {"vocab": torch.cat([torch.arange(s, s + run) for s in np.linspace(0, V - run, 64).astype(int)]),
            "sorted": perm[:N], "strided": strided[:N]}
    res["emulate"] = {}
    for name, rows in sets.items():
        Wn = W[rows].T.contiguous()                            # (5120, N): ourq's (in, out)
        f = Ft[rows]
        def cost(wq):
            E = Wn - wq; return float(((H @ E) * E).sum(0) @ f)
        for K in (2, 3):
            one = cost(cpuq.quant(qm, Wn, H * 1000, K=K)[0])
            for so in (0.01, 0.1):
                two = cost(cpuq.quant(qm, Wn, H * 1000, K=K, H_out=f, sigma_reg_out=so)[0])
                res["emulate"][f"{name}/K{K}/sigma_out={so:g}"] = two / one
                print(f"emulate {name:6s} K{K} sigma_out {so:<5g}: sum F err two-sided / one-sided {two / one:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
