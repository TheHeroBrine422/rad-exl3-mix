#!/usr/bin/env python3
"""gscale.py LAYER TENSOR OUT.jsonl [ROWS] -- would per-group weight scales help EXL3 at low K? Hessian-weighted (CPU).

Recheck of the Oct-30 "per-group gain scales" null result (amgm.py: unweighted, high-rate formula, K <= 2 untested),
for V3's sub-2 menu. Same setup as trellis.py (ROWS sampled output rows, input dim 128-Hadamard rotated, per-row RMS
scale, global factor searched, block LDLQ with the bitshift trellis on mul1), plus a variant where each (row, G input
columns) group gets its own scale: at the group's first 16-column block the group's RMS in the *current*
(feedback-updated) weight is taken, rounded to an 8-bit log grid (steps of 2^(1/16)), and the group's tiles are
quantized as s * trellis(x / s). Error metric: the hproxy relative error tr(dW H dW^T) / tr(W H W^T) on the same rows.

Per K: baseline (no groups) at K0.5 ... K2.25 in quarter steps, group variants (G = 64, 128) at K0.5 / 1 / 1.5 / 2.
Overhead: 8 / G bit per weight (0.125 / 0.0625). report: gscalereport.py (group rel vs the baseline interpolated at
K + overhead: < 1 means groups win at equal bits). Env OMP_NUM_THREADS, E3_CKPT, E3_HESS."""
import json, math, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3
import trellis as T

layer, tensor, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ROWS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BASE = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
GRP = [(g, k) for g in (64, 128) for k in (0.5, 1, 1.5, 2)]
FAC = (0.85, 1.0, 1.15)


def q8(s):   # 8-bit log-grid scale: 2^(n/16), n integer
    return torch.exp2(torch.round(torch.log2(s.clamp(min=1e-12)) * 16) / 16)


def ldlq_group(W, H, cb, K, G, damp=0.01):
    """trellis.ldlq_trellis with a scale per (row, G columns), taken from the feedback-updated weight"""
    W = W.clone(); R, Kd = W.shape
    H = H.clone(); dead = H.diagonal() == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W); s = None
    for a in range(0, Kd, 16):
        if a % G == 0:
            s = q8(W[:, a:a + G].pow(2).mean(1, keepdim=True).sqrt())
        blk = W[:, a:a + 16]
        tiles = (blk / s).reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()
        q = torch.from_numpy(T.vit(tiles, cb, K).astype(np.float64)).reshape(R, 16) * s
        Q[:, a:a + 16] = q
        if a + 16 < Kd:
            E = (blk - q) @ torch.linalg.inv(U[a:a + 16, a:a + 16])
            W[:, a + 16:] -= E @ U[a:a + 16, a + 16:]
    return Q


names, hn = e3.ROUTE[tensor]
W0 = e3.load_w(layer, names); H0 = e3.load_h(layer, hn)
N, Kd = W0.shape
g = torch.Generator().manual_seed(7 + layer)
idx = torch.randperm(N, generator=g)[:ROWS].sort()[0]
rk = e3.had_blocks(Kd, 1000 + layer)
Wr = rk(W0[idx]); Hr = rk(rk(H0).T).T
rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
cb = np.ascontiguousarray(T.codebook("mul1"), dtype=np.float32)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l); done.add((d["G"], d["bits"]))
for G, K in [(0, k) for k in BASE] + GRP:
    if (G, K) in done: continue
    t0 = time.time()
    Wn = torch.cat([Wr / (rms * f) for f in FAC], 0)
    Qs = T.ldlq_trellis(Wn, Hr, cb, K) if G == 0 else ldlq_group(Wn, Hr, cb, K, G)
    best = min((e3.rel(Wr, Qs[i * ROWS:(i + 1) * ROWS] * rms * f, Hr), f) for i, f in enumerate(FAC))
    o = {"layer": layer, "w": tensor, "G": G, "bits": K, "overhead": 8 / G if G else 0.0, "rel": best[0],
         "scale_f": best[1], "rows": ROWS, "sec": round(time.time() - t0, 1)}
    print(json.dumps(o), flush=True)
    with open(OUT, "a") as fh: fh.write(json.dumps(o) + "\n")
