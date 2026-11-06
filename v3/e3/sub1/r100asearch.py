#!/usr/bin/env python3
"""r100asearch.py NTILES KS TILESEED (rand:SEED:COUNT | A,A,...) OUT.jsonl -- #100 side search (codebook only, no
weights, no calib): a global multiplier A for IQR's computed code, A an odd 24-bit constant (the decode's
v_mad_u32_u24 takes any A < 2^24 at the same cost), B fixed to IQR's. Score: plain bitshift-trellis MSE on NTILES iid
N(0,1) 16x16 tiles (seed TILESEED), best of scale factors 0.9 / 1.0 / 1.1, per K in KS; mul1_i8 and IQR's own A are
scored on the same tiles first (ratio = MSE / mul1_i8's). One JSON line per (A, K). One thread: OMP_NUM_THREADS=1."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import trellis, trellis4

n, KS, ts, spec, out = int(sys.argv[1]), [float(k) for k in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4], sys.argv[5]
x = np.random.default_rng(ts).standard_normal((n, 256)).astype(np.float32)
if spec.startswith("rand:"):
    _, sd, cnt = spec.split(":"); A = [int(a) for a in (np.random.default_rng(int(sd)).integers(1 << 20, 1 << 24, int(cnt)) | 1)]
else:
    A = [int(a, 0) for a in spec.split(",")]


def mse(cb, K):
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    return min(float(((trellis.vit(x / f, cb, K) * f - x) ** 2).mean()) for f in (0.9, 1.0, 1.1))


ref = {K: mse(trellis4.codebook("mul1_i8"), K) for K in KS}
with open(out, "a") as fh:
    for a in [trellis4.IQR_A] + A:
        for K in KS:
            m = mse(trellis4.codebook(f"iqrh:{a:#x}:{trellis4.IQR_B:#x}"), K)
            fh.write(json.dumps({"A": a, "B": trellis4.IQR_B, "K": K, "mse": m, "ratio": m / ref[K], "ntiles": n,
                                 "tileseed": ts, "iqr": a == trellis4.IQR_A}) + "\n"); fh.flush()
