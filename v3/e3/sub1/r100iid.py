#!/usr/bin/env python3
"""r100iid.py [NTILES] [K,...] -- #100 side check (codebook only, no LDLQ): plain bitshift-trellis MSE on iid N(0,1)
16x16 tiles for mul1_i8, iqrh (IQR's A / B) and iqrh with other odd 24-bit multipliers A (B fixed), best of a small
scale grid. Shows whether IQR's constant is typical among 24-bit multipliers (i.e. whether a tuned global A could
close a gap). Seconds a K on a few cores."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import trellis, trellis4

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
KS = [float(k) for k in (sys.argv[2] if len(sys.argv) > 2 else "2,4").split(",")]
x = np.random.default_rng(0).standard_normal((n, 256)).astype(np.float32)
rng = np.random.default_rng(1)
cbs = ["mul1_i8", "iqrh"] + [f"iqrh:{a:#x}:{trellis4.IQR_B:#x}" for a in (rng.integers(1 << 22, 1 << 24, 8) | 1)]
for K in KS:
    res = []
    for name in cbs:
        cb = np.ascontiguousarray(trellis4.codebook(name), dtype=np.float32)
        best = min(float(((trellis.vit(x / f, cb, K) * f - x) ** 2).mean()) for f in (0.9, 1.0, 1.1))
        res.append((name, best))
    ref = res[0][1]
    print(f"K{K:g}: " + "  ".join(f"{nm} {m:.5f} ({m / ref:.3f})" for nm, m in res), flush=True)
