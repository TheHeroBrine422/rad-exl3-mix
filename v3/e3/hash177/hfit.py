#!/usr/bin/env python3
"""hfit.py A B IB K NTILES SEED ITERS OUT.npy [L] [TARGETS.npy] -- #177's HYB codebook (QTIP-style hash -> LUT, one value a
state): value(s) = LUT[top IB bits of (s * A + B) mod 2^32], the 2^IB LUT fitted Lloyd-style at rate K: Viterbi with the
current LUT over the training tiles (iid N(0,1), seed SEED, or TARGETS [n, 256] float32, e.g. real rotated weight tiles
at unit RMS), then each entry = the mean of the targets coded through it; ITERS rounds. Saves the float LUT to OUT.npy and
its int8-grid copy (step max|LUT| / 127) to OUT-i8.npy; prints the training MSE a round. (w139)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hscreen

A, B, IB, K = int(sys.argv[1], 0), int(sys.argv[2], 0), int(sys.argv[3]), int(sys.argv[4])
n, seed, iters, out = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
L = int(sys.argv[9]) if len(sys.argv) > 9 else 16
x = np.load(sys.argv[10]).astype(np.float32)[:n] if len(sys.argv) > 10 else hscreen.tiles(n, seed)
idx = (hscreen.hashv(A, B, L) >> np.uint64(32 - IB)).astype(np.int64)
lut = hscreen.gauss_lut(IB)   # hscreen's unfitted hyb table
for it in range(iters):
    q, st = hscreen.vit(x, lut[idx], K, L, states=True)
    e = idx[st.ravel()]
    s = np.bincount(e, weights=x.ravel().astype(np.float64), minlength=1 << IB)
    c = np.bincount(e, minlength=1 << IB)
    print(json.dumps({"it": it, "mse": float(((q - x) ** 2).mean()), "unused": int((c == 0).sum())}), flush=True)
    lut = np.where(c > 0, s / np.maximum(c, 1), lut)
np.save(out, lut)
st8 = np.abs(lut).max() / 127
np.save(out[:-4] + "-i8.npy", np.round(lut / st8) * st8)
