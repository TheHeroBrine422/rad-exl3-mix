#!/usr/bin/env python3
"""fitlevels.py CONTAINER OFFSET ROWS COLS BITS... -- Lloyd-Max codebooks for the token table under fwht128 and a
scale per 128 (each group normalised by its absmax), fitted on every 160th row; one line a width, `B v0;v1;..`,
ready for rtn's table=."""
import sys
import numpy as np
path, off, rows, cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
H = np.array([[1.0]])
while H.shape[0] < 128:
    H = np.block([[H, H], [H, -H]])
mm = np.memmap(path, np.uint16, "r", offset=off, shape=(rows, cols))
w = (np.asarray(mm[::160]).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
v = (w.reshape(-1, cols // 128, 128) @ H / 128)
x = np.sort((v / np.maximum(np.abs(v).max(2, keepdims=True), 1e-30)).ravel())
for B in map(int, sys.argv[5:]):
    n = 2 ** B
    L = np.quantile(x, (np.arange(n) + 0.5) / n)
    for _ in range(50):
        e = (L[1:] + L[:-1]) / 2
        i = np.searchsorted(e, x)
        s = np.bincount(i, x, n); c = np.bincount(i, None, n)
        L = np.where(c > 0, s / np.maximum(c, 1), L)
    print(B, ";".join(f"{t:.7g}" for t in L))
