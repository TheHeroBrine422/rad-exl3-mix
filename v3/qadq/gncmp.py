#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gncmp.py NAME=RUNDIR/gradnoise.npz [NAME=...] [--n 40] -- signal vs noise of the KL gradient per layer block for several start
states (lead 11-22): per scale (a trainable log multiplier), significant when |mean| > z * std / sqrt(n) (pure noise: 5 % at z 1.96,
0.27 % at z 3). Prints, per block of 16 layers and overall: share significant (z 1.96 / z 3), rms mean gradient, median |mean| / std."""
import re, sys
import numpy as np
a = sys.argv[1:]; n = int(a[a.index("--n") + 1]) if "--n" in a else 40
runs = [x.split("=", 1) for x in a if "=" in x]
BL = [(0, 15), (16, 31), (32, 47), (48, 63), (0, 63)]
print("| state | " + " | ".join(f"L{b0}-{b1}: sig 1.96 / 3 · rms g · med z" for b0, b1 in BL) + " |")
print("|---|" + "---|" * len(BL))
for name, f in runs:
    z = np.load(f); D = []
    for k in z.files:
        if not k.endswith("|mean"): continue
        m = re.search(r"\.layers\.(\d+)\.", k)
        if m: D.append((int(m.group(1)), z[k].astype(np.float64), z[k[:-5] + "|std"].astype(np.float64)))
    cells = []
    for b0, b1 in BL:
        sel = [d for d in D if b0 <= d[0] <= b1]
        if not sel: cells.append("-"); continue
        mu = np.concatenate([d[1] for d in sel]); sd = np.concatenate([d[2] for d in sel]); t = np.abs(mu) / np.maximum(sd / np.sqrt(n), 1e-30)
        cells.append(f"{(t > 1.96).mean() * 100:.1f} % / {(t > 3).mean() * 100:.2f} % · {np.sqrt((mu ** 2).mean()):.2e} · {np.median(t):.2f}")
    print(f"| {name} | " + " | ".join(cells) + " |")
