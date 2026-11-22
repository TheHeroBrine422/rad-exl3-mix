#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gn222.py MSE_GRADNOISE.npz KL_GRADNOISE.npz [--n 40] [--lo 0 --hi 31] -- the C1-MSE diagnostic (w222, w217's method): per scale (a trainable log multiplier of layers lo..hi) the share whose mean gradient over the
N batches is significant (|mean| > 1.96 * std / sqrt(N); pure noise gives ~5 %; also z 3), for the MSE run (--local-mse block, layers 0-31 only) and the KL-only run of w217 (`runs/gn-2.50/gradnoise.npz`, same 40 batches of
w214's order), overall and per block of layers, plus the lambda that makes the two gradients the same size: lambda = rms(KL mean gradient) / rms(MSE mean gradient) over the same scales (Adam is scale-free, so lambda
only matters when a KL term meets the MSE term)."""
import re, sys
import numpy as np
a = sys.argv[1:]; n = int(a[a.index("--n") + 1]) if "--n" in a else 40; lo = int(a[a.index("--lo") + 1]) if "--lo" in a else 0; hi = int(a[a.index("--hi") + 1]) if "--hi" in a else 31
def load(f):
    z = np.load(f); out = {}
    for k in z.files:
        if k.endswith("|mean"):
            nm = k[:-5]; m = re.search(r"\.layers\.(\d+)\.", nm)
            if m and lo <= int(m.group(1)) <= hi: out[nm] = (int(m.group(1)), z[k].astype(np.float64), z[nm + "|std"].astype(np.float64))
    return out
M, K = load(a[0]), load(a[1]); names = sorted(set(M) & set(K))
print(f"{len(names)} scale vectors in layers {lo}-{hi} (MSE run {len(M)}, KL run {len(K)})")
def share(D, sel, z):
    m = np.concatenate([D[x][1] for x in sel]); s = np.concatenate([D[x][2] for x in sel]); return float((np.abs(m) > z * s / np.sqrt(n)).mean()), float(np.sqrt((m ** 2).mean())), float(np.median(np.abs(m)) / max(np.median(s), 1e-30))
print("| layers | scales | MSE: significant (z 1.96 / z 3) | KL-only (w217): significant (z 1.96 / z 3) | rms mean grad MSE | rms mean grad KL | median |mean| / std MSE / KL |\n|---|---|---|---|---|---|---|")
for b0, b1 in [(lo, hi)] + [(x, min(x + 7, hi)) for x in range(lo, hi + 1, 8)]:
    sel = [x for x in names if b0 <= M[x][0] <= b1]
    if not sel: continue
    m196, rm, rr = share(M, sel, 1.96); m3, _, _ = share(M, sel, 3.0); k196, rk, kr = share(K, sel, 1.96); k3, _, _ = share(K, sel, 3.0)
    print(f"| {b0}-{b1} | {sum(len(M[x][1]) for x in sel)} | {m196 * 100:.1f} % / {m3 * 100:.1f} % | {k196 * 100:.1f} % / {k3 * 100:.1f} % | {rm:.3e} | {rk:.3e} | {rr:.3f} / {kr:.3f} |")
sel = names; _, rm, _ = share(M, sel, 1.96); _, rk, _ = share(K, sel, 1.96)
print(f"\nlambda (rms KL mean gradient / rms MSE mean gradient over layers {lo}-{hi}) = {rk / rm:.4g}   [MSE run was lambda = 1]")
