#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cos218.py GDIR GDIR [GDIR ...] -- #218: the honest gradient-SNR check, independent of the step-level variance estimate (steps of one
document are correlated): per probe tensor, the cosine between the gradients of disjoint document chunks (at the same codes). For
chunk gradients G_i = g + n_i (n_i independent), E cos = |g|^2 / (|g|^2 + |n|^2): the signal share of ONE chunk; the merged mean of
c chunks then has signal share |g|^2 / (|g|^2 + |n|^2 / c)."""
import itertools, json, os, sys

import numpy as np

dirs = sys.argv[1:]
gm = json.load(open(os.path.join(dirs[0], "grad.json")))["tensors"]
allc = []
for k, v in sorted(gm.items()):
    G = [np.fromfile(os.path.join(d, k + ".g"), np.float32).astype(np.float64).ravel() for d in dirs]
    cs = [float(a @ b / np.sqrt((a @ a) * (b @ b))) for a, b in itertools.combinations(G, 2)]
    allc.append(np.mean(cs))
    print(f"{k.split('layers.')[1]:30s} cos {' '.join(f'{c:.3f}' for c in cs)}  mean {np.mean(cs):.3f}", flush=True)
c = float(np.mean(allc)); n = len(dirs)
print(f"mean cos over tensors {c:.4f} -> signal share of one chunk {c:.3f}, of the {n}-chunk mean {c / (c + (1 - c) / n):.3f}")
