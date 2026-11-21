#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""chkstu214.py GOLDEN STUDENT.npz RADIANCE.rows -- #214's loader check: the torch student (tscore191.py, A8 on, the released container's decoded pieces)
vs the served model (radiance rows file) on the same gen rows: means, per-row difference (max, p99.9, rms), correlation, per-document means."""
import json, sys
import numpy as np
G, NPZ, ROWS = sys.argv[1:4]
m = json.load(open(f"{G}/kld.json")); gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool)
K = np.fromfile(ROWS, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
z = np.load(NPZ); kl, doc, pos = z["kl"], z["doc"], z["pos"]
row = np.array([m["docs"][d]["row0"] + p - m["docs"][d]["score_from"] for d, p in zip(doc, pos)])
assert gen[row].all() and gen.sum() == len(row)
r = K[row]; d = kl - r
print(f"rows {len(row)}: student mean KL {kl.mean():.6f}, radiance {r.mean():.6f}, ratio {kl.mean() / r.mean():.4f}")
print(f"per-row student - radiance: mean {d.mean():+.2e}, rms {np.sqrt((d ** 2).mean()):.4f}, max |d| {np.abs(d).max():.3f}, p99.9 |d| {np.percentile(np.abs(d), 99.9):.3f}; corr {np.corrcoef(kl, r)[0, 1]:.5f}")
ds = [(kl[doc == i].mean(), r[doc == i].mean()) for i in np.unique(doc)]
a = np.array(ds); print(f"per-document means ({len(a)} docs): corr {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.5f}, max ratio dev {np.abs(a[:, 0] / a[:, 1] - 1).max():.3f}, median ratio {np.median(a[:, 0] / a[:, 1]):.4f}")
