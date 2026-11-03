#!/usr/bin/env python3
"""summ_synth.py DIR -- synthetic f32 files (constant KL per arm and seed) for a dry run of summ.py: DIR/{g,D,C}."""
import os, sys
import numpy as np
d = sys.argv[1]; n = 1000
for s in ("g", "D", "C"): os.makedirs(f"{d}/{s}", exist_ok=True)
gen = (np.arange(n) % 3 > 0).astype(np.uint8); gen.tofile(f"{d}/g/gen.u8")
def w(sub, tag, m):
    a = np.zeros((n, 4), np.float32); a[:, 0] = m; a[:, 2] = gen; a.tofile(f"{d}/{sub}/{tag}.f32")
for K, (gm, eb, b) in {3: ([0.088, 0.089, 0.087], [0.0905, 0.091, 0.0902], [0.0989, 0.0994, 0.0974]),
                       2: ([0.50, 0.51, 0.505], [0.503, 0.508, 0.509], [0.535, 0.541, 0.5225])}.items():
    for i, s in enumerate(["", "_s1", "_s2"]):
        w("D", f"k{K}gm{s}", gm[i]); w("C" if i else "D", f"k{K}eB{s}", eb[i]); w("C", f"k{K}base{s}", b[i])
