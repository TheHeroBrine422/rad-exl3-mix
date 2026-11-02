#!/usr/bin/env python3
"""diaghalves.py SNAPDIR -- split-half agreement of the diagonal H_out on the 0.8B snapshots (mean files): halves
A = snap(n) and B = 2 snap(2n) - snap(n) are independent n-sequence estimates; cosine per linear, summarized by type.
Reference for the 27B collector's stopping rule (#33-17)."""
import glob, os, sys, numpy as np, torch
from safetensors import safe_open
S = sys.argv[1]
def diag(n, k):
    with safe_open(f"{S}/snap{n}/hout/{k}.safetensors", "pt") as f: return f.get_tensor("hout").diagonal().double()
keys = sorted(os.path.basename(f)[:-12] for f in glob.glob(f"{S}/snap8192/hout/*.safetensors"))
cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
for n in (1024, 2048, 4096):
    r = {}
    for k in keys:
        a = diag(n, k); b = 2 * diag(2 * n, k) - a; r[k] = cos(a, b)
    v = np.array(list(r.values())); t = {}
    for k, c in r.items(): t.setdefault(k.split("layers.")[1].split(".", 1)[1], []).append(c)
    print(f"halves of {n} seqs: all min {v.min():.5f} p5 {np.percentile(v, 5):.5f} median {np.median(v):.5f}; by type min/median: " +
          ", ".join(f"{ty} {min(c):.4f}/{np.median(c):.4f}" for ty, c in sorted(t.items())))
