#!/usr/bin/env python3
"""strata94.py DATA -- REPORT-ONLY (after #125's G2 failed): per-stratum arm ratios vs B. Each stratum's KL (only that
stratum's sampled rows changed) is exact for those rows, so the ratios need no cross-stratum estimate."""
import sys, numpy as np
D = sys.argv[1]
def gm(s): return np.fromfile(f"{D}/meta/{s}/gen.u8", np.uint8).astype(bool), np.fromfile(f"{D}/meta/{s}/doc.i32", np.int32)
M = {s: gm(s) for s in ("search", "qad")}
SETS = {"search-gen": ("search", M["search"][0]), "qad-even": ("qad", M["qad"][0] & (M["qad"][1] % 2 == 0))}
print("arm / B per stratum (A = blocks 0-15 all, B = 24 of 16-255, C = 24 of 256-1939); real = today's GPU donor (no mean-row) / C0")
for K in (3, 4):
    for tag in ("real_one", "raw_one", "mr_two0.01", "mr_two0.1", "mr_two1", "mr-r4_one", "mr-r8_one"):
        out = []
        for rs, (s, m) in SETS.items():
            ref = "raw_one" if tag == "real_one" else "mr_one"
            z, b = np.load(f"{D}/kl/{s}/{tag}_K{K}.npz"), np.load(f"{D}/kl/{s}/{ref}_K{K}.npz")
            out.append(rs + " " + " ".join(f"{st} {z['k' + st][m].sum() / b['k' + st][m].sum():.3f}" for st in "ABC"))
        print(f"K{K} {tag:11s} vs {'C0' if tag == 'real_one' else 'B '}: " + " | ".join(out))
