#!/usr/bin/env python3
"""mkblocks94.py OUT.json -- hq94.py's stratified block sample (sensitivity order, 1,940 blocks of 128 rows), fixed
before data. A = blocks 0-15 (all; ~65% of the Fisher-weighted K3 error), B = 24 of blocks 16-255 (systematic, random
offset), C = 24 of blocks 256-1939 (systematic, random offset). Estimate = KL_A + wB KL_B + wC KL_C."""
import json, sys, numpy as np
rng = np.random.default_rng(9494); NB = 248320 // 128
A = list(range(16))
oB = rng.uniform(); B = [16 + int((i + oB) * 240 / 24) for i in range(24)]
oC = rng.uniform(); C = [256 + int((i + oC) * (NB - 256) / 24) for i in range(24)]
assert len(set(A + B + C)) == 64 and max(C) < NB
json.dump({"A": A, "B": B, "C": C, "all": A + B + C, "wB": 240 / 24, "wC": (NB - 256) / 24, "seed": 9494}, open(sys.argv[1], "w"))
print(B, C)
