#!/usr/bin/env python3
"""timing71.py K OUT.jsonl -- seconds per tile and run of each (L, M) search at K (tail-biting = 2 runs a tile)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "/data/work/w71/code/settle/w71"); sys.path.insert(0, "/data/work/w71/code")
import trellisL as T
K, out = int(sys.argv[1]), sys.argv[2]
X = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32) * 0.9
for L, M, n in [(16, 0, 32), (18, 0, 16), (20, 0, 4), (16, 256, 32), (16, 1024, 16), (16, 4096, 8), (18, 1024, 16),
                (18, 4096, 8), (20, 256, 32), (20, 1024, 16), (20, 4096, 8)]:
    cb = T.codebook_L("mul1_i8", L)
    t0 = time.time(); T.search(X[:n], cb, L, K, M, 1); dt = (time.time() - t0) / n
    o = {"K": K, "L": L, "M": M, "sec_tile_tb": dt}; print(json.dumps(o), flush=True)
    open(out, "a").write(json.dumps(o) + "\n")
