#!/usr/bin/env python3
"""qtf2sub1.py SRC -- sub-1 (KA 0) throughput by launch batch (its costs live in global, 2 blocks a CU: the batch sets
the occupancy). K0.5, i8, 1024 tiles."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3frac2
ext = exl3frac2.load(sys.argv[1], "i8"); ka, mask = exl3frac2.frac_k(0.5)
T = torch.randn(1024, 256, device="cuda"); oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
for bs in (32, 64, 128, 256):
    tc = torch.empty(bs, 2, 65536, dtype=torch.half, device="cuda"); te = torch.empty(bs, 256, 65536, dtype=torch.short, device="cuda")
    ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask); torch.cuda.synchronize(); t = time.perf_counter()
    ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask); torch.cuda.synchronize()
    print(json.dumps({"K": 0.5, "batch": bs, "tps": round(1024 / (time.perf_counter() - t))}), flush=True)
    del tc, te; torch.cuda.empty_cache()
