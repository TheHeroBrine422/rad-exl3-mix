#!/usr/bin/env python3
"""speed.py TREE SRC VARIANT [BATCHES] -- #142 speed: tiles/s of every KA = 0 instance of the exl3frac2 build in TREE
(qtf2sub1.py's method: 1,024 Gaussian tiles, one warm call then timed calls; best of 3), by launch batch.
One JSON line per (K, batch). Env as fusecheck.py (W108_STUB, PYTORCH_ROCM_ARCH, QTF2_DEFS, QTF2_LUT)."""
import json, os, sys, time
sys.path[:0] = [os.environ["W108_STUB"], sys.argv[1]]
import torch
import exl3frac2
ext = exl3frac2.load(sys.argv[2], sys.argv[3])
batches = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "32,64,128,256").split(",")]
ks = [float(x) for x in os.environ.get("W108_K", "0.125,0.25,0.375,0.5,0.625,0.75,0.875").split(",")]
torch.manual_seed(0)
T = torch.randn(1024, 256, device="cuda")
oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
for bs in batches:
    tc = torch.empty(bs, 2, 65536, dtype=torch.half, device="cuda")
    te = torch.empty(bs, 256, 65536, dtype=torch.short, device="cuda")
    for K in ks:
        ka, mask = exl3frac2.frac_k(K)
        ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask); torch.cuda.synchronize()
        best = 0.0
        for _ in range(3):
            t = time.perf_counter(); ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask); torch.cuda.synchronize()
            best = max(best, len(T) / (time.perf_counter() - t))
        print(json.dumps({"tree": os.path.basename(sys.argv[1].rstrip("/")), "cb": sys.argv[3], "defs": os.environ.get("QTF2_DEFS", ""),
                          "lut": os.environ.get("QTF2_LUT", "1"), "K": K, "batch": bs, "tps": round(best)}), flush=True)
    del tc, te; torch.cuda.empty_cache()
