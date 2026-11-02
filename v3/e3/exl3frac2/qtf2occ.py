#!/usr/bin/env python3
"""qtf2occ.py SRC OUT.jsonl -- qtf2 throughput by tiles per call (the LDLQ calls one 16-row block at a time: 64 / 320 /
384 / 640 / 768 / 1088 tiles for this model's widths) and by launch batch (get_temp_buffers_frac: 128), K2.5 (i8).
Shows the per-call and per-launch tails; synchronised after every call, as ldlq's next block depends on it."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3frac2
SRC, OUT = sys.argv[1:3]
ext = exl3frac2.load(SRC, "i8"); ka, mask = exl3frac2.frac_k(2.5); em = 65536 >> ka
out = open(OUT, "a")
for bs in (128, 256, 1088):
    tc = torch.empty(bs, 2, em, dtype=torch.half, device="cuda"); te = torch.empty(bs, 256, em, dtype=torch.short, device="cuda")
    for n in (64, 128, 320, 384, 640, 768, 1088, 2176):
        T = torch.randn(n, 256, device="cuda"); oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
        reps = max(2, 4096 // n)
        ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps): ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask); torch.cuda.synchronize()
        r = {"batch": bs, "tiles_per_call": n, "tps": round(n * reps / (time.perf_counter() - t))}
        print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n")
    del tc, te; torch.cuda.empty_cache()
