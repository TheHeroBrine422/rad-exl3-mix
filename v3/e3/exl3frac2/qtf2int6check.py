#!/usr/bin/env python3
"""qtf2int6check.py SRC N -- qtf2 at integer K (KA = K, MASK 0) vs the integer kernels the ladder and the heads use:
exl3cb i8 (EXL3_QT_OPTIMIZED=1, trunk levels) and exllamav3's prebuilt mul1 (head donors), K1-4. Indices and values
must be identical for a K to move to qtf2; tiles / s of both. One JSON line per (codebook, K)."""
import json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "exl3frac"), os.path.join(HERE, "..", "exl3cb")]
import exl3frac2, exl3cb
import exllamav3.modules.quant.exl3_lib.quantize as Q
SRC, N = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0); d = torch.device("cuda", 0)
T = torch.cat([torch.randn(N, 256, device=d), torch.distributions.Laplace(0., 0.7).sample((N // 4, 256)).to(d),
               torch.round(torch.randn(N // 8, 256, device=d) * 2) / 2, torch.zeros(8, 256, device=d),
               torch.full((8, 256), 3.5, device=d), torch.randn(8, 256, device=d) * 12]).contiguous()
def tput(fn, n):
    fn(); torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); return n / (time.perf_counter() - t)
def qtf2(ext, X, ka):
    em = 65536 >> ka; oq, oi = torch.zeros_like(X), torch.zeros_like(X, dtype=torch.short)
    tc = torch.empty(128, 2, em, dtype=torch.half, device=d); te = torch.empty(128, 256, em, dtype=torch.short, device=d)
    ext.quantize_tiles_frac(X, oq, oi, tc, te, ka, 0); return oq, oi
fails = 0
for cb in ("mul1", "i8"):
    if cb == "i8": exl3cb.install(SRC, "i8")         # integer reference: the ladder's kernel; mul1: the prebuilt one
    ext = exl3frac2.load(SRC, cb)
    for K in (1, 2, 3, 4):
        rq, ri = Q.quantize_tiles(T, {"K": K, "mul1": True}); nq, ni = qtf2(ext, T, K)
        di, dv = int((ri != ni).sum()), int((rq != nq).sum()); fails += di + dv
        T2 = T[:2048].contiguous()
        print(json.dumps({"check": "int6", "cb": cb, "K": K, "idx_diff": di, "val_diff": dv,
                          "int_tps": round(tput(lambda: Q.quantize_tiles(T2, {"K": K, "mul1": True}), len(T2))),
                          "qtf2_tps": round(tput(lambda: qtf2(ext, T2, K), len(T2)))}), flush=True)
print(json.dumps({"check": "int6_summary", "fails": fails}), flush=True)
