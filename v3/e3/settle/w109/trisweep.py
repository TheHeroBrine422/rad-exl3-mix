# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""trisweep.py RXQ_PIPELINE -- #144: rxq.swjtri._grid_sums at other BJ / BC / num_warps (BT 16 kept): identity + time"""
import itertools, sys, time

import torch
import triton

sys.path.insert(0, sys.argv[1])
from rxq import swjtri

torch.manual_seed(0); J = 17408; grid = torch.arange(0.6, 1.4 + 0.005, 0.01, device="cuda"); NG = len(grid); T = 1280
g = torch.randn(T, J, device="cuda"); u = torch.randn(T, J, device="cuda"); a = torch.nn.functional.silu(g * 1.03) * u * 0.98


def run(BJ, BC, nw):
    sap = torch.empty(NG, J, device="cuda"); spp = torch.empty_like(sap)
    swjtri._grid_sums[(triton.cdiv(J, BJ), triton.cdiv(NG, BC))](a, g, u, grid, sap, spp, T, J, NG, BT=16, BJ=BJ, BC=BC, num_warps=nw)
    return sap, spp


def tm(f):
    y = f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5): y = f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / 5, y


t0, y0 = tm(lambda: swjtri.grid_sums(a, g, u, grid))
print(f"today (BJ 64 BC 8 nw default): {t0 * 1e3:.2f} ms")
for BJ, BC, nw in itertools.product((32, 64, 128), (4, 8, 16), (2, 4, 8)):
    try:
        t, y = tm(lambda: run(BJ, BC, nw))
        print(f"BJ {BJ:3d} BC {BC:2d} nw {nw}: {t * 1e3:.2f} ms x{t0 / t:.2f} identical {all(torch.equal(p, q) for p, q in zip(y, y0))}", flush=True)
    except Exception as e:
        print(f"BJ {BJ} BC {BC} nw {nw}: failed {str(e)[:60]}")
