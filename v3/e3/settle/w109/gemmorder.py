# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gemmorder.py -- #144: which accumulation order gives torch's f32 GEMM bits (x bf16-valued f32 [T, 5120] @ W f32
[17408, 5120].T on the card)? Emulated on the CPU for a few outputs: sequential FMA, mul-then-add, A interleaved
accumulators (A = 2 .. 64, combined in order or as a tree), K chunks of B (inner sequential FMA). Synthetic data."""
import math, sys

import numpy as np
import torch

torch.manual_seed(0); K, N, T = 5120, 17408, 4
x = torch.randn(T, K).bfloat16().float(); W = torch.randn(N, K) * 0.02
y = (x.cuda() @ W.cuda().T).cpu()
f32 = np.float32
pts = [(0, 0), (1, 5), (2, 17000), (3, 999), (0, 4096), (1, 12345), (2, 1), (3, 17407)]


def fma32(a, b, c): return f32(math.fma(float(a), float(b), float(c)))


def seq(xr, wr, k0=0, k1=K, step=1):
    acc = f32(0)
    for k in range(k0, k1, step): acc = fma32(xr[k], wr[k], acc)
    return acc


def mul_add(xr, wr):
    acc = f32(0)
    for k in range(K): acc = f32(acc + f32(xr[k] * wr[k]))
    return acc


def inter(xr, wr, A, tree):
    parts = [seq(xr, wr, a, K, A) for a in range(A)]
    if not tree:
        s = parts[0]
        for p in parts[1:]: s = f32(s + p)
        return s
    while len(parts) > 1: parts = [f32(parts[i] + parts[i + 1]) for i in range(0, len(parts), 2)]
    return parts[0]


def chunks(xr, wr, B):
    s = f32(0)
    for c in range(0, K, B): s = f32(s + seq(xr, wr, c, c + B))
    return s


cands = {"seq_fma": lambda a, b: seq(a, b), "mul_add": mul_add}
for A in (2, 4, 8, 16, 32, 64):
    cands[f"inter{A}"] = lambda a, b, A=A: inter(a, b, A, False); cands[f"inter{A}tree"] = lambda a, b, A=A: inter(a, b, A, True)
for B in (8, 16, 32, 64, 128, 256, 512, 1024, 2560):
    cands[f"chunk{B}"] = lambda a, b, B=B: chunks(a, b, B)
xs, ws = x.numpy(), W.numpy()
for name, f in cands.items():
    hit = sum(f(xs[i], ws[j]) == f32(y[i, j]) for i, j in pts)
    print(f"{name:12s} {hit}/{len(pts)} equal", flush=True)
