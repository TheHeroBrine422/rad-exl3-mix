#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""benchdx215.py -- w215 microbench: the trainer's dX GEMM (dy [M, n] times W^T, W [k, n] bf16 contiguous) as torch
computes it (dy @ W.t()) vs the transposed product (W @ dy.t()).t() (+ the copy back to [M, k] row-major) vs dy @ W^T
held contiguous; ms, TFLOP/s, max relative difference to the current form."""
import time

import torch


def t(f, n=10):
    f(); f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n


torch.manual_seed(0)
for (k, n) in ((5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120), (5120, 6144)):
    W = torch.randint(-127, 128, (k, n), device="cuda").to(torch.bfloat16); Wt = W.t().contiguous()
    for M in (1024, 694, 1105, 256):
        dy = torch.randn(M, n, device="cuda").to(torch.bfloat16)
        ref = dy @ W.t()
        fl = 2 * M * k * n
        alt = (W @ dy.t()).t().contiguous()
        rel = float((alt.float() - ref.float()).abs().max() / ref.float().abs().max())
        ta = t(lambda: dy @ W.t()); tb = t(lambda: (W @ dy.t()).t().contiguous()); tc = t(lambda: dy @ Wt)
        print(f"dX M {M} k {k} n {n}: dy@W.t() {ta * 1e3:.3f} ms ({fl / ta / 1e12:.0f} TF/s), (W@dy.t()).t() {tb * 1e3:.3f} ms "
              f"({fl / tb / 1e12:.0f}), dy@Wt {tc * 1e3:.3f} ms ({fl / tc / 1e12:.0f}); max rel {rel:.1e}", flush=True)
