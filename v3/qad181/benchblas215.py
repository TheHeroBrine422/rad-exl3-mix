#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""benchblas215.py MODE -- w215 microbench of the trainer's bf16 GEMM layouts (fwd x @ W, dX dy @ W^T with W [k, n]
contiguous) under MODE = lt (hipBLASLt, torch's default here) | rocblas (preferred_blas_library cublas) | tunable
(TunableOp over both, tuned in-process); ms, TFLOP/s, max relative difference to a f32 reference."""
import sys, time

import torch

mode = sys.argv[1]
if mode == "rocblas": torch.backends.cuda.preferred_blas_library("cublas")
if mode == "tunable":
    torch.cuda.tunable.enable(True); torch.cuda.tunable.tuning_enable(True)
    torch.cuda.tunable.set_filename(f"/data/w215/runs/tunable215.csv")
print("blas", torch.backends.cuda.preferred_blas_library(), "tunable", torch.cuda.tunable.is_enabled(), flush=True)


def t(f, n=10):
    f(); f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n


torch.manual_seed(0)
for (k, n) in ((5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120)):
    W = torch.randint(-127, 128, (k, n), device="cuda").to(torch.bfloat16)
    for M in (1024, 694):
        for what in ("fwd", "dX"):
            B = W if what == "fwd" else W.t()
            x = torch.randn(M, B.shape[0], device="cuda").to(torch.bfloat16)
            ref = x.float() @ B.float()
            rel = float((x @ B).float().sub(ref).abs().max() / ref.abs().max())
            tt = t(lambda: x @ B)
            print(f"{mode} [{M}x{B.shape[0]}]@[{B.shape[0]}x{B.shape[1]}] {what}: {tt * 1e3:.3f} ms "
                  f"({2 * M * B.shape[0] * B.shape[1] / tt / 1e12:.0f} TF/s), max rel vs f32 {rel:.1e}", flush=True)
