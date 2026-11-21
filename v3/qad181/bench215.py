#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""bench215.py -- w215 microbench: int8 -> bf16 weight conversion, torch's .to(bfloat16) vs qmodel.i8_to_bf16 (Triton),
on the trunk's weight shapes; checks bitwise equality and prints GB/s (read 1 B + write 2 B per element)."""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qad"))
import torch  # noqa: E402
import qmodel  # noqa: E402


def t(f, n=20):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n


for shp in ((5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120)):
    w = torch.randint(-127, 128, shp, dtype=torch.int8, device="cuda")
    a, b = w.to(torch.bfloat16), qmodel.i8_to_bf16(w)
    assert torch.equal(a, b) and b.dtype == torch.bfloat16 and b.shape == w.shape
    ta, tb = t(lambda: w.to(torch.bfloat16)), t(lambda: qmodel.i8_to_bf16(w))
    gb = w.numel() * 3 / 1e9
    print(f"{shp}: torch {ta * 1e3:.3f} ms ({gb / ta:.0f} GB/s), triton {tb * 1e3:.3f} ms ({gb / tb:.0f} GB/s), equal")
