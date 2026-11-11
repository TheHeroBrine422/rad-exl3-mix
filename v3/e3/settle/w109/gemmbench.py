# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gemmbench.py [OUT.json] -- #144: swjfit's hook GEMM (x f32 [T, 5120] from bf16 @ W f32 [17408, 5120].T) at the
27B shape: time and bit identity (torch.equal vs today's call) of layouts / merged N / stacked documents / BLAS
backends. Synthetic data (randn bf16 x, f32 W)."""
import json, sys, time

import torch

dev = torch.device("cuda"); torch.manual_seed(0)
K, N = 5120, 17408
W = torch.randn(N, K, device=dev) * 0.02; W2 = torch.randn(N, K, device=dev) * 0.02
Wt = W.T.contiguous(); WW = torch.cat([W, W2]); Wb = W.bfloat16()
res = {"backend_default": str(torch.backends.cuda.preferred_blas_library()), "rows": []}


def tm(f, n=5):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): y = f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n, y


def row(name, T, sec, same):
    r = {"case": name, "T": T, "ms": round(sec * 1e3, 3), "tflops": round(2 * T * K * N / sec / 1e12, 2), "identical": same}
    res["rows"].append(r); print(r, flush=True)


for T in (400, 1280, 4096):
    x = torch.randn(T, K, device=dev).bfloat16().float()
    s0, y0 = tm(lambda: x @ W.T); row("today x@W.T", T, s0, True)
    s, y = tm(lambda: x @ Wt); row("x@Wt (K,N contiguous)", T, s, torch.equal(y, y0))
    s, y = tm(lambda: x @ WW.T); row("merged N=2x (per GEMM)", T, s / 2, torch.equal(y[:, :N], y0))
    s, y = tm(lambda: torch.mm(x, W.t())); row("torch.mm", T, s, torch.equal(y, y0))
    xs = torch.cat([x, torch.randn(1000, K, device=dev).bfloat16().float()])
    s, y = tm(lambda: xs @ W.T); row("stacked docs M+1000 (per row x T)", T, s * T / xs.shape[0], torch.equal(y[:T], y0))
    s, y = tm(lambda: (x.bfloat16() @ Wb.T).float()); row("bf16 GEMM (NOT same math: W rounded)", T, s, torch.equal(y, y0))
    for lib in ("hipblaslt", "rocblas", "cublas", "cublaslt", "ck"):
        try:
            torch.backends.cuda.preferred_blas_library(lib)
            s, y = tm(lambda: x @ W.T); row(f"backend {lib}", T, s, torch.equal(y, y0))
        except Exception as e:
            print(lib, "n/a", str(e)[:100])
    torch.backends.cuda.preferred_blas_library(res["backend_default"].split(".")[-1].lower())
if len(sys.argv) > 1: json.dump(res, open(sys.argv[1], "w"), indent=1)
