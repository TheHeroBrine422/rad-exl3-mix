#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""benchmm215.py -- w215 microbench: is a fused int8-weight x bf16-activation Triton GEMM (dequant in the kernel, f32
accumulate) on gfx1201 faster than (int8 -> bf16 conversion) + hipBLASLt bf16 GEMM? Trunk shapes, forward (x @ W) and
dX (dy @ W^T); prints ms, TFLOP/s and the max relative difference to torch.mm on the converted weight."""
import os, sys, time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qad"))
import qmodel  # noqa: E402

CFG = [triton.Config({"BM": bm, "BN": bn, "BK": bk, "G": 8}, num_warps=w, num_stages=s)
       for bm, bn, bk, w, s in ((128, 128, 32, 4, 2), (128, 128, 64, 8, 2), (128, 256, 32, 8, 2), (256, 128, 32, 8, 2),
                                (128, 128, 32, 8, 2), (64, 128, 64, 4, 2), (128, 64, 64, 4, 2), (256, 256, 32, 8, 2))]


@triton.autotune(configs=CFG, key=["M", "N", "K", "sbk", "sbn"])
@triton.jit
def _mm_i8(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (G * npn); fm = gid * G; gs = min(npm - fm, G)
    pm = fm + (pid % gs); pn = (pid % (G * npn)) // gs
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = a + rm[:, None] * sam + rk[None, :] * sak
    bp = b + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        km = rk < K - k * BK
        x = tl.load(ap, mask=(rm[:, None] < M) & km[None, :], other=0.0)
        w = tl.load(bp, mask=km[:, None] & (rn[None, :] < N), other=0).to(tl.bfloat16)
        acc = tl.dot(x, w, acc)
        ap += BK * sak; bp += BK * sbk
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def mm_i8(x, w):
    M, K = x.shape; N = w.shape[1]
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    grid = lambda m: (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)
    _mm_i8[grid](x, w, y, M, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1))
    return y


def t(f, n=10):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n


torch.manual_seed(0)
for (k, n) in ((5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120)):
    w = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="cuda")
    for M in (1024, 4096):
        for what in ("fwd", "dX"):
            if what == "fwd": x = torch.randn(M, k, device="cuda").to(torch.bfloat16); W = w
            else: x = torch.randn(M, n, device="cuda").to(torch.bfloat16); W = w.t()
            fl = 2 * M * W.shape[0] * W.shape[1]
            ref = x @ W.to(torch.bfloat16)
            got = mm_i8(x, W)
            rel = float((got.float() - ref.float()).abs().max() / ref.float().abs().max())
            Wb = W.to(torch.bfloat16)
            tg = t(lambda: x @ Wb); tc = t(lambda: x @ (qmodel.i8_to_bf16(w) if what == "fwd" else qmodel.i8_to_bf16(w).t())); tt = t(lambda: mm_i8(x, W))
            print(f"[{M}x{W.shape[0]}]@[{W.shape[0]}x{W.shape[1]}] {what}: hipBLASLt {tg * 1e3:.3f} ms ({fl / tg / 1e12:.0f} TF/s), "
                  f"+conv {tc * 1e3:.3f} ms, triton-i8 {tt * 1e3:.3f} ms ({fl / tt / 1e12:.0f} TF/s) cfg {_mm_i8.best_config}, max rel {rel:.1e}",
                  flush=True)
