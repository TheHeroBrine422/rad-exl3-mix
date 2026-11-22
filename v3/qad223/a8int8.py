# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""a8int8.py (w223) -- the QAD trainer's a8g128 GEMM as a real int8 x int8 -> int32 product.

  quant_a8(x)          x bf16 [M, K] -> (q int8 [M, K], sa f32 [M, K/128]): per row and 128-group absmax * f32(1/127) (torch's
                       tensor / scalar), q = round(x / sa) clamped to +-127 (round half to even, IEEE division): q and sa bitwise =
                       qad220's _A8
  mm_a8(q, sa, w)      y[m, n] = sum_g sa[m, g] * sum_{j in g} q[m, j] w[j, n]  (int32 per group, f32 across groups) -> bf16
  a8_matmul(x, w)      autograd: forward = quant_a8 + mm_a8; backward dX = dy @ w^T in bf16 (the STE of qad220's --a8: the
                       fake-quant GEMM's input gradient); w (int8 codebook integers, frozen) gets no gradient

Numerics vs qad220's --a8 fake quant (x_fq = bf16(q sa), then x_fq @ bf16(w) with f32 accumulation): the int8 path multiplies q sa
exactly (no bf16 rounding of the dequantized activation, 2^-9 relative per element) and sums in a different order. This is the
served exl3i A8 arithmetic (int8 dot per group, f32 group scale)."""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BF = torch.bfloat16
_INV127 = float(np.float32(1.0) / np.float32(127.0))


@triton.jit
def _quant_a8_k(x, q, sa, M, K, sxm, sqm, ssm, INV127: tl.constexpr, BR: tl.constexpr):
    """one program = BR rows x one 128-group (groups fastest in launch order: neighbouring programs read neighbouring bytes)"""
    g = tl.program_id(0); pr = tl.program_id(1)
    rm = pr * BR + tl.arange(0, BR); rk = g * 128 + tl.arange(0, 128)
    msk = rm[:, None] < M
    v = tl.load(x + rm[:, None].to(tl.int64) * sxm + rk[None, :], mask=msk, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-30)
    s = s * INV127                                  # torch's tensor / python scalar = multiply by the f32 reciprocal
    r = libdevice.rint(tl.math.div_rn(v, s[:, None]))
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    tl.store(q + rm[:, None].to(tl.int64) * sqm + rk[None, :], r.to(tl.int8), mask=msk)
    tl.store(sa + rm.to(tl.int64) * ssm + g, s, mask=rm < M)


def quant_a8(x):
    M, K = x.shape
    assert K % 128 == 0 and x.stride(1) == 1, (x.shape, x.stride())
    q = torch.empty(M, K, dtype=torch.int8, device=x.device)
    sa = torch.empty(M, K // 128, dtype=torch.float32, device=x.device)
    BR = 32
    _quant_a8_k[(K // 128, triton.cdiv(M, BR))](x, q, sa, M, K, x.stride(0), q.stride(0), sa.stride(0), INV127=_INV127, BR=BR, num_warps=4)
    return q, sa


def _cfgs():
    out = []
    for bm, bn, gpi, w, st in ((128, 128, 1, 8, 2), (128, 128, 2, 8, 2), (128, 128, 1, 8, 3), (128, 128, 2, 4, 2), (64, 256, 1, 8, 2),
                               (256, 128, 1, 8, 2), (256, 128, 2, 8, 2), (128, 256, 2, 8, 2), (64, 128, 1, 4, 2), (128, 64, 1, 4, 2),
                               (64, 256, 2, 8, 2), (256, 256, 1, 8, 2)):
        out.append(triton.Config({"BM": bm, "BN": bn, "GPI": gpi, "G": 8}, num_warps=w, num_stages=st))
    return out


@triton.autotune(configs=_cfgs(), key=["M", "N", "K"], prune_configs_by={"early_config_prune": lambda cfgs, nargs, **kw: [c for c in cfgs if nargs["K"] % (128 * c.kwargs["GPI"]) == 0]})
@triton.jit
def _mm_a8_k(a, sa, b, c, M, N, K, sam, ssm, sbk, scm, BM: tl.constexpr, BN: tl.constexpr, GPI: tl.constexpr, G: tl.constexpr):
    """GPI = 128-groups per loop iteration (K % (128 GPI) == 0 required: the autotuner prunes)"""
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (G * npn); fm = gid * G; gs = min(npm - fm, G)
    pm = fm + (pid % gs); pn = (pid % (G * npn)) // gs
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, 128)
    mm = rm < M; mn = rn < N
    ap = a + rm[:, None].to(tl.int64) * sam + rk[None, :]
    bp = b + rk[:, None].to(tl.int64) * sbk + rn[None, :]
    sp = sa + rm.to(tl.int64) * ssm
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, K // (128 * GPI)):
        for h in tl.static_range(GPI):
            x = tl.load(ap + h * 128, mask=mm[:, None], other=0)
            w = tl.load(bp + h * 128 * sbk, mask=mn[None, :], other=0)
            s = tl.load(sp + g * GPI + h, mask=mm, other=0.0)
            part = tl.dot(x, w, out_dtype=tl.int32)
            acc += part.to(tl.float32) * s[:, None]
        ap += 128 * GPI; bp += 128 * GPI * sbk
    tl.store(c + rm[:, None].to(tl.int64) * scm + rn[None, :], acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


def mm_a8(q, sa, w):
    M, K = q.shape; N = w.shape[1]
    assert w.shape[0] == K and w.dtype == torch.int8 and q.stride(1) == 1 and w.stride(1) == 1 and K % 128 == 0
    y = torch.empty(M, N, dtype=BF, device=q.device)
    grid = lambda m: (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)
    _mm_a8_k[grid](q, sa, w, y, M, N, K, q.stride(0), sa.stride(0), w.stride(0), y.stride(0))
    return y


class A8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, to_bf16):
        ctx.save_for_backward(w); ctx.to_bf16 = to_bf16
        q, sa = quant_a8(x.contiguous())
        return mm_a8(q, sa, w)

    @staticmethod
    def backward(ctx, dy):
        w, = ctx.saved_tensors
        return dy @ ctx.to_bf16(w).t(), None, None


def a8_matmul(x, w, to_bf16=None):
    """x bf16 [M, K] (the rotated GEMM input), w int8 [K, N] -> bf16 [M, N]; to_bf16 = the int8 -> bf16 weight copy used by the backward"""
    return A8Matmul.apply(x, w, to_bf16 or (lambda t: t.to(BF)))
