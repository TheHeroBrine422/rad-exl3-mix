# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""trigemm.py -- #144: can a Triton f32 GEMM (tl.dot ieee: FMA lowering, k sequential within and across K blocks)
reproduce torch's x @ W.T bits (sequential FMA over k, gemmorder.py) and beat its speed at the swj shape?"""
import itertools, sys, time

import torch
import triton
import triton.language as tl


@triton.jit
def _mm(X, W, Y, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    gsz = GM * npn; g = pid // gsz; fm = g * GM; gm = min(npm - fm, GM)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    om = pm * BM + tl.arange(0, BM); on = pn * BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(X + om[:, None] * K + (k0 + ok)[None, :], mask=(om[:, None] < M), other=0.0)
        b = tl.load(W + on[None, :] * K + (k0 + ok)[:, None], mask=(on[None, :] < N), other=0.0)   # W [N, K]: b = W.T tile
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(Y + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))


def mm(x, w, BM, BN, BK, GM=8, nw=4, ns=2):
    M, K = x.shape; N = w.shape[0]
    y = torch.empty(M, N, device=x.device)
    _mm[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](x, w, y, M, N, K, BM=BM, BN=BN, BK=BK, GM=GM, num_warps=nw, num_stages=ns)
    return y


if __name__ == "__main__":
    torch.manual_seed(0); K, N = 5120, 17408
    W = (torch.randn(N, K) * 0.02).cuda()
    for T in (1280, 4096):
        x = torch.randn(T, K).bfloat16().float().cuda()
        y0 = x @ W.T; torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(3): y0 = x @ W.T
        torch.cuda.synchronize(); tref = (time.perf_counter() - t) / 3
        print(f"T {T}: torch {tref * 1e3:.2f} ms ({2 * T * K * N / tref / 1e12:.1f} TF)", flush=True)
        for BM, BN, BK, nw, ns in itertools.product((64, 128), (64, 128), (8, 16, 32), (4, 8), (1, 2)):
            try:
                y = mm(x, W, BM, BN, BK, nw=nw, ns=ns); torch.cuda.synchronize(); t = time.perf_counter()
                for _ in range(3): y = mm(x, W, BM, BN, BK, nw=nw, ns=ns)
                torch.cuda.synchronize(); s = (time.perf_counter() - t) / 3
                print(f"  BM {BM} BN {BN} BK {BK} w {nw} s {ns}: {s * 1e3:.2f} ms ({2 * T * K * N / s / 1e12:.1f} TF) identical {torch.equal(y, y0)}"
                      + ("" if torch.equal(y, y0) else f" ({int((y != y0).sum())} differ)"), flush=True)
            except Exception as e:
                print(f"  BM {BM} BN {BN} BK {BK} w {nw} s {ns}: failed {str(e)[:80]}", flush=True)
