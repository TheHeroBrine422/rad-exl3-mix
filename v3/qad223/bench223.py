#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""bench223.py (w223) -- int8 a8g128 GEMM (a8int8.py) vs hipBLASLt bf16 on the 27B trunk shapes (k x n of x @ W), M = segment rows.
Prints per shape / M: hipBLASLt bf16 GEMM alone, qad220's --a8 forward (torch fake quant + int8 -> bf16 copy + bf16 GEMM), the int8
path (quant_a8 + mm_a8) and mm_a8 alone (TFLOP/s of 2 M k n), plus the check vs the fake-quant path: q bitwise, max abs / rel of y.
  python bench223.py [--m 512,1024,2048,4096] [--shapes 5120x17408,...] [--n-iter 20]"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a8int8  # noqa: E402

BF = torch.bfloat16
SHAPES = "5120x17408,17408x5120,5120x10240,5120x6144,6144x5120,5120x12288,5120x1024"   # gate/up, down, GDN qkv, z, out/o, q, k/v


def fq(x):
    """qad220's _A8 forward"""
    g = x.float().view(*x.shape[:-1], -1, 128)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
    return (torch.round(g / s).clamp(-127, 127) * s).view_as(x).to(x.dtype), torch.round(g / s).clamp(-127, 127).view_as(x), s[..., 0]


def tm(f, n):
    for _ in range(3): f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="512,1024,2048,4096"); ap.add_argument("--shapes", default=SHAPES); ap.add_argument("--n-iter", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    print(f"device {torch.cuda.get_device_name(0)}; torch {torch.__version__}")
    print("| k x n | M | bf16 GEMM ms (TF/s) | --a8 fwd ms | int8 fwd ms (TOPS) | mm_a8 ms (TOPS) | x vs bf16 GEMM | x vs --a8 fwd | q bitwise | max abs dy | max rel dy | f64 err int8 / --a8 | best cfg |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sh in a.shapes.split(","):
        k, n = map(int, sh.split("x"))
        w = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="cuda")
        wb = w.to(BF)
        for M in map(int, a.m.split(",")):
            x = (torch.randn(M, k, device="cuda") * torch.rand(M, 1, device="cuda") * 3).to(BF)
            fl = 2 * M * k * n
            xf, qr, sr = fq(x)
            ref = xf @ wb
            q, sa = a8int8.quant_a8(x)
            got = a8int8.mm_a8(q, sa, w)
            qbit = torch.equal(q.float(), qr) and torch.equal(sa, sr)
            d = (got.float() - ref.float()).abs()
            mab = float(d.max()); mrel = float(d.max() / ref.float().abs().max())
            ex = ((qr.double().view(M, -1, 128) * sr.double()[..., None]).view(M, k) @ w.double()) if M <= 1024 else None
            fe = f"{float((got.double() - ex).abs().max() / ex.abs().max()):.2e} / {float((ref.double() - ex).abs().max() / ex.abs().max()):.2e}" if ex is not None else "-"
            tg = tm(lambda: x @ wb, a.n_iter)
            tf = tm(lambda: fq(x)[0] @ w.to(BF), a.n_iter)
            ti = tm(lambda: a8int8.mm_a8(*a8int8.quant_a8(x), w), a.n_iter)
            tmm = tm(lambda: a8int8.mm_a8(q, sa, w), a.n_iter)
            print(f"| {k}x{n} | {M} | {tg*1e3:.3f} ({fl/tg/1e12:.0f}) | {tf*1e3:.3f} | {ti*1e3:.3f} ({fl/ti/1e12:.0f}) | {tmm*1e3:.3f} ({fl/tmm/1e12:.0f}) | "
                  f"{tg/ti:.2f} | {tf/ti:.2f} | {qbit} | {mab:.3g} | {mrel:.2e} | {fe} | {a8int8._mm_a8_k.best_config} |", flush=True)


if __name__ == "__main__":
    main()
