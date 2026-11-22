#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test223.py (w223) -- gate (i): one trunk linear's --a8-int8 forward / backward vs qad220's --a8 fake-quant path, eager (use_a8) and
compiled (use_compiled_scales), on every 27B trunk shape. QLinears with random int8 codebook integers (|m| <= 126), the i8 step,
suh / svh ~ the stored scales' spread, random r / c, inputs with outlier channels. Prints per shape and M:
  out  max |y_int8 - y_a8| and that / max |y_a8|; the share of outputs that differ; both paths' error vs the f64 product of the same
       quantized input (q sa exactly, no bf16 rounding of the activation) -- the int8 path should be the closer one
       rel L2 vs the f64 product per path; rel L2 of y_int8 - y_a8 next to rel L2 of y_a8 - y (no A8): the int8 change against the size of the A8 effect itself
  grad rel L2 of dx, dr, dc (int8 vs a8; dX is the same bf16 GEMM, so only the forward's difference reaches it through dy)
  cfg  mm_a8 bitwise the same under every autotune config (int32 group dots exact, f32 accumulation over groups in order)
  python test223.py [--m 279,700,1105] [--shapes ...]"""
import argparse, os, sys, types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad220"))
import qad220  # noqa: E402
import qmodel  # noqa: E402
from rxq import deq  # noqa: E402

BF = torch.bfloat16
SHAPES = "5120x17408,17408x5120,5120x10240,5120x6144,6144x5120,5120x12288,5120x1024"


def mk(k, n, g):
    q = qmodel.QLinear.__new__(qmodel.QLinear); torch.nn.Module.__init__(q)
    q.key = f"t{k}x{n}"; q.k, q.n = k, n; q.la = q.lb = None
    q.register_buffer("wq", torch.randint(-126, 127, (k, n), generator=g, dtype=torch.int8).cuda())
    q.s = qmodel.int_form("i8"); q.exact = True
    q.register_buffer("suh", (torch.rand(k, generator=g) * 0.02 + 0.005).cuda())
    q.register_buffer("svh", (torch.rand(n, generator=g) * 0.02 + 0.005).cuda())
    q.register_buffer("Hb", deq.hadamard128("cpu").to(BF).cuda())
    q.r = torch.nn.Parameter((torch.randn(k, generator=g) * 0.05).cuda()); q.c = torch.nn.Parameter((torch.randn(n, generator=g) * 0.05).cuda())
    return q


def run(q, fwd, x, G):
    q.forward = types.MethodType(fwd, q)
    for p in (q.r, q.c): p.grad = None
    xx = x.clone().requires_grad_(True)
    y = q(xx)
    (y.float() * G).sum().backward()
    return y.detach(), xx.grad.detach().float(), q.r.grad.detach().clone(), q.c.grad.detach().clone()


def grab_fwd(use, *args):
    holder = {}
    class Cap:                                                      # use_* bind fwd onto each module: capture the function
        la = None
        def __setattr__(self, k, v): holder[k] = v
    use(types.SimpleNamespace(qlin={"x": Cap()}), *args)
    return holder["forward"].__func__


def rl2(a, b): return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--m", default="279,700,1105"); ap.add_argument("--shapes", default=SHAPES)
    a = ap.parse_args()
    noa8 = grab_fwd(qad220.use_compiled_scales, False)
    paths = {"eager": (grab_fwd(qad220.use_a8), grab_fwd(qad220.use_a8, True)),
             "compiled": (grab_fwd(qad220.use_compiled_scales, True), grab_fwd(qad220.use_compiled_scales, True, True))}
    g = torch.Generator().manual_seed(0)
    print("| path | k x n | M | max abs dy | max abs dy / max abs y | outputs differing | f64 err int8 / a8 (max rel) | rel L2 vs f64 int8 / a8 | rel L2 int8 vs a8 | rel L2 a8 vs no-A8 | dx rel L2 | dr rel L2 | dc rel L2 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    worst = 0.0
    for sh in a.shapes.split(","):
        k, n = map(int, sh.split("x")); q = mk(k, n, g)
        for M in map(int, a.m.split(",")):
            chs = torch.ones(k); chs[torch.randperm(k, generator=g)[:k // 100]] = 20.0     # 1 % outlier channels
            x = (torch.randn(M, k, generator=g) * chs * (torch.rand(M, 1, generator=g) * 3 + 0.1)).to(BF).cuda()
            G = torch.randn(M, n, generator=g).cuda()
            # f64 reference: the same quantized input (q sa exactly) times the codebook integers, then had_n, (sv s)
            with torch.no_grad():
                xr = (x.float() * q.su()).to(BF); xr = (xr.view(-1, 128) @ q.Hb).view(-1, k)
                gg = xr.float().view(M, -1, 128); s = gg.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
                xq = (torch.round(gg / s).clamp(-127, 127).double() * s.double()).view(M, k)
                ex = ((xq @ q.wq.double()).view(-1, 128) @ q.Hb.double()).view(M, n) * (q.sv().double() * q.s)
            y0 = run(q, noa8, x, G)[0].float()
            for name, (fa, fi) in paths.items():
                ya, dxa, dra, dca = run(q, fa, x, G)
                yi, dxi, dri, dci = run(q, fi, x, G)
                d = (yi.float() - ya.float()).abs(); rel = float(d.max() / ya.float().abs().max()); worst = max(worst, rel)
                ei = float((yi.double() - ex).abs().max() / ex.abs().max()); ea = float((ya.double() - ex).abs().max() / ex.abs().max())
                print(f"| {name} | {k}x{n} | {M} | {float(d.max()):.3g} | {rel:.2e} | {float((d > 0).float().mean()) * 100:.1f} % | {ei:.2e} / {ea:.2e} | {rl2(yi.double(), ex):.2e} / {rl2(ya.double(), ex):.2e} | {rl2(yi.float(), ya.float()):.2e} | {rl2(ya.float(), y0):.2e} | "
                      f"{rl2(dxi, dxa):.2e} | {rl2(dri, dra):.2e} | {rl2(dci, dca):.2e} |", flush=True)
        del q; torch.cuda.empty_cache()
    print(f"worst max abs dy / max abs y: {worst:.2e}")
    import triton, a8int8
    for (k, n) in ((5120, 17408), (17408, 5120), (5120, 1024)):
        w = torch.randint(-126, 127, (k, n), generator=g, dtype=torch.int8).cuda()
        for M in (279, 1105):
            qx, sa = a8int8.quant_a8((torch.randn(M, k, generator=g) * 3).to(BF).cuda())
            outs = []
            for c in a8int8._cfgs():
                if k % (128 * c.kwargs["GPI"]): continue
                y = torch.empty(M, n, dtype=BF, device="cuda")
                grid = (triton.cdiv(M, c.kwargs["BM"]) * triton.cdiv(n, c.kwargs["BN"]),)
                try: a8int8._mm_a8_k.fn[grid](qx, sa, w, y, M, n, k, 0, qx.stride(0), sa.stride(0), w.stride(0), y.stride(0), **c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                except triton.runtime.errors.OutOfResources: continue                  # the autotuner skips these too
                outs.append(y)
            print(f"cfg {k}x{n} M {M}: {len(outs)} configs, all bitwise equal: {all(torch.equal(o, outs[0]) for o in outs)}", flush=True)


if __name__ == "__main__":
    main()
