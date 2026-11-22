#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""memql223.py (w223) -- peak memory of a checkpointed stack of MLP-shaped QLinears (gate / up 5120 -> 17408, down 17408 -> 5120;
silu(g) * u; plus the GDN / attention shapes on the same input) under qad220's compiled forwards --a8 vs --a8-int8: one training-like pass (forward with grad under checkpoint +
backward) at M rows, after a warm-up pass (compile / autotune), peak over the resident weights.
  python memql223.py [--layers 6] [--m 1105] [--int8-first] [--nograd-first] [--only a8|a8-int8]   (run with an empty Triton cache to see the first pass's autotune)"""
import argparse, os, sys, types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test223  # noqa: E402
import qad220  # noqa: E402

BF = torch.bfloat16


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--m", type=int, default=1105); ap.add_argument("--int8-first", action="store_true"); ap.add_argument("--nograd-first", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    L = [(test223.mk(5120, 17408, g), test223.mk(5120, 17408, g), test223.mk(17408, 5120, g), test223.mk(5120, 10240, g), test223.mk(5120, 6144, g),
          test223.mk(6144, 5120, g), test223.mk(5120, 12288, g), test223.mk(5120, 1024, g)) for _ in range(a.layers)]
    x0 = torch.randn(a.m, 5120, generator=g).to(BF).cuda()
    for name, args in ((("a8-int8", (True, True)), ("a8", (True,))) if a.int8_first else (("a8", (True,)), ("a8-int8", (True, True)))):
        if a.only and name != a.only: continue
        f = test223.grab_fwd(qad220.use_compiled_scales, *args)
        for ly in L:
            for q in ly: q.forward = types.MethodType(f, q)
        def layer(h, ly):                                           # MLP + every other trunk shape (GDN qkv / z / out, q, k / v) on the same input
            z = ly[4](h); o = ly[5](F.silu(z)); qq = ly[3](h)[:, :5120] + ly[6](h)[:, :5120] + ly[7](h).repeat(1, 5)
            return h + ly[2](F.silu(ly[0](h)) * ly[1](h)) + o + qq
        def once():
            h = x0.detach().requires_grad_(True)
            for ly in L: h = checkpoint(layer, h, ly, use_reentrant=False)
            h.float().pow(2).mean().backward()
        if a.nograd_first:                                          # the trainer's order: no-grad passes (eval / carry) first
            for m in (2048, 1024, 300):
                with torch.no_grad():
                    h = torch.randn(m, 5120, generator=g).to(BF).cuda()
                    for ly in L: h = layer(h, ly)
        base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        once(); torch.cuda.synchronize(); first = torch.cuda.max_memory_allocated() - base
        torch.cuda.reset_peak_memory_stats()
        once(); torch.cuda.synchronize()
        print(f"{name}: peak over resident {(torch.cuda.max_memory_allocated() - base) / 2**20:.0f} MiB (first pass, compile / autotune: "
              f"{first / 2**20:.0f} MiB; {a.layers} MLP layers, M {a.m})", flush=True)


if __name__ == "__main__":
    main()
