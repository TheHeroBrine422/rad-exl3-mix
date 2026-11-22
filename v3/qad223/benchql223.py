#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""benchql223.py (w223) -- one trunk QLinear as the trainer runs it under activation checkpointing: forward under no_grad (the
checkpointed pass) + forward with grad (the recompute) + backward, per shape at M rows, for qad220's compiled forwards: no A8,
--a8 (fake quant) and --a8-int8. Prints ms per linear-segment and the per-layer sum (one GDN layer's / one attention layer's
linears + the MLP), weighted to the 27B (48 GDN + 16 attention layers).
  python benchql223.py [--m 1024] [--n-iter 10]"""
import argparse, os, sys, types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test223  # noqa: E402
import qad220  # noqa: E402

BF = torch.bfloat16
GDN = {"5120x10240": 1, "5120x6144": 1, "6144x5120": 1}            # in_proj_qkv, in_proj_z, out_proj
ATT = {"5120x12288": 1, "5120x1024": 2, "6144x5120": 1}            # q (with gate), k + v, o
MLP = {"5120x17408": 2, "17408x5120": 1}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=1024); ap.add_argument("--n-iter", type=int, default=10)
    a = ap.parse_args()
    import torch._dynamo.config
    torch._dynamo.config.recompile_limit = 64                   # 3 compiled forwards x 7 shapes x grad modes in one process (the trainer has one)
    fw = {"none": test223.grab_fwd(qad220.use_compiled_scales, False), "a8": test223.grab_fwd(qad220.use_compiled_scales, True),
          "a8-int8": test223.grab_fwd(qad220.use_compiled_scales, True, True)}
    g = torch.Generator().manual_seed(0)
    ms = {}
    print(f"| k x n | M | " + " | ".join(f"{k} ms" for k in fw) + " | int8 vs a8 | int8 vs none |"); print("|---|---|---|---|---|---|---|")
    for sh in sorted(set(GDN) | set(ATT) | set(MLP)):
        k, n = map(int, sh.split("x")); q = test223.mk(k, n, g)
        x = torch.randn(a.m, k, generator=g).to(BF).cuda(); G = torch.randn(a.m, n, generator=g).to(BF).cuda()
        row = {}
        for name, f in fw.items():
            q.forward = types.MethodType(f, q)
            def once():
                with torch.no_grad(): q(x)
                xx = x.detach().requires_grad_(True); y = q(xx); y.backward(G)
            for _ in range(3): once()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(a.n_iter): once()
            e1.record(); torch.cuda.synchronize()
            row[name] = e0.elapsed_time(e1) / a.n_iter
        ms[sh] = row
        print(f"| {sh} | {a.m} | " + " | ".join(f"{row[k]:.3f}" for k in fw) + f" | {row['a8'] / row['a8-int8']:.2f} | {row['none'] / row['a8-int8']:.2f} |", flush=True)
        del q; torch.cuda.empty_cache()
    tot = {k: sum(48 * c * ms[s][k] for s, c in GDN.items()) + sum(16 * c * ms[s][k] for s, c in ATT.items()) + sum(64 * c * ms[s][k] for s, c in MLP.items()) for k in fw}
    print("27B trunk linears per segment of M rows (fwd + recompute + bwd): " + ", ".join(f"{k} {v:.0f} ms" for k, v in tot.items())
          + f"; int8 saves {tot['a8'] - tot['a8-int8']:.0f} ms vs a8 ({tot['a8'] / tot['a8-int8']:.2f}x), {tot['none'] - tot['a8-int8']:.0f} ms vs none")


if __name__ == "__main__":
    main()
