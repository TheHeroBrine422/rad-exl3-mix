#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""testew215.py -- w215: whole-student check of qmodel.use_compiled_elementwise's patches (shift conv, MLP silu * up), one
at a time vs eager: for qad-pool documents whose first segment has ~279 .. 1,105 tokens, the segment's KL / row (relative
difference) and its gradient's cosine to eager's (env MODEL, LEVEL, CALIB, POOLS)."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import qad181  # noqa: E402
import qmodel  # noqa: E402
import data181 as D  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35  # noqa: E402

E = os.environ
qad181.mad_int_form(); qmodel.use_shift_conv()
C = D.Corpus([], 8192, calib=E["CALIB"], pools=E["POOLS"], pool="qad")
stu = qmodel.Student(E["MODEL"], E["LEVEL"], ["cuda:0"]); qad181.use_seg_attention(stu)
G0 = {}
orig = (q35.causal_conv1d_fn, q35.Qwen3_5MLP.forward)
qmodel.use_compiled_elementwise()
comp = (q35.causal_conv1d_fn, q35.Qwen3_5MLP.forward)


def setp(which):
    q35.causal_conv1d_fn = comp[0] if "conv" in which else orig[0]
    q35.Qwen3_5MLP.forward = comp[1] if "mlp" in which else orig[1]


docs = [(gi, i, n) for gi, i, n, nr in C.items if len(C.g[gi].scored(i, 0, n)[0]) >= 100]
pick = []
for L in (279, 566, 694, 1105, 1024, 300, 777):                 # documents whose whole first segment has about these lengths
    best = min((d for d in docs if d[2] <= 1200), key=lambda d: abs(d[2] - L)); pick.append(best)
for which in ((), ("conv",), ("mlp",), ("conv", "mlp")):
    setp(which); res = []
    for k, (gi, i, n) in enumerate(pick):
        kl, m, _ = qad181.seg_loss(stu, C.g, (gi, i), 0, n, None); (kl / m).backward()
        g = torch.cat([p.grad.reshape(-1).double() for p in stu.P.values()]); [setattr(p, "grad", None) for p in stu.P.values()]
        if not which: G0[k] = (g, float(kl) / m)
        res.append(f"L{n}: KL {float(kl) / m / G0[k][1] - 1:+.2e} cos {float(g @ G0[k][0] / g.norm() / G0[k][0].norm()):.5f}")
    print(which or "eager", "; ".join(res), flush=True)
