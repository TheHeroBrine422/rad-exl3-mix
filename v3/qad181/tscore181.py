#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""tscore181.py OUT.npz --model BF16 --level LEVEL --golden G [--head DIR --head-perm P] [--init PARAMS] [--seg 2048]
-- the #181 student's per-row KL on every document of a golden (torch path; documents in carried segments of <= --seg
tokens): OUT.npz with per-row kl / doc / position and per-doc sums, for comparisons with radiance's rows."""
import argparse, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qad"))
import qad181 as Q  # noqa: E402
import data181 as D  # noqa: E402
import qmodel  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--model", required=True); ap.add_argument("--level", required=True)
ap.add_argument("--golden", required=True); ap.add_argument("--head"); ap.add_argument("--head-perm"); ap.add_argument("--init")
ap.add_argument("--seg", type=int, default=2048)
ap.add_argument("--docs", default=None, help="comma list of doc indices (default all)"); ap.add_argument("--cap", type=int, default=1 << 30)
ap.add_argument("--genmask", default=None, help="gen mask file for the golden (genmask181.py)")
a = ap.parse_args()
Q.mad_int_form(); qmodel.use_shift_conv()
stu = qmodel.Student(a.model, a.level, ["cuda:0"], head_dir=a.head, head_perm=a.head_perm); Q.use_seg_attention(stu)
if a.init: stu.load_state(torch.load(a.init, weights_only=False)["state"])
g = D.Gold(a.golden, genfile=a.genmask); KL, DOC, POS, S, N = [], [], [], [], []; t0 = time.time()
docs = [int(x) for x in a.docs.split(",")] if a.docs else range(len(g.docs))
with torch.no_grad():
    for i in docs:
        n = min(a.cap, g.ntok(i)); st = None; p = 0; s = 0.0; m = 0
        while p < n:
            q = min(n, p + a.seg); k, mm, st = Q.seg_loss(stu, [g], (0, i), p, q, st, per_row=True)
            if mm: KL.append(k.double().cpu().numpy()); j = g.scored(i, p, q)[0]; POS.append(j); DOC.append(np.full(len(j), i)); s += float(k.sum()); m += mm
            p = q
        S.append(s); N.append(m)
np.savez(a.out, kl=np.concatenate(KL), doc=np.concatenate(DOC), pos=np.concatenate(POS), sums=np.array(S), rows=np.array(N))
print(f"{a.out}: KL/row {sum(S) / sum(N):.6f} over {sum(N)} rows, {len(N)} docs, {time.time() - t0:.0f} s")
