#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""tscore196.py (#196: tscore191.py + the switches of the checkpoint: R factors, U un-shrink, N norm gains, replayed
before the checkpoint state loads) OUT.npz --golden G [--ckpt CKPT_DIR] [--no-a8] [--seg 2048] [student args as run191.sh's] -- #191's
response on the torch path (RULES191): a checkpoint's student (the 2.5 incumbent's served form + the checkpoint's
scales, and its adapters when it has them) with A8 (a8g128 fake quant on every trunk linear, as served) scores every
document of a golden over its gen rows (gen.u8 of the golden dir), documents in carried segments of <= --seg tokens.
No --ckpt = the control (untrained scales). OUT.npz: per-row kl / doc / position, per-doc sums and rows. Writes
OUT.part.npz, then renames."""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qad196 as Q  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--golden", required=True); ap.add_argument("--ckpt")
ap.add_argument("--model", required=True); ap.add_argument("--level", required=True); ap.add_argument("--head"); ap.add_argument("--head-perm")
ap.add_argument("--emb"); ap.add_argument("--ab"); ap.add_argument("--build-lock"); ap.add_argument("--no-a8", action="store_true")
ap.add_argument("--seg", type=int, default=2048); ap.add_argument("--cap", type=int, default=1 << 30)
a = ap.parse_args()
Q.mad_int_form(); Q.qmodel.use_shift_conv()
z = torch.load(os.path.join(a.ckpt, "params.pt"), weights_only=False) if a.ckpt else None
a.lora = int(z["meta"]["args"].get("lora", 0)) if z else 0
stu = Q.build_student(a, "cuda:0", print)
sw = SimpleNamespace(factors="all", unshrink=None, norms=False)
if z:
    ca = z["meta"]["args"]; sw = SimpleNamespace(factors=ca.get("factors", "all"), unshrink=ca.get("unshrink"), norms=bool(ca.get("norms", False)))
Q.apply196(sw, stu, print)                       # #196 switches of the checkpoint (frozen un-shrink starts, norm gains)
Q.use_seg_attention(stu)
if z: stu.load_state(z["state"]); assert set(z["state"]) == set(stu.P), "checkpoint / student parameters differ"
if not a.no_a8: Q.use_a8(stu)                    # (its adapter GEMMs in f32, as trained)
elif a.lora: Q.use_lora32(stu)
g = Q.D.Gold(a.golden); KL, DOC, POS, S, N = [], [], [], [], []; t0 = time.time()
with torch.no_grad():
    for i in range(len(g.docs)):
        n = min(a.cap, g.ntok(i)); st = None; p = 0; s = 0.0; m = 0
        while p < n:
            q = min(n, p + a.seg); k, mm, st = Q.seg_loss(stu, [g], (0, i), p, q, st, per_row=True)
            if mm: KL.append(k.double().cpu().numpy()); j = g.scored(i, p, q)[0]; POS.append(j); DOC.append(np.full(len(j), i)); s += float(k.sum()); m += mm
            p = q
        S.append(s); N.append(m)
np.savez(a.out + ".part.npz", kl=np.concatenate(KL), doc=np.concatenate(DOC), pos=np.concatenate(POS), sums=np.array(S), rows=np.array(N),
         meta=json.dumps({"golden": a.golden, "ckpt": a.ckpt, "a8": not a.no_a8, "seg": a.seg}))
os.replace(a.out + ".part.npz", a.out)
print(f"{a.out}: KL/row {sum(S) / sum(N):.6f} over {sum(N)} rows, {len(N)} docs, A8 {not a.no_a8}, {time.time() - t0:.0f} s")
