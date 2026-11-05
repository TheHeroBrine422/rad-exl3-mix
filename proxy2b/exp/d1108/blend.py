#!/usr/bin/env python3
"""blend.py hout A B W OUT [ONLY] | hin A B W OUT [ONLY] -- d1108: mix two statistics directories, weight W on B
(ONLY: a regex on the file name; files not matching are linked from A unchanged: per-family blends).
  hout: Σdy² diagonals (ourq --hout files, "hout_diag" = a per-sequence mean). Each is normalised to mean 1 per tensor
        first (ourq rescales H_out by its diagonal mean anyway; the corpora's scored-position counts differ), then
        (1 - W) a + W b. W = 1 copies B's shape.
  hin:  input Hessians in the challenge layout ("hessian_upper", metadata k / rows: a sum over `rows` tokens): the
        per-token means mixed, (1 - W) A/rows_A + W B/rows_B, stored as a sum over rows_A tokens (A's scale, so ourq
        treats it like the original). Files only in A are linked unchanged."""
import glob, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import re
mode, A, B, W, O = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
ONLY = re.compile(sys.argv[6]) if len(sys.argv) > 6 else None
os.makedirs(O, exist_ok=True); n = 0
for f in sorted(glob.glob(f"{A}/*.safetensors")):
    name = os.path.basename(f); fb = f"{B}/{name}"
    if not os.path.exists(fb) or (ONLY and not ONLY.search(name)):
        if not os.path.exists(f"{O}/{name}"): os.symlink(os.path.realpath(f), f"{O}/{name}")
        continue
    with safe_open(f, "pt") as x, safe_open(fb, "pt") as y:
        if mode == "hout":
            a, b = x.get_tensor("hout_diag").double(), y.get_tensor("hout_diag").double()
            v = (1 - W) * a / a.mean() + W * b / b.mean()
            save_file({"hout_diag": v.float().contiguous()}, f"{O}/{name}", metadata={"blend": f"{W} on {B}"})
        else:
            ma, mb = x.metadata(), y.metadata(); ra, rb = float(ma["rows"]), float(mb["rows"])
            assert ma["k"] == mb["k"], name
            u = ((1 - W) * x.get_tensor("hessian_upper").double() / ra + W * y.get_tensor("hessian_upper").double() / rb) * ra
            save_file({"hessian_upper": u.float().contiguous()}, f"{O}/{name}", metadata=ma)
    n += 1
open(f"{O}/.done", "w").close()
print(f"BLEND DONE {mode} W={W}: {n} blended -> {O}")
