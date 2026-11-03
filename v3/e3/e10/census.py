#!/usr/bin/env python3
"""census.py HESSDIR OUT.jsonl -- E10 (faraway): dead / near-dead input units of the writer linears (down_proj,
linear_attn.out_proj, self_attn.o_proj) from the calib input Hessians' diagonals E[h_j^2]. Per tensor: units below
1e-6 / 1e-4 / 1e-2 x the median diagonal, the share of total input energy in the bottom 1% / 10% of units, and the top
unit's share. Dead units are free bpw (their down_proj column can be dropped / coded at K0)."""
import glob, json, os, sys
import numpy as np
from safetensors import safe_open
hd, out = sys.argv[1], sys.argv[2]
fs = sorted(f for f in glob.glob(f"{hd}/*.safetensors") if any(s in f for s in ("down_proj.input", "out_proj.input", "o_proj.input")))
with open(out, "w") as fo:
    for f in fs:
        with safe_open(f, "np") as x:
            m = x.metadata(); k = int(m["k"]); up = x.get_tensor("hessian_upper")
        i = np.arange(k, dtype=np.int64); d = up[i * k - i * (i - 1) // 2].astype(np.float64) / int(m.get("rows", 1))
        s = np.sort(d); med = np.median(d); tot = d.sum()
        r = {"tensor": os.path.basename(f).replace(".input.safetensors", ""), "k": k,
             **{f"below_{t:g}": int((d < t * med).sum()) for t in (1e-6, 1e-4, 1e-2)},
             "bottom1pct_energy": float(s[: k // 100].sum() / tot), "bottom10pct_energy": float(s[: k // 10].sum() / tot),
             "top_unit_share": float(s[-1] / tot), "min_over_median": float(s[0] / med)}
        fo.write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)
