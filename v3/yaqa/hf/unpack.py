#!/usr/bin/env python3
"""unpack.py MERGED.safetensors OUTDIR -- split one merged H_out file (one tensor per linear, named by checkpoint key)
into the per-tensor layout ourq.py --hout reads: OUTDIR/<key>.safetensors with tensor "hout_diag"."""
import os, sys
from safetensors import safe_open
from safetensors.torch import save_file
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
with safe_open(src, "pt") as f:
    for k in f.keys(): save_file({"hout_diag": f.get_tensor(k).contiguous()}, f"{out}/{k}.safetensors")
print("UNPACKED", len(list(os.scandir(out))), "->", out)
