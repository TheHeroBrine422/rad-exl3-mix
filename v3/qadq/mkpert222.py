#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkpert222.py noise|revert REF_CKPT OUT_CKPT [--seed S] [--family gdn|att|mlp] -- checkpoints for the blow-up fragility tests (w222 D1 follow-up). REF_CKPT = a derailing checkpoint (scales/<linear key>.safetensors with r / c multipliers).
noise: every scale vector the reference changed gets an independent random log multiplier N(0, sigma) with sigma = the reference's rms log multiplier on that side (r: input, c: output), the other vectors stay 1: the same rms
as the trained scales, random direction. revert: the reference's multipliers, except the family's linears (gdn = linear_attn.*, att = self_attn.*, mlp = mlp.*) put back to the released scales (multiplier 1).
Writes OUT_CKPT/scales/*.safetensors and OUT_CKPT/.done (the format swap222.py reads)."""
import glob, os, sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
mode, ref, out = sys.argv[1:4]; a = sys.argv[4:]
seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 0; fam = a[a.index("--family") + 1] if "--family" in a else None
FAM = {"gdn": ".linear_attn.", "att": ".self_attn.", "mlp": ".mlp."}
os.makedirs(f"{out}/scales", exist_ok=True)
fs = sorted(glob.glob(f"{ref}/scales/*.safetensors")); ld = {}
for f in fs:
    with safe_open(f, "pt") as z: ld[os.path.basename(f)] = (z.get_tensor("r").float(), z.get_tensor("c").float())
def rms(side):
    v = [torch.log(x[side]) for x in ld.values() if bool((x[side] != 1).any())]
    v = torch.cat(v); return float(v.pow(2).mean().sqrt()), len(v)
sig = {"r": rms(0), "c": rms(1)}
print(f"{mode} {os.path.basename(os.path.dirname(ref.rstrip('/')))}: rms log r {sig['r'][0]:.5f} ({len(ld)} linears), rms log c {sig['c'][0]:.5f}")
g = torch.Generator().manual_seed(seed); n = 0
for name, (r, c) in ld.items():
    if mode == "noise":
        nr = torch.exp(sig["r"][0] * torch.randn(r.shape, generator=g)) if bool((r != 1).any()) else torch.ones_like(r)
        nc = torch.exp(sig["c"][0] * torch.randn(c.shape, generator=g)) if bool((c != 1).any()) else torch.ones_like(c)
    else:
        key = name[:-len(".safetensors")]
        if FAM[fam] in key + ".": nr, nc = torch.ones_like(r), torch.ones_like(c); n += 1
        else: nr, nc = r, c
    save_file({"r": nr.contiguous(), "c": nc.contiguous()}, f"{out}/scales/{name}")
open(f"{out}/.done", "w").close()
if mode == "revert": print(f"family {fam}: {n} linears put back to the released scales")
