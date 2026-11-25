#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmpck231.py CKPT_A CKPT_B -- w231 equivalence gate: the scales of two checkpoints (qad220 write_ckpt: scales/<linear>.safetensors with r / c = exp(log
multiplier)): max |log(a) - log(b)| over every value (the trained parameter's own difference), the tensor holding it, the RMS difference, and how many values
differ at all. Also the norms (norms.safetensors) when present."""
import glob, os, sys
import torch
from safetensors.torch import load_file
A, B = sys.argv[1:3]
fa = sorted(glob.glob(f"{A}/scales/*.safetensors")); assert fa, A
mx, arg, ss, n, nd = 0.0, None, 0.0, 0, 0
for f in fa:
    za, zb = load_file(f), load_file(f"{B}/scales/{os.path.basename(f)}")
    for k in za:
        d = (za[k].double().log() - zb[k].double().log()).abs()
        if float(d.max()) > mx: mx, arg = float(d.max()), f"{os.path.basename(f)}:{k}"
        ss += float((d ** 2).sum()); n += d.numel(); nd += int((d > 0).sum())
print(f"scales: {len(fa)} files, {n} values; max |diff of log multipliers| {mx:.3e} ({arg}); RMS {(ss / n) ** 0.5:.3e}; values differing {nd} ({nd / n:.1%})")
if os.path.exists(f"{A}/norms.safetensors"):
    za, zb = load_file(f"{A}/norms.safetensors"), load_file(f"{B}/norms.safetensors")
    print("norms: max |diff|", max(float((za[k].float() - zb[k].float()).abs().max()) for k in za))
