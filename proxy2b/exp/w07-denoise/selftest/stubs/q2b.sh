#!/bin/bash
# q2b.sh stub: wq files copied from the matching base quant (same K and seed offset; offsets >= 3 -> seed 0's), perturbed
# unless the --hin dir is a no-op one; --only restricts the keys
python3 - "$@" <<'PY'
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.environ["STUB"]))
from common_stub import keys, put
from safetensors.torch import load_file
M, K, CB, OUT = sys.argv[1:5]; a = sys.argv[5:]
opt = lambda k: a[a.index(k) + 1] if k in a else None
so = int(opt("--seed-offset") or 0); Y = os.path.dirname(os.path.dirname(os.path.abspath(OUT)))
base = f"{Y}/q-k{K}base{'' if so == 0 or so >= 3 else f'_s{so}'}/wq"
ks = [k for l in range(24) for k in keys(l)]
if opt("--only"): only = set(open(opt("--only")).read().split()); ks = [k for k in ks if k in only]
for f in ("--hin", "--wpre"):
    if opt(f): assert os.path.isdir(opt(f)), f"{f} {opt(f)} missing"
noop = (opt("--hin") or "").find("noop") >= 0 or (opt("--hin") is None and so < 3)
for k in ks:
    w = load_file(f"{base}/{k}.safetensors")["weight"]
    put(f"{OUT}/wq/{k}.safetensors", {"weight": w if noop else w + 0.5})
print("done")
PY
