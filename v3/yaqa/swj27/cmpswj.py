#!/usr/bin/env python3
"""cmpswj.py REF NEW -- two swj outputs (swiglufit.py's dst_joint or swjfit27.py's OUT/NAME): scales/ and swiglu.jsonl.
Prints whether they are bit-identical; if not, how far apart: share of channels whose c_g grid point differs, max and
mean |Δc|, and the per-layer records' largest relative difference. Last line: CMPSWJ IDENTICAL | CMPSWJ DIFFER ..."""
import json, os, sys
import torch
from safetensors import safe_open

ref, new = sys.argv[1:3]
ks = sorted(f for f in os.listdir(f"{ref}/scales") if f.endswith(".safetensors"))
kn = sorted(f for f in os.listdir(f"{new}/scales") if f.endswith(".safetensors"))
assert ks == kn, f"different tensors: {len(ks)} vs {len(kn)}"
ident, st = True, {"gate": [0, 0, 0.0, 0.0], "up": [0, 0, 0.0, 0.0]}   # channels, differing, max |dc|, sum |dc|
for f in ks:
    t = {}
    for d in (ref, new):
        with safe_open(f"{d}/scales/{f}", "pt") as h: t[d] = {k: h.get_tensor(k) for k in h.keys()}
    same = t[ref].keys() == t[new].keys() and all(torch.equal(t[ref][k], t[new][k]) for k in t[ref])
    ident &= same
    if not torch.equal(t[ref]["r"], t[new]["r"]): print(f"{f}: r differs")
    c0, c1 = t[ref]["c"].double(), t[new]["c"].double(); dc = (c0 - c1).abs()
    s = st["gate" if ".gate_proj" in f else "up"]
    s[0] += len(c0); s[1] += int((dc > 0).sum()); s[2] = max(s[2], float(dc.max())); s[3] += float(dc.sum())
ja = [json.loads(l) for l in open(f"{ref}/swiglu.jsonl")]; jb = [json.loads(l) for l in open(f"{new}/swiglu.jsonl")]
jsame = ja == jb; ident &= jsame
worst = max((abs(x[k] - y[k]) / max(abs(x[k]), 1e-30), k, x["layer"]) for x, y in zip(ja, jb) for k in x if k != "layer")
for n, s in st.items():
    print(f"{n:4s} {s[0]} channels: {s[1]} differ ({s[1] / max(s[0], 1):.2%}), max |dc| {s[2]:.3g}, mean |dc| {s[3] / max(s[0], 1):.2e}")
print(f"swiglu.jsonl {'identical' if jsame else 'differs'}: largest relative difference {worst[0]:.2e} ({worst[1]}, layer {worst[2]})")
print("CMPSWJ IDENTICAL" if ident else f"CMPSWJ DIFFER (scales {'identical' if all(s[1] == 0 for s in st.values()) else 'differ'})")
