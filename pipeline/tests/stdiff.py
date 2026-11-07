#!/usr/bin/env python3
"""stdiff.py A B -- two safetensors files: metadata, tensor names / dtypes / shapes, values (equal, max |diff|)"""
import json, struct, sys
import torch
from safetensors import safe_open
def hdr(p):
    with open(p, "rb") as f: n = struct.unpack("<Q", f.read(8))[0]; return json.loads(f.read(n))
a, b = sys.argv[1:3]; ha, hb = hdr(a), hdr(b)
ma, mb = ha.pop("__metadata__", None), hb.pop("__metadata__", None)
print("metadata equal:", ma == mb, "" if ma == mb else {k: (ma.get(k) if ma else None, mb.get(k) if mb else None) for k in set(ma or {}) | set(mb or {}) if (ma or {}).get(k) != (mb or {}).get(k)})
print("names equal:", sorted(ha) == sorted(hb))
with safe_open(a, "pt") as fa, safe_open(b, "pt") as fb:
    for n in sorted(set(ha) & set(hb)):
        x, y = fa.get_tensor(n), fb.get_tensor(n)
        print(n, x.dtype, tuple(x.shape), "equal" if torch.equal(x, y) else f"DIFF max {(x.float() - y.float()).abs().max().item():.3e} frac {(x != y).float().mean().item():.3e}")
