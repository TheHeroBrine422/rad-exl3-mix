#!/usr/bin/env python3
"""cmp.py A B NAME -- two ourq outputs (layers 3-4): shard tensors byte-identical and proxy rows equal -> PASS"""
import json, sys, torch
from safetensors import safe_open
a, b, name = sys.argv[1:4]; diff = n = 0
for f in ("q-L03.safetensors", "q-L04.safetensors"):
    with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
        assert set(x.keys()) == set(y.keys()), f
        for k in x.keys():
            n += 1; ta, tb = x.get_tensor(k), y.get_tensor(k)
            if ta.shape != tb.shape or not bool((ta.contiguous().view(-1).view(torch.uint8) == tb.contiguous().view(-1).view(torch.uint8)).all()): diff += 1
pa = [json.loads(l) for l in open(f"{a}/proxy.jsonl")]; pb = [json.loads(l) for l in open(f"{b}/proxy.jsonl")]
pd = sum(1 for p, q in zip(pa, pb) if (p["key"], p["proxy_err"], p["sigma_reg"]) != (q["key"], q["proxy_err"], q["sigma_reg"]))
ok = diff == 0 and pd == 0 and len(pa) == len(pb)
print(f"  {name}: {n} tensors, {diff} differ; proxy rows {len(pa)}/{len(pb)}, {pd} differ -> {'PASS' if ok else 'FAIL'}")
