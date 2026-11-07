#!/usr/bin/env python3
"""cmpjobs.py A B -- two rxq job dirs: every t-<key>.safetensors of A equal in B (tensor names, dtypes, values) and the
log records' proxy_err / g_scale / sigma_reg equal. Last line "JOBS SAME|DIFF n/m"."""
import json, os, sys
import torch
from safetensors import safe_open

A, B = sys.argv[1:3]
logs = lambda d: {r["key"]: r for r in map(json.loads, open(os.path.join(d, "log.jsonl")))}
la, lb = logs(A), logs(B)
keys = sorted(f[2:-12] for f in os.listdir(A) if f.startswith("t-") and f.endswith(".safetensors"))
ok = 0
for k in keys:
    pb = os.path.join(B, f"t-{k}.safetensors")
    if not os.path.exists(pb): print(f"DIFF {k}: missing in B"); continue
    with safe_open(os.path.join(A, f"t-{k}.safetensors"), "pt") as x, safe_open(pb, "pt") as y:
        same = sorted(x.keys()) == sorted(y.keys()) and all(x.get_tensor(n).dtype == y.get_tensor(n).dtype and
                                                             torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys())
    lg = [f for f in ("proxy_err", "g_scale", "sigma_reg") if la.get(k, {}).get(f) != lb.get(k, {}).get(f)]
    ok += same and not lg
    print(f"{'SAME' if same and not lg else 'DIFF'} {k}: tensors {'equal' if same else 'differ'}; log fields differ {lg}")
print(f"JOBS {'SAME' if keys and ok == len(keys) else 'DIFF'} {ok}/{len(keys)}")
