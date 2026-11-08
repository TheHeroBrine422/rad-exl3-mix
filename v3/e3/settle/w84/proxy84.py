#!/usr/bin/env python3
"""proxy84.py CONTROL_PROXY.jsonl ARM_LOG.jsonl -- the readers' one-sided proxy_err (ourq / rxq log records, after the
scale refit) arm / control, geometric mean overall and per family (#117, ../w84.md: reported, not deciding)."""
import json, math, sys
fam = lambda k: next((f for s, f in (("q_proj", "attn_qg"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("in_proj_qkv", "ssm_qkv"),
                                     ("in_proj_z", "ssm_z"), ("gate_proj", "ffn_gate"), ("up_proj", "ffn_up")) if k.endswith(s)), None)
rd = lambda f: {j["key"]: j["proxy_err"] for j in map(json.loads, open(f)) if fam(j["key"])}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
r = {k: b[k] / a[k] for k in b if k in a}
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
fams = sorted({fam(k) for k in r})
print(" ".join(f"{f} {geo([v for k, v in r.items() if fam(k) == f]):.4f}" for f in fams))
print(f"{len(r)} readers: geo {geo(list(r.values())):.4f} (" + ", ".join(f"{f} {geo([v for k, v in r.items() if fam(k) == f]):.4f}" for f in fams)
      + f"), < 1: {sum(v < 1 for v in r.values())}/{len(r)}")
