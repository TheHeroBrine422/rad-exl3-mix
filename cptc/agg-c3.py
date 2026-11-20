#!/usr/bin/env python3
"""Aggregate C3 rows (thor + box2 + colo c3.jsonl) into the CPTC-vs-baselines table.
Usage: python3 agg-c3.py c3.jsonl [more.jsonl ...]"""
import json, sys
from collections import defaultdict

BASELINES = ["mul1", "mul1_i8", "gauss_i8", "int4grid"]
rows = []
for f in sys.argv[1:]:
    for line in open(f):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
# dedupe by (layer, w): first occurrence wins
seen, uniq = set(), []
for r in rows:
    k = (r.get("layer"), r.get("w"))
    if k not in seen:
        seen.add(k); uniq.append(r)

cls = {}
for r in uniq:
    w = r["w"]
    c = ("attn_full" if w.startswith("attn") else
         "ssm" if w.startswith("ssm") else "ffn_down" if w == "ffn_down" else "ffn_gate_up")
    cls[(r["layer"], r["w"])] = c

byK = defaultdict(list)
for r in uniq:
    for K, d in r.get("per_K", {}).items():
        cptc = d.get(f"cptc{K}")
        if cptc is None: continue
        base = {b: d.get(b) for b in BASELINES if d.get(b) is not None}
        best = min(base.items(), key=lambda kv: kv[1])
        byK[K].append((cls[(r["layer"], r["w"])], r["w"], r.get("layer"), cptc, best[1], best[0],
                       cptc / best[1] - 1.0, len(base)))

print(f"{'K':>2} {'class':<12} {'layer/w':<16} {'cptc':>9} {'bestbase':>9} {'base':<10} {'vs best':>8} {'n':>2}")
for K in sorted(byK):
    for (c, w, L, cptc, bb, bname, vs, n) in sorted(byK[K]):
        print(f"{K:>2} {c:<12} {str(L)+'/'+w:<16} {cptc:9.4%} {bb:9.4%} {bname:<10} {vs:+8.1%} {n:>2}")
    wins = sum(1 for t in byK[K] if t[6] < -0.015)
    ties = sum(1 for t in byK[K] if -0.015 <= t[6] <= 0.015)
    losses = sum(1 for t in byK[K] if t[6] > 0.015)
    print(f"  K{K}: {len(byK[K])} rows -- wins {wins} / ties {ties} / losses {losses} (±1.5% = paired-SE noise bar)")
