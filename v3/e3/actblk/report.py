#!/usr/bin/env python3
"""actblk/report.py DIR [R95.jsonl] -- #103 tables from run.sh's data/*.jsonl (README.md "Rule"): ffn_down proxy error
blk128 / natural per (layer, K), paired rows; the rule on #95's tensors (L10, L30), the rest reported. With R95 given,
the natural arm is checked against #95's numbers (same harness and settings: should match to rounding)."""
import glob, json, math, os, sys
from collections import defaultdict

rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))) for l in open(f)]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
D = defaultdict(dict)
for r in rows:
    if r["method"].startswith("ldlq_"): D[(r["layer"], r["bits"])][r["order"]] = r
ref = {}
if len(sys.argv) > 2:
    for l in open(sys.argv[2]):
        r = json.loads(l)
        if r["w"] == "ffn_down" and r.get("order") == "natural" and r["method"].startswith("ldlq_"): ref[(r["layer"], r["bits"])] = r["rel"]
print("| layer | K | rel natural | rel blk128 | blk128 / natural | 128-blocks moved | block trace max/min | #95 natural (match) |")
print("|---" * 8 + "|")
for k in sorted(D, key=lambda k: (k[1], k[0])):
    d = D[k]
    if "natural" not in d or "blk128" not in d: continue
    a, b = d["natural"], d["blk128"]
    m = f"{ref[k]:.6e} ({a['rel'] / ref[k]:.6f})" if k in ref else "–"
    print(f"| {k[0]} | {k[1]:g} | {a['rel']:.4e} | {b['rel']:.4e} | {b['rel'] / a['rel']:.4f} | {b['moved']:.2f} | {b['mass_ratio']:.0f} | {m} |")
print()
for name, Ls in (("#95's down tensors (L10, L30): the rule", (10, 30)), ("all down tensors run", None)):
    print(f"{name}: per K geometric mean of blk128 / natural")
    res = {}
    for K in sorted({k[1] for k in D}):
        v = [D[k]["blk128"]["rel"] / D[k]["natural"]["rel"] for k in D if k[1] == K and len(D[k]) == 2 and (Ls is None or k[0] in Ls)]
        if v: res[K] = gm(v); print(f"  K{K:g}: {res[K]:.4f} over {len(v)} tensors (max {max(v):.4f})")
    if Ls:
        n98 = sum(v <= 0.98 for v in res.values()); worst = max(res.values()) if res else 9
        ok = len(res) == 3 and n98 >= 2 and worst <= 1.00
        print(f"  RULE (<= 0.98 at >= 2 of K2/3/4, none > 1.00): {n98} K <= 0.98, worst {worst:.4f} -> {'ADOPT' if ok else 'DO NOT ADOPT'}")
