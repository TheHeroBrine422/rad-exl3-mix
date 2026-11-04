#!/usr/bin/env python3
"""eqsize.py RESULTS.json REF CMP [CMP ...] [--md] -- equal-quality size: for every entry of each CMP series, the size
the REF series needs for the same mean KL (log-KL interpolated linearly in bpw between REF's neighbouring entries; no
extrapolation past REF's ends), and how much smaller / larger CMP is: Δbpw and the size ratio. A series is a board
group id (e.g. exl3v2, iqr4, iqr3, ud, exl3) or a name prefix (e.g. "EXL3-SC"). bpw = bytes x 8 / params.
Public framing (user, 2026-11-04): "X% smaller at the same quality" reads better than KL ratios."""
import json, math, sys

a = [x for x in sys.argv[1:] if not x.startswith("--")]; md = "--md" in sys.argv
d = json.load(open(a[0])); P = d["params"]

def series(s):
    pts = [(e["bytes"] * 8 / P, e["scores"]["kld_mean"], e["name"]) for e in d["entries"]
           if (e.get("group") == s or e["name"].startswith(s)) and e.get("scores", {}).get("kld_mean")]
    return sorted(pts)

def bpw_at(ref, k):
    """REF's bpw at mean KL k (KL falls with bpw); None outside REF's range"""
    for (b0, k0, _), (b1, k1, _) in zip(ref, ref[1:]):
        if k1 <= k <= k0 or k0 <= k <= k1:
            if k0 == k1: return b0
            return b0 + (math.log(k0) - math.log(k)) / (math.log(k0) - math.log(k1)) * (b1 - b0)
    return None

ref = series(a[1])
assert ref, f"no entries for {a[1]}"
for s in a[2:]:
    rows = []
    for b, k, n in series(s):
        rb = bpw_at(ref, k)
        if rb is None: continue
        rows.append((n, b, k, rb, b - rb, b / rb))
    if md:
        print(f"\n| {s} entry | bpw | KL | {a[1]} at that KL | Δbpw | size vs {a[1]} |\n|---|---|---|---|---|---|")
        for n, b, k, rb, db, r in rows: print(f"| {n} | {b:.2f} | {k:.5f} | {rb:.2f} | {db:+.2f} | {100 * (r - 1):+.1f}% |")
    else:
        print(f"== {s} vs {a[1]} (same KL)")
        for n, b, k, rb, db, r in rows: print(f"  {n:28s} {b:5.2f} bpw  KL {k:.5f}  {a[1]} needs {rb:5.2f}  Δ {db:+.2f} bpw  ({100 * (r - 1):+.1f}%)")
