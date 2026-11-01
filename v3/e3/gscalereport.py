#!/usr/bin/env python3
"""gscalereport.py GS.jsonl -- per-group scales at equal bits: group rel at K vs the no-group baseline interpolated
(log-linear) at K + overhead. ratio < 1: groups win. Also as bits: (log2 baseline-slope) equivalent bits saved."""
import collections, json, math, statistics as st, sys

B = collections.defaultdict(dict); G = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    d = json.loads(l); key = (d["layer"], d["w"])
    (B[key] if d["G"] == 0 else G[key])[(d["G"], d["bits"])] = (d["rel"], d["overhead"])
def base_at(b, x):
    ks = sorted(k for _, k in b)
    lo = max(k for k in ks if k <= x); hi = min(k for k in ks if k >= x)
    if lo == hi: return b[(0, lo)][0]
    a, c = math.log(b[(0, lo)][0]), math.log(b[(0, hi)][0])
    return math.exp(a + (x - lo) / (hi - lo) * (c - a))
rows = collections.defaultdict(list)
for key, g in G.items():
    for (gg, k), (rel, ov) in g.items():
        if (0, k) not in B[key]: continue
        bl = base_at(B[key], k + ov); r = rel / bl
        rows[(gg, k)].append((r, key, rel / B[key][(0, k)][0]))
print("G    K    ratio vs baseline at K+overhead: median [min, max] | raw ratio vs baseline at K (no overhead charged)")
for (gg, k) in sorted(rows):
    v = rows[(gg, k)]; r = [x[0] for x in v]; raw = [x[2] for x in v]
    print(f"{gg:<4} {k:<4} {st.median(r):.3f} [{min(r):.3f}, {max(r):.3f}]  n={len(r)} | raw {st.median(raw):.3f}")
for (gg, k) in sorted(rows):
    w = min(rows[(gg, k)])
    print(f"  best tensor G{gg} K{k}: {w[1]} ratio {w[0]:.3f}")
