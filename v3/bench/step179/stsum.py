#!/usr/bin/env python3
"""stsum.py [--kind decode|prefill] [--top K] A.csv [B.csv ...] -- RADIANCE_AQL_STAMP CSVs (stamp.sh) side by side: per run
the pass kind (records a pass) taken -- decode: the most frequent kind; prefill: the kinds with more than 2000 records,
summed over a request (one of each chunk) -- then wall / busy / idle ms a pass, dispatches, and per kernel name ms and
launches a pass (median over complete passes), sorted by the first run's time. Uses scripts/stampgaps.py's reading
rules (a pass shorter than its kind's longest is left out). Standard library only. (w152, #179)"""
import csv, statistics as st, sys
from collections import defaultdict, Counter
a = sys.argv[1:]
kind = "decode"; top = 40
if "--kind" in a: i = a.index("--kind"); kind = a[i + 1]; del a[i:i + 2]
if "--top" in a: i = a.index("--top"); top = int(a[i + 1]); del a[i:i + 2]

def union(iv):
    tot, ca, cb = 0, None, None
    for x, y in sorted(iv):
        if cb is None or x > cb:
            if cb is not None: tot += cb - ca
            ca, cb = x, y
        elif y > cb: cb = y
    return tot + (cb - ca if cb is not None else 0)

def load(path):
    P = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")): P[int(r["pass"])].append(r)
    kinds = defaultdict(list)
    for p, rows in sorted(P.items()): kinds[int(rows[0]["records"])].append(rows)
    def stats(rows):
        if any(not r["start_ns"] or not r["end_ns"] for r in rows): return None
        iv = [(int(r["start_ns"]), int(r["end_ns"])) for r in rows]
        k, n = defaultdict(int), Counter()
        for r, (x, y) in zip(rows, iv): k[r["kernel"]] += y - x; n[r["kernel"]] += 1
        wall = max(y for _, y in iv) - min(x for x, _ in iv)
        return {"wall": wall, "busy": union(iv), "n": len(rows), "k": k, "c": n}
    if kind == "decode":
        ks = [max(kinds, key=lambda q: len(kinds[q]))]
    else:
        ks = sorted(q for q in kinds if q > 2000)
    out = {"wall": 0, "busy": 0, "n": 0, "k": defaultdict(float), "c": defaultdict(float), "np": []}
    for q in ks:
        full = max(len(r) for r in kinds[q])
        res = [s for s in (stats(r) if len(r) == full else None for r in kinds[q]) if s]
        if not res: continue
        if kind == "prefill": res = res[-1:]   # the request's own chunk (startup's warm passes come first)
        out["np"].append(len(res))
        for f in ("wall", "busy", "n"): out[f] += st.median(s[f] for s in res)
        names = set().union(*(s["k"] for s in res))
        for nm in names:
            out["k"][nm] += st.median(s["k"].get(nm, 0) for s in res)
            out["c"][nm] += st.median(s["c"].get(nm, 0) for s in res)
    return out

R = [load(p) for p in a]
lab = [p.rsplit("/", 1)[-1][:-4][-22:] for p in a]
print("run".ljust(46) + "".join(f"{l:>24s}" for l in lab))
for f, nm in (("wall", "wall ms a pass"), ("busy", "busy ms (union)")):
    print(nm.ljust(46) + "".join(f"{r[f] / 1e6:24.3f}" for r in R))
print("idle ms".ljust(46) + "".join(f"{(r['wall'] - r['busy']) / 1e6:24.3f}" for r in R))
print("dispatches a pass".ljust(46) + "".join(f"{r['n']:24.0f}" for r in R))
print("complete passes used".ljust(46) + "".join(f"{'+'.join(map(str, r['np'])):>24s}" for r in R))
names = sorted(set().union(*(r["k"] for r in R)), key=lambda k: -max(r["k"].get(k, 0) for r in R))
print(f"\nkernel (µs a pass / launches)".ljust(46))
for k in names[:top]:
    print(k[:45].ljust(46) + "".join(f"{r['k'].get(k, 0) / 1e3:15.1f} / {r['c'].get(k, 0):6.0f}" for r in R))
rest = names[top:]
if rest:
    print(f"({len(rest)} more)".ljust(46) + "".join(f"{sum(r['k'].get(k, 0) for k in rest) / 1e3:15.1f} / {sum(r['c'].get(k, 0) for k in rest):6.0f}" for r in R))
