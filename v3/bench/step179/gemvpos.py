#!/usr/bin/env python3
"""gemvpos.py A.csv B.csv -- RADIANCE_AQL_STAMP decode passes (most frequent kind) of two runs: the k-th exl3i GEMV of a
pass matched between runs (same launch order), median duration each; grouped by the kernel just before it in B (short
name) and by its own duration class, the mean B - A. Tells a clock / power effect (every GEMV slower alike) from a
neighbour effect (only those after a fused op). Standard library only. (w152, #179)"""
import csv, re, statistics as st, sys
from collections import defaultdict
def load(path):
    P = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")): P[int(r["pass"])].append(r)
    cnt = defaultdict(int)
    for v in P.values(): cnt[int(v[0]["records"])] += 1
    rec = max(cnt, key=cnt.get)
    durs, prev = defaultdict(list), {}
    for v in P.values():
        if int(v[0]["records"]) != rec or any(not r["end_ns"] for r in v): continue
        v = sorted(v, key=lambda r: int(r["seq"])); k = 0
        for i, r in enumerate(v):
            if "exl3i_gemm" in r["kernel"]:
                durs[k].append(int(r["end_ns"]) - int(r["start_ns"]))
                pk = v[i - 1]["kernel"] if i else "-"
                m = re.search(r"(\d+)([A-Za-z_][A-Za-z_0-9]*?)(_kernel|Ev|I)", pk)
                prev[k] = (m.group(2) if m else pk)[:28]
                k += 1
    return {k: st.median(x) for k, x in durs.items()}, prev
A, _ = load(sys.argv[1]); B, prev = load(sys.argv[2])
g = defaultdict(list)
for k in B:
    if k in A: g[prev[k]].append((A[k], B[k]))
print(f"{'kernel before the GEMV (in B)':30s} {'n':>4s} {'A us':>8s} {'B us':>8s} {'B-A us':>8s} {'%':>6s}")
for p, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
    a = sum(x for x, _ in v) / len(v) / 1e3; b = sum(y for _, y in v) / len(v) / 1e3
    print(f"{p:30s} {len(v):4d} {a:8.2f} {b:8.2f} {b - a:8.2f} {100 * (b - a) / a:6.1f}")
for lo, hi in ((0, 30e3), (30e3, 60e3), (60e3, 1e9)):
    v = [(A[k], B[k]) for k in B if k in A and lo <= A[k] < hi]
    if v: print(f"A in [{lo/1e3:.0f}, {hi/1e3:.0f}) us: n {len(v)}, mean B-A {sum(y - x for x, y in v) / len(v) / 1e3:+.2f} us, {100 * sum(y - x for x, y in v) / sum(x for x, _ in v):+.1f} %")
