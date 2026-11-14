#!/usr/bin/env python3
"""dprof.py LOG [LOG] -- decode µs a step per weight family from dprof.sh's --profile-ops dumps (last − first dump; step
counts from the dumps' own step numbers if present, else 64 a dump)."""
import re, sys, collections
ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
def dumps(p):
    out, cur = [], None
    for l in open(p, errors="replace"):
        if "per-op timing, rank 0" in l: cur = {}; out.append(cur); continue
        if cur is None: continue
        m = ROW.match(l)
        if m: cur[(m.group(7), m.group(8), int(m.group(9)), m.group(10))] = (int(m.group(1)), float(m.group(2)))
    return out
def fam(op, w):
    m = re.match(r"(?:blk\.\d+\.)?([a-z_]+)", w)
    if m and w != "-": return op.replace("exl3i_dense", "gemm") + " " + m.group(1)
    return "other " + op
res = {}
for p in sys.argv[1:]:
    d = dumps(p)
    if len(d) < 2: print(p, "dumps", len(d)); continue
    a, b = d[1], d[-1]; steps = 64 * (len(d) - 2)
    t = collections.Counter()
    for k, (n, us) in b.items():
        n0, us0 = a.get(k, (0, 0.0)); t[fam(k[0], k[3])] += 1000 * (us - us0) / steps
    res[p] = t
keys = sorted(set().union(*[set(t) for t in res.values()]), key=lambda k: -max(t.get(k, 0) for t in res.values()))
print(f"{'family':28s} " + " ".join(f"{p.split('/')[-1][:14]:>14s}" for p in res))
for k in keys:
    if max(t.get(k, 0) for t in res.values()) < 20: continue
    print(f"{k[:28]:28s} " + " ".join(f"{t.get(k, 0):14.1f}" for t in res.values()))
print(f"{'TOTAL µs/step':28s} " + " ".join(f"{sum(t.values()):14.1f}" for t in res.values()))
