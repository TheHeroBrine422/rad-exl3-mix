#!/usr/bin/env python3
"""dprof2.py [--prefill] [--ops] LOG... -- per-op µs a step from radiance --profile-ops dumps (dprof.sh logs).
Decode (default): last − first dump, the step count from the head's calls (one logits launch a step), so a short final
interval counts right (the lead's dprof.py assumes 64 steps a dump). --prefill: the last dump whole, per prefill (the head
launches once a request). Rows: weight families (gemm X = exl3i_dense / exl3_dense on blk.N.X) and other ops by name;
--ops: by op name only. Also launches a step and the profile's `wait us` (lane idle before the op, a step). (w152, #179;
from the lead's v3/bench/iqrcmp/dprof.py)"""
import re, sys, collections
ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+|-)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PRE, OPS = "--prefill" in sys.argv, "--ops" in sys.argv
def dumps(p):
    out, cur = [], None
    for l in open(p, errors="replace"):
        if "per-op timing, rank 0" in l: cur = {}; out.append(cur); continue
        if cur is None: continue
        m = ROW.match(l)
        if m:
            w = float(m.group(6)) if m.group(6) != "-" else 0.0
            cur[(m.group(7), m.group(8), int(m.group(9)), m.group(10))] = (int(m.group(1)), float(m.group(2)), w)
    return out
def fam(op, w):
    if OPS: return op
    m = re.match(r"(?:blk\.\d+\.)?([a-z_]+)", w)
    if m and w != "-": return op.replace("exl3i_dense", "gemm").replace("exl3_dense", "gemm") + " " + m.group(1)
    return "other " + op
res, meta = {}, {}
for p in args:
    d = dumps(p)
    if not d: print(p, "no dumps"); continue
    if PRE: a, b = {}, d[-1]
    else:
        if len(d) < 2: print(p, "dumps", len(d)); continue
        a, b = d[0], d[-1]
    head = [k for k in b if k[3].startswith("output.weight")]
    steps = sum(b[k][0] - a.get(k, (0, 0, 0))[0] for k in head) or 1
    t, nl, wt = collections.Counter(), 0, 0.0
    for k, (n, ms, w) in b.items():
        n0, ms0, w0 = a.get(k, (0, 0.0, 0.0))
        t[fam(k[0], k[3])] += 1000 * (ms - ms0) / steps
        nl += (n - n0); wt += w * n - w0 * n0
    res[p] = t; meta[p] = (steps, nl / steps, wt / steps)
keys = sorted(set().union(*[set(t) for t in res.values()]), key=lambda k: -max(t.get(k, 0) for t in res.values()))
nm = lambda p: p.split("/")[-1].replace(".log", "")[-16:]
print(f"{'family (µs / ' + ('prefill' if PRE else 'step') + ')':34s} " + " ".join(f"{nm(p):>16s}" for p in res))
for k in keys:
    if max(t.get(k, 0) for t in res.values()) < 20: continue
    print(f"{k[:34]:34s} " + " ".join(f"{t.get(k, 0):16.1f}" for t in res.values()))
print(f"{'TOTAL':34s} " + " ".join(f"{sum(t.values()):16.1f}" for t in res.values()))
print(f"{'steps / launches a step':34s} " + " ".join(f"{meta[p][0]:7d} / {meta[p][1]:6.0f}" for p in res))
print(f"{'wait µs a step (lane idle)':34s} " + " ".join(f"{meta[p][2]:16.1f}" for p in res))
