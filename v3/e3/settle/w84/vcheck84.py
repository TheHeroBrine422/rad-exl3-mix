#!/usr/bin/env python3
"""vcheck84.py A -- the lead's routine verify check (../w84.md item 3): K2.5 H_out arms vs their plain controls on
golden-verify27 (pair84.py). H_out-everywhere stands if the geo of the three ratios is < 1."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair84 import pair, fmt
A = sys.argv[1]; GV = "/data/verify27/golden-verify27"; V = "/data/verify27/verify"
P = [("base", f"{V}/base.json.rows", "bn"), ("s1", f"{V}/s1.json.rows", "bns1"), ("s2", f"{A}/vchk/s2.json.rows", "bns2")]
rs, ds, ses, out = [], [], [], {}
for c, cr, a in P:
    p = pair([(GV, cr, f"{A}/vchk/{a}.json.rows")]); out[f"{c}->{a}"] = p
    rs.append(p["r"]); ds.append(p["diff"]); ses.append(p["se"]); print(f"{c} -> {a}: {fmt(p)}")
g = math.exp(sum(map(math.log, rs)) / len(rs)); z = sum(ds) / math.sqrt(sum(s * s for s in ses))
v = "STANDS" if g < 1 else "DOES NOT STAND"
json.dump({"pairs": out, "geo": g, "z_pooled": z, "verdict": v}, open(f"{A}/vchk/vcheck84.json", "w"), indent=1)
print(f"VCHECK H_out K2.5 on golden-verify27: geo {g:.4f} ({', '.join(f'{r:.4f}' for r in rs)}), pooled z {z:+.2f} -> H_out-everywhere {v} (rule: geo < 1)")
