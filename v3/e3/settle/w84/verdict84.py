#!/usr/bin/env python3
"""verdict84.py A ARM [ARM ...] -- ../w84.md's rule (amendment 1) over the seed pairs w84.sh wrote. Per arm: control ->
permuted-order arm on golden-search27b's two parts (A/b27/<tag>-{x,c}.json.rows; pair84.py: per part and pooled,
position-weighted, stratified document bootstrap), and the golden-search27 pair (A/<ARM>/pair.json, report only).
ADOPT if every pooled r < 1 AND the geo over the pairs of r_x < 1 and of r_c < 1 (same sign on both parts);
a pooled pass with a part's geo >= 1 is FLAGGED (not adopted); NOT ADOPTED if every pooled r >= 1; else AMBIGUOUS."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair84 import pair, fmt
A, arms = sys.argv[1], sys.argv[2:]
GX, GC, G27 = f"{A}/golden-search27b-x", f"{A}/golden-search27b-c", "/data/swap27/golden-search27"
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
R = {"pooled": [], "x": [], "c": [], "s27": []}; out = {}
for arm in arms:
    ctl = open(f"{A}/{arm}/ctl.tag").read().strip()
    px = pair([(GX, f"{A}/b27/{ctl}-x.json.rows", f"{A}/b27/{arm}perm-x.json.rows")])
    pc = pair([(GC, f"{A}/b27/{ctl}-c.json.rows", f"{A}/b27/{arm}perm-c.json.rows")])
    pp = pair([(GX, f"{A}/b27/{ctl}-x.json.rows", f"{A}/b27/{arm}perm-x.json.rows"), (GC, f"{A}/b27/{ctl}-c.json.rows", f"{A}/b27/{arm}perm-c.json.rows")])
    m = json.load(open(f"{A}/{arm}/pair-meta.json"))
    p27 = pair([(G27, m["control"], m["perm"])])
    for k, p in (("pooled", pp), ("x", px), ("c", pc), ("s27", p27)): R[k].append(p["r"])
    out[arm] = {"control": ctl, "pooled": pp, "xcorp": px, "calib": pc, "search27": p27}
    print(f"{arm} (control {ctl}):\n  golden-search27b pooled  {fmt(pp)}\n  xcorp part (65 docs)     {fmt(px)}\n"
          f"  calib part (36 docs)     {fmt(pc)}\n  golden-search27 (report) {fmt(p27)}")
zs = sum(out[a]["pooled"]["diff"] for a in arms) / math.sqrt(sum(out[a]["pooled"]["se"] ** 2 for a in arms))
gp, gx, gc = geo(R["pooled"]), geo(R["x"]), geo(R["c"])
if len(arms) >= 2 and all(r < 1 for r in R["pooled"]):
    v = "ADOPT" if gx < 1 and gc < 1 else "FLAGGED (pooled pass, a part's geo >= 1: not adopted)"
elif all(r >= 1 for r in R["pooled"]): v = "NOT ADOPTED"
else: v = "AMBIGUOUS (K4 pair on the lead's go)"
json.dump({"arms": out, "geo_pooled": gp, "geo_x": gx, "geo_c": gc, "geo_s27": geo(R["s27"]), "z_pooled_over_pairs": zs, "verdict": v},
          open(f"{A}/verdict84.json", "w"), indent=1)
print(f"W84 VERDICT {v}: pooled r = {', '.join(f'{r:.4f}' for r in R['pooled'])} (geo {gp:.4f}, pooled z {zs:+.2f}); "
      f"part geo xcorp {gx:.4f}, calib {gc:.4f}; golden-search27 geo {geo(R['s27']):.4f} (report)")
