#!/usr/bin/env python3
"""verdict92.py A [--k1] -- ../w92.md's gate over the K2.5 seed pairs w92.sh scored on golden-search27c (two parts), with
w84's pair84.py (position-weighted, document bootstrap stratified by part). Per seed pair (control, full stack, bn+swj):
control -> fs pooled / xcorp / calib, control -> bn+swj (the bar), bn+swj -> fs (direct, report). PASS if geo r(ctl -> fs)
<= geo r(ctl -> bn+swj) and every pooled and part ratio of ctl -> fs <= 1.0; else FAIL, labelled "integration alarm" (a
pooled r(ctl -> fs) > 1 or a direct bn+swj -> fs pooled z > +2) or "bar, within noise" (only the bar missed, direct |z| <= 2).
A bn+swj mix without a hygiene pass vs its control is replaced by 0.924 (#104 amendment 1). One pair: "provisional (1 pair)".
--k1: the K1 i4 level's scores (finite rows, KLD / top-1 / p99 per part and pooled; no gate)."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w84"))
from pair84 import pair, fmt, doc_sums
A = sys.argv[1]; K1 = "--k1" in sys.argv
GX, GC = f"{A}/golden-search27c-x", f"{A}/golden-search27c-c"
rows = lambda t, p: f"{A}/scores/{t}-{p}.json.rows"
have = lambda t: all(os.path.exists(rows(t, p)) for p in "xc")
hyg = lambda a, b: os.path.exists(f"{A}/mix/hyg-{a}-vs-{b}.ok")
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
nx, nc = (len(json.load(open(f"{g}/kld.json"))["docs"]) for g in (GX, GC))

if K1:
    out = {}
    for part, g in (("x", GX), ("c", GC)):
        s, n, t, R = doc_sums(g, rows("k1", part))
        out[part] = {"kld": float(s.sum() / n.sum()), "top1": float(t.sum() / n.sum()), "p99": float(np.percentile(R[:, 0], 99)),
                     "finite": bool(np.isfinite(R[:, 0]).all()), "positions": int(n.sum()), "docs": len(s), "sum": float(s.sum())}
        print(f"k1 {part}: KLD {out[part]['kld']:.5f} top-1 {out[part]['top1']:.4f} p99 {out[part]['p99']:.3f} "
              f"finite {out[part]['finite']} ({out[part]['docs']} docs, {out[part]['positions']} pos)")
    P = sum(o["sum"] for o in out.values()) / sum(o["positions"] for o in out.values())
    ok = all(o["finite"] for o in out.values())
    json.dump({"parts": out, "pooled_kld": P, "finite": ok}, open(f"{A}/k1.json", "w"), indent=1)
    print(f"W92 K1 {'END-TO-END OK' if ok else 'NON-FINITE KLD'}: pooled KLD {P:.5f} (xcorp {out['x']['kld']:.5f}, calib {out['c']['kld']:.5f})")
    sys.exit(0)

res, R, B = {}, [], []
for ctl, fs, ref in (("basectl", "fs0", "bnswj"), ("s1ctl", "fs1", "bns1swj")):
    if not (have(ctl) and have(fs)): continue
    if not hyg(fs, ctl): sys.exit(f"{fs} vs {ctl}: no hygiene pass, the pair is not read")
    parts = lambda a, b: [(GX, rows(a, "x"), rows(b, "x")), (GC, rows(a, "c"), rows(b, "c"))]
    pp, px, pc = pair(parts(ctl, fs)), pair(parts(ctl, fs)[:1]), pair(parts(ctl, fs)[1:])
    r = {"control": ctl, "arm": fs, "pooled": pp, "xcorp": px, "calib": pc}
    print(f"{ctl} -> {fs}:\n  pooled  {fmt(pp)}\n  xcorp ({nx}) {fmt(px)}\n  calib ({nc}) {fmt(pc)}")
    if have(ref) and hyg(ref, ctl):
        pb, pd = pair(parts(ctl, ref)), pair(parts(ref, fs))
        pbx, pbc = pair(parts(ctl, ref)[:1]), pair(parts(ctl, ref)[1:])
        r.update({"ref": ref, "bar": pb["r"], "ref_pooled": pb, "ref_xcorp": pbx, "ref_calib": pbc, "direct": pd})
        print(f"{ctl} -> {ref} (bar):\n  pooled  {fmt(pb)}\n  xcorp   {fmt(pbx)}\n  calib   {fmt(pbc)}\n"
              f"{ref} -> {fs} (direct):\n  pooled  {fmt(pd)}")
    else:
        r.update({"ref": None, "bar": 0.924}); print(f"  {ref}: no hygienic mix scored: bar 0.924 (#104 amendment 1)")
    res[fs] = r; R.append(pp["r"]); B.append(r["bar"])
if not res: sys.exit("no scored pair")
g, gb = geo(R), geo(B)
c1 = g <= gb
c2 = all(res[a][k]["r"] <= 1.0 for a in res for k in ("pooled", "xcorp", "calib"))
alarm = any(res[a]["pooled"]["r"] > 1 for a in res) or any(res[a].get("direct", {}).get("z", 0) > 2 for a in res)
dz = [res[a]["direct"]["z"] for a in res if "direct" in res[a]]
if c1 and c2: v = "PASS"
elif alarm: v = "FAIL (integration alarm)"
elif c2 and all(abs(z) <= 2 for z in dz): v = "FAIL (bar, within noise)"
else: v = "FAIL"
if len(res) == 1: v += " provisional (1 pair)"
geox, geoc = geo([res[a]["xcorp"]["r"] for a in res]), geo([res[a]["calib"]["r"] for a in res])
json.dump({"pairs": res, "geo_fs": g, "geo_bar": gb, "geo_x": geox, "geo_c": geoc, "c1_bar": c1, "c2_le1": c2, "verdict": v},
          open(f"{A}/verdict92.json", "w"), indent=1)
print(f"W92 VERDICT {v}: ctl -> fs pooled r = {', '.join(f'{x:.4f}' for x in R)} (geo {g:.4f}; xcorp geo {geox:.4f}, calib geo {geoc:.4f}) "
      f"vs bar {', '.join(f'{x:.4f}' for x in B)} (geo {gb:.4f}); direct z {', '.join(f'{z:+.2f}' for z in dz)}")
