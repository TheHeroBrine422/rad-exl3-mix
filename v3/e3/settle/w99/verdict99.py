#!/usr/bin/env python3
"""verdict99.py A h|k6 -- ../w99.md's rules on the rows w99.sh scored on golden-search27c (two parts), with w84's pair84.py.
  h   #128: per seed pair S (hpS -> hbS) the pooled / xcorp / calib ratios; geo over the pairs; HOLDS / FIRES /
      INCONCLUSIVE; combined SE of geo = sqrt(SE_doc(geo)^2 + 0.0086^2 / 2) (SE_doc from one joint document bootstrap of
      both pairs, stratified by part; 0.0086 = #116's same-seed pair SD)
  k6  #129: k6c9 -> k6swj pooled / parts; KEEP (r <= 0.995) or PROPOSE k_max 4.5
Writes A/verdict-<item>.json; the last line is the verdict."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w84"))
from pair84 import pair, fmt, doc_sums

A, item = sys.argv[1], sys.argv[2]
G = {"x": f"{A}/golden-search27c-x", "c": f"{A}/golden-search27c-c"}
rows = lambda t, p: f"{A}/scores/{t}-{p}.json.rows"
parts = lambda a, b, ps="xc": [(G[p], rows(a, p), rows(b, p)) for p in ps]
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
hyg = lambda a, b: os.path.exists(f"{A}/mix/hyg-{a}-vs-{b}.ok")

if item == "k6":
    if not hyg("k6swj", "k6c9"): sys.exit("k6swj vs k6c9: no hygiene pass, not read")
    p, px, pc = pair(parts("k6c9", "k6swj")), pair(parts("k6c9", "k6swj", "x")), pair(parts("k6c9", "k6swj", "c"))
    print(f"k6c9 -> k6swj:\n  pooled  {fmt(p)}\n  xcorp   {fmt(px)}\n  calib   {fmt(pc)}")
    v = "KEEP swj above K4.5" if p["r"] <= 0.995 else "PROPOSE k_max = 4.5"
    json.dump({"pooled": p, "xcorp": px, "calib": pc, "verdict": v}, open(f"{A}/verdict-k6.json", "w"), indent=1)
    print(f"W99 #129 VERDICT {v}: r = {p['r']:.4f} (xcorp {px['r']:.4f}, calib {pc['r']:.4f}), z {p['z']:+.2f}")
    sys.exit(0)

res = {}
for s in "01":
    a, b = f"hp{s}", f"hb{s}"
    if not all(os.path.exists(rows(t, p)) for t in (a, b) for p in "xc"): continue
    if not hyg(b, a): sys.exit(f"{b} vs {a}: no hygiene pass, the pair is not read")
    res[s] = {"pooled": pair(parts(a, b)), "xcorp": pair(parts(a, b, "x")), "calib": pair(parts(a, b, "c"))}
    print(f"seed {s}: {a} -> {b}\n" + "".join(f"  {k:7s} {fmt(res[s][k])}\n" for k in ("pooled", "xcorp", "calib")), end="")
if not res: sys.exit("no scored pair")
R = [res[s]["pooled"]["r"] for s in res]
g = geo(R)
gx, gc = geo([res[s]["xcorp"]["r"] for s in res]), geo([res[s]["calib"]["r"] for s in res])

# joint document bootstrap of geo (both pairs resampled on the same documents, stratified by part)
S = {}
for s in res:
    for p in "xc":
        S[(s, p, "a")] = doc_sums(G[p], rows(f"hp{s}", p))[0]; S[(s, p, "b")] = doc_sums(G[p], rows(f"hb{s}", p))[0]
rng, bs = np.random.default_rng(0), []
nd = {p: len(S[(next(iter(res)), p, "a")]) for p in "xc"}
for _ in range(2000):
    idx = {p: rng.integers(0, nd[p], nd[p]) for p in "xc"}
    lr = [math.log(sum(S[(s, p, "b")][idx[p]].sum() for p in "xc") / sum(S[(s, p, "a")][idx[p]].sum() for p in "xc")) for s in res]
    bs.append(sum(lr) / len(lr))
se_doc = g * float(np.std(bs))
se = math.sqrt(se_doc ** 2 + 0.0086 ** 2 / len(res))
if g >= 1.0: v = "FIRES"
elif g <= 0.985 and all(r < 1 for r in R): v = "HOLDS"
else: v = "INCONCLUSIVE"
if len(res) == 1: v += " provisional (1 pair)"
json.dump({"pairs": res, "geo": g, "geo_x": gx, "geo_c": gc, "se_doc": se_doc, "se_comb": se, "verdict": v},
          open(f"{A}/verdict-h.json", "w"), indent=1)
print(f"W99 #128 VERDICT {v}: hp -> hb pooled r = {', '.join(f'{x:.4f}' for x in R)}, geo {g:.4f} "
      f"(xcorp {gx:.4f}, calib {gc:.4f}); SE doc {se_doc:.4f}, combined {se:.4f}: z(vs 1) {(g - 1) / se:+.2f}, "
      f"z(vs 0.985) {(g - 0.985) / se:+.2f}")
