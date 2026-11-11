#!/usr/bin/env python3
"""verdict130c.py C GOLDDIR -- w101 #130-03 verdict (../w101.md amendment 1), pair84.py's arithmetic as verdict99.py.
Per seed pair S (N_S -> P_S) on golden-search27c (GOLDDIR/golden-search27c-{x,c}; rows C/scores/<arm><S>-<part>.json.rows):
pooled / xcorp / calib ratios; PASS iff geo of the pooled ratios <= 0.995 and the geo over the pairs of the xcorp and of the
calib ratios are both < 1. Combined SE = sqrt(SE_doc(geo)^2 + 0.0086^2 / n_pairs) (joint document bootstrap stratified by
part, #116's same-seed pair SD). Needs C/hyg-S.ok (gates G-c1..3) for each pair. Writes C/verdict130c.json; last line = verdict."""
import json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w84"))
from pair84 import pair, fmt, doc_sums  # noqa: E402

C, GD = sys.argv[1:3]
G = {"x": f"{GD}/golden-search27c-x", "c": f"{GD}/golden-search27c-c"}
rows = lambda t, p: f"{C}/scores/{t}-{p}.json.rows"
parts = lambda a, b, ps="xc": [(G[p], rows(a, p), rows(b, p)) for p in ps]
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))

res = {}
for s in "01":
    a, b = f"N{s}", f"P{s}"
    if not all(os.path.exists(rows(t, p)) for t in (a, b) for p in "xc"): continue
    if not os.path.exists(f"{C}/hyg-{s}.ok"): sys.exit(f"seed {s}: no gate pass, the pair is not read")
    res[s] = {"pooled": pair(parts(a, b)), "xcorp": pair(parts(a, b, "x")), "calib": pair(parts(a, b, "c"))}
    print(f"seed {s}: {a} -> {b}\n" + "".join(f"  {k:7s} {fmt(res[s][k])}\n" for k in ("pooled", "xcorp", "calib")), end="")
if not res: sys.exit("no scored pair")
R = [res[s]["pooled"]["r"] for s in res]
g, gx, gc = geo(R), geo([res[s]["xcorp"]["r"] for s in res]), geo([res[s]["calib"]["r"] for s in res])
S = {(s, p, t): doc_sums(G[p], rows(f"{t}{s}", p))[0] for s in res for p in "xc" for t in "NP"}
rng, bs = np.random.default_rng(0), []
nd = {p: len(S[(next(iter(res)), p, "N")]) for p in "xc"}
for _ in range(2000):
    idx = {p: rng.integers(0, nd[p], nd[p]) for p in "xc"}
    lr = [math.log(sum(S[(s, p, "P")][idx[p]].sum() for p in "xc") / sum(S[(s, p, "N")][idx[p]].sum() for p in "xc")) for s in res]
    bs.append(sum(lr) / len(lr))
se_doc = g * float(np.std(bs)); se = math.sqrt(se_doc ** 2 + 0.0086 ** 2 / len(res))
v = "PASS" if g <= 0.995 and gx < 1 and gc < 1 else "FAIL"
if len(res) < 2: v += " provisional (1 pair)"
json.dump({"pairs": res, "geo": g, "geo_x": gx, "geo_c": gc, "se_doc": se_doc, "se_comb": se, "verdict": v},
          open(f"{C}/verdict130c.json", "w"), indent=1)
print(f"W101 #130-03 VERDICT {v}: N -> P16 pooled r = {', '.join(f'{x:.4f}' for x in R)}, geo {g:.4f} "
      f"(xcorp {gx:.4f}, calib {gc:.4f}), SE comb {se:.4f} (doc {se_doc:.4f}), z {(g - 1) / se:+.2f}")
