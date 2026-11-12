#!/usr/bin/env python3
"""verdict115.py B ARM GOLD_X GOLD_C -- #152-02 / #153-02 verdict (../w115.md), w113's verdict arithmetic (pair84.py).
Rows B/scores/<tag>-{x,c}.json.rows, tags N<S> (control) and <ARM><S> (R = re-LDLQ, C = tile CD families), S = 0, 1.
Needs B/mix/hyg-<ARM><S>.ok for both seeds. Writes B/verdict-<ARM>.json, B/docstat-N<S>-<ARM><S>.txt; last line = verdict.
Adopt-proposal iff pooled geo <= 0.995, seed-geo of the xcorp ratios < 1 and of the calib ratios < 1, and the seed-geo of
the per-document geo (both parts, docstat "== both") < 1."""
import json, math, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w84"))
from pair84 import pair, fmt, doc_sums  # noqa: E402

B, ARM, GX, GC = sys.argv[1:5]
G = {"x": GX, "c": GC}
rows = lambda t, p: f"{B}/scores/{t}-{p}.json.rows"
parts = lambda a, b, ps="xc": [(G[p], rows(a, p), rows(b, p)) for p in ps]
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
for s in "01":
    if not os.path.exists(f"{B}/mix/hyg-{ARM}{s}.ok"): sys.exit(f"seed {s}: no hygiene pass, nothing is read")
res, pdg = {}, {}
for s in "01":
    a, b = f"N{s}", f"{ARM}{s}"
    res[s] = {"pooled": pair(parts(a, b)), "xcorp": pair(parts(a, b, "x")), "calib": pair(parts(a, b, "c"))}
    print(f"seed {s}: {a} -> {b}\n" + "".join(f"  {k:7s} {fmt(res[s][k])}\n" for k in ("pooled", "xcorp", "calib")), end="")
    ds = os.path.join(HERE, "..", "..", "..", "..", "eval", "docstat", "docstat.py")
    out = subprocess.run([sys.executable, ds, GX, GC, rows(a, "x"), rows(a, "c"), rows(b, "x"), rows(b, "c")],
                         capture_output=True, text=True).stdout
    open(f"{B}/docstat-{a}-{b}.txt", "w").write(out)
    print("".join("  " + l + "\n" for l in out.splitlines() if l.startswith("==")), end="")
    pdg[s] = float(re.search(r"== both: .*per-doc geo ([0-9.]+)", out).group(1))
g = geo([res[s]["pooled"]["r"] for s in res])
gx, gc = geo([res[s]["xcorp"]["r"] for s in res]), geo([res[s]["calib"]["r"] for s in res])
gd = geo(list(pdg.values()))
S = {(s, p, t): doc_sums(G[p], rows(f, p))[0] for s in res for p in "xc" for t, f in (("A", f"N{s}"), ("B", f"{ARM}{s}"))}
rng, bs = np.random.default_rng(0), []
nd = {p: len(S[("0", p, "A")]) for p in "xc"}
for _ in range(2000):
    idx = {p: rng.integers(0, nd[p], nd[p]) for p in "xc"}
    lr = [math.log(sum(S[(s, p, "B")][idx[p]].sum() for p in "xc") / sum(S[(s, p, "A")][idx[p]].sum() for p in "xc")) for s in res]
    bs.append(sum(lr) / len(lr))
se_doc = g * float(np.std(bs)); se = math.sqrt(se_doc ** 2 + 0.0086 ** 2 / len(res))
v = g <= 0.995 and gx < 1 and gc < 1 and gd < 1
json.dump({"arm": ARM, "pairs": res, "geo": g, "geo_x": gx, "geo_c": gc, "perdoc_geo": pdg, "perdoc_geo_seeds": gd,
           "se_doc": se_doc, "se_comb": se, "z": (g - 1) / se, "pass": v}, open(f"{B}/verdict-{ARM}.json", "w"), indent=1)
rr = ", ".join("%.4f" % res[s]["pooled"]["r"] for s in "01")
print(f"W115 {ARM} VERDICT {'PASS (adopt-proposal)' if v else 'FAIL'}: N -> {ARM} pooled r = {rr}, geo {g:.4f} (xcorp {gx:.4f}, "
      f"calib {gc:.4f}), per-doc geo {gd:.4f}, SE comb {se:.4f}, z {(g - 1) / se:+.2f}")
