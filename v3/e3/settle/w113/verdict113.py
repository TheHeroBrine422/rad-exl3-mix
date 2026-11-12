#!/usr/bin/env python3
"""verdict113.py C GOLD_X GOLD_C -- w113 #148-02 verdict (../w113.md), w111 verdict111.py's arithmetic (pair84.py).
Rows C/scores/<tag>-{x,c}.json.rows, tags N<S>-v (w111's natural mixes), T<S> (P16-TP, deciding), P<S>-v (full P16,
reported). Needs C/hyg-S.ok for both seeds. Writes C/verdict113.json and C/docstat-<a>-<b>.txt; last line = the verdict.
Joint z with w101's 27c pair (/data/w101/c/verdict130c.json) and w111's 27d pair D (/data/w111/verdict111.json)."""
import json, math, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w84"))
from pair84 import pair, fmt, doc_sums  # noqa: E402

C, GX, GC = sys.argv[1:4]
G = {"x": GX, "c": GC}
rows = lambda t, p: f"{C}/scores/{t}-{p}.json.rows"
parts = lambda a, b, ps="xc": [(G[p], rows(a, p), rows(b, p)) for p in ps]
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
for s in "01":
    if not os.path.exists(f"{C}/hyg-{s}.ok"): sys.exit(f"seed {s}: no gate pass, nothing is read")


def scope(nt, pt):
    res = {}
    for s in "01":
        a, b = nt.format(s), pt.format(s)
        res[s] = {"pooled": pair(parts(a, b)), "xcorp": pair(parts(a, b, "x")), "calib": pair(parts(a, b, "c"))}
        print(f"  seed {s}: {a} -> {b}\n" + "".join(f"    {k:7s} {fmt(res[s][k])}\n" for k in ("pooled", "xcorp", "calib")), end="")
        ds = os.path.join(HERE, "..", "..", "..", "..", "eval", "docstat", "docstat.py")
        out = subprocess.run([sys.executable, ds, GX, GC, rows(a, "x"), rows(a, "c"), rows(b, "x"), rows(b, "c")],
                             capture_output=True, text=True).stdout
        open(f"{C}/docstat-{a}-{b}.txt", "w").write(out)
        print("".join("    " + l + "\n" for l in out.splitlines() if l.startswith("==")), end="")
    g = geo([res[s]["pooled"]["r"] for s in res])
    gx, gc = geo([res[s]["xcorp"]["r"] for s in res]), geo([res[s]["calib"]["r"] for s in res])
    S = {(s, p, t): doc_sums(G[p], rows(f, p))[0] for s in res for p in "xc" for t, f in (("A", nt.format(s)), ("B", pt.format(s)))}
    rng, bs = np.random.default_rng(0), []
    nd = {p: len(S[("0", p, "A")]) for p in "xc"}
    for _ in range(2000):
        idx = {p: rng.integers(0, nd[p], nd[p]) for p in "xc"}
        lr = [math.log(sum(S[(s, p, "B")][idx[p]].sum() for p in "xc") / sum(S[(s, p, "A")][idx[p]].sum() for p in "xc")) for s in res]
        bs.append(sum(lr) / len(lr))
    se_doc = g * float(np.std(bs)); se = math.sqrt(se_doc ** 2 + 0.0086 ** 2 / len(res))
    return {"pairs": res, "geo": g, "geo_x": gx, "geo_c": gc, "se_doc": se_doc, "se_comb": se, "z": (g - 1) / se}


def joint(*xs):   # inverse-variance combination of ln geo; xs = (geo, se)
    l = [math.log(g) for g, _ in xs]; w = [1 / (se / g) ** 2 for g, se in xs]
    m = sum(a * b for a, b in zip(w, l)) / sum(w); se = 1 / math.sqrt(sum(w))
    return {"geo": math.exp(m), "se_ln": se, "z": m / se}


print("DECIDING: N -> T (P16-TP):"); TN = scope("N{}-v", "T{}")
print("reported: N -> P (full P16, 128 tensors; third slice):"); PN = scope("N{}-v", "P{}-v")
print("reported: P -> T:"); TP = scope("P{}-v", "T{}")
w101 = json.load(open("/data/w101/c/verdict130c.json")); w111 = json.load(open("/data/w111/verdict111.json"))["D"]
a, b = (w101["geo"], w101["se_comb"]), (w111["geo"], w111["se_comb"])
J_T = joint(a, b, (TN["geo"], TN["se_comb"])); J_P = joint(a, b, (PN["geo"], PN["se_comb"]))
J_P2 = joint(b, (PN["geo"], PN["se_comb"]))
v = TN["geo"] <= 0.995 and TN["geo_x"] < 1 and TN["geo_c"] < 1
for name, r in (("N->T", TN), ("N->P", PN), ("P->T", TP)):
    print(f"{name}: geo {r['geo']:.4f} (xcorp {r['geo_x']:.4f}, calib {r['geo_c']:.4f}), SE comb {r['se_comb']:.4f}, z {r['z']:+.2f}")
print(f"joint (w101 27c 112 P16 {a[0]:.4f}±{a[1]:.4f}, w111 27d 128 P16 {b[0]:.4f}±{b[1]:.4f}) + this slice's N->T: "
      f"geo {J_T['geo']:.4f} z {J_T['z']:+.2f}  [arms differ: P16 on 27c / 27d, P16-TP here]")
print(f"joint full P16 over three slices (27c 112, 27d 128, 27e 128): geo {J_P['geo']:.4f} z {J_P['z']:+.2f}; "
      f"128-tensor P16 on 27d + 27e: geo {J_P2['geo']:.4f} z {J_P2['z']:+.2f}")
json.dump({"NT": TN, "NP": PN, "PT": TP, "joint_T": J_T, "joint_P3": J_P, "joint_P128": J_P2, "w101": a, "w111": b,
           "pass": v}, open(f"{C}/verdict113.json", "w"), indent=1)
rT = ", ".join("%.4f" % TN["pairs"][s]["pooled"]["r"] for s in "01")
print(f"W113 #148-02 VERDICT {'PASS (adopt-proposal)' if v else 'FAIL'}: N -> P16-TP pooled r = {rT}, geo {TN['geo']:.4f} "
      f"(xcorp {TN['geo_x']:.4f}, calib {TN['geo_c']:.4f}), SE comb {TN['se_comb']:.4f}, z {TN['z']:+.2f}")
