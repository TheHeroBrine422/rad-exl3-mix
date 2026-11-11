#!/usr/bin/env python3
"""verdict111.py C W101C GOLD_X GOLD_C -- w111 #146-02 verdict (../w111.md), pair84.py's arithmetic as w101's verdict130c.py.
Rows C/scores/<tag>-{x,c}.json.rows, tags N<S>-v / P<S>-v (decision pair D, 128 tensors) and N<S> / P<S> (w101's 112-tensor
mixes, reported). Needs C/hyg-S.ok for both seeds. Writes C/verdict111.json and C/docstat-<pair>.txt (docstat.py);
last line = the verdict."""
import json, math, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w84"))
from pair84 import pair, fmt, doc_sums  # noqa: E402

C, W101C, GX, GC = sys.argv[1:5]
G = {"x": GX, "c": GC}
rows = lambda t, p: f"{C}/scores/{t}-{p}.json.rows"
parts = lambda a, b, ps="xc": [(G[p], rows(a, p), rows(b, p)) for p in ps]
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
for s in "01":
    if not os.path.exists(f"{C}/hyg-{s}.ok"): sys.exit(f"seed {s}: no gate pass, nothing is read")


def scope(nt, pt):
    """nt / pt: tag formats for N / P; per seed pooled / xcorp / calib, the geo and the combined SE (w101's)."""
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
    S = {(s, p, t): doc_sums(G[p], rows(f, p))[0] for s in res for p in "xc" for t, f in (("N", nt.format(s)), ("P", pt.format(s)))}
    rng, bs = np.random.default_rng(0), []
    nd = {p: len(S[("0", p, "N")]) for p in "xc"}
    for _ in range(2000):
        idx = {p: rng.integers(0, nd[p], nd[p]) for p in "xc"}
        lr = [math.log(sum(S[(s, p, "P")][idx[p]].sum() for p in "xc") / sum(S[(s, p, "N")][idx[p]].sum() for p in "xc")) for s in res]
        bs.append(sum(lr) / len(lr))
    se_doc = g * float(np.std(bs)); se = math.sqrt(se_doc ** 2 + 0.0086 ** 2 / len(res))
    return {"pairs": res, "geo": g, "geo_x": gx, "geo_c": gc, "se_doc": se_doc, "se_comb": se, "z": (g - 1) / se}


print("D (128 tensors: ssm_qkv, ssm_z, attn_qg, attn_v), N -> P16:")
D = scope("N{}-v", "P{}-v")
print("112 scope (w101's mixes as they are), N -> P16:")
O = scope("N{}", "P{}")
inc = {s: D["pairs"][s]["pooled"]["r"] / O["pairs"][s]["pooled"]["r"] for s in "01"}
ginc = geo(list(inc.values()))


def joint(a, sa, b, sb):   # inverse-variance combination of ln geo
    la, lb, va, vb = math.log(a), math.log(b), (sa / a) ** 2, (sb / b) ** 2
    w = (1 / va, 1 / vb); m = (w[0] * la + w[1] * lb) / sum(w); se = 1 / math.sqrt(sum(w))
    return {"geo": math.exp(m), "se_ln": se, "z": m / se}


W101 = json.load(open(f"{W101C}/verdict130c.json")); w_g, w_se = W101["geo"], W101["se_comb"]
J1, J2 = joint(w_g, w_se, O["geo"], O["se_comb"]), joint(w_g, w_se, D["geo"], D["se_comb"])
vD = D["geo"] <= 0.995 and D["geo_x"] < 1 and D["geo_c"] < 1
v112 = O["geo"] <= 0.995 and O["geo_x"] < 1 and O["geo_c"] < 1
vjoin = vD and ginc <= 1.0
print(f"v_proj increment r(128) / r(112): seed 0 {inc['0']:.4f}, seed 1 {inc['1']:.4f}, geo {ginc:.4f}")
print(f"joint with w101 (geo {w_g:.4f}, SE {w_se:.4f}): (i) 112 scope geo {J1['geo']:.4f} z {J1['z']:+.2f}; "
      f"(ii) w101 112 + D geo {J2['geo']:.4f} z {J2['z']:+.2f}")
json.dump({"D": D, "s112": O, "v_increment": inc, "v_increment_geo": ginc, "joint_112": J1, "joint_D": J2,
           "w101": {"geo": w_g, "se_comb": w_se}, "D_pass": vD, "s112_pass": v112, "attn_v_joins": vjoin},
          open(f"{C}/verdict111.json", "w"), indent=1)
print(f"112 scope on this slice: geo {O['geo']:.4f} (xcorp {O['geo_x']:.4f}, calib {O['geo_c']:.4f}), z {O['z']:+.2f} -> "
      f"{'PASS' if v112 else 'FAIL'} (reported)")
rD = ", ".join("%.4f" % D["pairs"][s]["pooled"]["r"] for s in "01")
print(f"W111 #146-02 VERDICT D {'PASS' if vD else 'FAIL'}: N -> P16 (128) pooled r = "
      f"{rD}, geo {D['geo']:.4f} (xcorp {D['geo_x']:.4f}, "
      f"calib {D['geo_c']:.4f}), SE comb {D['se_comb']:.4f} (doc {D['se_doc']:.4f}), z {D['z']:+.2f}; "
      f"attn_v {'joins' if vjoin else 'does not join'} (increment geo {ginc:.4f})")
