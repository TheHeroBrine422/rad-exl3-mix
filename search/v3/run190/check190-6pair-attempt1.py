# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check190.py PLAN.json CALIB_STRATA.json XSUPPLY.jsonl GM_B1.u8 -- checks of draw190's plan before anything is built
(RULES-cooldown §1, RULES-strata §3; check186.py with the 2x sizes). Exit 1 on any failed check. Checks: six c and six x parts
(x: 260 pieces each); every x piece distinct and outside the exclusion set (draw190.exclusion: s17*, s186a-f, t183, t186n,
final-x, golden-final750, registry ready / active / staged, xsupply's used flags); every c document in fit + search and outside
the exclusion set, distinct within a part (reuse across parts is reported, allowed); N gen share in [0.155, 0.19] (c) and
[0.15, 0.20] (x); >= 6 scripts; no script above 1/3 of N gen; sizes: c scored positions >= 1.9 x the matching s186 part, x
pieces >= 1.9 x 130; x_short must be null (the supply gives all six x parts)."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draw190 as D  # noqa: E402  (exclusion() and the paths; read-only use)

plan = json.load(open(sys.argv[1])); K = json.load(open(sys.argv[2]))["docs"]
X = [json.loads(l) for l in open(sys.argv[3])]; GM = np.fromfile(sys.argv[4], np.uint8)
XD = json.load(open(D.XB1))["docs"]
P = json.load(open(D.POOLS))["pool"]
ex_c, ex_x = D.exclusion(json.load(open(D.REG)))
S186PLAN = "/data/q27/w164/strata/plan186.json"
s186 = {c["name"]: c for c in json.load(open(S186PLAN))["c"]}
GEN = np.fromfile(D.CALGEN, np.uint8)
bad = []
ok = lambda name, cond, msg="": (print(("PASS " if cond else "FAIL ") + name, msg), cond or bad.append(name))
b1 = {x["idx"]: x for x in X if x["batch"] == "xc8k-b1"}


def xg(i, cut): return int(GM[XD[i]["row0"]:XD[i]["row0"] + min(XD[i]["tokens"] - 1 - XD[i]["score_from"], cut or 10**9)].sum())
def cg(i):    # a c document as the part scores it: cut at 4,096 tokens (as draw190)
    n = min(K[i]["tokens"], D.C_CAP) - 1 - K[i]["score_from"]; return int(GEN[K[i]["row0"]:K[i]["row0"] + n].sum())


def gen_checks(name, ids, gfun, stratum_of, script_of, lo, hi):
    tot = sum(gfun(i) for i in ids); N = [i for i in ids if stratum_of(i) == "N"]
    gN = sum(gfun(i) for i in N); sg = collections.Counter()
    for i in N: sg[script_of(i)] += gfun(i)
    share = gN / max(tot, 1); mx = max(sg.values()) / max(gN, 1) if sg else 1.0
    ok(f"{name} N gen share in [{lo}, {hi}]", lo <= share <= hi, f"{share:.4f} (N gen {gN} of {tot})")
    ok(f"{name} >= 6 scripts", len(sg) >= 6, f"{len(sg)} scripts")
    ok(f"{name} max script <= 1/3 of N gen", mx <= 1 / 3 + 1e-9, f"{mx:.3f} {dict(sg.most_common(3))}")


ok("six c parts", len(plan["c"]) == 6, f"{len(plan['c'])}")
ok("six x parts (the supply gives all six)", len(plan["x"]) == 6, f"{len(plan['x'])} built; x_short = {plan.get('x_short')}")
ok("x_short is null", plan.get("x_short") is None, str(plan.get("x_short")))
xs = [i for p in plan["x"] for i in p["idx"]]
ok("x parts: 260 pieces each", all(len(p["idx"]) == 260 for p in plan["x"]))
ok("x pieces distinct across the parts", len(xs) == len(set(xs)), f"{len(xs)} pieces, {len(set(xs))} distinct")
ok("x pieces outside the exclusion set (earlier x parts, s186, t186n, final-x, golden-final750)", not (set(b1[i]["doc"] for i in xs) & ex_x))
ok("x pieces are xc8k-b1", all(i in b1 for i in xs))
for p in plan["x"]:
    capd = {int(k): v for k, v in p["caps"].items()}
    gen_checks(f"{p['name']}-x", p["idx"], lambda i: xg(i, capd.get(i, 0)), lambda i: b1[i]["stratum"], lambda i: b1[i]["script"], 0.15, 0.20)
    ok(f"{p['name']}-x N pieces are N", all(b1[i]["stratum"] == "N" for i in p["N"]))
    ok(f"{p['name']}-x size >= 1.9 x 130 pieces", len(p["idx"]) >= 1.9 * 130)
cs = [i for p in plan["c"] for i in p["idx"]]
ok("c documents distinct within each part", all(len(p["idx"]) == len(set(p["idx"])) for p in plan["c"]))
print(f"INFO c documents: {len(cs)} uses, {len(set(cs))} distinct, {len(cs) - len(set(cs))} repeated uses (reuse across parts)")
ok("c documents: fit or search pool", all(P[i] in ("fit", "search") for i in cs))
ok("c documents outside the exclusion set (s17*, s186a-f, t183, t186n, golden-final750, registry)", not (set(cs) & ex_c))
for p in plan["c"]:
    gen_checks(f"{p['name']}-c", p["idx"], cg, lambda i: K[i]["stratum"], lambda i: K[i]["script"], 0.155, 0.19)
    ok(f"{p['name']}-c N docs are N", all(K[i]["stratum"] == "N" for i in p["N"]))
    pos = sum(min(K[i]["tokens"], D.C_CAP) - 1 - K[i]["score_from"] for i in p["idx"])
    ok(f"{p['name']}-c size >= 1.9 x {p['name'].replace('s190', 's186')} scored positions", pos >= 1.9 * s186[p["name"].replace("s190", "s186")]["pos"], f"{pos} vs {s186[p['name'].replace('s190', 's186')]['pos']}")
print("ALL PASS" if not bad else f"FAILED: {bad}")
sys.exit(1 if bad else 0)
