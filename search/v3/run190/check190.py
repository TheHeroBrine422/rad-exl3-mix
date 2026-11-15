# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check190.py PLAN.json CALIB_STRATA.json XSUPPLY.jsonl GM_B1.u8 -- checks of draw190's plan before anything is built (RULES-cooldown
§1 as amended 11-14 11:40 PM CST; check186.py for the rest). Exit 1 on any failed check.
Checks: two c parts (s190a search, s190b confirm) and two x parts; every c document in fit + search and outside draw190.exclusion()
(s17*, s186a-f, t183, t186n, golden-final750, registry ready / active / staged / confirm / reselect / tracking); c documents distinct
within each part and across the pair; the N documents of the two parts are exactly the free non-Latin calib documents (each once);
every x piece is xc8k-b1, outside the exclusion set (earlier x parts, s186, t186n, final-x, golden-final750), distinct within each part
and across the pair; the N pieces of the two parts are exactly the unused non-Latin b1 pieces (each once); >= 4 non-Latin scripts per
part; no script above half of a part's N gen; L gen positions of each part >= 1.9 x the matching s186 part's (c: cut at 4,096; x: the
caps). The N share of each part is printed, not checked (the amendment lets it fall to what the supply gives)."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draw190 as D  # noqa: E402  (exclusion() and the paths; read-only use)

plan = json.load(open(sys.argv[1])); K = json.load(open(sys.argv[2]))["docs"]
X = [json.loads(l) for l in open(sys.argv[3])]; GM = np.fromfile(sys.argv[4], np.uint8)
XD = json.load(open(D.XB1))["docs"]
P = json.load(open(D.POOLS))["pool"]
ex_c, ex_x = D.exclusion(json.load(open(D.REG)))
ref = json.load(open(D.PLAN186)); r_c = {c["name"]: c for c in ref["c"]}; r_x = {x["name"]: x for x in ref["x"]}
GEN = np.fromfile(D.CALGEN, np.uint8)
bad = []
ok = lambda name, cond, msg="": (print(("PASS " if cond else "FAIL ") + name, msg), cond or bad.append(name))
b1 = {x["idx"]: x for x in X if x["batch"] == "xc8k-b1"}
sc = lambda i: min(K[i]["tokens"], D.C_CAP) - 1 - K[i]["score_from"]


def xg(i, cut): return int(GM[XD[i]["row0"]:XD[i]["row0"] + min(XD[i]["tokens"] - 1 - XD[i]["score_from"], cut or 10**9)].sum())
def cg(i):    # a c document as the part scores it: cut at 4,096 tokens (as draw190)
    n = sc(i); return int(GEN[K[i]["row0"]:K[i]["row0"] + n].sum())


def gen_checks(name, ids, gfun, stratum_of, script_of):
    """the amended N checks: >= 4 non-Latin scripts, no script above half of the part's N gen (the N share is printed)"""
    tot = sum(gfun(i) for i in ids); N = [i for i in ids if stratum_of(i) == "N"]
    gN = sum(gfun(i) for i in N); sg = collections.Counter()
    for i in N: sg[script_of(i)] += gfun(i)
    mx = max(sg.values()) / max(gN, 1) if sg else 1.0
    ok(f"{name} >= 4 non-Latin scripts", len(sg) >= 4, f"{len(sg)} scripts {dict(sg.most_common())}")
    ok(f"{name} max script <= 1/2 of N gen", mx <= 0.5 + 1e-9, f"{mx:.3f}")
    print(f"INFO {name} N gen share {gN / max(tot, 1):.4f} (N gen {gN} of {tot}; {len(N)} N units)")


names_c = [c["name"] for c in plan["c"]]; names_x = [x["name"] for x in plan["x"]]
ok("c parts: s190a, s190b", names_c == ["s190a", "s190b"], str(names_c))
ok("x parts: s190a, s190b", names_x == ["s190a", "s190b"], str(names_x))
ok("x_short absent or null", plan.get("x_short") is None, str(plan.get("x_short")))
# ------------------------------------------------------------------ x side
xs = [i for p in plan["x"] for i in p["idx"]]
ok("x pieces distinct within each part and across the pair", len(xs) == len(set(xs)), f"{len(xs)} pieces, {len(set(xs))} distinct")
ok("x pieces are xc8k-b1", all(i in b1 for i in xs))
ok("x pieces outside the exclusion set (earlier x parts, s186, t186n, final-x, golden-final750)", not (set(b1[i]["doc"] for i in xs) & ex_x))
unused_N = {x["idx"] for x in X if x["batch"] == "xc8k-b1" and x["doc"] not in ex_x and x["stratum"] == "N"}
N_x_all = [i for p in plan["x"] for i in p["N"]]
ok("x N pieces: each unused non-Latin b1 piece once (the two parts together)", sorted(N_x_all) == sorted(unused_N),
   f"{len(N_x_all)} used, {len(unused_N)} unused N in b1")
ok("x N pieces are N", all(b1[i]["stratum"] == "N" for i in N_x_all))
for p in plan["x"]:
    capd = {int(k): v for k, v in p["caps"].items()}
    gen_checks(f"{p['name']}-x", p["idx"], lambda i: xg(i, capd.get(i, 0)), lambda i: b1[i]["stratum"], lambda i: b1[i]["script"])
    gl = sum(xg(i, capd.get(i, 0)) for i in p["idx"] if b1[i]["stratum"] == "L")
    m = r_x[p["name"].replace("s190", "s186")]
    ok(f"{p['name']}-x L gen >= 1.9 x {m['name']}", gl >= 1.9 * m["gen_L"], f"{gl} vs {m['gen_L']} ({gl / m['gen_L']:.3f} x)")
    print(f"INFO {p['name']}-x pieces {len(p['idx'])} (N {len(p['N'])}, L {len(p['idx']) - len(p['N'])}); s186 {m['name']}: {len(m['idx'])} pieces")
# ------------------------------------------------------------------ c side
cs = [i for p in plan["c"] for i in p["idx"]]
ok("c documents distinct within each part", all(len(p["idx"]) == len(set(p["idx"])) for p in plan["c"]))
ok("c documents distinct across the pair", len(cs) == len(set(cs)), f"{len(cs)} uses, {len(set(cs))} distinct")
ok("c documents: fit or search pool", all(P[i] in ("fit", "search") for i in cs))
ok("c documents outside the exclusion set (s17*, s186a-f, t183, t186n, golden-final750, registry)", not (set(cs) & ex_c))
free_N = {i for i, p in enumerate(P) if p in ("fit", "search") and i not in ex_c and K[i]["stratum"] == "N" and sc(i) >= 16}
N_c_all = [i for p in plan["c"] for i in p["N"]]
ok("c N documents: each free non-Latin calib document once (the two parts together)", sorted(N_c_all) == sorted(free_N),
   f"{len(N_c_all)} used, {len(free_N)} free N")
ok("c N documents are N", all(K[i]["stratum"] == "N" for i in N_c_all))
for p in plan["c"]:
    gen_checks(f"{p['name']}-c", p["idx"], cg, lambda i: K[i]["stratum"], lambda i: K[i]["script"])
    gl = sum(cg(i) for i in p["idx"] if K[i]["stratum"] == "L")
    m = r_c[p["name"].replace("s190", "s186")]
    ok(f"{p['name']}-c L gen >= 1.9 x {m['name']}", gl >= 1.9 * m["gen_L"], f"{gl} vs {m['gen_L']} ({gl / m['gen_L']:.3f} x)")
    print(f"INFO {p['name']}-c docs {len(p['idx'])} (N {len(p['N'])}, L {len(p['idx']) - len(p['N'])}), scored positions "
          f"{sum(sc(i) for i in p['idx'])}; s186 {m['name']}: {len(m['idx'])} docs, {m['pos']} positions")
print("ALL PASS" if not bad else f"FAILED: {bad}")
sys.exit(1 if bad else 0)
