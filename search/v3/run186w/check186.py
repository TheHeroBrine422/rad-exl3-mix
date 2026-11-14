# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check186.py PLAN.json CALIB_STRATA.json XSUPPLY.jsonl GM_B1.u8 -- independent checks of draw186's plan (RULES-strata §3, §7),
run before anything is built. Exit 1 on any failed check. Checks: every x piece is unused by every earlier x part (the
xsupply 'used' flags, which cover every golden-s17*-x, t183-x, final-x, s129-x and w129's search27 excl files) and
distinct from the other parts; every c document is in fit + search, not in a ready / active slice or t183's c part,
and distinct; N gen share of each part in [0.15, 0.20]; >= 6 scripts per part; no script above 1/3 of a part's N gen;
t186n: >= 6 scripts, ~30k gen positions, no document in a c part."""
import collections, json, sys

import numpy as np

plan = json.load(open(sys.argv[1])); K = json.load(open(sys.argv[2]))["docs"]
X = [json.loads(l) for l in open(sys.argv[3])]; GM = np.fromfile(sys.argv[4], np.uint8)
XD = json.load(open("/data/q27/goldens/xcorp/xc8k-b1/kld.json"))["docs"]
P = json.load(open("/data/q27/w164/strata/pools-v3.json"))["pool"]
REG = json.load(open("/data/q27/w164/strata/registry.json"))
T183 = {d["orig"] for d in json.load(open("/data/q27/w131/slices/golden-t183-c/kld.json"))["docs"] if d.get("orig") is not None}
busy = {int(d["id"][2:]) for s in REG["slices"].values() if s["status"] in ("ready", "active") for d in s["docs"] if d["part"] == "c"}
bad = []; ok = lambda name, cond, msg="": (print(("PASS " if cond else "FAIL ") + name, msg), cond or bad.append(name))
b1 = {x["idx"]: x for x in X if x["batch"] == "xc8k-b1"}
xs = [i for p in plan["x"] for i in p["idx"]]
ok("x pieces distinct across the six parts", len(xs) == len(set(xs)) == 130 * 6, f"{len(xs)} pieces, {len(set(xs))} distinct")
ok("x pieces all unused by earlier x parts (xsupply flags)", all(not b1[i]["used"] for i in xs))
ok("x parts: 130 pieces each", all(len(p["idx"]) == 130 for p in plan["x"]))
GEN = np.fromfile("/data/q27/challenge/golden/calib/gen.u8", np.uint8)
def xg(i, cut): return int(GM[XD[i]["row0"]:XD[i]["row0"] + min(XD[i]["tokens"] - 1 - XD[i]["score_from"], cut or 10**9)].sum())
def cg(i):    # a c document as the part scores it: cut at 4,096 tokens (golden_pick idx:4096)
    n = min(K[i]["tokens"], 4096) - 1 - K[i]["score_from"]; return int(GEN[K[i]["row0"]:K[i]["row0"] + n].sum())
def gen_checks(name, parts_doc, gfun, stratum_of, script_of):
    """parts_doc: ids of the part; gfun(id) gen positions; RULES-strata §3: N share in [0.15, 0.20], >= 6 scripts,
    no script above a third of the part's N gen positions (all gen-based)"""
    tot = sum(gfun(i) for i in parts_doc); N = [i for i in parts_doc if stratum_of(i) == "N"]
    gN = sum(gfun(i) for i in N); sg = collections.Counter()
    for i in N: sg[script_of(i)] += gfun(i)
    share = gN / max(tot, 1); mx = max(sg.values()) / max(gN, 1)
    ok(f"{name} N gen share", 0.15 <= share <= 0.20, f"{share:.4f} (N gen {gN} of {tot})")
    ok(f"{name} >= 6 scripts", len(sg) >= 6, f"{len(sg)} scripts")
    ok(f"{name} max script <= 1/3 of N gen", mx <= 1 / 3, f"{mx:.3f} {dict(sg.most_common(3))}")
    return share
for p in plan["x"]:
    capd = {int(k): v for k, v in p["caps"].items()}
    gen_checks(f"{p['name']}-x", p["idx"], lambda i: xg(i, capd.get(i, 0)), lambda i: b1[i]["stratum"], lambda i: b1[i]["script"])
    ok(f"{p['name']}-x N pieces are N", all(b1[i]["stratum"] == "N" for i in p["N"]))
cs = [i for p in plan["c"] for i in p["idx"]]
ok("c documents distinct within each part", all(len(p["idx"]) == len(set(p["idx"])) for p in plan["c"]))
print(f"INFO c documents: {len(cs)} uses, {len(set(cs))} distinct, {len(cs) - len(set(cs))} repeated uses (least used first)")
ok("c documents: fit or search pool", all(P[i] in ("fit", "search") for i in cs))
ok("c documents: not busy (ready / active slices), not t183", not (set(cs) & (busy | T183)), f"busy {len(busy)}, t183 {len(T183)}")
for p in plan["c"]:
    gen_checks(f"{p['name']}-c", p["idx"], cg, lambda i: K[i]["stratum"], lambda i: K[i]["script"])
    ok(f"{p['name']}-c N docs are N", all(K[i]["stratum"] == "N" for i in p["N"]))
t = plan["t186n"]
ok("t186n: >= 6 scripts", len(t["scripts"]) >= 6, str(t["scripts"]))
ok("t186n: ~30k gen positions", 28000 <= t["gen"] <= 34000, str(t["gen"]))
ok("t186n: every document is N", all(K[i]["stratum"] == "N" for i in t["calib"]) and all(b1[i]["stratum"] == "N" for i in t["xcorp"]))
ok("t186n: disjoint from every c part", not (set(t["calib"]) & set(cs)))
ok("t186n: disjoint from x pieces", not (set(t["xcorp"]) & set(xs)))
ok("t186n calib docs: not busy, not t183", not (set(t["calib"]) & (busy | T183)))
print("ALL PASS" if not bad else f"FAILED: {bad}")
sys.exit(1 if bad else 0)
