# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check195.py PLAN.json -- an independent recount of plan195.json against RULES-slices195 §2 (#195, w180), before the build. Reads the
goldens' metadata, gen masks, pools, the registry / ledger snapshots (its own loading code; from draw195 only the path constants and name
lists). Prints one PASS / FAIL line per check (INFO lines are numbers for the report) and exits 1 on any FAIL."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draw195 as D  # noqa: E402  (constants only)

plan = json.load(open(sys.argv[1])); fails = 0


def chk(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg); fails += 0 if ok else 1


def gdocs(p):
    return json.load(open(f"{p}/kld.json"))["docs"]


C = {c["name"]: c for c in plan["c"]}; X = {x["name"]: x for x in plan["x"]}
names = D.SEARCH + D.CONFIRM
chk(sorted(C) == sorted(names) and sorted(X) == sorted(names) and len(names) == 15, "15 slices: s195s1-5, s195c1-10, each with a c and an x part")
chk(plan["pairs"] == [[f"s195s{(k - 1) % 5 + 1}", f"s195c{k}"] for k in range(1, 11)], "pairing: s195s((k-1) mod 5 + 1) with s195ck, k = 1..10")
chk(all(C[n]["role"] == ("confirm" if n in D.CONFIRM else "search") for n in names), "roles: s* search, c* confirm")
# ---- the exclusion sets, rebuilt here
K = json.load(open(D.CAL))["docs"]; P = json.load(open(D.POOLS))["pool"]; GEN = np.fromfile(D.CALGEN, np.uint8)
REGJ = json.load(open(D.REG))
never_c, never_x, ret_x, any_x = set(), set(), collections.defaultdict(set), set()
for n, r in D.ROOT.items():
    co = {d["orig"] for d in gdocs(f"{r}/golden-{n}-c")}; xo = {d["xdoc"] for d in gdocs(f"{r}/golden-{n}-x")}
    any_x |= xo
    if n in D.NEVER: never_c |= co; never_x |= xo
    else:
        for d in xo: ret_x[d].add(n)
for g in D.FINAL:
    for d in gdocs(g):
        if "xdoc" in d: never_x.add(d["xdoc"]); any_x.add(d["xdoc"])
        else: never_c.add(d["orig"])
live = [n for n, s in REGJ["slices"].items() if s["status"] != "retired" and n not in D.RETIRED_OK]
for n in live: never_c |= {int(d["id"][2:]) for d in REGJ["slices"][n]["docs"] if d["part"] == "c"}
chk(set(live) <= set(D.NEVER), f"every non-retired registry slice ({sorted(live)}) is in the NEVER list")
chk(not (set(D.NEVER) & set(D.RETIRED)), "NEVER and RETIRED lists are disjoint")
# ---- calib parts
allc = [i for n in names for i in C[n]["idx"]]
chk(len(allc) == len(set(allc)), f"calib documents distinct over the 15 slices ({len(allc)} uses, {len(set(allc))} documents)")
chk(all(P[i] in ("fit", "search") for i in allc), "calib documents in the merged fit + search pool (no verify / qad)")
chk(all(K[i]["stratum"] == "L" for i in allc), "calib documents all Latin (N comes from the fresh batch only)")
chk(not (set(allc) & never_c), "no calib document of t183, t186n, final750 / final-c3, s173f, s190d-f (live / confirming)")
sc = lambda i: min(K[i]["tokens"], D.C_CAP) - 1 - K[i]["score_from"]
gc = lambda i: int(GEN[K[i]["row0"]:K[i]["row0"] + sc(i)].sum())
chk(all(sum(gc(i) for i in C[n]["idx"]) == C[n]["gen_L"] for n in names), "calib gen positions (cut at 4,096) = the plan's")
use3 = collections.Counter(); mine = {f"golden-{k}-c" for k in REGJ["slices"]}
for l in open(D.LED):
    r = json.loads(l)
    if r["doc"] == "all" or r["use"] in mine or r["use"].endswith(("-headx-search", "-headx-qad", "-headx-fit", "-headx-verify")): continue
    if str(r.get("line", "")).startswith("v3"):
        dd = r.get("decisions"); use3[int(r["doc"])] += dd if isinstance(dd, (int, float)) and not isinstance(dd, bool) else 1
for k, v in REGJ["doc_decisions"].items():
    if k.startswith("c:"): use3[int(k[2:])] += v
mx_conf = max(use3[i] for n in D.CONFIRM for i in C[n]["idx"]); mn_srch = min(use3[i] for n in D.SEARCH for i in C[n]["idx"])
chk(mx_conf <= mn_srch, f"confirm slices hold the least used calib documents (max V3 decisions in confirm {mx_conf} <= min in search {mn_srch})")
pool_ok = [i for i, p in enumerate(P) if p in ("fit", "search") and K[i]["stratum"] == "L" and sc(i) >= 16 and gc(i) >= 16 and i not in never_c]
chk(set(pool_ok) == set(allc), f"the whole allowed calib Latin pool is used ({len(pool_ok)} documents, {sum(gc(i) for i in pool_ok)} gen): no fresher document left out")
# ---- x parts
XS = {(x["batch"], x["idx"]): x for f in D.XSUP for x in map(json.loads, open(f))}
XD, G, META = {}, {}, {}
for b in D.BATCH_L + [D.BATCH_N]:
    XD[b] = gdocs(f"{D.XV}/{b}"); G[b] = np.fromfile(f"{D.XV}/{b}/gm.u8", np.uint8)
    META[b] = [json.loads(l) for l in open(f"{D.XV}/{b}/corpus.meta.jsonl")]
    assert len(META[b]) == len(XD[b]) and len(G[b]) == json.load(open(f"{D.XV}/{b}/kld.json"))["rows"], b
allp = [(p["batch"], p["idx"]) for n in names for p in X[n]["pieces"]]; alld = [p["doc"] for n in names for p in X[n]["pieces"]]
chk(len(allp) == len(set(allp)) and len(alld) == len(set(alld)), f"xcorp pieces distinct over the 15 slices by (batch, idx) and by xdoc ({len(allp)} pieces)")
chk(all(META[b][i]["doc"] == XS[(b, i)]["doc"] for b, i in allp) and all(p["doc"] == XS[(p["batch"], p["idx"])]["doc"] for n in names for p in X[n]["pieces"]),
    "each piece's xdoc = its batch's corpus meta = the xsupply row")
chk(not (set(alld) & never_x), "no xcorp piece of t183, t186n, final-x / final750, s173f, s190d-f (live / confirming)")
ndocs_b = {b: {m["doc"] for m in META[b]} for b in META}
chk(not (ndocs_b[D.BATCH_N] & set().union(*[ndocs_b[b] for b in D.BATCH_L])) and not (ndocs_b[D.BATCH_N] & any_x),
    "the fresh batch xn8k-b195 shares no xcorp document with xc8k-b1 / b2 or with any earlier x golden")
okN = okLc = okLs = okcap = okgen = oklab = True; reuse_cnt = collections.Counter()
for n in names:
    for p in X[n]["pieces"]:
        s = XS[(p["batch"], p["idx"])]; d = XD[p["batch"]][p["idx"]]; rows = d["tokens"] - 1 - d["score_from"]
        if p["stratum"] != s["stratum"] or p["script"] != s["script"] or p["category"] != s["category"]: oklab = False
        k = p["cap"] or rows
        if k > rows or k < 1: okcap = False
        if p["stratum"] == "L" and p["batch"] != D.BATCH_N and s["category"] in D.HEAVY and rows > D.CUT and p["cap"] != D.CUT: okcap = False
        if int(G[p["batch"]][d["row0"]:d["row0"] + k].sum()) != p["gen"]: okgen = False
        if p["stratum"] == "N":
            if p["batch"] != D.BATCH_N or s["used"] or p["doc"] in any_x: okN = False
        elif n in D.CONFIRM:
            if s["used"] or p["doc"] in any_x or p["batch"] not in D.BATCH_L: okLc = False
        else:
            h = ret_x.get(p["doc"], set())
            if not (s["used"] or h) or not h <= set(D.RETIRED) or p["doc"] in never_x: okLs = False
            reuse_cnt["flag only (search27 / s129)" if not h else "retired slices"] += 1
chk(oklab, "piece labels (stratum, script, category) = the xsupply tables'")
chk(okcap, "caps: heavy Latin categories cut at 2,048 scored positions; every cap inside its piece")
chk(okgen, "each piece's gen positions (with its cap, from the batch gen mask) = the plan's")
chk(okN, "N pieces: all from xn8k-b195, never used, in no earlier golden")
chk(okLc, "confirm Latin pieces: fresh (no used flag, in no earlier golden)")
chk(okLs, f"search Latin pieces: reused from retired slices only ({dict(reuse_cnt)})")
# ---- sizes and shares (RULES-slices195 §2)
ref = json.load(open(D.PLAN186)); rc = {c["name"]: c["gen_L"] for c in ref["c"]}; rx = {x["name"]: x["gen_L"] for x in ref["x"]}
for n in names:
    m = D.MATCH[n]; gLc = C[n]["gen_L"]; gLx = sum(p["gen"] for p in X[n]["pieces"] if p["stratum"] == "L")
    sg = collections.Counter()
    for p in X[n]["pieces"]:
        if p["stratum"] == "N": sg[p["script"]] += p["gen"]
    gN = sum(sg.values()); share = gN / (gN + gLc + gLx); ratio = (gLc + gLx) / (rc[m] + rx[m])
    chk(ratio >= 1.9 and gLx >= 1.9 * rx[m], f"{n}: Latin gen {gLc + gLx} = {ratio:.3f} x {m}'s (c + x {rc[m] + rx[m]}); x part {gLx / rx[m]:.2f} x {m}-x")
    chk(0.09 <= share <= 0.12, f"{n}: N share of gen positions {share:.4f} in [0.09, 0.12] (N gen {gN})")
    chk(len([v for v in sg.values() if v > 0]) >= 5 and max(sg.values()) <= gN / 3, f"{n}: {len(sg)} scripts, largest {max(sg.values()) / gN:.3f} of N gen (<= 1/3)")
    print(f"INFO {n}: c part Latin gen {gLc} = {gLc / rc[m]:.2f} x {m}-c (the calib pool is short: the shortfall is in the x part)")
print("ALL PASS" if not fails else f"{fails} FAIL")
sys.exit(1 if fails else 0)
