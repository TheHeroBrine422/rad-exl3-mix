# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check190ef.py PLAN.json CALIB_STRATA.json XSUPPLY.jsonl -- checks of draw190ef's plan before anything is built (#190-03, w179; the
RULES-cooldown amendment 2026-11-15 4:25 PM CST: pair s190e (search) / s190f (confirm)): the draw190cd / check190cd checks with the
amended sources (N reused from the retired search / confirm / reselect slices, now also s190a and s190b, s190b once clear).
check190cd.py's checks, with the amended N source:
  - two c parts (s190c search, s190d confirm) and two x parts; c and x distinct within each part and across the pair;
  - Latin (L) documents / pieces are fresh: outside every earlier and non-retired slice (draw190's exclusion, the s190a / s190b x
    pieces included, the retired slices' documents excluded) and in the fit + search pool (c) or unused xc8k-b1 (x);
  - calib N documents: held only by RETIRED slices (s173a-e, s186a, s186c, and s186b / d / e / f under the 5:40 amendment) and by no
    other slice (not s190a / s190b, s173f, s171a-d, t183, t186n, golden-final750, any other non-retired registry slice); fit + search;
  - xcorp N pieces are reused from RETIRED search slices only (s173a-e, s186a, s186c) or are unused xc8k b2 N pieces;
    never a piece held by a non-retired slice (confirm / reselect / tracking / staged / active / final / t183 / t186n / s171),
    never a verify-pool document;
  - N of each part: >= 4 non-Latin scripts, no script above half of the part's N gen (the N share is printed, not banded);
  - L gen of each part >= 1.9 x the matching s186 part's L gen (c: cut at 4,096; x: caps as pickx);
  - the c N part is checked too: a plan whose c part is blocked fails here (exit 1), as it must not be built.
Exit 1 on any failed check. Reads the same snapshots as draw190ef (ds8 paths)."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draw190ef as D  # noqa: E402  (exclusion sets and paths; read-only use)

plan = json.load(open(sys.argv[1])); K = json.load(open(sys.argv[2]))["docs"]
X = [json.loads(l) for l in open(sys.argv[3])]
P = json.load(open(D.POOLS))["pool"]; GEN = np.fromfile(D.CALGEN, np.uint8)
REGJ = json.load(open(D.REG))
nr_c_set, reg_nr_c, ret_c_map, nr_x_set, ret_x_map = D.exclusion(REGJ)
ex_n, ret_n = D.n_calib_sets(REGJ)
ret_c_all = set(ret_c_map)
XD = {b: json.load(open(f"{D.XB[b]}/kld.json"))["docs"] for b in D.XB}
G = {b: np.fromfile(D.XGM[b], np.uint8) for b in D.XGM}
ref = json.load(open(D.PLAN186)); r_c = {c["name"]: c for c in ref["c"]}; r_x = {x["name"]: x for x in ref["x"]}
xrow = {(x["batch"], x["idx"]): x for x in X}
bad = []
ok = lambda name, cond, msg="": (print(("PASS " if cond else "FAIL ") + name, msg), cond or bad.append(name))
sc = lambda i: min(K[i]["tokens"], D.C_CAP) - 1 - K[i]["score_from"]
gc = lambda i: int(GEN[K[i]["row0"]:K[i]["row0"] + sc(i)].sum())


def nrow(b, i):
    d = XD[b][i]; return d["tokens"] - 1 - d["score_from"]


def xg(b, i, cap):
    n = min(nrow(b, i), cap) if cap else nrow(b, i)
    r0 = XD[b][i]["row0"]
    return int(G[b][r0:r0 + n].sum())


def ok_scripts(name, pieces, gen_of, stratum_of, script_of):
    N = [k for k in pieces if stratum_of(k) == "N"]
    gN = sum(gen_of(k) for k in N); sg = collections.Counter()
    for k in N: sg[script_of(k)] += gen_of(k)
    sg = collections.Counter({s: v for s, v in sg.items() if v > 0})       # a script counts only with N gen
    mx = max(sg.values()) / max(gN, 1) if sg else 1.0
    ok(f"{name} >= 4 non-Latin scripts", len(sg) >= 4, f"{len(sg)} scripts {dict(sg.most_common())}")
    ok(f"{name} max script <= 1/2 of N gen", mx <= 0.5 + 1e-9, f"{mx:.3f}")
    tot = sum(gen_of(k) for k in pieces)
    print(f"INFO {name} N gen share {gN / max(tot, 1):.4f} (N gen {gN} of {tot}; {len(N)} N units)")


names_c = [c["name"] for c in plan["c"]]; names_x = [x["name"] for x in plan["x"]]
ok("c parts: s190e, s190f", names_c == ["s190e", "s190f"], str(names_c))
ok("x parts: s190e, s190f", names_x == ["s190e", "s190f"], str(names_x))
ok("c part not blocked (N supply)", all(not c.get("blocked") for c in plan["c"]), str(plan.get("blocked"))[:300])
ok("c N policy is the 4:25 PM amendment (amend425)", plan.get("n_policy") == "amend425", str(plan.get("n_policy")))
# ------------------------------------------------------------------ x side
xk = lambda p: (p["batch"], p["idx"])
allx = [xk(p) for part in plan["x"] for p in part["pieces"]]
ok("x pieces distinct within each part and across the pair", len(allx) == len(set(allx)), f"{len(allx)} pieces, {len(set(allx))} distinct")
for part in plan["x"]:
    name = part["name"]; m = r_x[part["match"]]        # the s186 part the new part doubles (draw190ef PAIR: s190e <- s186c, s190f <- s186d)
    caps = {tuple([k.split(":")[0], int(k.split(":")[1])]): v for k, v in part["caps"].items()}
    miss = [p for p in part["pieces"] if xrow.get(xk(p)) is None or xrow[xk(p)]["doc"] != p["doc"]]
    ok(f"{name}-x pieces are xsupply rows (batch, idx, doc)", not miss, f"{len(miss)} missing")
    fresh_bad = [p for p in part["pieces"] if not ret_x_map.get(p["doc"]) and xrow[xk(p)]["used"]]
    ok(f"{name}-x fresh pieces unused (not in any earlier x part)", not fresh_bad, f"{len(fresh_bad)} used")
    nr_bad = [p for p in part["pieces"] if p["doc"] in nr_x_set]
    ok(f"{name}-x no piece of a non-retired slice / final / t183 / t186n / s171 / s190a-b", not nr_bad, f"{len(nr_bad)} held")
    ret_holders_bad = [p for p in part["pieces"] if ret_x_map.get(p["doc"]) and not set(ret_x_map[p["doc"]]) <= set(D.RETIRED)]
    ok(f"{name}-x reused pieces held only by retired slices (5:15 rule)", not ret_holders_bad, f"{len(ret_holders_bad)} bad")
    lbad = [p for p in part["pieces"] if p["stratum"] == "L" and (p["batch"] != "xc8k-b1" or ret_x_map.get(p["doc"]))]
    ok(f"{name}-x L pieces: fresh xc8k-b1 only", not lbad, f"{len(lbad)} bad")
    nN = [p for p in part["pieces"] if p["stratum"] == "N"]
    ok(f"{name}-x N pieces are N in xsupply", all(xrow[xk(p)]["stratum"] == "N" for p in nN), f"{len(nN)} N pieces")
    gfun = lambda p: xg(p["batch"], p["idx"], caps.get(xk(p), 0))
    ok_scripts(f"{name}-x", part["pieces"], gfun, lambda p: p["stratum"], lambda p: p["script"])
    gl = sum(gfun(p) for p in part["pieces"] if p["stratum"] == "L")
    # 4:25 PM amendment: the x part carries the shortfall of its c part, so the check is on the slice (c + x L gen), below
    print(f"INFO {name}-x L gen {gl} (own 2x target {m['gen_L'] * 2}; the slice total is checked below)")
    print(f"INFO {name}-x pieces {len(part['pieces'])} (N {len(nN)} = fresh {sum(1 for p in nN if not ret_x_map.get(p['doc']))} + reused "
          f"{sum(1 for p in nN if ret_x_map.get(p['doc']))}; L {len(part['pieces']) - len(nN)}); cats L {part.get('cats_L')}")
fresh_all = [x for x in X if x["batch"] != "xc8k-b1" and x["stratum"] == "N" and not x["used"] and x["doc"] not in nr_x_set]
used_fresh = {xk(p) for part in plan["x"] for p in part["pieces"] if not ret_x_map.get(p["doc"]) and p["stratum"] == "N"}
ok("every unused non-Latin b2 piece (N source) is in a pair part", all((x["batch"], x["idx"]) in used_fresh for x in fresh_all),
   f"{len(fresh_all)} unused N b2 pieces, {len(used_fresh)} used fresh")
# ------------------------------------------------------------------ c side
cs = [i for c in plan["c"] for i in c["idx"]]
ok("c documents distinct within each part", all(len(c["idx"]) == len(set(c["idx"])) for c in plan["c"]))
ok("c documents distinct across the pair", len(cs) == len(set(cs)), f"{len(cs)} uses, {len(set(cs))} distinct")
ok("c documents: fit or search pool", all(P[i] in ("fit", "search") for i in cs))
Ls = [i for i in cs if K[i]["stratum"] == "L"]; Ns = [i for i in cs if K[i]["stratum"] == "N"]
SR = D.search_ret_docs(); V2 = D.v2_docs(P)          # 4:40 PM: s190e's Latin may come from these (the search side only)
fresh_c = lambda i: i not in nr_c_set and i not in ret_c_all
ok("c Latin documents never from a non-retired / excluded slice (ex_n, the live pair, t183, t186n, final750, s171, s173f)",
   all(i not in ex_n for i in Ls), f"{sum(1 for i in Ls if i in ex_n)} excluded")
ok("c N documents held only by retired-ok slices (4:25 PM amendment), by no other slice", all(i in ret_n and i not in ex_n for i in Ns),
   f"{sum(1 for i in Ns if i in ex_n or i not in ret_n)} not allowed")
ok("c N documents: no stratum mismatch (N in calib strata)", all(K[i]["stratum"] == "N" for i in Ns))
ok("c N documents carry N gen (gen > 0)", all(gc(i) > 0 for i in Ns), f"{sum(1 for i in Ns if gc(i) == 0)} with 0 gen")
holders = collections.Counter(s for i in Ns for s in sorted(ret_n[i]))
print(f"INFO c N holders (slices holding each N document, counts): {dict(holders)}")
for c in plan["c"]:
    Lc = [i for i in c["idx"] if K[i]["stratum"] == "L"]
    if c["name"] == "s190f":           # confirm: fresh Latin only (the freshest first)
        ok("s190f-c Latin documents fresh (not in any slice)", all(fresh_c(i) for i in Lc), f"{sum(1 for i in Lc if not fresh_c(i))} not fresh")
    else:                              # search (4:40 PM): fresh, or a retired search document, or a V2-era fit document
        ok("s190e-c Latin documents fresh, retired search or V2-era (4:40 PM)", all(fresh_c(i) or i in SR or i in V2 for i in Lc),
           f"{sum(1 for i in Lc if not (fresh_c(i) or i in SR or i in V2))} other")
        ok("s190e-c Latin reused documents: none held by a non-retired or confirm / reselect slice unless retired-search or V2",
           all(set(ret_c_map.get(i, [])) <= set(D.RETIRED_C_N) for i in Lc), "")
    Nc = [i for i in c["idx"] if K[i]["stratum"] == "N"]
    ok(f"{c['name']}-c N documents reused from retired-ok slices only", all(i in ret_n and i not in ex_n and set(ret_n[i]) <= set(D.RETIRED_C_N) for i in Nc),
       f"{len(Nc)} N documents")
    if Nc or c.get("blocked") is None:
        ok_scripts(f"{c['name']}-c", c["idx"], gc, lambda i: K[i]["stratum"], lambda i: K[i]["script"])
    m = r_c[c["match"]]
    gl = sum(gc(i) for i in c["idx"] if K[i]["stratum"] == "L")
    print(f"INFO {c['name']}-c L gen {gl} vs own 2x target {2 * m['gen_L']} ({gl / m['gen_L']:.3f} x s186; the calib Latin pool is short)")
    print(f"INFO {c['name']}-c docs {len(c['idx'])} (N {len(Nc)}, L {len(Lc)}), scored positions {sum(sc(i) for i in c['idx'])}")
# slice totals: the c + x L gen of each slice >= 1.9 x the matching s186 slice's c + x L gen (4:25 PM: the shortfall goes to the x part)
for c, x in zip(plan["c"], plan["x"]):
    m_c = r_c[c["match"]]; m_x = r_x[c["match"]]
    cl = sum(gc(i) for i in c["idx"] if K[i]["stratum"] == "L")
    caps_x = {tuple([k.split(":")[0], int(k.split(":")[1])]): v for k, v in x["caps"].items()}
    xl = sum(xg(p["batch"], p["idx"], caps_x.get((p["batch"], p["idx"]), 0)) for p in x["pieces"] if p["stratum"] == "L")
    tot = m_c["gen_L"] + m_x["gen_L"]
    ok(f"{c['name']} slice L gen (c + x) >= 1.9 x {c['match']} (c + x)", cl + xl >= 1.9 * tot, f"{cl + xl} vs {tot} ({(cl + xl) / tot:.3f} x)")
# calib Latin order (4:40 PM): s190f takes the freshest documents first; if the free pool is not enough for its 2x, every free document
# goes to the pair (s190f first, then s190e)
cLs = {c["name"]: sum(gc(i) for i in c["idx"] if K[i]["stratum"] == "L") for c in plan["c"]}
free_L = {i for i in range(len(P)) if P[i] in ("fit", "search") and fresh_c(i) and K[i]["stratum"] == "L" and sc(i) >= 16}
used_fresh = {i for c in plan["c"] for i in c["idx"] if i in free_L}
f_part = next(c for c in plan["c"] if c["name"] == "s190f")
ok("s190f holds the freshest Latin: all free documents are used by the pair, or s190f has its 2x",
   used_fresh == free_L or cLs["s190f"] >= 2 * r_c[f_part["match"]]["gen_L"],
   f"{len(used_fresh)} of {len(free_L)} free used; s190f c L {cLs['s190f']}, s190e c L {cLs['s190e']}")
reuse_pool = {i for i in range(len(P)) if P[i] in ("fit", "search") and i not in ex_n and K[i]["stratum"] == "L" and sc(i) >= 16
              and (i in SR or i in V2)}
s_part = next(c for c in plan["c"] if c["name"] == "s190e")
ok("calib Latin: s190e reaches its 2x, or every reuse document is in use",
   cLs["s190e"] >= 2 * r_c[s_part["match"]]["gen_L"] or reuse_pool <= set(s_part["idx"]) | used_fresh,
   f"s190e c L {cLs['s190e']} vs 2x {2 * r_c[s_part['match']]['gen_L']}; reuse pool {len(reuse_pool)}")
print("ALL PASS" if not bad else f"FAILED: {bad}")
sys.exit(1 if bad else 0)
