# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""draw190ef.py -- the draw of RULES-cooldown 'Amendment 2026-11-15 4:25 PM CST (user: rotate): pair s190e / s190f' (w179), built on the
draw190cd.py pipeline (w176; the cd scripts are unchanged). The pair s190e (search) and s190f (confirm):

  Latin (L), calib: 2 x the matching s186 part's L gen (s190e <- s186c, s190f <- s186d: the second s186 pair). Fresh = outside every
      earlier slice (retired ones included, s186b/d/e/f, s190a-d, t183, t186n, s171, s173f, final-x, verify). 4:40 PM amendment: s190f
      (confirm) takes the fresh documents first; s190e (search) takes what is left of the fresh pool, then retired search slices' and
      V2-era fit + search documents (not in any excluded slice), least used first by V3 decisions (ledger v3 / v3.1 + doc_decisions).
      A c part short of its 2x is filled in its x part (4:25 PM: the slice's total L gen is 2x the s186 slice's c + x).
      Latin (L), xcorp: the unused xc8k-b1 L pieces (fresh), as before; they carry the shortfall of the two c parts.
  N, calib (4:25 amendment, "N reused from retired search / confirm / reselect slices, now also s190a and s190b"): documents of the
      retired s173a-e, s186a-f, s190a, and s190b once it has no pending confirmation (the driver state, checked at draw time; a pending
      one leaves s190b out). Never a document held by any other slice: s190c / s190d (live pair), s173f (staged), s171a-d, t183, t186n,
      golden-final750 (c1 / c3), or any other non-retired registry slice. Verify is never in the pool. Least used first.
  N, xcorp (same source as calib, the 4:25 text covers N reuse in general): pieces of the retired s173a-e, s186a-f, s190a and s190b (when
      clear), and the unused xc8k-b2 N pieces (gen masks by genmask181.py; all 11 were used by s190c / s190d, so none is left). Never a
      piece of s190c / s190d, s173f, s171a-d, t183, t186n, final-x, or verify.
  Each N part targets the N gen of the s190c / s190d parts (c 6,388; x 25,120, the same targets as w176: their means were 6,530 / 25,172),
  at least 4 non-Latin scripts per part, no script above half of a part's N gen. Documents / pieces are distinct within the pair.

Runs on ds8 (CPU, nice -n 19). Reads the calib golden, the xcorp goldens' metadata, the registry and ledger snapshots (copies taken on
ds8; paths below), the xsupply table, the xc8k gen masks and the s190b pending file. Writes only the plan (indices and counts).
  python3 -I draw190ef.py PLAN.json SEED [--supply] [--n-policy amend425|strict]
--supply: print and write the supply counts only (no plan). strict = the 5:15 rule for the calib N (retired s173a-e / s186a / s186c only)."""
import collections, json, os, random, sys

import numpy as np

L_MULT = 2.0                                        # the Latin part doubles (RULES-cooldown 1' / 1'')
C_CAP = 4096                                        # draw171's cut for calib documents
CUT = 2048                                          # pickx's cap for the heavy categories
HEAVY = {"agentic", "tools", "long", "reasoning"}
PAIR = [("s190e", "s186c"), ("s190f", "s186d")]    # (new part, matching s186 part): search, then confirm
RETIRED_BASE = ["s173a", "s173b", "s173c", "s173d", "s173e", "s186a", "s186b", "s186c", "s186d", "s186e", "s186f", "s190a"]
S190B_PENDING_FILE = "/data/q27/w179/in/s190b-pending.json"   # written from the driver state on the VM (tentative without confirm)
def s190b_clear():
    return json.load(open(S190B_PENDING_FILE))["pending"] == 0
RETIRED = RETIRED_BASE + (["s190b"] if s190b_clear() else [])   # the retired-ok slices (x and calib N sources)
RETIRED_C_N = RETIRED
AMEND_OK = {"s186b", "s186d", "s186e", "s186f"} | ({"s190b"} if s190b_clear() else set())   # staged in the registry, counted as retired
N_GEN_C = 6388                                      # N gen target of the s190c / s190d c parts (6,368 / 6,693; the s190a/b mean)
N_GEN_X = 25120                                     # N gen target of the x parts (28,152 / 22,191 for s190c / d)
MIN_SCRIPTS = 4

CAL = "/data/q27/w164/strata/calib-kld-strata.json"
POOLS = "/data/q27/w164/strata/pools-v3.json"
CALGEN = "/data/q27/challenge/golden/calib/gen.u8"
LED = "/data/q27/w179/in/ledger.jsonl"              # snapshot of eval/pools/ledger.jsonl at the draw (w179 copies it; see README)
REG = "/data/q27/w179/in/registry-live.json"        # snapshot of the live registry (lead-srch/run/registry.json) at the draw
S = "/data/q27/w131/slices"; S186 = "/data/q27/w164/slices"; S190 = "/data/q27/w169/slices"; S190CD = "/data/q27/w176/slices"
PLAN186 = "/data/q27/w164/strata/plan186.json"
XSUP = "/data/q27/w164/strata/xsupply186b.jsonl"
XB = {"xc8k-b1": "/data/q27/goldens/xcorp/xc8k-b1", "xc8k-b2-c0": "/data/q27/goldens/xcorp/xc8k-b2-c0",
      "xc8k-b2-c1": "/data/q27/goldens/xcorp/xc8k-b2-c1", "xc8k-b2-c2": "/data/q27/goldens/xcorp/xc8k-b2-c2"}
XGM = {"xc8k-b1": "/data/q27/w164/strata/gm-b1.u8", "xc8k-b2-c0": "/data/q27/w176/strata/gm-b2-c0.u8",
       "xc8k-b2-c1": "/data/q27/w176/strata/gm-b2-c1.u8", "xc8k-b2-c2": "/data/q27/w176/strata/gm-b2-c2.u8"}
FINAL_C = "/data/q27/final/golden-final750"; FINAL_X = "/data/q27/final/golden-final-x"


def docs(p):
    return json.load(open(f"{p}/kld.json"))["docs"]


SEARCH_RET_SLICES = ["s173a", "s173b", "s173c", "s173d", "s173e", "s186a", "s186c", "s190a"]   # retired search slices (4:40 PM: s190e Latin)


def search_ret_docs():
    """calib documents of the retired search slices' c goldens (the 4:40 PM Latin reuse source for s190e)"""
    out = set()
    for name in SEARCH_RET_SLICES:
        out |= {d["orig"] for d in docs(slice_dir(name)) if d.get("orig") is not None}
    return out


def v2_docs(pool):
    """fit + search documents with a ledger row from the V2 era (line v2, v1,v2 or v2-emb; the 4:40 PM Latin reuse source for s190e)"""
    out = set()
    for l in open(LED):
        r = json.loads(l)
        if "v2" in r["line"] and r["doc"] != "all" and pool[int(r["doc"])] in ("fit", "search"): out.add(int(r["doc"]))
    return out


def slice_dir(name):
    """the calib c golden directory of a slice name (s173a-e: w131; s186a-f: w164; s190a/b: w169; s190c/d: w176)"""
    if name.startswith("s173"): return f"{S}/golden-{name}-c"
    if name in ("s190a", "s190b"): return f"{S190}/golden-{name}-c"
    if name in ("s190c", "s190d"): return f"{S190CD}/golden-{name}-c"
    return f"{S186}/golden-{name}-c"


def xdir(name):
    """the x golden directory of a slice name (same roots as slice_dir)"""
    return slice_dir(name)[:-len("-c")] + "-x"


def gold_sets():
    """the golden directories by role (x side and the Latin freshness; RULES-cooldown 4:25 PM amendment, w179).
    ret_*: the retired-ok slices (RETIRED: their documents / pieces may be reused as N, never as L);
    nr_*: everything an earlier or non-retired slice holds (never reused, never fresh)"""
    ret_c = [slice_dir(n) for n in RETIRED]
    ret_x = [xdir(n) for n in RETIRED]
    # everything an earlier or non-retired slice holds (never reused, never fresh)
    nr_c = [f"{S}/golden-s171{k}-c" for k in "abcd"] + [f"{S}/golden-s173f-c"] + \
           [f"{S}/golden-t183-c", f"{S186}/golden-t186n-c", f"{S190CD}/golden-s190c-c", f"{S190CD}/golden-s190d-c"]
    nr_x = [f"{S}/golden-s171{k}-x" for k in "abcd"] + [f"{S}/golden-s173f-x"] + \
           [f"{S}/golden-t183-x", f"{S186}/golden-t186n-x", FINAL_X, f"{S190CD}/golden-s190c-x", f"{S190CD}/golden-s190d-x"]
    if "s190b" not in RETIRED:                       # pending confirmation: its documents are left out
        nr_c.append(slice_dir("s190b")); nr_x.append(xdir("s190b"))
    return ret_c, ret_x, nr_c, nr_x


def exclusion(REGJ):
    """returns (fresh_c, reg_nr_c, retired c orig->slices (s173a-e, s186a, s186c), nr_x_set, retired x doc->slices)"""
    ret_c, ret_x, nr_c, nr_x = gold_sets()
    # registry: every non-retired slice's calib documents (status ready / active / staged / confirm / reselect / tracking)
    reg_nr_c = set()
    for name, sl in REGJ["slices"].items():
        if sl["status"] != "retired":
            reg_nr_c |= {int(d["id"][2:]) for d in sl["docs"] if d["part"] == "c"}
    nr_c_set = set(reg_nr_c)
    for g in nr_c:
        nr_c_set |= {d["orig"] for d in docs(g) if d.get("orig") is not None}
    nr_c_set |= {d["orig"] for d in docs(FINAL_C) if d.get("part") in ("c1", "c3")}
    ret_c_map = collections.defaultdict(set)
    for name in RETIRED:
        if name not in AMEND_OK and REGJ["slices"][name]["status"] != "retired": raise SystemExit(f"{name} is not retired in the registry snapshot")
    for g in ret_c:
        sl = os.path.basename(g)[len("golden-"):-len("-c")]
        for d in docs(g):
            if d.get("orig") is not None: ret_c_map[d["orig"]].add(sl)
    nr_x_set = set()
    for g in nr_x:
        if os.path.exists(f"{g}/kld.json"): nr_x_set |= {d["xdoc"] for d in docs(g) if "xdoc" in d}
    ret_x_map = collections.defaultdict(set)
    for g in ret_x:
        sl = os.path.basename(g)[len("golden-"):-len("-x")]
        for d in docs(g):
            if "xdoc" in d: ret_x_map[d["xdoc"]].add(sl)
    return nr_c_set, reg_nr_c, dict(ret_c_map), nr_x_set, dict(ret_x_map)


def n_calib_sets(REGJ):
    """the calib N pool of the 4:25 PM amendment (w179): (excluded documents, {orig: the retired-ok slices holding it}).
    Excluded: every document of a registry slice that is not retired and not in AMEND_OK, plus the goldens of s173f, s171a-d, t183,
    t186n, s190a / s190b (c) and golden-final750 (c1 / c3). Verify is outside the fit + search pool (checked by the caller)."""
    for name in AMEND_OK:
        if name not in REGJ["slices"]: raise SystemExit(f"{name} missing from the registry snapshot")
    ex = set()
    for name, sl in REGJ["slices"].items():
        if sl["status"] != "retired" and name not in AMEND_OK:
            ex |= {int(d["id"][2:]) for d in sl["docs"] if d["part"] == "c"}
    for g in [f"{S}/golden-t183-c", f"{S186}/golden-t186n-c", f"{S}/golden-s173f-c"] + [f"{S}/golden-s171{k}-c" for k in "abcd"] + \
             [f"{S190CD}/golden-s190c-c", f"{S190CD}/golden-s190d-c"]:
        ex |= {d["orig"] for d in docs(g) if d.get("orig") is not None}
    if "s190b" not in RETIRED:
        ex |= {d["orig"] for d in docs(slice_dir("s190b")) if d.get("orig") is not None}
    ex |= {d["orig"] for d in docs(FINAL_C) if d.get("part") in ("c1", "c3")}
    ret = collections.defaultdict(set)
    for name in RETIRED_C_N:
        if name not in AMEND_OK and REGJ["slices"][name]["status"] != "retired": raise SystemExit(f"{name} is not retired in the snapshot")
        for d in docs(slice_dir(name)):
            if d.get("orig") is not None: ret[d["orig"]].add(name)
    return ex, dict(ret)


def split_pair(items, gen_of, script_of):
    """the items of one stratum split into the two parts of the pair (each item in one part only): the scripts are taken
    in order of size; the first two items of a script go one to each part (so every script with >= 2 items is in both
    parts), the rest to the part with less gen so far (balanced by gen positions). As draw190.split_pair."""
    by = collections.defaultdict(list)
    for i in items: by[script_of(i)].append(i)
    parts, gg = ([], []), [0, 0]
    for s in sorted(by, key=lambda s: (-len(by[s]), s)):
        first = None
        for k, i in enumerate(sorted(by[s], key=lambda i: (-gen_of(i), i))):
            p = 1 - first if k == 1 else gg.index(min(gg))
            if k == 0: first = p
            parts[p].append(i); gg[p] += gen_of(i)
    return parts


def main(plan_path, seed, supply_only=False, n_policy="amend425"):
    # ---------------------------------------------------------------- calib
    K = json.load(open(CAL))["docs"]; P = json.load(open(POOLS))["pool"]; GEN = np.fromfile(CALGEN, np.uint8)
    sc = lambda i: min(K[i]["tokens"], C_CAP) - 1 - K[i]["score_from"]
    gen_c = {}
    for i, d in enumerate(K):
        n = sc(i); gen_c[i] = int(GEN[d["row0"]:d["row0"] + n].sum())
    REGJ = json.load(open(REG))
    nr_c_set, reg_nr_c, ret_c_map, nr_x_set, ret_x_map = exclusion(REGJ)
    ref = json.load(open(PLAN186)); r_c = {c["name"]: c for c in ref["c"]}; r_x = {x["name"]: x for x in ref["x"]}
    mine = {f"golden-{k}-c" for k in REGJ["slices"]}
    use = collections.Counter()
    for l in open(LED):
        r = json.loads(l)
        if r["doc"] == "all" or r["use"] in mine: continue
        dd = r.get("decisions"); k = dd if isinstance(dd, (int, float)) and not isinstance(dd, bool) else 1
        if r["use"] in ("w94-headx-search", "w94-headx-qad", "w119-headx-search", "w119-headx-fit", "w119-headx-verify"): continue
        use[int(r["doc"])] += k
    for k, v in REGJ["doc_decisions"].items():
        if k.startswith("c:"): use[int(k[2:])] += v
    # V3 decision count (4:40 PM: "least-used first by V3 decision count (ledger)"): the ledger's v3 / v3.1 decisions (own goldens
    # skipped, as above) plus the registry's doc_decisions
    use_v3 = collections.Counter()
    for l in open(LED):
        r = json.loads(l)
        if r["doc"] == "all" or r["use"] in mine or r["line"] not in ("v3", "v3.1"): continue
        dd = r.get("decisions"); k = dd if isinstance(dd, (int, float)) and not isinstance(dd, bool) else 0
        use_v3[int(r["doc"])] += k
    for k, v in REGJ["doc_decisions"].items():
        if k.startswith("c:"): use_v3[int(k[2:])] += v
    fit_search = lambda i: P[i] in ("fit", "search")
    # fresh Latin: outside every earlier and non-retired slice AND outside the retired ones (their documents are reused only as N)
    ret_c_all = set(ret_c_map)
    free = [i for i in range(len(P)) if fit_search(i) and i not in nr_c_set and i not in ret_c_all]
    L_free = [i for i in free if K[i]["stratum"] == "L" and sc(i) >= 16]
    # calib N: the 5:15 strict pool (retired s173a-e / s186a / s186c, not held by any non-retired slice) and the 5:40 pool
    N_reuse = [i for i in ret_c_map if i not in nr_c_set and fit_search(i) and K[i]["stratum"] == "N" and sc(i) >= 16]
    ex_n, ret_n = n_calib_sets(REGJ)
    # N units need N gen (a document with no gen positions adds scored rows, not N signal: 520 had 4,095 rows and 0 gen)
    N_540 = [i for i in ret_n if i not in ex_n and fit_search(i) and K[i]["stratum"] == "N" and sc(i) >= 16 and gen_c[i] > 0]
    all_N = [i for i in range(len(P)) if fit_search(i) and K[i]["stratum"] == "N" and sc(i) >= 16]
    rng2 = random.Random(seed + 1); rnd = {i: rng2.random() for i in range(len(P))}
    stats = {"calib": {"fit_search_docs": sum(1 for p in P if p in ("fit", "search")), "nonretired_c_docs": len(nr_c_set),
                       "registry_nonretired_c": len(reg_nr_c), "retired_c_docs": len(ret_c_all),
                       "L_fresh_docs": len(L_free), "L_fresh_gen": sum(gen_c[i] for i in L_free),
                       "N_fit_search_docs": len(all_N), "N_fit_search_docs_held_by_excluded": sum(1 for i in all_N if i in ex_n),
                       "N_pool_strict": {"docs": len(N_reuse), "scripts": dict(collections.Counter(K[i]["script"] for i in N_reuse)),
                                         "gen": sum(gen_c[i] for i in N_reuse)},
                       "N_pool_540": {"docs": len(N_540), "scripts": dict(collections.Counter(K[i]["script"] for i in N_540)),
                                      "gen": sum(gen_c[i] for i in N_540),
                                      "holders": dict(collections.Counter(s for i in N_540 for s in sorted(ret_n[i])))}}}
    SR0 = search_ret_docs(); V20 = v2_docs(P)
    Lr0 = [i for i in range(len(P)) if fit_search(i) and i not in ex_n and K[i]["stratum"] == "L" and sc(i) >= 16 and (i in SR0 or i in V20)]
    stats["calib"]["L_reuse_for_s190e"] = {"docs": len(Lr0), "gen": sum(gen_c[i] for i in Lr0), "search_retired_docs": sum(1 for i in Lr0 if i in SR0),
                                           "v2_docs": sum(1 for i in Lr0 if i in V20), "v2_docs_all": len(V20)}
    # ---------------------------------------------------------------- xcorp
    X = [json.loads(l) for l in open(XSUP)]
    XD = {b: json.load(open(f"{XB[b]}/kld.json"))["docs"] for b in XB}
    G = {b: np.fromfile(XGM[b], np.uint8) for b in XGM}
    by_doc = collections.defaultdict(list)
    for x in X: by_doc[x["doc"]].append(x)
    ndup = sum(1 for v in by_doc.values() if len(v) > 1)
    def nrow(b, i):
        d = XD[b][i]; return d["tokens"] - 1 - d["score_from"]
    def cutof(x):
        return CUT if x["category"] in HEAVY and nrow(x["batch"], x["idx"]) > CUT else 0
    def xgen(x):
        cut = cutof(x); n = min(nrow(x["batch"], x["idx"]), cut) if cut else nrow(x["batch"], x["idx"])
        r0 = XD[x["batch"]][x["idx"]]["row0"]
        return int(G[x["batch"]][r0:r0 + n].sum())
    xgen_of = {}
    for x in X:
        if x["batch"] in G: xgen_of[(x["batch"], x["idx"])] = xgen(x)
    # used by a non-retired slice / an earlier x part (xsupply flags, and the non-retired goldens' xdocs)
    def x_excluded(x):
        return x["used"] or x["doc"] in nr_x_set
    fresh_L = [x for x in X if x["batch"] == "xc8k-b1" and x["stratum"] == "L" and not x_excluded(x) and x["doc"] not in ret_x_map]
    fresh_N = [x for x in X if x["stratum"] == "N" and not x_excluded(x) and x["doc"] not in ret_x_map]
    reuse_N = [x for x in X if x["doc"] in ret_x_map and x["stratum"] == "N" and x["doc"] not in nr_x_set]
    n_ret_x_unmapped = len(ret_x_map) - len({x["doc"] for x in X if x["doc"] in ret_x_map})
    # least used: the sum of the decisions of the retired slices that hold the piece (fewest decided first)
    dec = {n: sum(REGJ["slices"][n]["decisions"].values()) if REGJ["slices"][n].get("decisions") else 0 for n in RETIRED}
    def x_use(x):
        return sum(dec.get(s, 0) for s in ret_x_map.get(x["doc"], []))
    stats["xcorp"] = {"fresh_L_b1": len(fresh_L), "fresh_N": len(fresh_N),
                      "fresh_N_by_batch": dict(collections.Counter(x["batch"] for x in fresh_N)),
                      "fresh_N_gen": sum(xgen_of[(x["batch"], x["idx"])] for x in fresh_N),
                      "reuse_N_retired_pieces": len(reuse_N), "reuse_N_by_batch": dict(collections.Counter(x["batch"] for x in reuse_N)),
                      "reuse_N_gen": sum(xgen_of[(x["batch"], x["idx"])] for x in reuse_N),
                      "reuse_N_scripts": dict(collections.Counter(x["script"] for x in reuse_N)),
                      "retired_x_docs": len(ret_x_map), "retired_x_docs_without_xsupply_row": n_ret_x_unmapped,
                      "xdoc_in_two_batches": ndup,
                      "L_fresh_gen": sum(xgen_of[(x["batch"], x["idx"])] for x in fresh_L)}
    print(json.dumps(stats, indent=1), file=sys.stderr)
    if supply_only:
        json.dump(stats, open(plan_path, "w"), indent=1); return
    # ---------------------------------------------------------------- N, calib: least used first until both parts reach their target
    def ok_part(ids, gen, script):
        sg = collections.Counter()
        for i in ids: sg[script(i)] += gen(i)
        tot = sum(sg.values()); sg = {s: v for s, v in sg.items() if v > 0}     # a script counts only with N gen
        return len(sg) >= MIN_SCRIPTS and max(sg.values()) <= 0.5 * tot

    def pick_N(pool, gen, script, target, key):
        """the least used units first until both parts can reach the target; None when the supply cannot (no part is built then)"""
        order = sorted(pool, key=key); chosen = []
        for i in order:
            chosen.append(i)
            if sum(gen(j) for j in chosen) < 2 * target: continue
            parts = split_pair(chosen, gen, script)
            if all(ok_part(p, gen, script) for p in parts): return parts, chosen
        return None

    gen_cN = lambda i: gen_c[i]
    if n_policy == "strict":
        pool_c, holders_c = N_reuse, ret_c_map
    elif n_policy == "amend425":
        pool_c, holders_c = N_540, ret_n
    else:
        raise SystemExit(f"unknown --n-policy {n_policy}")
    res_c = pick_N(pool_c, gen_cN, lambda i: K[i]["script"], N_GEN_C, key=lambda i: (use[i], rnd[i]))
    if res_c is None:
        blocked = (f"calib N supply ({n_policy}): {len(pool_c)} documents, gen {sum(gen_c[i] for i in pool_c)}, scripts "
                   f"{dict(collections.Counter(K[i]['script'] for i in pool_c))}: two c parts of ~{N_GEN_C} N gen with >= {MIN_SCRIPTS} "
                   f"scripts each cannot be drawn. No c N part is drawn; the Latin c parts and the x parts are still drawn.")
        print("BLOCKED: " + blocked, file=sys.stderr)
        N_parts_c = [[], []]
    else:
        N_parts_c = res_c[0]; blocked = None
    plan = {"seed": seed, "n_policy": n_policy, "blocked": blocked, "c": [], "x": [], "stats": stats}
    # calib Latin, two amendments: 4:25 PM (the calib Latin shortfall of a c part is filled with xcorp Latin pieces in its x part, so the
    # slice's total L gen is 2x the matching s186 slice's c + x L gen; "same total L gen positions per slice") and 4:40 PM (for s190e,
    # the search side only, Latin may also reuse the retired search slices' and the V2-era fit + search documents, least used first by
    # the V3 decision count; s190f, the confirm side, keeps the freshest Latin). Order: s190f takes the fresh documents first (least
    # used, until its own 2x); s190e takes what is left of the fresh pool, then the reuse pool, by V3 decisions, until its own 2x.
    # A c part's shortfall is filled in its x part (below).
    SEARCH_RET = search_ret_docs(); V2 = v2_docs(P)
    L_reuse = [i for i in range(len(P)) if fit_search(i) and i not in ex_n and K[i]["stratum"] == "L" and sc(i) >= 16
               and (i in SEARCH_RET or i in V2)]
    tgt_c = {name: L_MULT * r_c[match]["gen_L"] for name, match in PAIR}
    c_L = {name: [] for name, _ in PAIR}; c_g = {name: 0 for name, _ in PAIR}; taken_L = set()
    def fill_c(name, pool):
        for i in sorted(pool, key=lambda i: (use_v3[i], rnd[i])):
            if c_g[name] >= tgt_c[name]: break
            if i in taken_L: continue
            c_L[name].append(i); c_g[name] += gen_c[i]; taken_L.add(i)
    fill_c("s190f", L_free)                                   # the confirm part: the freshest Latin first
    fill_c("s190e", L_free + L_reuse)                         # the search part: fresh left over, then reuse (by V3 decisions)
    slice_L_target = {name: L_MULT * (r_c[match]["gen_L"] + r_x[match]["gen_L"]) for name, match in PAIR}
    for k, (name, match) in enumerate(PAIR):
        pick_l = c_L[name]
        pn = N_parts_c[k]
        ch = sorted(pn + pick_l)
        gl = c_g[name]; gn = sum(gen_c[i] for i in pn)
        sg = collections.Counter()
        for i in pn: sg[K[i]["script"]] += gen_c[i]
        plan["c"].append({"name": name, "match": match, "idx": ch, "N": sorted(pn), "gen_N": gn, "gen_L": gl,
                          "share_N": gn / max(gn + gl, 1), "L_target": L_MULT * r_c[match]["gen_L"], "docs_N": len(pn), "docs_L": len(pick_l),
                          "N_reused_from": {str(i): sorted(holders_c.get(i, [])) for i in pn},
                          "scripts": dict(collections.Counter(K[i]["script"] for i in pn)),
                          "max_script_share": max(sg.values()) / gn if gn else 0.0, "pos": sum(sc(i) for i in ch),
                          "blocked": blocked})
    # ---------------------------------------------------------------- N, xcorp: fresh b2 N pieces first (no use), then least used
    fresh_key = lambda x: (x_use(x), rnd_x(x))
    rng3 = random.Random(seed + 2)
    rx = {}
    def rnd_x(x):
        k = (x["batch"], x["idx"])
        if k not in rx: rx[k] = rng3.random()
        return rx[k]
    pool_N = fresh_N + [x for x in reuse_N if (x["batch"], x["idx"]) not in {(y["batch"], y["idx"]) for y in fresh_N}]
    xgen_f = lambda x: xgen_of[(x["batch"], x["idx"])]
    xscript = lambda x: x["script"]
    res_x = pick_N(pool_N, xgen_f, xscript, N_GEN_X, key=fresh_key)
    if res_x is None: raise SystemExit("x N supply cannot give two parts of 25,120 N gen with >= 4 scripts each")
    N_parts_x = res_x[0]
    # ---------------------------------------------------------------- L, xcorp: stratified by category, grown to the target
    def pick_l(pool, n, seed_n):
        cnt = collections.Counter(x["category"] for x in pool); tot = sum(cnt.values())
        want = {c: n * v / tot for c, v in cnt.items()}
        take = {c: min(cnt[c], int(want[c])) for c in cnt}
        for c in sorted(cnt, key=lambda c: (-(want[c] - int(want[c])), c)):
            if sum(take.values()) >= n: break
            if take[c] < cnt[c]: take[c] += 1
        rl = random.Random(seed + 100 + seed_n); pl = []
        for c in sorted(take):
            pool_c = [x for x in pool if x["category"] == c]
            rl.shuffle(pool_c); pl += pool_c[:take[c]]
        return pl
    taken_x = set()
    for k, (name, match) in enumerate(PAIR):
        # the x part's L target: the slice's 2x total minus the c part's calib L (the shortfall of the c part is filled here)
        target = slice_L_target[name] - c_g[name]; nL0 = len(r_x[match]["idx"]) - r_x[match]["nN"]
        pool = [x for x in fresh_L if (x["batch"], x["idx"]) not in taken_x]
        for n in range(nL0, nL0 + 1000):
            pl = pick_l(pool, n, k)
            gl = sum(xgen_f(x) for x in pl)
            if gl >= target: break
        taken_x |= {(x["batch"], x["idx"]) for x in pl}
        pn = N_parts_x[k]
        gN = sum(xgen_f(x) for x in pn)
        caps = {}
        pieces = []
        for x in pn + pl:
            c = cutof(x)
            if c: caps[f"{x['batch']}:{x['idx']}"] = c
            pieces.append({"batch": x["batch"], "idx": x["idx"], "doc": x["doc"], "stratum": x["stratum"], "script": x["script"],
                           "category": x["category"], "cap": c, "gen": xgen_f(x), "reused_from": sorted(ret_x_map.get(x["doc"], []))})
        pieces.sort(key=lambda p: (p["batch"], p["idx"]))
        sg = collections.Counter()
        for x in pn: sg[x["script"]] += xgen_f(x)
        plan["x"].append({"name": name, "match": match, "pieces": pieces, "caps": caps, "nN": len(pn), "nL": len(pl),
                          "gen_N": gN, "gen_L": gl, "L_target": target, "L_target_own_2x": L_MULT * r_x[match]["gen_L"],
                          "slice_L_target": slice_L_target[name], "share_N": gN / max(gN + gl, 1),
                          "scripts": dict(collections.Counter(x["script"] for x in pn)),
                          "max_script_share": max(sg.values()) / gN if gN else 0.0,
                          "cats_L": dict(collections.Counter(x["category"] for x in pl)),
                          "N_fresh": sum(1 for x in pn if not ret_x_map.get(x["doc"])),
                          "N_reused": sum(1 for x in pn if ret_x_map.get(x["doc"]))})
    json.dump(plan, open(plan_path + ".part", "w"), indent=1); os.replace(plan_path + ".part", plan_path)
    for c in plan["c"]:
        print(f"{c['name']}: c {len(c['idx'])} docs (N {c['docs_N']}, L {c['docs_L']}, scripts {len(c['scripts'])}), N gen {c['gen_N']} "
              f"of {c['gen_N'] + c['gen_L']} = {c['share_N']:.3f}, L gen {c['gen_L']} (target {c['L_target']:.0f}), pos {c['pos']}", file=sys.stderr)
    for x in plan["x"]:
        print(f"{x['name']}: x {len(x['pieces'])} pieces (N {x['nN']} [fresh {x['N_fresh']}, reused {x['N_reused']}], L {x['nL']}, "
              f"scripts {len(x['scripts'])}), N gen {x['gen_N']} of {x['gen_N'] + x['gen_L']} = {x['share_N']:.3f}, "
              f"L gen {x['gen_L']} (target {x['L_target']:.0f})", file=sys.stderr)


if __name__ == "__main__":
    args = sys.argv[1:]
    sup = "--supply" in args
    pol = "amend425"
    if "--n-policy" in args:
        pol = args[args.index("--n-policy") + 1]
        args = [a for a in args if a not in ("--n-policy", pol)]
    args = [a for a in args if a != "--supply"]
    main(args[0], int(args[1]), supply_only=sup, n_policy=pol)
