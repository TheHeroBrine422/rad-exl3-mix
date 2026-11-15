# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""draw190cd.py -- the draw of RULES-cooldown 1'' (amendment 2026-11-15 5:15 AM CST, user: "rotate") as widened by the amendment of
5:40 AM CST (user "c", option A): the pair s190c (search) and s190d (confirm). A copy of draw190.py (the s190a/b draw, unchanged) with
the N source of the amendments:

  Latin (L): fresh, as draw190: the free calib L documents of the merged fit + search pool (least used first) and the unused
      xc8k-b1 L pieces, each part until its L gen reaches 2 x the matching s186 part's L gen (s190c <- s186a, s190d <- s186b).
      Fresh = outside every earlier slice (retired ones included, and s186b/d/e/f): draw190's exclusion, PLUS the s190a / s190b x
      pieces (draw190's exclusion misses their xdocs). No Latin document or piece is shared with any earlier slice.
  N, calib (5:40 amendment): documents of the RETIRED slices s173a-e, s186a, s186c and of the retired s186b, s186d, s186e, s186f
      (AMEND_OK: staged in the registry, but their pairs / the re-selection are finished). Never a document held by any other slice:
      s190a / s190b (live), s173f (staged), s171a-d, t183, t186n, golden-final750 (c1 / c3), or any other non-retired registry slice.
      Verify is never in the pool (fit + search only). Least used first (ledger + registry decision counts).
  N, xcorp (5:15 rule, unchanged): pieces of the retired s173a-e / s186a / s186c only, and the 11 unused xc8k-b2 N pieces (gen masks by
      genmask181.py, built first; no use yet, so they go first). The 5:40 amendment names calib documents only, so the x parts are the
      ones built on 11-15 (their plan is identical on the 11-15 registry; see README).
  Each N part targets the mean N gen of s190a / s190b's parts (c 6,388; x 25,120), at least 4 non-Latin scripts per part, no script above
  half of a part's N gen. Documents / pieces are distinct within the pair. The N share is printed, not aimed at a band.

Runs on ds8 (CPU, nice -n 19). Reads the calib golden, the xcorp goldens' metadata, the registry and ledger snapshots (copies taken on
ds8; paths below), the xsupply table and the xc8k-b2 gen masks. Writes only the plan (indices and counts).
  python3 -I draw190cd.py PLAN.json SEED [--supply] [--n-policy amend540|strict]
--supply: print and write the supply counts only (no plan). strict = the 5:15 rule (retired s173a-e / s186a / s186c only): blocked."""
import collections, json, os, random, sys

import numpy as np

L_MULT = 2.0                                        # the Latin part doubles (RULES-cooldown 1' / 1'')
C_CAP = 4096                                        # draw171's cut for calib documents
CUT = 2048                                          # pickx's cap for the heavy categories
HEAVY = {"agentic", "tools", "long", "reasoning"}
PAIR = [("s190c", "s186a"), ("s190d", "s186b")]    # (new part, matching s186 part): search, then confirm
RETIRED = ["s173a", "s173b", "s173c", "s173d", "s173e", "s186a", "s186c"]    # the retired slices of the 5:15 rule (x side, strict N)
RETIRED_C_N = RETIRED + ["s186b", "s186d", "s186e", "s186f"]                  # calib N sources of the 5:40 amendment
AMEND_OK = {"s186b", "s186d", "s186e", "s186f"}     # registry slices that count as retired under the 5:40 amendment
N_GEN_C = 6388                                      # mean N gen of s190a / s190b's c parts (6,488 / 6,287)
N_GEN_X = 25120                                     # mean N gen of s190a / s190b's x parts (25,526 / 24,714)
MIN_SCRIPTS = 4

CAL = "/data/q27/w164/strata/calib-kld-strata.json"
POOLS = "/data/q27/w164/strata/pools-v3.json"
CALGEN = "/data/q27/challenge/golden/calib/gen.u8"
LED = "/data/q27/w176/in/ledger.jsonl"              # snapshot of eval/pools/ledger.jsonl (md5 1164c1d9, 11-15)
REG = "/data/q27/w176/in/registry-live-fa61256a.json"   # snapshot of the live registry (lead-srch/run/registry.json, md5 fa61256a, 11-15)
S = "/data/q27/w131/slices"; S186 = "/data/q27/w164/slices"; S190 = "/data/q27/w169/slices"
PLAN186 = "/data/q27/w164/strata/plan186.json"
XSUP = "/data/q27/w164/strata/xsupply186b.jsonl"
XB = {"xc8k-b1": "/data/q27/goldens/xcorp/xc8k-b1", "xc8k-b2-c0": "/data/q27/goldens/xcorp/xc8k-b2-c0",
      "xc8k-b2-c1": "/data/q27/goldens/xcorp/xc8k-b2-c1", "xc8k-b2-c2": "/data/q27/goldens/xcorp/xc8k-b2-c2"}
XGM = {"xc8k-b1": "/data/q27/w164/strata/gm-b1.u8", "xc8k-b2-c0": "/data/q27/w176/strata/gm-b2-c0.u8",
       "xc8k-b2-c1": "/data/q27/w176/strata/gm-b2-c1.u8", "xc8k-b2-c2": "/data/q27/w176/strata/gm-b2-c2.u8"}
FINAL_C = "/data/q27/final/golden-final750"; FINAL_X = "/data/q27/final/golden-final-x"


def docs(p):
    return json.load(open(f"{p}/kld.json"))["docs"]


def slice_dir(name):
    """the calib c golden directory of a slice name (s173a-e: w131; s186a-f: w164)"""
    return f"{S}/golden-{name}-c" if name.startswith("s173") else f"{S186}/golden-{name}-c"


def gold_sets():
    """the golden directories by role (x side and the Latin freshness; RULES-cooldown 1'' / 1''-amendment)"""
    ret_c = [f"{S}/golden-s173{k}-c" for k in "abcde"] + [f"{S186}/golden-s186{k}-c" for k in "ac"]
    ret_x = [f"{S}/golden-s173{k}-x" for k in "abcde"] + [f"{S186}/golden-s186{k}-x" for k in "ac"]
    # everything an earlier or non-retired slice holds (never reused, never fresh)
    nr_c = [f"{S}/golden-s171{k}-c" for k in "abcd"] + [f"{S}/golden-s173f-c"] + [f"{S186}/golden-s186{k}-c" for k in "bdef"] + \
           [f"{S}/golden-t183-c", f"{S186}/golden-t186n-c", f"{S190}/golden-s190a-c", f"{S190}/golden-s190b-c"]
    nr_x = [f"{S}/golden-s171{k}-x" for k in "abcd"] + [f"{S}/golden-s173f-x"] + [f"{S186}/golden-s186{k}-x" for k in "bdef"] + \
           [f"{S}/golden-t183-x", f"{S186}/golden-t186n-x", FINAL_X, f"{S190}/golden-s190a-x", f"{S190}/golden-s190b-x"]
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
        if REGJ["slices"][name]["status"] != "retired": raise SystemExit(f"{name} is not retired in the registry snapshot")
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
    """the calib N pool of the 5:40 amendment: (excluded documents, {orig: the retired-ok slices holding it}).
    Excluded: every document of a registry slice that is not retired and not in AMEND_OK, plus the goldens of s173f, s171a-d, t183,
    t186n, s190a / s190b (c) and golden-final750 (c1 / c3). Verify is outside the fit + search pool (checked by the caller)."""
    for name in AMEND_OK:
        if name not in REGJ["slices"]: raise SystemExit(f"{name} missing from the registry snapshot")
    ex = set()
    for name, sl in REGJ["slices"].items():
        if sl["status"] != "retired" and name not in AMEND_OK:
            ex |= {int(d["id"][2:]) for d in sl["docs"] if d["part"] == "c"}
    for g in [f"{S}/golden-t183-c", f"{S186}/golden-t186n-c", f"{S}/golden-s173f-c"] + [f"{S}/golden-s171{k}-c" for k in "abcd"] + \
             [f"{S190}/golden-s190a-c", f"{S190}/golden-s190b-c"]:
        ex |= {d["orig"] for d in docs(g) if d.get("orig") is not None}
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


def main(plan_path, seed, supply_only=False, n_policy="amend540"):
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
    elif n_policy == "amend540":
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
    taken = set()
    for k, (name, match) in enumerate(PAIR):
        target = L_MULT * r_c[match]["gen_L"]
        pick_l, g = [], 0
        for i in sorted(L_free, key=lambda i: (use[i], rnd[i])):
            if g >= target: break
            if i in taken: continue
            pick_l.append(i); g += gen_c[i]
        taken |= set(pick_l)
        pn = N_parts_c[k]
        ch = sorted(pn + pick_l)
        gl = sum(gen_c[i] for i in pick_l); gn = sum(gen_c[i] for i in pn)
        sg = collections.Counter()
        for i in pn: sg[K[i]["script"]] += gen_c[i]
        plan["c"].append({"name": name, "match": match, "idx": ch, "N": sorted(pn), "gen_N": gn, "gen_L": gl,
                          "share_N": gn / max(gn + gl, 1), "L_target": target, "docs_N": len(pn), "docs_L": len(pick_l),
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
        target = L_MULT * r_x[match]["gen_L"]; nL0 = len(r_x[match]["idx"]) - r_x[match]["nN"]
        pool = [x for x in fresh_L if (x["batch"], x["idx"]) not in taken_x]
        for n in range(nL0, nL0 + 400):
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
                          "gen_N": gN, "gen_L": gl, "L_target": target, "share_N": gN / max(gN + gl, 1),
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
    pol = "amend540"
    if "--n-policy" in args:
        pol = args[args.index("--n-policy") + 1]
        args = [a for a in args if a not in ("--n-policy", pol)]
    args = [a for a in args if a != "--supply"]
    main(args[0], int(args[1]), supply_only=sup, n_policy=pol)
