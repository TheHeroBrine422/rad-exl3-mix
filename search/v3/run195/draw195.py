# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""draw195.py PLAN.json SEED [--supply] -- the draw of RULES-slices195 §2 (#195, w180): 15 slices, 5 search (s195s1-5) and 10 confirm
(s195c1-10), each 2 x the matching s186 slice's Latin gen positions with ~10 % of its gen positions non-Latin, all N from the fresh
batch xn8k-b195. Pairing: search s195s((k-1) mod 5 + 1) with confirm s195ck; slice s195s<i> and its two confirm slices match
s186{a..e}[i]: the slice's Latin target is L_MULT x (that s186 slice's c + x Latin gen).

  NEVER (any slice): t183, t186n, golden-final750 / final-x / final-c3, verify and public (outside the fit + search pool), the live
      slices (s190e / s190f; s190d, still a confirm slice) and every registry slice that is not retired and not RETIRED_OK (s173f, staged).
  calib Latin: every other Latin document of the merged fit + search pool, cut at 4,096, ordered least used first: (V3 decision count =
      ledger rows of a v3* line + the registry's doc_decisions, every-line count, seeded tie). The pool cannot give fifteen disjoint
      2 x c parts (see the README), so, as the 4:25 PM RULES-cooldown amendment did for s190e/f, each c part takes an equal share and
      the shortfall is filled with xcorp Latin in the x part (same Latin total per slice). The least used CONFIRM_SHARE of the pool's gen
      goes to the confirm slices (the freshest documents), the rest to the search slices; inside a group each document goes to the slice
      with the least gen so far. No document is in two slices.
  xcorp Latin, confirm: fresh pieces only (in no golden, no used flag) of xc8k-b1, xc8k-b2-c0..c4, stratified by category.
  xcorp Latin, search: pieces of retired slices (s171a-d, s173a-e, s186a-f, s190a-c) and of the older search27 / s129 x parts (the
      xsupply used flag), least used first (the holders' decisions); the least used prefix that covers the need is stratified by
      category. No piece is in two slices.
  N: xn8k-b195 pieces only, each used once over the 15 slices. A slice's N target is (its Latin gen) / 9 (10 % of gen), split equally
      over the ten scripts of the batch (KATAKANA, 9 short pieces, is left out); whole pieces in a seeded order per script while they
      fit, then one piece cut (a prefix) so the script meets its quota; a script with few pieces (BENGALI 20, DEVANAGARI 25) gives
      each slice at most pieces_left // slices_left pieces, so it lasts for all 15 slices.
Runs on ds8 (CPU, nice). Reads goldens' metadata, gen masks, the registry / ledger snapshots; writes only the plan."""
import collections, json, os, random, sys

import numpy as np

L_MULT = 2.0
C_CAP = 4096
CUT = 2048
HEAVY = {"agentic", "tools", "long", "reasoning"}
N_SHARE = 0.10
CONFIRM_SHARE = 10 / 15
SEARCH = [f"s195s{i}" for i in range(1, 6)]; CONFIRM = [f"s195c{k}" for k in range(1, 11)]
MATCH = {**{f"s195s{i}": "s186" + "abcde"[i - 1] for i in range(1, 6)}, **{f"s195c{k}": "s186" + "abcde"[(k - 1) % 5] for k in range(1, 11)}}
PAIRS = [[f"s195s{(k - 1) % 5 + 1}", f"s195c{k}"] for k in range(1, 11)]
SCRIPTS = ["ARABIC", "BENGALI", "CJK", "CYRILLIC", "DEVANAGARI", "GREEK", "HANGUL", "HEBREW", "HIRAGANA", "THAI"]
MIN_FILL = 256

Q = "/data/q27"; W = os.environ.get("W195", f"{Q}/w180")
CAL = f"{Q}/w164/strata/calib-kld-strata.json"; POOLS = f"{Q}/w164/strata/pools-v3.json"; CALGEN = f"{Q}/challenge/golden/calib/gen.u8"
LED = f"{W}/in/ledger.jsonl"; REG = f"{W}/in/registry-live.json"
PLAN186 = f"{Q}/w164/strata/plan186.json"
XSUP = [f"{Q}/w164/strata/xsupply186b.jsonl", f"{W}/strata/xsupply195.jsonl"]
XV = f"{W}/xview"
BATCH_L = ["xc8k-b1", "xc8k-b2-c0", "xc8k-b2-c1", "xc8k-b2-c2", "xc8k-b2-c3", "xc8k-b2-c4"]; BATCH_N = "xn8k-b195"
S = f"{Q}/w131/slices"; S186 = f"{Q}/w164/slices"
ROOT = {**{f"s171{k}": S for k in "abcd"}, **{f"s173{k}": S for k in "abcdef"}, "t183": S, **{f"s186{k}": S186 for k in "abcdef"}, "t186n": S186,
        "s190a": f"{Q}/w169/slices", "s190b": f"{Q}/w169/slices", "s190c": f"{Q}/w176/slices", "s190d": f"{Q}/w176/slices",
        "s190e": f"{Q}/w179/slices", "s190f": f"{Q}/w179/slices"}
NEVER = ["t183", "t186n", "s173f", "s190d", "s190e", "s190f"]    # + final750 / final-x / final-c3 below; s190e / f live, s190d confirming
RETIRED = [f"s171{k}" for k in "abcd"] + [f"s173{k}" for k in "abcde"] + [f"s186{k}" for k in "abcdef"] + ["s190a", "s190b", "s190c"]
RETIRED_OK = {"s186b", "s186d", "s186e", "s186f", "s190b"}      # staged in the registry, finished (RULES-cooldown 5:40 AM / 4:25 PM)
FINAL = [f"{Q}/final/golden-final750", f"{Q}/final/golden-final-x", f"{Q}/final/golden-final-c3"]
PRE171_USE = 10                                    # decisions assumed for a piece held only by the search27 / s129 x parts (used flag)


def docs(p):
    return json.load(open(f"{p}/kld.json"))["docs"]


def held_sets():
    """({slice: calib origs}, {slice: xdocs}) of every earlier slice golden, plus the finals"""
    c, x = {}, {}
    for n, r in ROOT.items():
        c[n] = {d["orig"] for d in docs(f"{r}/golden-{n}-c") if d.get("orig") is not None}
        x[n] = {d["xdoc"] for d in docs(f"{r}/golden-{n}-x") if "xdoc" in d}
    c["final"], x["final"] = set(), set()
    for g in FINAL:
        for d in docs(g):
            if "xdoc" in d: x["final"].add(d["xdoc"])
            elif d.get("orig") is not None: c["final"].add(d["orig"])
    return c, x


def load_x():
    """the xsupply rows of every batch with the gen positions a slice would score (heavy categories cut at CUT) and the gen mask"""
    X = [json.loads(l) for f in XSUP for l in open(f)]
    XD = {b: docs(f"{XV}/{b}") for b in BATCH_L + [BATCH_N]}
    G = {b: np.fromfile(f"{XV}/{b}/gm.u8", np.uint8) for b in XD}
    out = []
    for x in X:
        if x["batch"] not in XD: continue
        d = XD[x["batch"]][x["idx"]]; n = d["tokens"] - 1 - d["score_from"]
        x["rows"] = n; x["row0"] = d["row0"]
        x["cap"] = CUT if x["category"] in HEAVY and n > CUT and x["batch"] != BATCH_N else 0
        x["gen"] = int(G[x["batch"]][d["row0"]:d["row0"] + (x["cap"] or n)].sum())
        out.append(x)
    return out, G


def pick_l(pool, n, rseed):
    """n pieces of pool, stratified by category (largest remainder of the pool's category shares); draw190's pick_l"""
    cnt = collections.Counter(x["category"] for x in pool); tot = sum(cnt.values())
    want = {c: n * v / tot for c, v in cnt.items()}
    take = {c: min(cnt[c], int(want[c])) for c in cnt}
    for c in sorted(cnt, key=lambda c: (-(want[c] - int(want[c])), c)):
        if sum(take.values()) >= n: break
        if take[c] < cnt[c]: take[c] += 1
    rl = random.Random(rseed); pl = []
    for c in sorted(take):
        pc = [x for x in pool if x["category"] == c]
        rl.shuffle(pc); pl += pc[:take[c]]
    return pl


def grow(pool, target, rseed):
    """the smallest stratified pick whose gen reaches target (None when the pool cannot)"""
    if sum(x["gen"] for x in pool) < target: return None
    mean = sum(x["gen"] for x in pool) / len(pool)
    n = max(1, int(0.7 * target / mean)); step = 8
    while True:                                     # coarse steps up, then one piece at a time down to the first n that reaches it
        pl = pick_l(pool, min(n, len(pool)), rseed)
        if sum(x["gen"] for x in pl) >= target or n >= len(pool): break
        n += step
    while n > 1:
        p2 = pick_l(pool, n - 1, rseed)
        if sum(x["gen"] for x in p2) < target: break
        pl, n = p2, n - 1
    return pl


def main(plan_path, seed, supply_only=False):
    K = json.load(open(CAL))["docs"]; P = json.load(open(POOLS))["pool"]; GEN = np.fromfile(CALGEN, np.uint8)
    sc = lambda i: min(K[i]["tokens"], C_CAP) - 1 - K[i]["score_from"]
    gen_c = {i: int(GEN[d["row0"]:d["row0"] + max(sc(i), 0)].sum()) for i, d in enumerate(K)}
    REGJ = json.load(open(REG)); hc, hx = held_sets()
    for n in RETIRED:
        if n in REGJ["slices"] and REGJ["slices"][n]["status"] != "retired" and n not in RETIRED_OK:
            raise SystemExit(f"{n} is {REGJ['slices'][n]['status']} in the registry snapshot, not retired")
    never_c = set(hc["final"]); never_x = set(hx["final"])
    for n in NEVER: never_c |= hc[n]; never_x |= hx[n]
    for n, sl in REGJ["slices"].items():            # any other non-retired registry slice (none today)
        if sl["status"] != "retired" and n not in RETIRED_OK:
            never_c |= {int(d["id"][2:]) for d in sl["docs"] if d["part"] == "c"}
            if n not in NEVER: raise SystemExit(f"registry slice {n} ({sl['status']}) is not in NEVER / RETIRED: add it before a draw")
    mine = {f"golden-{k}-c" for k in REGJ["slices"]}
    use, use3 = collections.Counter(), collections.Counter()
    for l in open(LED):
        r = json.loads(l)
        if r["doc"] == "all" or r["use"] in mine: continue
        if r["use"] in ("w94-headx-search", "w94-headx-qad", "w119-headx-search", "w119-headx-fit", "w119-headx-verify"): continue
        dd = r.get("decisions"); k = dd if isinstance(dd, (int, float)) and not isinstance(dd, bool) else 1
        use[int(r["doc"])] += k
        if str(r.get("line", "")).startswith("v3"): use3[int(r["doc"])] += k
    for k, v in REGJ["doc_decisions"].items():
        if k.startswith("c:"): use[int(k[2:])] += v; use3[int(k[2:])] += v
    rng = random.Random(seed); rnd = {i: rng.random() for i in range(len(P))}
    pool_L = [i for i, p in enumerate(P) if p in ("fit", "search") and K[i]["stratum"] == "L" and sc(i) >= 16 and gen_c[i] >= 16]
    avail = sorted([i for i in pool_L if i not in never_c], key=lambda i: (use3[i], use[i], rnd[i]))
    tot_c = sum(gen_c[i] for i in avail)
    ref = json.load(open(PLAN186)); r_c = {c["name"]: c for c in ref["c"]}; r_x = {x["name"]: x for x in ref["x"]}
    T = {n: L_MULT * (r_c[m]["gen_L"] + r_x[m]["gen_L"]) for n, m in MATCH.items()}
    # ---- xcorp supply
    X, G = load_x()
    held = collections.defaultdict(set)
    for n, s in hx.items():
        for d in s: held[d].add(n)
    dec = {n: sum(REGJ["slices"][n]["decisions"].values()) if n in REGJ["slices"] and REGJ["slices"][n].get("decisions") else 0 for n in RETIRED}
    d171 = collections.defaultdict(dict)            # s171a-d left the registry: their decisions from the ledger (golden-s171X-c*, each use once)
    for l in open(LED):
        r = json.loads(l)
        if r["use"].startswith("golden-s171") and r["use"][12:14] == "-c" and isinstance(r.get("decisions"), (int, float)):
            d171[r["use"][7:12]][r["use"]] = r["decisions"]
    for n, u in d171.items(): dec[n] = sum(u.values())
    rng_x = random.Random(seed + 2)
    for x in X: x["rnd"] = rng_x.random(); x["held"] = sorted(held.get(x["doc"], ()))
    XL = [x for x in X if x["batch"] in BATCH_L and x["stratum"] == "L" and x["gen"] >= 16]
    fresh_L = [x for x in XL if not x["used"] and not x["held"]]
    reuse_L = [x for x in XL if (x["used"] or x["held"]) and x["doc"] not in never_x and all(h in RETIRED for h in x["held"])]
    for x in reuse_L: x["use"] = sum(dec.get(h, 0) for h in x["held"]) if x["held"] else PRE171_USE
    reuse_L.sort(key=lambda x: (x["use"], x["rnd"]))
    XN = [x for x in X if x["batch"] == BATCH_N and x["stratum"] == "N" and not x["used"] and not x["held"] and x["doc"] not in never_x]
    dup = collections.Counter(x["doc"] for x in X); assert max(dup.values()) == 1, "an xcorp document is in two batches"
    tab = lambda R: {"pieces": len(R), "gen": sum(x["gen"] for x in R)}
    by = lambda R, k: {v: tab([x for x in R if x[k] == v]) for v in sorted({x[k] for x in R})}
    stats = {"calib": {"pool_L_docs": len(pool_L), "pool_L_gen": sum(gen_c[i] for i in pool_L), "never_docs_in_pool": sum(i in never_c for i in pool_L),
                       "avail_docs": len(avail), "avail_gen": tot_c, "avail_v3_decisions_0": sum(use3[i] == 0 for i in avail),
                       "need_gen_2x_15": sum(L_MULT * r_c[m]["gen_L"] for m in MATCH.values()),
                       "need_gen_2x_confirm_only": sum(L_MULT * r_c[MATCH[n]]["gen_L"] for n in CONFIRM)},
             "xcorp_L": {"fresh": tab(fresh_L), "fresh_by_batch": by(fresh_L, "batch"), "reuse": tab(reuse_L), "reuse_by_batch": by(reuse_L, "batch"),
                         "holder_decisions": dec, "pre171_use": PRE171_USE},
             "xcorp_N": {"batch": tab(XN), "by_script": by(XN, "script")},
             "targets": {"slice_L": T, "sum_L": sum(T.values()), "sum_L_confirm": sum(T[n] for n in CONFIRM), "sum_L_search": sum(T[n] for n in SEARCH),
                         "sum_N": sum(T.values()) * N_SHARE / (1 - N_SHARE)}}
    print(json.dumps(stats, indent=1), file=sys.stderr)
    if supply_only:
        json.dump(stats, open(plan_path, "w"), indent=1); return
    # ---- calib Latin: the least used CONFIRM_SHARE (by gen) to the confirm slices, the rest to the search slices
    c_docs = {n: [] for n in SEARCH + CONFIRM}; c_g = {n: 0 for n in c_docs}
    cum = 0
    for i in avail:
        grp = CONFIRM if cum < CONFIRM_SHARE * tot_c else SEARCH
        n = min(grp, key=lambda n: (c_g[n], n)); c_docs[n].append(i); c_g[n] += gen_c[i]; cum += gen_c[i]
    # ---- xcorp Latin: confirm from the fresh pieces, search from the least used retired pieces
    x_L = {}; taken = set()
    key = lambda x: (x["batch"], x["idx"])
    for k, n in enumerate(CONFIRM):
        pool = [x for x in fresh_L if key(x) not in taken]
        pl = grow(pool, T[n] - c_g[n], seed + 100 + k)
        if pl is None: raise SystemExit(f"{n}: the fresh xcorp Latin supply ({sum(x['gen'] for x in pool)} gen left) cannot give {T[n] - c_g[n]:.0f}")
        x_L[n] = pl; taken |= {key(x) for x in pl}
    need_s = sum(T[n] - c_g[n] for n in SEARCH)
    pre, g = [], 0
    for x in reuse_L:                               # the least used prefix with 15 % slack for the stratification
        if g >= 1.15 * need_s: break
        pre.append(x); g += x["gen"]
    if g < need_s: raise SystemExit(f"search: the retired xcorp Latin supply ({g} gen) cannot give {need_s:.0f}")
    for k, n in enumerate(SEARCH):
        pool = [x for x in pre if key(x) not in taken]
        pl = grow(pool, T[n] - c_g[n], seed + 200 + k)
        if pl is None: raise SystemExit(f"{n}: the retired xcorp Latin prefix cannot give {T[n] - c_g[n]:.0f}")
        x_L[n] = pl; taken |= {key(x) for x in pl}
    # ---- N: the fresh batch, equal quotas over SCRIPTS, each piece once
    GN = G[BATCH_N]
    byS = {s: sorted([x for x in XN if x["script"] == s], key=lambda x: x["rnd"]) for s in SCRIPTS}
    x_N = {}
    for n in SEARCH + CONFIRM:
        gl = c_g[n] + sum(x["gen"] for x in x_L[n]); q = gl * N_SHARE / (1 - N_SHARE) / len(SCRIPTS)
        pn = []

        def fill(s, quota, budget):
            """pieces of script s up to quota gen, at most budget pieces: whole pieces that fit, in the seeded order; the last
            allowed piece, or when none fits, is the smallest piece that covers the rest, cut to it (a prefix: the least waste),
            else the largest piece left. Returns the gen taken."""
            got = k = 0
            while byS[s] and quota - got >= MIN_FILL and k < budget:
                need = int(round(quota - got))
                fit = [j for j, x in enumerate(byS[s]) if x["gen"] <= need]
                big = [j for j, x in enumerate(byS[s]) if x["gen"] > need]
                if fit and (k < budget - 1 or not big): j = fit[0] if k < budget - 1 else max(fit, key=lambda j: (byS[s][j]["gen"], -byS[s][j]["rnd"]))
                else: j = min(big, key=lambda j: (byS[s][j]["gen"], byS[s][j]["rnd"]))
                x = dict(byS[s].pop(j))
                if x["gen"] > need:
                    cs = np.cumsum(GN[x["row0"]:x["row0"] + x["rows"]]); cap = int(np.searchsorted(cs, need) + 1)
                    x["cap"] = cap; x["gen"] = int(cs[cap - 1])
                pn.append(x); got += x["gen"]; k += 1
            return got

        left_slices = len(SEARCH + CONFIRM) - len(x_N)          # this slice included
        short = 0
        for s in SCRIPTS:                           # a scarce script keeps enough pieces for the slices still to come
            g = fill(s, q, max(1, len(byS[s]) // left_slices))
            if q - g >= MIN_FILL: short += q - g
        while short >= MIN_FILL:                    # a script fell short: the script with the most supply left takes the rest
            s = max(SCRIPTS, key=lambda s: (sum(x["gen"] for x in byS[s]), s))
            g = fill(s, min(short, q / 2), 10 ** 6)
            if g == 0: break
            short -= g
        x_N[n] = pn
    # ---- the plan
    plan = {"seed": seed, "rule": "RULES-slices195 §2", "pairs": PAIRS, "c": [], "x": [], "slices": [], "stats": stats}
    for n in SEARCH + CONFIRM:
        m = MATCH[n]; idx = sorted(c_docs[n]); pn, pl = x_N[n], x_L[n]
        gN = sum(x["gen"] for x in pn); gLx = sum(x["gen"] for x in pl); gLc = c_g[n]
        sg = collections.Counter()
        for x in pn: sg[x["script"]] += x["gen"]
        role = "confirm" if n in CONFIRM else "search"
        plan["c"].append({"name": n, "role": role, "match": m, "idx": idx, "N": [], "gen_N": 0, "gen_L": gLc, "docs_L": len(idx),
                          "L_2x_own": L_MULT * r_c[m]["gen_L"], "ratio_L_own": gLc / r_c[m]["gen_L"], "pos": sum(sc(i) for i in idx),
                          "v3_decisions": {"min": min(use3[i] for i in idx), "max": max(use3[i] for i in idx), "mean": sum(use3[i] for i in idx) / len(idx),
                                           "zero": sum(use3[i] == 0 for i in idx)}})
        pieces = [{"batch": x["batch"], "idx": x["idx"], "doc": x["doc"], "stratum": x["stratum"], "script": x["script"], "category": x["category"],
                   "cap": x["cap"], "gen": x["gen"], "reused_from": x["held"] if x["stratum"] == "L" and role == "search" else [],
                   "used_flag": bool(x["used"])} for x in pn + pl]
        pieces.sort(key=lambda p: (p["batch"], p["idx"]))
        plan["x"].append({"name": n, "role": role, "match": m, "pieces": pieces, "nN": len(pn), "nL": len(pl), "gen_N": gN, "gen_L": gLx,
                          "L_target": T[n] - gLc, "ratio_L_own": gLx / r_x[m]["gen_L"], "scripts": dict(sorted(sg.items())),
                          "cats_L": dict(collections.Counter(x["category"] for x in pl)), "cats_N": dict(collections.Counter(x["category"] for x in pn)),
                          "batches": dict(collections.Counter(x["batch"] for x in pn + pl))})
        plan["slices"].append({"name": n, "role": role, "match": m, "gen_L": gLc + gLx, "gen_N": gN, "gen": gLc + gLx + gN, "L_target": T[n],
                               "ratio_L": (gLc + gLx) / (r_c[m]["gen_L"] + r_x[m]["gen_L"]), "share_N": gN / (gLc + gLx + gN),
                               "n_scripts": len(sg), "max_script_share": max(sg.values()) / gN})
    left = {s: tab(v) for s, v in byS.items()}
    plan["left"] = {"N_batch_unused": {"pieces": sum(v["pieces"] for v in left.values()), "gen": sum(v["gen"] for v in left.values()), "by_script": left},
                    "fresh_L_unused": tab([x for x in fresh_L if key(x) not in taken]), "reuse_L_unused": tab([x for x in reuse_L if key(x) not in taken])}
    json.dump(plan, open(plan_path + ".part", "w"), indent=1); os.replace(plan_path + ".part", plan_path)
    for s in plan["slices"]:
        c = next(c for c in plan["c"] if c["name"] == s["name"]); x = next(x for x in plan["x"] if x["name"] == s["name"])
        print(f"{s['name']} ({s['role']}, {s['match']}): c {c['docs_L']} docs L gen {c['gen_L']} ({c['ratio_L_own']:.2f}x), x {x['nL']} L + {x['nN']} N pieces, "
              f"L gen {s['gen_L']} ({s['ratio_L']:.3f}x), N gen {s['gen_N']} = {s['share_N']:.4f}, scripts {s['n_scripts']}, max {s['max_script_share']:.3f}", file=sys.stderr)
    print(json.dumps(plan["left"]), file=sys.stderr)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--supply"]
    main(a[0], int(a[1]), supply_only="--supply" in sys.argv[1:])
