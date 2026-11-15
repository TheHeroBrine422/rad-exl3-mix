# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""draw190.py -- the draws of RULES-cooldown §1 (#190, w169): six 2x-size slices s190a-f (c and x parts; no monitor).
draw186.py with the targets as parameters (X_PIECES, C_POS), the exclusion set of RULES-cooldown §1 (exclusion()) and a
stop, not a crash, when the x supply runs out (x_short in the plan).

Runs on ds8 (CPU; reads the calib golden, the xcorp goldens, the registry and the ledger; writes only the plan).
  x parts: 260 pieces of xc8k-b1 that no earlier x part used (its meta from colo's copy, verified line by line
           against the golden: tokens and source), N pieces by round-robin over their scripts until their gen
           positions reach the share TARGET_X of the part's gen positions, the rest L pieces stratified by category
           (largest remainder over the unused L pieces; the heavy categories cut to 2,048 scored positions, as pickx).
  c parts: calib documents of the merged fit + search pool (least used first: ledger decision counts + the registry's own
           decisions, ties random by SEED), never a document of an earlier slice (exclusion()); N documents
           by round-robin over their scripts (capped at 4,096 tokens as draw171), then L documents to TARGET_C scored
           positions; the N share of the part's gen positions aimed at TARGET_C_SHARE.
Output: one JSON (plan) + a summary on stderr. Pieces and documents are indices; no text leaves the machine.
python3 -I draw190.py PLAN.json SEED"""
import collections, json, os, random, sys

import numpy as np

TARGET_X = 0.16          # aimed N share of an x part's gen positions (band 15-20 %)
TARGET_C_SHARE = 0.16    # the same for a c part
X_PIECES = 260           # 2x s186's 130 (RULES-cooldown §1)
C_POS = 90000            # 2x draw171's 45k (scored positions, cap 4,096)
C_CAP = 4096
RESERVE_MINOR = False    # w164's reserve of the scripts with < 20 unused N pieces (off for s190, see the x loop)
HEAVY = {"agentic", "tools", "long", "reasoning"}
CUT = 2048               # pickx's --cap-cat for the heavy categories

CAL = "/data/q27/w164/strata/calib-kld-strata.json"
POOLS = "/data/q27/w164/strata/pools-v3.json"
CALGEN = "/data/q27/challenge/golden/calib/gen.u8"
LED = "/data/q27/w169/in/ledger.jsonl"                  # snapshot of eval/pools/ledger.jsonl (w169, 11-15)
REG = "/data/q27/w169/in/registry-live.json"             # snapshot of the live registry (w169, 11-15)
S = "/data/q27/w131/slices"; S186 = "/data/q27/w164/slices"
XSUP = "/data/q27/w164/strata/xsupply186b.jsonl"
XGM = "/data/q27/w164/strata/gm-b1.u8"
XB1 = "/data/q27/goldens/xcorp/xc8k-b1/kld.json"


def docs(p):
    return json.load(open(f"{p}/kld.json"))["docs"]


def exclusion(REGJ):
    """RULES-cooldown §1: the calib documents and xcorp pieces of every earlier slice (s17*, s186a-f, t183, t186n, final-x,
    golden-final750; c1 / c3 of final750 are calib c parts), every ready / active / staged part of the registry, and xsupply186's
    used flags (every earlier x part). Returns (calib indices, xcorp doc ids)."""
    gold_c = [f"{S}/golden-s171{k}-c" for k in "abcd"] + [f"{S}/golden-s173{k}-c" for k in "abcdef"] + \
             [f"{S186}/golden-s186{k}-c" for k in "abcdef"] + [f"{S}/golden-t183-c", f"{S186}/golden-t186n-c"]
    gold_x = [f"{S}/golden-s171{k}-x" for k in "abcd"] + [f"{S}/golden-s173{k}-x" for k in "abcdef"] + \
             [f"{S186}/golden-s186{k}-x" for k in "abcdef"] + [f"{S}/golden-t183-x", f"{S186}/golden-t186n-x",
             "/data/q27/final/golden-final-x", "/data/q27/final/golden-final750"]
    ex_c = set()
    for g in gold_c:
        ex_c |= {d["orig"] for d in docs(g) if d.get("orig") is not None}
    ex_c |= {d["orig"] for d in docs("/data/q27/final/golden-final750") if d.get("part") in ("c1", "c3")}
    for sl in REGJ["slices"].values():
        if sl["status"] in ("ready", "active", "staged", "confirm", "reselect", "tracking"):
            ex_c |= {int(d["id"][2:]) for d in sl["docs"] if d["part"] == "c"}
    ex_x = {json.loads(l)["doc"] for l in open(XSUP) if json.loads(l)["used"]}
    for g in gold_x:
        if os.path.exists(f"{g}/kld.json"): ex_x |= {d["xdoc"] for d in docs(g) if "xdoc" in d}
    return ex_c, ex_x


def pick_balanced(pool, script_of, gen_of, n=None, gen=None, max_scripts=6, reserve=None):
    """N documents from `pool` (the caller's preference order: least used / shuffled first), balanced over the scripts by
    gen positions (RULES-strata §3: at most a third of a part's N positions from one script).
    Phase 1: one document from each of the `max_scripts` scripts with the most supply, so a part has >= 6 scripts.
    Phase 2 takes the next document from the script with the fewest gen positions picked so far, until `n` documents or
    `gen` gen positions are reached; a script keeps reserve[script] documents back (the parts still to draw take their turn)."""
    by = collections.defaultdict(list)
    for i in pool: by[script_of(i)].append(i)
    reserve = reserve or {}
    picks, got, gg, ng = [], collections.Counter(), collections.Counter(), 0
    def done(): return (n is not None and len(picks) >= n) or (gen is not None and ng >= gen)
    ph1 = [s for s in sorted(by, key=lambda s: (-len(by[s]), s)) if by[s]][:max_scripts]
    for s in ph1:
        i = by[s].pop(0); picks.append(i); got[s] += 1; gg[s] += gen_of(i); ng += gen_of(i)
    while not done():
        cand = [s for s in by if len(by[s]) > reserve.get(s, 0)]
        if not cand: break
        s = min(cand, key=lambda s: (gg[s], -len(by[s]), s))
        i = by[s].pop(0); picks.append(i); got[s] += 1; gg[s] += gen_of(i); ng += gen_of(i)
    return picks


def main(plan_path, seed):
    rng = random.Random(seed)
    # ---------------------------------------------------------------- calib
    K = json.load(open(CAL))["docs"]; P = json.load(open(POOLS))["pool"]; GEN = np.fromfile(CALGEN, np.uint8)
    gen_c = {}
    for i, d in enumerate(K):                       # the gen positions the c part scores (documents cut at C_CAP)
        n = min(d["tokens"], C_CAP) - 1 - d["score_from"]; gen_c[i] = int(GEN[d["row0"]:d["row0"] + n].sum())
    REGJ = json.load(open(REG))
    ex_c, ex_x = exclusion(REGJ)
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
    free = [i for i, p in enumerate(P) if p in ("fit", "search") and i not in ex_c]
    sc = lambda i: min(K[i]["tokens"], C_CAP) - 1 - K[i]["score_from"]
    # ---------------------------------------------------------------- c parts (merged pool)
    N_free = [i for i in free if K[i]["stratum"] == "N" and sc(i) >= 16]
    L_free = [i for i in free if K[i]["stratum"] == "L" and sc(i) >= 16]
    rng2 = random.Random(seed + 1); rnd = {i: rng2.random() for i in free}
    key_c = lambda i: (use[i], rnd[i])
    plan = {"seed": seed, "c": [], "x": [], "x_short": None, "stats": {}}
    # the six c parts draw least used first and count their own uses as they go (a document may recur in a later part once
    # the less used ones are spent: the N pool is ~58 documents and 6 parts need >= 6 scripts each; the recurrences are reported)
    c_used = set(); recur = collections.Counter()
    for slice_ in "abcdef":
        Lc = sorted(L_free, key=key_c)
        Nc = sorted(N_free, key=key_c)
        pick_l, lp = [], 0
        for i in Lc:                                    # L documents first: ~ (1 - share) of the part's positions
            if lp >= C_POS * (1 - TARGET_C_SHARE - 0.01): break
            pick_l.append(i); lp += sc(i)
        gl = sum(gen_c[i] for i in pick_l)
        # N gen needed for the share: gn / (gn + gl) = TARGET_C_SHARE
        pick_n = pick_balanced(Nc, lambda i: K[i]["script"], lambda i: gen_c[i], gen=TARGET_C_SHARE / (1 - TARGET_C_SHARE) * gl)
        Lrest = [i for i in Lc if i not in set(pick_l)]; Nrest = [i for i in Nc if i not in set(pick_n)]
        def stats():
            gn_ = sum(gen_c[i] for i in pick_n); gl_ = sum(gen_c[i] for i in pick_l); sg_ = collections.Counter()
            for i in pick_n: sg_[K[i]["script"]] += gen_c[i]
            return gn_, gl_, gn_ / max(gn_ + gl_, 1), (max(sg_.values()) / gn_ if gn_ else 1.0), sg_
        # repair (RULES-strata §3: >= 6 scripts, at most a third of the N gen positions from one script, N share in band):
        # a script too big gets an N document from another script; a share too high gets an L document, too low an N one
        for _ in range(400):
            gn_, gl_, share_, mx_, sg_ = stats()
            if len(sg_) < 6 or mx_ > 1 / 3 + 1e-9:
                top = max(sg_, key=lambda s_: sg_[s_]) if sg_ else None
                cand = [i for i in Nrest if (K[i]["script"] not in sg_) if len(sg_) < 6] or [i for i in Nrest if K[i]["script"] != top]
                if not cand: break
                pick_n.append(cand[0]); Nrest.remove(cand[0])
            elif share_ > 0.19 and Lrest:
                pick_l.append(Lrest.pop(0))
            elif share_ < 0.155 and Nrest:
                pick_n.append(Nrest.pop(0))
            elif share_ < 0.155 and pick_l:              # N pool exhausted (w169): trim L docs back into the band
                pick_l.pop()
            else:
                break
        gl = sum(gen_c[i] for i in pick_l); gn = sum(gen_c[i] for i in pick_n)
        ch = sorted(pick_n + pick_l)
        for i in ch:
            if use[i] > 0 and i in c_used: recur[i] += 1
            c_used.add(i); use[i] += 1
        scripts = collections.Counter(K[i]["script"] for i in pick_n)
        sg = collections.Counter()
        for i in pick_n: sg[K[i]["script"]] += gen_c[i]
        left = collections.Counter(K[i]["script"] for i in Nrest)
        plan["c"].append({"name": f"s190{slice_}", "idx": ch, "N": sorted(pick_n), "gen_N": gn, "gen_L": gl, "share_N": gn / max(gn + gl, 1),
                          "N_scripts_left_after": dict(left),
                          "scripts": dict(scripts), "max_script_share": max(sg.values()) / max(gn, 1) if gn else 0,
                          "pos": sum(sc(i) for i in ch), "docs_N": len(pick_n)})
    plan["stats"]["calib"] = {"merged_free_docs": len(free), "N_free": len(N_free), "N_free_gen": sum(gen_c[i] for i in N_free),
                              "c_recurrences": {str(k): v for k, v in recur.items()}, "c_distinct_docs": len(c_used),
                              "excluded_c": len(ex_c)}
    # ---------------------------------------------------------------- x parts: xc8k-b1 unused pieces
    X = [json.loads(l) for l in open(XSUP)]
    G = np.fromfile(XGM, np.uint8); XD = json.load(open(XB1))["docs"]
    r0 = [d["row0"] for d in XD]; nrow = [d["tokens"] - 1 - d["score_from"] for d in XD]
    b1 = [x for x in X if x["batch"] == "xc8k-b1"]
    unused = [x["idx"] for x in b1 if x["doc"] not in ex_x]
    stratum = {x["idx"]: x for x in b1}
    def xgen(i, cut):
        n = min(nrow[i], cut) if cut else nrow[i]
        return int(G[r0[i]:r0[i] + n].sum())
    def cutof(i):
        return CUT if stratum[i]["category"] in HEAVY and nrow[i] > CUT else 0
    N_x = [i for i in unused if stratum[i]["stratum"] == "N"]
    L_x = [i for i in unused if stratum[i]["stratum"] == "L"]
    # w164 kept the scripts with < 20 unused pieces (THAI, HANGUL, GREEK, HEBREW, DEVANAGARI) for one piece per part, for the
    # monitor. s190 has no monitor and only ~41 N pieces left: the reserve would block every part (every script is < 20 now), so
    # it is off (RESERVE_MINOR = False); the >= 6 scripts rule is checked per part by check190.py
    cnt0 = collections.Counter(stratum[i]["script"] for i in N_x); minor = {s_ for s_, v in cnt0.items() if v < 20}
    reserve = {s_: 10 ** 9 for s_ in minor} if RESERVE_MINOR else {}
    plan["stats"]["x"] = {"unused_b1": len(unused), "N": len(N_x), "N_gen": sum(xgen(i, cutof(i)) for i in N_x),
                          "scripts_N": dict(cnt0), "minor_scripts": sorted(minor)}
    for k, slice_ in enumerate("abcdef"):
        rr = random.Random(seed + 10 + k)
        Nxr = rr.sample(N_x, len(N_x))
        best = None
        for nN in range(30, 75):                       # the N piece count that puts the N share nearest TARGET_X
            pn = pick_balanced(Nxr, lambda i: stratum[i]["script"], lambda i: xgen(i, cutof(i)), n=nN, max_scripts=8, reserve=reserve)
            if len(pn) < nN: break
            spn = set(pn); gN = sum(xgen(i, cutof(i)) for i in pn)
            # L pieces: 130 - nN, category-stratified: largest remainder of the unused L pieces' category shares
            nL = X_PIECES - nN
            cnt = collections.Counter(stratum[i]["category"] for i in L_x if i not in spn)
            tot = sum(cnt.values()); want = {c: nL * v / tot for c, v in cnt.items()}
            take = {c: min(cnt[c], int(want[c])) for c in cnt}
            for c in sorted(cnt, key=lambda c: (-(want[c] - int(want[c])), c)):
                if sum(take.values()) >= nL: break
                if take[c] < cnt[c]: take[c] += 1
            rl = random.Random(seed + 100 + nN)
            pl = []
            for c in sorted(take):
                pool_c = [i for i in L_x if stratum[i]["category"] == c and i not in spn]
                rl.shuffle(pool_c); pl += pool_c[:take[c]]
            gL = sum(xgen(i, cutof(i)) for i in pl)
            share = gN / max(gN + gL, 1)
            cand = (abs(share - TARGET_X), nN, share, pn, pl, gN, gL)
            if best is None or cand[0] < best[0]: best = cand
        # feasible only if the N pieces fill the part (nN of them, >= 6 scripts) and its N share is in band; else the supply
        # is short: stop here, the parts built so far stay in the plan (never shrink a part to fit)
        if best is None or len(best[3]) < best[1] or not (0.15 <= best[2] <= 0.20) or \
                len({stratum[i]["script"] for i in best[3]}) < 6:
            plan["x_short"] = {"from_part": f"s190{slice_}", "built": len(plan["x"]), "N_left": len(N_x),
                               "N_gen_left": sum(xgen(i, cutof(i)) for i in N_x),
                               "best_share": None if best is None else round(best[2], 4),
                               "best_N_pieces": None if best is None else best[1]}
            break
        _, nN, share, pn, pl, gN, gL = best
        ids = sorted(pn + pl)
        caps = {str(i): cutof(i) for i in ids if cutof(i)}
        N_x = [i for i in N_x if i not in set(pn)]; L_x = [i for i in L_x if i not in set(pl)]
        sc_x = collections.Counter(stratum[i]["script"] for i in pn)
        sg = collections.Counter()
        for i in pn: sg[stratum[i]["script"]] += xgen(i, cutof(i))
        plan["x"].append({"name": f"s190{slice_}", "idx": ids, "caps": caps, "N": sorted(pn), "nN": nN, "gen_N": gN, "gen_L": gL,
                          "share_N": share, "scripts": dict(sc_x), "max_script_share": max(sg.values()) / max(gN, 1),
                          "cats_L": dict(collections.Counter(stratum[i]["category"] for i in pl))})
    json.dump(plan, open(plan_path + ".part", "w"), indent=1); os.replace(plan_path + ".part", plan_path)
    for c in plan["c"]:
        print(f"{c['name']}: c {len(c['idx'])} docs (N {c['docs_N']}, scripts {len(c['scripts'])}, max script {c['max_script_share']:.2f}), N gen {c['gen_N']} of {c['gen_N'] + c['gen_L']} = {c['share_N']:.3f}, pos {c['pos']}", file=sys.stderr)
    for x in plan["x"]:
        print(f"{x['name']}: x {len(x['idx'])} pieces (N {x['nN']}, scripts {len(x['scripts'])}, max script {x['max_script_share']:.2f}), N gen {x['gen_N']} of {x['gen_N'] + x['gen_L']} = {x['share_N']:.3f}", file=sys.stderr)
    print(f"x_short: {plan['x_short']}", file=sys.stderr)
    print(json.dumps(plan["stats"]), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
