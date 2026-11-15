# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""draw190.py -- the draw of RULES-cooldown §1 as amended 11-14 11:40 PM CST (#190, w169): ONE 2x-Latin pair, s190a (search)
and s190b (confirm), each matched to s186a / s186b. The amendment replaces the six-pair sizes: the supply has 15 free
non-Latin calib documents and 41 unused non-Latin xc8k-b1 pieces (the 11 in the b2 batches have no gen mask on ds8 and stay out).

draw186.py's draw with the targets as parameters (L_MULT, C_CAP, CUT) and the pair's sharing rules:
  c: the free calib N documents (merged fit + search pool, outside exclusion(): RULES-cooldown §1) are split between the two parts
     (a script with two or more documents goes to both parts, the rest to the part with less N gen); then L documents, least used
     first (ledger decision counts + the registry's own decisions, ties random by SEED), each part until its L gen positions reach
     L_MULT x the matching s186 part's L gen. The two parts share no document.
  x: the unused non-Latin xc8k-b1 pieces (exclusion(): earlier x parts, s186, t186n, final-x, golden-final750) split the same way;
     L pieces stratified by category (largest remainder of the unused L pieces' category shares), grown until their gen positions
     reach L_MULT x the matching s186 part's L gen; heavy categories cut to 2,048 scored positions (as pickx). No shared piece.
  No monitor. The N share is not aimed at a band (the amendment: it falls to what the supply gives); check190.py reports it.
Runs on ds8 (CPU; reads the calib golden, the xcorp golden, the registry and ledger snapshots; writes only the plan).
python3 -I draw190.py PLAN.json SEED"""
import collections, json, os, random, sys

import numpy as np

L_MULT = 2.0                                        # the Latin part doubles (RULES-cooldown amendment 11-14 11:40 PM CST)
C_CAP = 4096                                        # draw171's cut for calib documents
CUT = 2048                                          # pickx's cap for the heavy categories
HEAVY = {"agentic", "tools", "long", "reasoning"}
PAIR = [("s190a", "s186a"), ("s190b", "s186b")]    # (new part, matching s186 part): the search slice, then its confirmation slice

CAL = "/data/q27/w164/strata/calib-kld-strata.json"
POOLS = "/data/q27/w164/strata/pools-v3.json"
CALGEN = "/data/q27/challenge/golden/calib/gen.u8"
LED = "/data/q27/w169/in/ledger.jsonl"              # snapshot of eval/pools/ledger.jsonl (w169, 11-15)
REG = "/data/q27/w169/in/registry-live.json"        # snapshot of the live registry (lead-srch/run/registry.json)
S = "/data/q27/w131/slices"; S186 = "/data/q27/w164/slices"
PLAN186 = "/data/q27/w164/strata/plan186.json"
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


def split_pair(items, gen_of, script_of):
    """the items of one stratum split into the two parts of the pair (each item in one part only): the scripts are taken
    in order of size; the first two items of a script go one to each part (so every script with >= 2 items is in both
    parts), the rest to the part with less gen so far (balanced by gen positions)"""
    by = collections.defaultdict(list)
    for i in items: by[script_of(i)].append(i)
    parts, gg = ([], []), [0, 0]
    for s in sorted(by, key=lambda s: (-len(by[s]), s)):
        first = None
        for k, i in enumerate(sorted(by[s], key=lambda i: (-gen_of(i), i))):
            p = 1 - first if k == 1 else gg.index(min(gg))      # the second document of a script goes to the other part
            if k == 0: first = p
            parts[p].append(i); gg[p] += gen_of(i)
    return parts


def main(plan_path, seed):
    # ---------------------------------------------------------------- calib
    K = json.load(open(CAL))["docs"]; P = json.load(open(POOLS))["pool"]; GEN = np.fromfile(CALGEN, np.uint8)
    sc = lambda i: min(K[i]["tokens"], C_CAP) - 1 - K[i]["score_from"]
    gen_c = {}
    for i, d in enumerate(K):                       # the gen positions a c part scores (documents cut at C_CAP)
        n = sc(i); gen_c[i] = int(GEN[d["row0"]:d["row0"] + n].sum())
    REGJ = json.load(open(REG)); ex_c, ex_x = exclusion(REGJ)
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
    free = [i for i, p in enumerate(P) if p in ("fit", "search") and i not in ex_c]
    N_free = [i for i in free if K[i]["stratum"] == "N" and sc(i) >= 16]
    L_free = [i for i in free if K[i]["stratum"] == "L" and sc(i) >= 16]
    rng2 = random.Random(seed + 1); rnd = {i: rng2.random() for i in free}
    N_parts = split_pair(N_free, lambda i: gen_c[i], lambda i: K[i]["script"])
    plan = {"seed": seed, "c": [], "x": [], "stats": {}}
    taken = set()
    for k, (name, match) in enumerate(PAIR):
        target = L_MULT * r_c[match]["gen_L"]
        pick_l, g = [], 0
        for i in sorted(L_free, key=lambda i: (use[i], rnd[i])):     # least used first; the other part's documents are out
            if g >= target: break
            if i in taken: continue
            pick_l.append(i); g += gen_c[i]
        taken |= set(pick_l)
        ch = sorted(N_parts[k] + pick_l)
        gl = sum(gen_c[i] for i in pick_l); gn = sum(gen_c[i] for i in N_parts[k])
        sg = collections.Counter()
        for i in N_parts[k]: sg[K[i]["script"]] += gen_c[i]
        plan["c"].append({"name": name, "match": match, "idx": ch, "N": sorted(N_parts[k]), "gen_N": gn, "gen_L": gl,
                          "share_N": gn / max(gn + gl, 1), "L_target": target, "docs_N": len(N_parts[k]), "docs_L": len(pick_l),
                          "scripts": dict(collections.Counter(K[i]["script"] for i in N_parts[k])),
                          "max_script_share": max(sg.values()) / gn if gn else 0.0, "pos": sum(sc(i) for i in ch)})
    plan["stats"]["calib"] = {"merged_free_docs": len(free), "excluded_c": len(ex_c), "N_free": len(N_free),
                              "N_free_gen": sum(gen_c[i] for i in N_free), "N_free_scripts": dict(collections.Counter(K[i]["script"] for i in N_free)),
                              "L_free": len(L_free), "L_free_gen": sum(gen_c[i] for i in L_free)}
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
    N_xparts = split_pair(N_x, lambda i: xgen(i, cutof(i)), lambda i: stratum[i]["script"])
    plan["stats"]["x"] = {"unused_b1": len(unused), "N_b1": len(N_x), "N_gen_b1": sum(xgen(i, cutof(i)) for i in N_x),
                          "N_scripts_b1": dict(collections.Counter(stratum[i]["script"] for i in N_x)),
                          "N_unused_b2_not_used": sum(1 for x in X if x["batch"] != "xc8k-b1" and not x["used"] and x["stratum"] == "N"),
                          "L_b1": len(L_x)}

    def pick_l(pool, n, seed_n):
        """n L pieces of pool, stratified by category (largest remainder of the pool's category shares)"""
        cnt = collections.Counter(stratum[i]["category"] for i in pool); tot = sum(cnt.values())
        want = {c: n * v / tot for c, v in cnt.items()}
        take = {c: min(cnt[c], int(want[c])) for c in cnt}
        for c in sorted(cnt, key=lambda c: (-(want[c] - int(want[c])), c)):
            if sum(take.values()) >= n: break
            if take[c] < cnt[c]: take[c] += 1
        rl = random.Random(seed + 100 + seed_n); pl = []
        for c in sorted(take):
            pool_c = [i for i in pool if stratum[i]["category"] == c]
            rl.shuffle(pool_c); pl += pool_c[:take[c]]
        return pl

    taken_x = set()
    for k, (name, match) in enumerate(PAIR):
        target = L_MULT * r_x[match]["gen_L"]; nL0 = len(r_x[match]["idx"]) - r_x[match]["nN"]
        pool = [i for i in L_x if i not in taken_x]
        for n in range(nL0, nL0 + 400):                 # grown one piece at a time until the L gen positions reach the target
            pl = pick_l(pool, n, k)
            gl = sum(xgen(i, cutof(i)) for i in pl)
            if gl >= target: break
        taken_x |= set(pl)
        pn = N_xparts[k]; gN = sum(xgen(i, cutof(i)) for i in pn)
        ids = sorted(pn + pl)
        caps = {str(i): cutof(i) for i in ids if cutof(i)}
        sg = collections.Counter()
        for i in pn: sg[stratum[i]["script"]] += xgen(i, cutof(i))
        plan["x"].append({"name": name, "match": match, "idx": ids, "caps": caps, "N": sorted(pn), "nN": len(pn), "nL": len(pl),
                          "gen_N": gN, "gen_L": gl, "L_target": target, "share_N": gN / max(gN + gl, 1),
                          "scripts": dict(collections.Counter(stratum[i]["script"] for i in pn)),
                          "max_script_share": max(sg.values()) / gN if gN else 0.0,
                          "cats_L": dict(collections.Counter(stratum[i]["category"] for i in pl))})
    json.dump(plan, open(plan_path + ".part", "w"), indent=1); os.replace(plan_path + ".part", plan_path)
    for c in plan["c"]:
        print(f"{c['name']}: c {len(c['idx'])} docs (N {c['docs_N']}, L {c['docs_L']}, scripts {len(c['scripts'])}), N gen {c['gen_N']} "
              f"of {c['gen_N'] + c['gen_L']} = {c['share_N']:.3f}, L gen {c['gen_L']} (target {c['L_target']:.0f}), pos {c['pos']}", file=sys.stderr)
    for x in plan["x"]:
        print(f"{x['name']}: x {len(x['idx'])} pieces (N {x['nN']}, L {x['nL']}, scripts {len(x['scripts'])}), N gen {x['gen_N']} "
              f"of {x['gen_N'] + x['gen_L']} = {x['share_N']:.3f}, L gen {x['gen_L']} (target {x['L_target']:.0f})", file=sys.stderr)
    print(json.dumps(plan["stats"]), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
