# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""supply190.py -- how many 2x slices (s190a-f) the calib c side and the xcorp x side can give (#190, RULES-cooldown §1).

Run on ds8 (CPU, nice; reads only): the calib labels and gen mask, the xcorp supply table (xsupply186b.jsonl), the live
registry snapshot, the ledger snapshot and the golden files that define the exclusion set. Writes counts and the
exclusion-set sizes; no text and no per-document lists leave the machine.
  python3 -I supply190.py OUT.json
Exclusion set (RULES-cooldown §1: "No document or piece of any earlier ready / active / confirm / reselect / tracking
slice (s17*, s186a-f, t183, t186n, final-x, golden-final750)", plus xsupply186's used flags and s186's built parts):
  c: calib docs of s171a-d, s173a-f, s186a-f, t183, t186n, golden-final750 (c1, c3), and every ready / active /
     staged c doc of the registry snapshot.
  x: pieces flagged used in xsupply186b (every earlier x part), s186a-f x and t186n x (built), golden-final750's x part,
     and the xcorp docs of the other x goldens (x181, x181g, hd122-x, h177-x) as a sensitivity count only.
"""
import collections, json, os, sys

import numpy as np

W = "/data/q27/w164/strata"; S = "/data/q27/w131/slices"; S186 = "/data/q27/w164/slices"; IN = "/data/q27/w169/in"
CAL = f"{W}/calib-kld-strata.json"; POOLS = f"{W}/pools-v3.json"; CALGEN = "/data/q27/challenge/golden/calib/gen.u8"
XSUP = f"{W}/xsupply186b.jsonl"; XGM = f"{W}/gm-b1.u8"; XB1 = "/data/q27/goldens/xcorp/xc8k-b1/kld.json"
REG = f"{IN}/registry-live.json"; LED = f"{IN}/ledger.jsonl"
C_CAP = 4096; HEAVY = {"agentic", "tools", "long", "reasoning"}; CUT = 2048
GOLD_C = [f"{S}/golden-s171{k}-c" for k in "abcd"] + [f"{S}/golden-s173{k}-c" for k in "abcdef"] + \
         [f"{S186}/golden-s186{k}-c" for k in "abcdef"] + [f"{S}/golden-t183-c", f"{S186}/golden-t186n-c"]
GOLD_X = [f"{S}/golden-s171{k}-x" for k in "abcd"] + [f"{S}/golden-s173{k}-x" for k in "abcdef"] + \
         [f"{S186}/golden-s186{k}-x" for k in "abcdef"] + [f"{S}/golden-t183-x", f"{S186}/golden-t186n-x",
         "/data/q27/final/golden-final-x", "/data/q27/final/golden-final750"]
SENS_X = ["/data/q27/w154/slices/golden-x181", "/data/q27/w154/slices/golden-x181g", "/data/q27/heads/slices/golden-hd122-x",
          "/data/q27/w139s3/slices/golden-h177-x"]


def docs(p):
    return json.load(open(f"{p}/kld.json" if os.path.isdir(p) else p))["docs"]


out = {}
# ------------------------------------------------------------------ calib c side
K = json.load(open(CAL))["docs"]; P = json.load(open(POOLS))["pool"]; GEN = np.fromfile(CALGEN, np.uint8)
REGJ = json.load(open(REG))
ex = set()
src_sets = {}
for g in GOLD_C:
    s = {d["orig"] for d in docs(g) if d.get("orig") is not None}
    src_sets[os.path.basename(g)] = len(s); ex |= s
f750 = [d for d in docs("/data/q27/final/golden-final750") if d.get("part") in ("c1", "c3")]
s750 = {d["orig"] for d in f750}; src_sets["golden-final750-c"] = len(s750); ex |= s750
reg_c = set()
for name, sl in REGJ["slices"].items():
    if sl["status"] in ("ready", "active", "staged", "confirm", "reselect", "tracking"):
        reg_c |= {int(d["id"][2:]) for d in sl["docs"] if d["part"] == "c"}
src_sets["registry-c (ready/active/staged)"] = len(reg_c); ex |= reg_c
out["c_excl_sources"] = src_sets; out["c_excl_total"] = len(ex)
free = [i for i, p in enumerate(P) if p in ("fit", "search") and i not in ex]
sc = lambda i: min(K[i]["tokens"], C_CAP) - 1 - K[i]["score_from"]
gen_c = {}
for i in free:
    n = sc(i); gen_c[i] = int(GEN[K[i]["row0"]:K[i]["row0"] + n].sum())
Nf = [i for i in free if K[i]["stratum"] == "N" and sc(i) >= 16]
Lf = [i for i in free if K[i]["stratum"] == "L" and sc(i) >= 16]
scr = collections.Counter(K[i]["script"] for i in Nf)
gN = {s: sum(gen_c[i] for i in Nf if K[i]["script"] == s) for s in scr}
out["c"] = {"merged_pool_docs": sum(1 for p in P if p in ("fit", "search")), "free_docs": len(free),
            "N_free_docs": len(Nf), "N_free_gen": sum(gen_c[i] for i in Nf), "N_scripts": dict(scr),
            "N_gen_by_script": gN, "L_free_docs": len(Lf), "L_free_gen": sum(gen_c[i] for i in Lf)}
# how many 2x c parts: each needs ~90k scored positions with N share in [15.5, 19] % of gen (reuse across parts allowed,
# so each part may draw the free N documents again): per part the N gen needed = share * part gen
PART_POS = 90000
L_pos_per_part = PART_POS * (1 - 0.17)
L_gen_per_pos = sum(gen_c[i] for i in Lf) / max(sum(sc(i) for i in Lf), 1)
N_gen_per_pos = sum(gen_c[i] for i in Nf) / max(sum(sc(i) for i in Nf), 1)
out["c"]["L_gen_per_scored_pos"] = round(L_gen_per_pos, 4); out["c"]["N_gen_per_scored_pos"] = round(N_gen_per_pos, 4)
out["c"]["N_pos_per_part_needed_at_17pct"] = round(0.17 / 0.83 * L_pos_per_part * L_gen_per_pos / max(N_gen_per_pos, 1e-9))
out["c"]["N_positions_available_if_no_reuse"] = sum(sc(i) for i in Nf)
out["c"]["N_docs_available_for_a_part_with_6_scripts"] = len(Nf) >= 9 and len(scr) >= 6
# ------------------------------------------------------------------ x side
X = [json.loads(l) for l in open(XSUP)]
used = {x["doc"] for x in X if x["used"]}
xb1 = {x["idx"]: x for x in X if x["batch"] == "xc8k-b1"}
added = collections.Counter()
for g in GOLD_X:
    try: D = docs(g)
    except FileNotFoundError: continue
    for d in D:
        if "xdoc" in d and d["xdoc"] not in used:
            used.add(d["xdoc"]); added[os.path.basename(g)] += 1
out["x_added_used_by_built_or_final_parts"] = dict(added)
sens = collections.Counter()
for g in SENS_X:
    for d in docs(g):
        if d.get("xdoc") is not None: sens[os.path.basename(g)] += 1
out["x_sensitivity_docs_in_other_xgoldens"] = dict(sens)
G = np.fromfile(XGM, np.uint8); XD = json.load(open(XB1))["docs"]
def xgen_b1(i, cap):
    n = XD[i]["tokens"] - 1 - XD[i]["score_from"]
    n = min(n, cap) if cap else n
    return int(G[XD[i]["row0"]:XD[i]["row0"] + n].sum())
def cap_of(cat, n): return CUT if cat in HEAVY and n > CUT else 0
rows = []
for x in X:
    if x["used"] or x["doc"] in used: continue
    if x["batch"] == "xc8k-b1":
        n = XD[x["idx"]]["tokens"] - 1 - XD[x["idx"]]["score_from"]; cap = cap_of(x["category"], n)
        g = xgen_b1(x["idx"], cap); gs = min(n, cap) if cap else n; exact = True
    else:                                   # b2 batches: no gen mask on ds8 yet; scored positions (upper bound)
        gs = x["tokens"] - 1 - x["score_from"]; cap = cap_of(x["category"], gs); gs = min(gs, cap) if cap else gs; g = gs; exact = False
    rows.append((x["batch"], x["stratum"], x["script"], g, gs, x["category"], exact))
Nx = [r for r in rows if r[1] == "N"]; Lx = [r for r in rows if r[1] == "L"]
out["x"] = {"unused_docs": len(rows), "unused_by_batch": dict(collections.Counter(r[0] for r in rows)),
            "N_docs": len(Nx), "N_by_batch": dict(collections.Counter(r[0] for r in Nx)),
            "N_gen": sum(r[3] for r in Nx), "N_scripts": dict(collections.Counter(r[2] for r in Nx)),
            "N_gen_exact_b1": sum(r[3] for r in Nx if r[6]), "L_docs": len(Lx), "L_gen": sum(r[3] for r in Lx),
            "L_gen_per_piece": round(sum(r[3] for r in Lx) / max(len(Lx), 1), 1),
            "N_gen_per_piece": round(sum(r[3] for r in Nx) / max(len(Nx), 1), 1)}
# an x part of 260 pieces at N share s needs gN = s/(1-s) * gL with gL = (260 - nN) * L gen per piece: the N gen of the whole
# pool bounds how many 2x x parts can be built (each part needs ~0.15-0.20 of its gen in N)
gL_piece = out["x"]["L_gen_per_piece"]; gN_total = out["x"]["N_gen"]
per_part_N = None
for share in (0.15, 0.175):
    # solve nN: nN*gN_pc / (nN*gN_pc + (260-nN)*gL_piece) = share, with gN_pc the mean N gen per piece
    gN_pc = out["x"]["N_gen_per_piece"]
    nN = share * 260 * gL_piece / (gN_pc * (1 - share) + share * gL_piece)
    per_part_N = (round(nN, 1), round(nN * gN_pc))
    out["x"][f"per_2x_part_N_pieces_and_gen_at_{share}"] = per_part_N
out["x"]["max_2x_parts_by_N_gen_at_15pct"] = int(gN_total // (per_part_N[1] or 1))
json.dump(out, open(sys.argv[1] + ".part", "w"), indent=1); os.replace(sys.argv[1] + ".part", sys.argv[1])
print(json.dumps(out, indent=1))
