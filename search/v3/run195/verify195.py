# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""verify195.py PLAN.json WORK -- checks the built goldens in WORK against plan195.json (after build195.py; w179's verify190ef.py with the
paths of #195, the same checks, plus RULES-slices195 §2 measured on the built mixes: N share of gen in [0.09, 0.12], >= 5 scripts, no
script above 1/3 of the slice's N gen, Latin gen >= 1.9 x the matching s186 slice's). Counts and shares only
(no text); exit 1 on any mismatch.
  x (golden-s190{c,d}-x): the pieces are the plan's (batch, idx) pairs, each once; the file sizes match the rows; for every piece, against
      its xcorp batch (w164/xview): tokens, top-k bytes and rows.f32 identical over the piece's rows; the built gen.u8 sum equals the
      plan's gen (cut as the plan says); stratum and script equal the xsupply table's.
  c (golden-s190{c,d}-c): the documents are the plan's calib indices, each once; each document is the calib document's first
      min(tokens, 4096) tokens (score_from equal); tokens, top-k bytes, rows.f32 and gen.u8 equal the calib golden's rows (the same
      prefix); stratum and script equal the calib strata's; the gen sums equal the plan's gen_N / gen_L.
  mix (golden-s190{c,d}): each part's documents once (keys: c (part, orig), x (part, batch, orig)); the rows are the parts' rows; the gen
      sum is the parts' planned gen."""
import collections, json, sys

import numpy as np

PLAN, W = sys.argv[1], sys.argv[2]
plan = json.load(open(PLAN))
import os
WD = os.environ.get("W195", "/data/q27/w180")
XV = f"{WD}/xview"; XSUP = ["/data/q27/w164/strata/xsupply186b.jsonl", f"{WD}/strata/xsupply195.jsonl"]
PLAN186 = "/data/q27/w164/strata/plan186.json"
CAL = "/data/q27/challenge/golden/calib"; CALSTRATA = "/data/q27/w164/strata/calib-kld-strata.json"
C_CAP = 4096
XS = {(x["batch"], x["idx"]): x for f in XSUP for x in map(json.loads, open(f))}
K = json.load(open(CALSTRATA))["docs"]
bad = []; out = {}
BATCH = {}


def batch(b):
    if b not in BATCH:
        m = json.load(open(f"{XV}/{b}/kld.json"))
        off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])]).astype(np.int64)
        BATCH[b] = (m, off, np.memmap(f"{XV}/{b}/tokens.i32", np.int32, "r"), np.memmap(f"{XV}/{b}/top.bin", np.uint8, "r"),
                    np.memmap(f"{XV}/{b}/rows.f32", np.float32, "r"))
    return BATCH[b]


def check_x(part):
    name = part["name"]; g = f"{W}/golden-{name}-x"
    m = json.load(open(f"{g}/kld.json")); D = m["docs"]; R = m["rows"]; top = m["top"]
    gen = np.fromfile(f"{g}/gen.u8", np.uint8)
    toks = np.memmap(f"{g}/tokens.i32", np.int32, "r"); tb = np.memmap(f"{g}/top.bin", np.uint8, "r")
    rw = np.memmap(f"{g}/rows.f32", np.float32, "r")
    plan_pieces = {(p["batch"], p["idx"]): p for p in part["pieces"]}
    have = [(d["batch"], d["orig"]) for d in D]
    if sorted(have) != sorted(plan_pieces): bad.append(f"{name}-x: pieces differ from the plan")
    if len(set(have)) != len(have): bad.append(f"{name}-x: a piece twice")
    rows_sum = sum(d["tokens"] - 1 - d["score_from"] for d in D)
    sizes_ok = (rows_sum == R and len(gen) == R and rw.size == 3 * R and tb.size == R * top * 8
                and toks.size == sum(d["tokens"] for d in D))
    if not sizes_ok: bad.append(f"{name}-x: size mismatch (rows {R}, sum {rows_sum}, gen {len(gen)}, tokens {toks.size})")
    poff = np.concatenate([[0], np.cumsum([d["tokens"] for d in D])]).astype(np.int64)
    n_tok_bad = n_top_bad = n_rw_bad = n_gen_bad = n_lab_bad = 0
    gN = collections.Counter(); gL = 0; nN = 0; plan_gen_tot = 0; built_gen_tot = 0
    for j, d in enumerate(D):
        b, i = d["batch"], d["orig"]
        mb, boff, btok, btop, brw = batch(b)
        nt, sf = d["tokens"], d["score_from"]; k = nt - 1 - sf
        bd = mb["docs"][i]
        if nt > bd["tokens"] or sf != bd["score_from"]: n_tok_bad += 1; continue
        if not np.array_equal(toks[poff[j]:poff[j] + nt], btok[boff[i]:boff[i] + nt]): n_tok_bad += 1
        r0, rb0 = d["row0"], bd["row0"]
        if not np.array_equal(tb[r0 * top * 8:(r0 + k) * top * 8], btop[rb0 * top * 8:(rb0 + k) * top * 8]): n_top_bad += 1
        if not np.array_equal(rw[r0 * 3:(r0 + k) * 3], brw[rb0 * 3:(rb0 + k) * 3]): n_rw_bad += 1
        gb = int(gen[r0:r0 + k].sum()); built_gen_tot += gb
        pg = plan_pieces[(b, i)]["gen"]; plan_gen_tot += pg
        if gb != pg: n_gen_bad += 1
        s = XS[(b, i)]
        if d["stratum"] != s["stratum"] or d["script"] != s["script"]: n_lab_bad += 1
        if d["stratum"] == "N": gN[d["script"]] += gb; nN += 1
        else: gL += gb
    tN = sum(gN.values())
    counts = {"tokens": n_tok_bad, "top_bytes": n_top_bad, "rows_f32": n_rw_bad, "gen_vs_plan": n_gen_bad, "labels": n_lab_bad}
    if any(counts.values()): bad.append(f"{name}-x: {counts}")
    out[f"{name}-x"] = {"pieces": len(D), "rows": R, "N_pieces": nN, "gen_N": tN, "gen_L": gL,
                        "N_share": round(tN / max(tN + gL, 1), 4), "N_scripts": len(gN),
                        "max_script_share_of_N": round(max(gN.values()) / tN, 4) if tN else 0.0,
                        "N_gen_by_script": dict(gN.most_common()), "plan_gen_total": plan_gen_tot, "built_gen_total": built_gen_tot,
                        "header": {k: m[k] for k in ("format", "model", "top", "n_vocab") if k in m}, "mismatch_counts": counts}


def check_c(part):
    name = part["name"]; g = f"{W}/golden-{name}-c"
    m = json.load(open(f"{g}/kld.json")); D = m["docs"]; R = m["rows"]; top = m["top"]
    CD = json.load(open(f"{CAL}/kld.json"))
    if m["top"] != CD["top"] or m["n_vocab"] != CD["n_vocab"]: bad.append(f"{name}-c: header differs from the calib golden")
    gen = np.fromfile(f"{g}/gen.u8", np.uint8)
    toks = np.memmap(f"{g}/tokens.i32", np.int32, "r"); tb = np.memmap(f"{g}/top.bin", np.uint8, "r")
    rw = np.memmap(f"{g}/rows.f32", np.float32, "r")
    CDD = CD["docs"]
    coff = np.concatenate([[0], np.cumsum([d["tokens"] for d in CDD])]).astype(np.int64)
    ctok = np.memmap(f"{CAL}/tokens.i32", np.int32, "r"); ctop = np.memmap(f"{CAL}/top.bin", np.uint8, "r")
    crw = np.memmap(f"{CAL}/rows.f32", np.float32, "r"); cgen = np.fromfile(f"{CAL}/gen.u8", np.uint8)
    ids = [d["orig"] for d in D]
    if sorted(ids) != sorted(part["idx"]): bad.append(f"{name}-c: documents differ from the plan")
    if len(set(ids)) != len(ids): bad.append(f"{name}-c: a document twice")
    rows_sum = sum(d["tokens"] - 1 - d["score_from"] for d in D)
    sizes_ok = (rows_sum == R and len(gen) == R and rw.size == 3 * R and tb.size == R * top * 8
                and toks.size == sum(d["tokens"] for d in D))
    if not sizes_ok: bad.append(f"{name}-c: size mismatch (rows {R}, sum {rows_sum}, gen {len(gen)}, tokens {toks.size})")
    poff = np.concatenate([[0], np.cumsum([d["tokens"] for d in D])]).astype(np.int64)
    cnt = collections.Counter()
    gN = collections.Counter(); gL = 0; nN = 0
    for j, d in enumerate(D):
        i = d["orig"]; cd = CDD[i]; nt, sf = d["tokens"], d["score_from"]; k = nt - 1 - sf
        if nt != min(cd["tokens"], C_CAP) or sf != cd["score_from"]: cnt["cut"] += 1; continue
        if not np.array_equal(toks[poff[j]:poff[j] + nt], ctok[coff[i]:coff[i] + nt]): cnt["tokens"] += 1
        r0, c0 = d["row0"], cd["row0"]
        if not np.array_equal(tb[r0 * top * 8:(r0 + k) * top * 8], ctop[c0 * top * 8:(c0 + k) * top * 8]): cnt["top_bytes"] += 1
        if not np.array_equal(rw[r0 * 3:(r0 + k) * 3], crw[c0 * 3:(c0 + k) * 3]): cnt["rows_f32"] += 1
        if not np.array_equal(gen[r0:r0 + k], cgen[c0:c0 + k]): cnt["gen_vs_calib"] += 1
        s = K[i]
        if d["stratum"] != s["stratum"] or d["script"] != s["script"]: cnt["labels"] += 1
        gb = int(gen[r0:r0 + k].sum())
        if d["stratum"] == "N": gN[d["script"]] += gb; nN += 1
        else: gL += gb
    tN = sum(gN.values())
    if any(cnt.values()): bad.append(f"{name}-c: {dict(cnt)}")
    if tN != part["gen_N"] or gL != part["gen_L"]: bad.append(f"{name}-c: gen {tN}/{gL} vs plan {part['gen_N']}/{part['gen_L']}")
    out[f"{name}-c"] = {"docs": len(D), "rows": R, "N_docs": nN, "gen_N": tN, "gen_L": gL, "N_share": round(tN / max(tN + gL, 1), 4),
                        "N_scripts_with_gen": sum(1 for v in gN.values() if v > 0),
                        "max_script_share_of_N": round(max(gN.values()) / tN, 4) if tN else 0.0,
                        "N_gen_by_script": dict(gN.most_common()), "mismatch_counts": dict(cnt)}


def check_mix(c_part, x_part):
    name = c_part["name"]; g = f"{W}/golden-{name}"
    m = json.load(open(f"{g}/kld.json")); D = m["docs"]; gen = np.fromfile(f"{g}/gen.u8", np.uint8)
    keys = [("c", d["orig"]) if d["part"] == "c" else ("x", d["batch"], d["orig"]) for d in D]
    if len(set(keys)) != len(keys): bad.append(f"{name}: a document twice")
    want = {("c", i) for i in c_part["idx"]} | {("x", p["batch"], p["idx"]) for p in x_part["pieces"]}
    if set(keys) != want: bad.append(f"{name}: mix documents differ from the two parts")
    if len(gen) != m["rows"] or sum(d["tokens"] - 1 - d["score_from"] for d in D) != m["rows"]: bad.append(f"{name}: mix rows / gen.u8 size")
    planned = c_part["gen_N"] + c_part["gen_L"] + sum(p["gen"] for p in x_part["pieces"])
    if int(gen.sum()) != planned: bad.append(f"{name}: mix gen {int(gen.sum())} vs planned {planned}")
    out[name] = {"docs": len(D), "rows": m["rows"], "gen_rows": int(gen.sum()), "planned_gen": planned,
                 "parts": m.get("goldmix", {}).get("parts", {})}


for part in plan["x"]: check_x(part)
for part in plan["c"]: check_c(part)
for c_part, x_part in zip(plan["c"], plan["x"]): check_mix(c_part, x_part)
ref = json.load(open(PLAN186)); rL = {c["name"]: c["gen_L"] for c in ref["c"]}
for x in ref["x"]: rL[x["name"]] += x["gen_L"]
for c_part in plan["c"]:                            # RULES-slices195 §2 on the built mix (gen.u8 and the docs' labels)
    name = c_part["name"]; g = f"{W}/golden-{name}"
    m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8)
    sg = collections.Counter(); gL = 0
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; v = int(gen[d["row0"]:d["row0"] + k].sum())
        if d["stratum"] == "N": sg[d["script"]] += v
        else: gL += v
    tN = sum(sg.values()); share = tN / (tN + gL); ratio = gL / rL[c_part["match"]]
    ok = 0.09 <= share <= 0.12 and len([s for s in sg.values() if s > 0]) >= 5 and max(sg.values()) <= tN / 3 and ratio >= 1.9
    if not ok: bad.append(f"{name}: rule (share {share:.4f}, scripts {len(sg)}, max {max(sg.values()) / tN:.3f}, L ratio {ratio:.3f})")
    out[name].update({"gen_L": gL, "gen_N": tN, "N_share": round(share, 4), "N_scripts": len(sg), "max_script_share_of_N": round(max(sg.values()) / tN, 4),
                      "L_ratio_to_match": round(ratio, 4), "match": c_part["match"], "role": c_part["role"]})
res = {"parts": out, "bad": bad}
print(json.dumps(res, indent=1))
sys.exit(1 if bad else 0)
