#!/usr/bin/env python3
"""report.py [--table gain-table.json] DIR... -- w43 results under the rules of README.md (written before data).
Reads every settle.py output (*.jsonl, recursively) under the DIRs; prints markdown for items 1 / 1b / 1c (gain),
2 (damping) and 3 (tile CD). With --table: writes the item-1 factor table and the per-ladder-K ourq --gain strings.
Incomplete cells (a family missing a tensor or a seed) are shown as such and never qualify."""
import argparse, collections, glob, json, math, os

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--table")
a = ap.parse_args()
FAM = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"]
SUF = {"ssm_qkv": ["in_proj_qkv"], "ssm_z": ["in_proj_z"], "ssm_out": ["linear_attn.out_proj"], "attn_qg": ["q_proj"],
       "attn_k": ["k_proj"], "attn_v": ["v_proj"], "attn_output": ["o_proj"], "ffn_gate_up": ["gate_proj", "up_proj"],
       "ffn_down": ["down_proj"]}
KMAIN = [1.5, 2, 2.5, 3, 4, 5, 6, 7]; KQ = [1.25, 1.75, 2.25, 2.75]
LADDER = [1 + 0.25 * i for i in range(17)] + [5.5, 6, 6.5, 7]
R = list({(r["exp"], r["layer"], r["w"], r["bits"], r["seed"], r.get("damp")): r for r in
          (json.loads(l) for d in a.dirs for f in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True))
           for l in open(f))}.values())                   # one line per job and damping (the same file pulled twice)
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
pct = lambda g: f"{100 * g:+.1f}%"
PAR = {}                                                   # tensor parameter counts (N x K) for weighting
for r in R: PAR[(r["layer"], r["w"])] = r["N"] * r["K"]

n = collections.Counter((r["exp"], "1b" if r["exp"] == "gain" and r["bits"] in (1.25, 1.75, 2.25, 2.75) else "",
                         r["seed"]) for r in R)
print("Jobs read (exp, part, seed: n): " + ", ".join(f"{e}{p} s{sd}: {c}" for (e, p, sd), c in sorted(n.items())) +
      " · full set: gain 216 a seed (seeds 0, 1, 2), 1b 108, damp 108, cd 135 a seed (seeds 0, 1); 1,350 in all\n")

# ---------------------------------------------------------------- item 1 / 1b / 1c: gain
G = {}                                                     # (w, layer, K, seed) -> {m: rel}, norefit, s_exl
for r in R:
    if r["exp"] == "gain": G[(r["w"], r["layer"], r["bits"], r["seed"])] = r
layers = collections.defaultdict(set)
for (w, l, K, s) in G: layers[w].add(l)
def ratios(w, K, s, m, key="rel"):
    out = []
    for l in sorted(layers[w]):
        r = G.get((w, l, K, s))
        if r is None: return None
        out.append(r[key][m] / r[key]["1.0"])
    return out if len(out) == 3 else None
cell = {}
for w in FAM:
    for K in KMAIN + KQ:
        r0 = [G.get((w, l, K, 0)) for l in sorted(layers[w])]
        if len(r0) != 3 or None in r0: continue
        ms = list(r0[0]["rel"])
        m = min(ms, key=lambda m: gm(ratios(w, K, 0, m)))
        s1 = ratios(w, K, 1, m)
        c = {"m": m, "g0": 1 - gm(ratios(w, K, 0, m)), "g1": None if s1 is None else 1 - gm(s1),
             "max": max(ratios(w, K, 0, m) + (s1 or [])), "edge": m in (ms[0], ms[-1]),
             "g1_norefit": None if ratios(w, K, 1, m, "rel_norefit") is None else 1 - gm(ratios(w, K, 1, m, "rel_norefit")),
             "s_exl": sorted(x["s_exl"] for x in r0)[1], "drift": r0[0]["drift"]}
        s2 = ratios(w, K, 2, m); c["max2"] = None if s2 is None else max(s2)
        c["q"] = s1 is not None and c["g1"] >= 0.01 and c["max"] <= 1.0
        c["q_final"] = c["q"] and (c["max2"] is None or c["max2"] <= 1.0)
        cell[(w, K)] = c
if cell:
    print("## Item 1 / 1b · gain on exllamav3's g_scale (pick m* on seed 0, gain judged on seed 1)\n")
    print("Cell: m* · held-out error change at m* (seed 1; negative = better) · max tensor ratio (seeds 0+1) · ✓ "
          "qualifies (rule 3); `e` = m* on a grid end; `x2` = fails on seed 2 (1c)\n")
    Ks = [K for K in sorted(KMAIN + KQ) if any((w, K) in cell for w in FAM)]
    print("| family | " + " | ".join(f"K{K:g}" for K in Ks) + " |"); print("|---" * (len(Ks) + 1) + "|")
    for w in FAM:
        row = []
        for K in Ks:
            c = cell.get((w, K))
            if c is None: row.append("—"); continue
            g1 = "n/a" if c["g1"] is None else pct(-c["g1"])
            row.append(f"{float(c['m']):g} · {g1} · {c['max']:.3f}{' ✓' if c['q'] else ''}{' e' if c['edge'] else ''}"
                       f"{' x2' if c['q'] and not c['q_final'] else ''}")
        print(f"| {w} | " + " | ".join(row) + " |")
    print("\nexllamav3's s_exl (median tensor) and drift per K, first family with data:")
    print("| family | " + " | ".join(f"K{K:g}" for K in Ks) + " |"); print("|---" * (len(Ks) + 1) + "|")
    for w in FAM:
        print(f"| {w} | " + " | ".join(f"{cell[(w, K)]['s_exl']:.3f}" if (w, K) in cell else "—" for K in Ks) + " |")
    print("\nHeld-out error change before the row refit (what the old RMS-factor proxy would have shown), same m*:")
    print("| family | " + " | ".join(f"K{K:g}" for K in Ks) + " |"); print("|---" * (len(Ks) + 1) + "|")
    for w in FAM:
        print(f"| {w} | " + " | ".join(pct(-cell[(w, K)]["g1_norefit"]) if (w, K) in cell and cell[(w, K)]["g1_norefit"]
                                        is not None else "—" for K in Ks) + " |")
    table = {}
    for w in FAM:
        qk = [K for K in KMAIN + KQ if (w, K) in cell and cell[(w, K)]["q_final"]]
        if len(qk) >= 2: table[w] = {f"{K:g}": float(cell[(w, K)]["m"]) for K in sorted(qk)}
    complete = all((w, K) in cell and cell[(w, K)]["g1"] is not None for w in FAM for K in KMAIN)
    print(f"\n**Verdict ({'all main cells complete' if complete else 'PROVISIONAL: main cells incomplete'}):** " +
          ("flat: keep 1.0 (no family qualifies at ≥ 2 K)" if not table else
           "factor table for " + ", ".join(f"{w} {table[w]}" for w in table)))
    edges = [f"{w} K{K:g}" for (w, K), c in cell.items() if c["edge"]]
    if edges: print(f"\nGrid-end picks (rule 7, extend the grid before use): {', '.join(edges)}")
    def m_at(w, K):
        pts = sorted({k for k in KMAIN + KQ if (w, k) in cell})
        if not pts or w not in table: return 1.0
        v = lambda k: table[w].get(f"{k:g}", 1.0)
        if K <= pts[0]: return v(pts[0])
        if K >= pts[-1]: return v(pts[-1])
        lo = max(k for k in pts if k <= K); hi = min(k for k in pts if k >= K)
        return v(lo) if lo == hi else v(lo) + (K - lo) / (hi - lo) * (v(hi) - v(lo))
    gains = {}
    for K in LADDER:
        parts = [f"{s}={m_at(w, K):.3f}" for w in FAM for s in SUF[w] if abs(m_at(w, K) - 1) > 1e-9]
        gains[f"{K:g}"] = ",".join(parts)
    if a.table:
        json.dump({"rule": "v3/e3/settle/README.md item 1", "complete": complete, "table": table,
                   "ourq_gain_by_K": gains, "cells": {f"{w} K{K:g}": c for (w, K), c in cell.items()}},
                  open(a.table, "w"), indent=1)
        print(f"\nwritten {a.table}")
    if table:
        print("\n`ourq.py --gain` per ladder K:\n")
        for K, s in gains.items(): print(f"- K{K}: `{s or '(none)'}`")
    print()

# ---------------------------------------------------------------- item 2: damping
D = collections.defaultdict(dict)                          # (w, layer, K, seed) -> damp -> line
for r in R:
    if r["exp"] == "damp": D[(r["w"], r["layer"], r["bits"], r["seed"])][r["damp"]] = r
if D:
    print("## Item 2 · damping at fractional K (ratio to the adopted value at m = 1; gmean over tensors × seeds)\n")
    grp = {"input": ["ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v"], "rest": ["ssm_out", "attn_output", "ffn_gate_up", "ffn_down"]}
    ADOPT = {"input": 0.005, "rest": 0.025}; ALT = {"input": [0.0025, 0.01], "rest": [0.0125, 0.05]}
    K2 = [1.5, 2.5, 3.5, 5.5]
    for g, ws in grp.items():
        print(f"**{g} group** (adopted {ADOPT[g]}): cell = error change at m = 1 (negative = better) · max tensor ratio · "
              f"n (tensor × seed) · ✓ passes at this K / error change at the best of m ∈ {{0.9, 1, 1.1}}\n")
        print("| d | " + " | ".join(f"K{K:g}" for K in K2) + " | verdict |"); print("|---" * 6 + "|")
        res = {}
        for d in ALT[g]:
            cells, npass, gains_ = [], 0, []
            for K in K2:
                rr, rb, n = [], [], 0
                for (w, l, KK, s), v in D.items():
                    if w in ws and KK == K and d in v and ADOPT[g] in v:
                        rr.append(v[d]["rel"]["1.0"] / v[ADOPT[g]]["rel"]["1.0"])
                        rb.append(min(v[d]["rel"].values()) / min(v[ADOPT[g]]["rel"].values()))
                full = len(rr) == 2 * 3 * len(ws)
                if not rr: cells.append("—"); continue
                gg = 1 - gm(rr); ok = full and gg >= 0.01 and max(rr) <= 1.0
                npass += ok; gains_.append(gg)
                cells.append(f"{pct(-gg)} · {max(rr):.3f} · {len(rr)}{'' if full else ' (incomplete)'}{' ✓' if ok else ''} / {pct(-(1 - gm(rb)))}")
            res[d] = (npass, sum(gains_) / max(len(gains_), 1))
            verdict = "passes (≥ 2 K)" if npass >= 2 else "keep adopted"
            if npass >= 2 and d < ADOPT[g]: verdict += " → less damping: held-out check first (rule)"
            print(f"| {d} | " + " | ".join(cells) + f" | {verdict} |")
        print()

# ---------------------------------------------------------------- item 3: tile CD
C = [r for r in R if r["exp"] == "cd"]
if C:
    print("## Item 3 · tile coordinate descent, ratio to LDLQ (after the row refit; before it in brackets)\n")
    slope = {}                                             # (w, K) -> d ln rel / dK at m = 1 from item 1 (gmean)
    for w in FAM:
        ks = sorted(K for K in KMAIN + KQ if all(G.get((w, l, K, 0)) for l in layers[w]) and layers[w])
        for i in range(len(ks) - 1):
            r1 = gm([G[(w, l, ks[i], 0)]["rel"]["1.0"] for l in layers[w]]); r2 = gm([G[(w, l, ks[i + 1], 0)]["rel"]["1.0"] for l in layers[w]])
            slope[(w, ks[i])] = math.log(r1 / r2) / (ks[i + 1] - ks[i])
    def sl(w, K):
        c = [k for (ww, k) in slope if ww == w]
        return slope[(w, min(c, key=lambda k: abs(k - K)))] if c else None
    print("| K | stage | mean | param-weighted | tensors ≤ 1 | ≈ bits saved (median) | accepted | CPU time ÷ LDLQ |")
    print("|---|---|---|---|---|---|---|---|")
    for K in sorted({r["bits"] for r in C}):
        rs = [r for r in C if r["bits"] == K]
        for st in ("cd1", "cd2", "cd3"):
            v = [(r["rel"][st] / r["rel"]["ldlq"], r["rel_norefit"][st] / r["rel_norefit"]["ldlq"], PAR[(r["layer"], r["w"])], r) for r in rs if st in r["rel"]]
            if not v: continue
            mean = sum(x[0] for x in v) / len(v); mean0 = sum(x[1] for x in v) / len(v)
            pw = sum(x[0] * x[2] for x in v) / sum(x[2] for x in v)
            bits = sorted(-math.log(x[0]) / sl(x[3]["w"], K) for x in v if sl(x[3]["w"], K))
            acc = sum(x[3]["accepted"][st] for x in v) / len(v)
            tr = sum((x[3]["sec_stage"]["ldlq"] + sum(x[3]["sec_stage"][f"cd{i}"] for i in range(1, int(st[2]) + 1))) /
                     x[3]["sec_stage"]["ldlq"] for x in v) / len(v)
            print(f"| {K:g} | {st} | {mean:.4f} ({mean0:.4f}) | {pw:.4f} | {sum(x[0] <= 1 for x in v)} / {len(v)} | "
                  f"{bits[len(bits) // 2]:.4f} | {acc:.2f} | {tr:.2f} |" if bits else
                  f"| {K:g} | {st} | {mean:.4f} ({mean0:.4f}) | {pw:.4f} | {sum(x[0] <= 1 for x in v)} / {len(v)} | — | {acc:.2f} | {tr:.2f} |")
    print("\ncd3 ÷ ldlq per family (mean over tensors × seeds):\n")
    Ks = sorted({r["bits"] for r in C})
    print("| family | " + " | ".join(f"K{K:g}" for K in Ks) + " |"); print("|---" * (len(Ks) + 1) + "|")
    for w in FAM:
        cells = []
        for K in Ks:
            v = [r["rel"]["cd3"] / r["rel"]["ldlq"] for r in C if r["w"] == w and r["bits"] == K and "cd3" in r["rel"]]
            cells.append(f"{sum(v) / len(v):.4f} (n {len(v)})" if v else "—")
        print(f"| {w} | " + " | ".join(cells) + " |")
