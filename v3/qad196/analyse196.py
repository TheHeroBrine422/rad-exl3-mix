#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""analyse196.py SERVED_DIR OUTDIR [--reps 2000] -- (w182) #196's readout (RULES196): per arm the served mean KL over the gen
rows of both judge sets (x181g, fe38; pooled by rows, and each set alone), and the contrasts of interest, each with a 95 %
CI from a per-document bootstrap (documents resampled within each set, the same resample for every arm, seed 0):
  2.5 family: every arm vs A (the reference) and vs the untrained control (ctl);
  4.0 family: every arm vs A4 and vs ctl40.
Also the factor contrasts (R vs A, UA vs A, UR vs R, RN vs R, AN vs A, U vs ctl, ...). Reads SERVED_DIR/TAG/result.json (the
eval196 result: slice -> rows file and golden). Writes OUTDIR/analysis.json, OUTDIR/perdoc.json, OUTDIR/tables.md."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad"))
from healeval import doc_sums  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("served"); ap.add_argument("out")
ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ARMS = {"2.5": ["ctl", "A", "R", "U", "UR", "RN", "AN", "UA"], "4.0": ["ctl40", "A4", "R4", "U4", "UR4"]}
SETS = ["x181g", "fe38"]
PAIRS = {  # (x, y): contrast x - y. RULES196: every arm vs the untrained control and vs the family's A (A / A4); plus the factor contrasts
    "2.5": [("A", "ctl"), ("R", "ctl"), ("U", "ctl"), ("UR", "ctl"), ("RN", "ctl"), ("AN", "ctl"), ("UA", "ctl"),
            ("R", "A"), ("U", "A"), ("UR", "A"), ("RN", "A"), ("AN", "A"), ("UA", "A"),
            ("UA", "A"), ("UR", "R"), ("UR", "U"), ("RN", "R"), ("RN", "AN")],
    "4.0": [("A4", "ctl40"), ("R4", "ctl40"), ("U4", "ctl40"), ("UR4", "ctl40"),
            ("R4", "A4"), ("U4", "A4"), ("UR4", "A4"), ("UR4", "R4")],
}
S, N, got = {}, {}, {}
for fam, tags in ARMS.items():
    for t in tags:
        p = os.path.join(a.served, t, "result.json")
        if not os.path.exists(p): continue
        res = json.load(open(p)); got[t] = res
        for st in SETS:
            sl = res["slices"][st]
            s, n = doc_sums(sl["golden"], sl["rows"])
            S[(t, st)] = s; N[(t, st)] = n
arms = [t for fam in ARMS for t in ARMS[fam] if t in got]
rng = np.random.default_rng(a.seed)
ndocs = {st: len(N[(arms[0], st)]) for st in SETS if (arms[0], st) in N}
B = {st: rng.integers(0, ndocs[st], size=(a.reps, ndocs[st])) for st in SETS}


def mean(t, idx=None):
    num = den = 0.0
    for st in SETS:
        s, n = S[(t, st)], N[(t, st)]
        if idx is not None: s, n = s[idx[st]], n[idx[st]]
        num += s.sum(); den += n.sum()
    return num / den


def mean_set(t, st, idx=None):
    s, n = S[(t, st)], N[(t, st)]
    if idx is not None: s, n = s[idx[st]], n[idx[st]]
    return s.sum() / n.sum()


def contrast(x, y):
    """x - y in KL units and x / y - 1 in %: point estimate and the bootstrap 95 % CI"""
    d0 = mean(x) - mean(y); r0 = mean(x) / mean(y) - 1
    dr = np.empty(a.reps); rr = np.empty(a.reps)
    for r in range(a.reps):
        idx = {st: B[st][r] for st in SETS}
        mx, my = mean(x, idx), mean(y, idx)
        dr[r] = mx - my; rr[r] = mx / my - 1
    lo, hi = np.percentile(rr, [2.5, 97.5]); dlo, dhi = np.percentile(dr, [2.5, 97.5])
    return {"x": x, "y": y, "dKL": d0, "dKL_ci": [dlo, dhi], "pct": r0, "pct_ci": [lo, hi],
            "verdict": ("below 0" if hi < 0 else "above 0" if lo > 0 else "crosses 0")}


out = {"arms": {}, "contrasts": {}, "perdoc_sets": {st: ndocs[st] for st in SETS}}
for t in arms:
    row = {"kl_pooled": mean(t), "kl": {st: mean_set(t, st) for st in SETS}, "rows": {st: int(N[(t, st)].sum()) for st in SETS},
           "ckpt": got[t].get("ckpt"), "template": got[t].get("template")}
    out["arms"][t] = row
for fam, prs in PAIRS.items():
    for x, y in prs:
        if x in got and y in got:
            k = f"{x} vs {y}"
            if k not in out["contrasts"]: out["contrasts"][k] = contrast(x, y)
json.dump(out, open(os.path.join(a.out, "analysis.json"), "w"), indent=1)
json.dump({t: {st: {"sum": S[(t, st)].tolist(), "rows": N[(t, st)].tolist()} for st in SETS} for t in arms},
          open(os.path.join(a.out, "perdoc.json"), "w"))
L = ["| arm | KL x181g | KL fe38 | KL pooled | rows (x181g / fe38) |", "|---|---|---|---|---|"]
for t in arms:
    r = out["arms"][t]
    L.append(f"| {t} | {r['kl']['x181g']:.5f} | {r['kl']['fe38']:.5f} | {r['kl_pooled']:.5f} | {r['rows']['x181g']} / {r['rows']['fe38']} |")
L += ["", "| contrast | dKL (KL units) | dKL 95% CI | % vs y | % CI | verdict |", "|---|---|---|---|---|---|"]
for k, c in out["contrasts"].items():
    L.append(f"| {k} | {c['dKL']:+.5f} | [{c['dKL_ci'][0]:+.5f}, {c['dKL_ci'][1]:+.5f}] | {100 * c['pct']:+.2f} % | "
             f"[{100 * c['pct_ci'][0]:+.2f}, {100 * c['pct_ci'][1]:+.2f}] | {c['verdict']} |")
open(os.path.join(a.out, "tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
