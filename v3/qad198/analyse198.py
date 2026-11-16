#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""analyse198.py OUT --arm TAG=RESULT_JSON ... [--pair X-Y ...] [--best X:M1,M2,...] [--reps 2000] [--seed 0]  (w186, #198)

The #198 readout (RULES198). Per arm: served mean KL on the judge sets x181g and fe38, pooled by gen rows (and each set
alone), plus the sha256 of each served rows file. Contrasts, each x - y with a 95 % CI from a per-document bootstrap
(documents resampled within each set, the same resample for every arm, seed 0, 2,000 reps):
  --pair X:Y        X minus Y
  --best X:M1,M2    X minus its best member (the member with the lowest pooled KL on these rows). The choice is made
                    on the same rows as the test, so it favours the members: a soup that wins against its best member
                    is the conservative reading.
Verdict (KL units): CI wholly below 0 = X better ("soup better" when X is a soup); wholly above 0 = worse; else crosses 0.
Writes OUT/analysis.json, OUT/perdoc.json (per-document sums and gen rows) and OUT/tables.md.
"""
import argparse, hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad"))
from healeval import doc_sums  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--arm", action="append", required=True)
ap.add_argument("--pair", action="append", default=[]); ap.add_argument("--best", action="append", default=[])
ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SETS = ["x181g", "fe38"]

ARM = {}
for s in a.arm:
    t, p = s.split("=", 1)
    ARM[t] = {"path": p, **json.load(open(p))}
tags = list(ARM)
S, N, sha = {}, {}, {}
for t in tags:
    for st in SETS:
        sl = ARM[t]["slices"][st]
        s, n = doc_sums(sl["golden"], sl["rows"])
        S[(t, st)], N[(t, st)] = s, n
        sha[(t, st)] = hashlib.sha256(open(sl["rows"], "rb").read()).hexdigest()
ndocs = {st: len(N[(tags[0], st)]) for st in SETS}
for t in tags:
    for st in SETS:
        assert len(N[(t, st)]) == ndocs[st] and np.array_equal(N[(t, st)], N[(tags[0], st)]), f"{t} {st}: different documents or gen rows"

rng = np.random.default_rng(a.seed)
B = {st: rng.integers(0, ndocs[st], size=(a.reps, ndocs[st])) for st in SETS}
SS, NN = {}, {}
kl0, kl_set = {}, {}
for t in tags:
    SS[t] = sum(S[(t, st)][B[st]].sum(1) for st in SETS)
    NN[t] = sum(N[(t, st)][B[st]].sum(1).astype(np.float64) for st in SETS)
    kl0[t] = sum(S[(t, st)].sum() for st in SETS) / sum(N[(t, st)].sum() for st in SETS)
    kl_set[t] = {st: S[(t, st)].sum() / N[(t, st)].sum() for st in SETS}
KLB = {t: SS[t] / NN[t] for t in tags}


def contrast(x, y, name=None):
    d0 = kl0[x] - kl0[y]; r0 = kl0[x] / kl0[y] - 1
    dr = KLB[x] - KLB[y]; rr = KLB[x] / KLB[y] - 1
    dlo, dhi = np.percentile(dr, [2.5, 97.5]); lo, hi = np.percentile(rr, [2.5, 97.5])
    v = "below 0" if dhi < 0 else ("above 0" if dlo > 0 else "crosses 0")
    return {"name": name or f"{x} vs {y}", "x": x, "y": y, "dKL": float(d0), "dKL_ci": [float(dlo), float(dhi)],
            "pct": float(100 * r0), "pct_ci": [float(100 * lo), float(100 * hi)], "verdict": v}


contrasts = []
for p in a.pair:
    x, y = p.split(":", 1)
    contrasts.append(contrast(x, y))
for b in a.best:
    x, ms = b.split(":", 1); ms = ms.split(",")
    m = min(ms, key=lambda t: kl0[t])
    c = contrast(x, m, name=f"{x} vs best member {m}")
    c["members"] = {t: kl0[t] for t in ms}
    contrasts.append(c)

out = {"arms": {}, "contrasts": contrasts, "docs": ndocs, "reps": a.reps, "seed": a.seed}
for t in tags:
    out["arms"][t] = {"path": ARM[t]["path"], "ckpt": ARM[t].get("ckpt"), "template": ARM[t].get("template"),
                      "kl_pooled": float(kl0[t]), "kl_set": {st: float(kl_set[t][st]) for st in SETS},
                      "gen_rows": {st: int(N[(t, st)].sum()) for st in SETS},
                      "rows_sha256": {st: sha[(t, st)] for st in SETS}}
json.dump(out, open(os.path.join(a.out, "analysis.json"), "w"), indent=1)
json.dump({t: {st: {"sum": S[(t, st)].tolist(), "rows": N[(t, st)].tolist()} for st in SETS} for t in tags},
          open(os.path.join(a.out, "perdoc.json"), "w"))
L = ["| arm | KL x181g | KL fe38 | KL pooled | gen rows (x181g / fe38) | rows sha256 x181g (first 12) |", "|---|---|---|---|---|---|"]
for t in tags:
    r = out["arms"][t]
    L.append(f"| {t} | {r['kl_set']['x181g']:.5f} | {r['kl_set']['fe38']:.5f} | {r['kl_pooled']:.5f} | "
             f"{r['gen_rows']['x181g']} / {r['gen_rows']['fe38']} | {r['rows_sha256']['x181g'][:12]} |")
L += ["", "| contrast | dKL (KL units) | dKL 95 % CI | % vs y | % CI | verdict |", "|---|---|---|---|---|---|"]
for c in contrasts:
    L.append(f"| {c['name']} | {c['dKL']:+.5f} | [{c['dKL_ci'][0]:+.5f}, {c['dKL_ci'][1]:+.5f}] | {c['pct']:+.2f} % | "
             f"[{c['pct_ci'][0]:+.2f}, {c['pct_ci'][1]:+.2f}] | {c['verdict']} |")
open(os.path.join(a.out, "tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
