#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""supp198.py ANALYSIS_DIR -- (w186, #198) supplementary contrasts from analysis/perdoc.json (per-document sums and gen rows
per arm and judge set) and analysis/analysis.json (reps, seed). Same resampling as analyse198.py (same generator, same
draw order), so the pre-registered contrasts reproduce exactly. Adds, for each soup, the contrast against the MEAN of its
members (the choice a team makes without knowing which seed is best; not a RULES198 contrast). Writes ANALYSIS_DIR/supp.md.
"""
import json, os, sys
import numpy as np

D = sys.argv[1]
P = json.load(open(os.path.join(D, "perdoc.json"))); AN = json.load(open(os.path.join(D, "analysis.json")))
SETS = ["x181g", "fe38"]; reps, seed = AN["reps"], AN["seed"]
rng = np.random.default_rng(seed)
nd = {st: len(P["ref"][st]["sum"]) for st in SETS}
B = {st: rng.integers(0, nd[st], size=(reps, nd[st])) for st in SETS}
S = {t: {st: np.array(P[t][st]["sum"], dtype=np.float64) for st in SETS} for t in P}
N = {t: {st: np.array(P[t][st]["rows"], dtype=np.float64) for st in SETS} for t in P}
KL0 = {t: sum(S[t][st].sum() for st in SETS) / sum(N[t][st].sum() for st in SETS) for t in P}
KLB = {t: sum(S[t][st][B[st]].sum(1) for st in SETS) / sum(N[t][st][B[st]].sum(1) for st in SETS) for t in P}


def row(name, x, mem):
    m0 = np.mean([KL0[t] for t in mem]); mb = np.mean([KLB[t] for t in mem], axis=0)
    d0 = KL0[x] - m0; dr = KLB[x] - mb
    lo, hi = np.percentile(dr, [2.5, 97.5])
    v = "below 0" if hi < 0 else ("above 0" if lo > 0 else "crosses 0")
    return f"| {name} | {100 * d0 / m0:+.2f} % | [{100 * lo / m0:+.2f}, {100 * hi / m0:+.2f}] | {v} |", (d0, lo, hi)


L = ["| contrast (supplementary: vs the mean of the members, not a RULES198 contrast) | dKL % vs mean | 95 % CI % | verdict |",
     "|---|---|---|---|"]
checks = []
for name, x, mem in [("S2 vs mean(ref, D1)", "S2", ["ref", "D1"]),
                     ("C2 vs mean(ref, D1)", "C2", ["ref", "D1"]),
                     ("S4 vs mean(ref, D1, Q1D0, T1)", "S4", ["ref", "D1", "Q1D0", "T1"]),
                     ("S2s vs mean(ref, As1)", "S2s", ["ref", "As1"]),
                     ("As1 vs ref (seed effect, same recipe)", "As1", ["ref"])]:
    r, _ = row(name, x, mem); L.append(r)
# consistency: the pre-registered contrast S2s vs ref, recomputed here, must equal analysis.json's
d0 = KL0["S2s"] - KL0["ref"]; dr = KLB["S2s"] - KLB["ref"]; lo, hi = np.percentile(dr, [2.5, 97.5])
ref_row = next(c for c in AN["contrasts"] if c["name"] == "S2s vs ref")
ok = abs(d0 - ref_row["dKL"]) < 1e-12 and abs(lo - ref_row["dKL_ci"][0]) < 1e-12 and abs(hi - ref_row["dKL_ci"][1]) < 1e-12
L += ["", f"consistency: S2s vs ref recomputed {d0:+.6f} [{lo:+.6f}, {hi:+.6f}] = analysis.json {ref_row['dKL']:+.6f} "
      f"{ref_row['dKL_ci']}: {'IDENTICAL' if ok else 'MISMATCH'}"]
assert ok, "supp198 does not reproduce analyse198"
open(os.path.join(D, "supp.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
