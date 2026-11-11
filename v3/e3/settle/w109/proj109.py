# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""proj109.py DATA -- #144 step 3: swj card-hours of the ladder (36 levels <= K4.5) and total card-days (w105's method:
quant 49-53 h on card 0 at today's stamp + swj passes), from
  - the R9700's production passes of today's code (gpubox timing.jsonl, DATA/r9700-passes.jsonl): t(S) = B + S P fitted;
  - dad1's dev runs (DATA/runs.jsonl, 8 layers x 143,099 tokens): t(S) for today's code and the (b) numerics arm, whose
    B / P ratios to today's scale the R9700 model.
Full pass = 64 layers x 927,611 tokens (dev x 51.86)."""
import json, sys

import numpy as np

D = sys.argv[1]
LEVELS, Q = 36, (49.0, 53.0)
SCALE = 64 / 8 * 927611 / 143099


def fit(pts):
    S, t = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    P, B = np.polyfit(S, t, 1); return B, P


R = [json.loads(l) for l in open(f"{D}/r9700-passes.jsonl")]
BR, PR = fit([(r["sources"], r["wall_s"]) for r in R])
runs = {r["tag"]: r for r in (json.loads(l) for l in open(f"{D}/runs.jsonl"))}
Bd, Pd = fit([(runs[t]["S"], runs[t]["wall"]) for t in ("ref1", "ref2b", "ref4", "ref8", "ref12")])
Bn, Pn = fit([(runs[t]["S"], runs[t]["wall"]) for t in ("patbf", "num8g", "patbf12")])   # the patch's runs + the dev copy's S=8
print(f"R9700 today's code, full pass: B {BR / 60:.1f} min + S x {PR / 60:.1f} min ({', '.join(f'S{r['sources']} {r['wall_s'] / 60:.0f}' for r in R)} min measured)")
print(f"dad1 9070 XT dev (x{SCALE:.2f} = full): today B {Bd:.1f} s + S x {Pd:.1f} s (full {Bd * SCALE / 60:.1f} + S x {Pd * SCALE / 60:.1f} min); "
      f"(b) numerics B {Bn:.1f} + S x {Pn:.1f} (ratios {Bn / Bd:.3f} / {Pn / Pd:.3f})")
cases = []
for S in (2, 4, 8, 12):
    cases.append((f"card 0, today's code, S={S}", "card0", LEVELS * (BR / S + PR) / 3600))
for S in (8, 12):
    cases.append((f"card 0, (b) numerics, S={S}", "card0", LEVELS * (BR * Bn / Bd / S + PR * Pn / Pd) / 3600))
for S in (2, 12):
    cases.append((f"dad1 9070 XT, today's code, S={S}", "dad1", LEVELS * (Bd / S + Pd) * SCALE / 3600))
cases.append(("dad1 9070 XT, (b) numerics, S=12", "dad1", LEVELS * (Bn / 12 + Pn) * SCALE / 3600))   # patbf12: peak 11.3 GiB
out = []
print(f"\n{'case':38s} {'swj h':>6s} {'card-0 days':>12s} {'swj share':>10s}")
for name, where, h in cases:
    c0 = [(q + (h if where == "card0" else 0)) / 24 for q in Q]
    share = [h / (q + h) for q in Q] if where == "card0" else [0.0, 0.0]
    print(f"{name:38s} {h:6.1f} {c0[0]:5.2f}-{c0[1]:4.2f} {share[1]:5.0%}-{share[0]:3.0%}" + ("  (+ dad1 in parallel)" if where == "dad1" else ""))
    out.append({"case": name, "where": where, "swj_h": round(h, 2), "card0_days": [round(x, 3) for x in c0], "swj_share": [round(x, 3) for x in share]})
json.dump({"R9700": {"B_s": BR, "P_s": PR}, "dad1_today": {"B_s": Bd, "P_s": Pd}, "dad1_numerics": {"B_s": Bn, "P_s": Pn},
           "scale": SCALE, "levels": LEVELS, "quant_h": Q, "cases": out}, open(f"{D}/projection.json", "w"), indent=1)
