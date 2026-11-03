# d1103 · follow-ups to the night of 2026-11-02/03 (0.8B, 4070), #87-02 … #33-23

Driver `run.sh` (one GPU job at a time, idempotent, steps in this order; `STEPS="…"` picks a subset). Results:
`/data/yb/d1103/result.txt`. Ratios are arm / control KL on g08-search1 unless marked `.fit` (g08-fit1), paired z
from `pcmp.py`. Seeds = `--seed-offset` 0 / 1 / 2 (base quants of the same seed).

Pass rules (2026-11-03 ~8:45 AM CST, written before data):

| step | # | what | pass | fail |
|---|---|---|---|---|
| stack | 87-02 | #87's sibling asym (λ 1, ρ 0.5, 48 writers) on top of edy (Σdy² H_out, also on the 48) vs edy | 3-seed mean ratio ≤ 0.97 and every seed < 1, at K3 and K2 | mean ≥ 1.00 at K3 |
| rho | 87-03 | ρ ∈ {0, 0.25, 0.75} vs 0.5, plain base, seed 0, screen on fit | screen only: a ρ that beats 0.5 by ≥ 1% on fit at both K gets seeds 0–2 on search later | — |
| gmdy | 83-02 | gen-masked loss for the **Σdy²** diagonal (the 27B estimator) vs edy | K3 mean ≤ 0.985 and every seed < 1; K2 mean ≤ 1.00 | K3 mean ≥ 1.00 |
| scale | 85-02 | B1 full 3 seeds (w02 rule); C9 α\* on top of C10 swj (base, seeds 0 / 1); swj then α\* on edy (seeds 0 / 1) | combo: ≤ 0.99 vs the better single refit, both seeds; on edy: ≤ 0.97 vs edy, both seeds | combo ≥ the better single on both seeds |
| allasym | 88-02 | asym (λ 1, ρ 0.5) for **all 150** tensors, moments through the same seed's quantized model (one-shot GPTAQ-like) vs base | mean ≤ 0.97 and every seed < 1 at K3 and K2; also reported vs #87's writers-only | mean ≥ 1.00 |
| k45 | 87-04 | #87 asym at K4 / K5 (seeds 0 / 1) | both seeds < 1 with z ≤ −2 at each K | — (descriptive otherwise) |
| heal | 89-02 | frozen-code healing (`heal.py` defaults, as k*all) on the best stack per K (edy + asym, seed 0) | ≤ 0.90 vs the stack at K2, ≤ 0.95 at K3 | — |
| cub | 78-04 | i8c + edy vs edy, seeds 3–5 (with #78-03: 6 seeds) | i8c kept for H_out quants if the 6-seed mean log ratio ≤ −2 SE at K3 | 6-seed mean ≥ 1.00 at K3: i8c only where no H_out |
| dose | 33-23 | Σdy² at 512 vs 1,024 windows (c18 snapshots) | 512 is enough for the 27B if the 3-seed mean ratio ≤ 1.01 at K3 | — |

Data discipline: moments / collections on the qad pool (as #83 / #87); α\* and swj fits on the fit golden; judged on
search. Nothing here touches verify or public.
