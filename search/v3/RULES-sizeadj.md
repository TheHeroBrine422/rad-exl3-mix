# #167 rule: size-adjusted vs raw-KLD accept on the CPU stand-in (w127)

Written 2026-11-12 7:30 PM CST, **before any A/B run** (committed alone; the smoke runs used seeds 98 / 99 only, which
are not A/B seeds, to check that the harness runs).

## Question
The V3 search driver accepts a candidate against its size's incumbent. Raw rule: pooled log(KLD_c / KLD_i) < 0 and on
every part. Size-adjusted rule (TODO 11-13): the same on log KLD + b·bpw, i.e. log r + b·(bpw_c − bpw_i) < 0 (also for
the hosts' early stop), b = the current fit's local slope −d log KLD / d bpw at the size (the fitted model's knapsack
optimum at size ∓ 0.1 bpw), updated each refit. Does the size-adjusted rule converge at least as well?

## Setup (fixed now)
- `search/v3/standin.py --real --layers 64` (realistic bytes: the real ladder's K grid K1–K7.5, 0.125 steps to K5, the real
  menu's per-tensor overhead, the 27B's parameter count, the base container's own head / embedding (donor-free mixes,
  so the driver calibrates its byte offset), every container w126's measured 8,255,936 B larger than its menu bytes;
  per-move byte deltas are the menu's), sizes 1.5 / 2.5 / 4.0 / 6.0, size_tol 0.02, 800 evals per run, `--par 2`,
  rotate after 25 decisions, 14 slices, boundary B p2 w5 (the pre-registered pick), every other CONF value as w126's
  stand-in.
- Arms: **raw** (`--size-adjust 0`) and **adj** (`--size-adjust 1`); **seeds 0–5** (6 seeds; the same seed = the same
  truth, menu, slices and driver seed in both arms). 12 runs on colo.
- Truth: per size the true knapsack frontier (the truth through `moves.mckp`, `standin.true_frontier`): log KL*(s) and
  b_true(s) = −d log KL* / d bpw (container bpw, s ∓ 0.1). The **true size-adjusted score** of an incumbent:
  S = log KL_true + b_true(s)·(bpw_container − s); the same b_true for both arms. Gap = S − log KL*(s).

## Measures (`search/v3/abreport.py`)
1. **Primary:** final ΔS = S_adj − S_raw per (seed, size) cell after 800 evals; mean over the 24 cells, 95% CI by a
   seed-cluster bootstrap (resample seeds, 10,000 draws).
2. **Convergence:** mean gap vs evals and vs card-hours (the executor's card-time model), per arm; ΔAUC = mean over a
   common grid (0 → the smaller arm's end, 40 points) of the per-cell gap difference adj − raw; CI as in 1.
3. Reported, no decision: the final true raw KL and the KL at the nominal size (exp S), bpw drift of the incumbents
   inside their bands (bpw_container − s over time and at the end), accepts / wrong accepts (true ΔS > +0.5% at the
   accept) per arm, early stops, card-hours, the fitted b vs b_true at the end.

## Decision
**Adopt size-adjusted (CONF `size_adjust` true, the driver's default) unless it is worse**, where worse means any of:
(a) primary mean ΔS > 0 with the CI's lower end > 0; (b) ΔAUC vs card-hours > 0 with the CI's lower end > 0;
(c) wrong accepts (true ΔS > +0.5%) in adj > 2 × raw's + 2 summed over runs.
If a run crashes or a stand-in check other than "rotation" fails in one arm and not the other, report it and rerun that
seed pair once; the decision uses the completed pairs (≥ 5 seeds required).
