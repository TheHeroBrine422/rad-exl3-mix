# RULES-187 — move prediction for the #183 search: replay protocol, arms, winner (w165, 11-15)

Written and committed before any replay arm is fitted or scored. The data pipeline was checked on the snapshot below
(counts and one reproduction of the lead's correlations); no arm, no reranker and no metric table existed yet.

**Why.** The search's proposals are ranked by the driver's per-band fit (fit.py / kfit3 form, `fit_groups` b1–b4).
Predicted vs measured log r on whole-slice candidates (state.jsonl `submit.pred_logr` vs `decision.logr`): b1 corr
0.72 (lead, 11-14), gain overstated 1.5×; b2 0.24, 3×; b3 0.18, 7×. Above 2.5 the proposals look close to random
against measurement. Goal: a better predictor of measured log r per band, and a reranker that orders candidates
before they are evaluated.

**Data cut.** Live driver snapshot taken 2026-11-15 00:22 UTC (6:22 PM CST, 11-14): `drv/state.jsonl` 1915 lines
(sha256 `ec82d3cecc5892d0…`), `multi/results.jsonl` 414 lines, `menu.json` (menu173m-v1b), `registry.json`,
`prior-obs-w1a8.jsonl`, `driver.json`, and the assignment files. The snapshot lives in `/workspace/rad-local/workers/w165/snap/`
(not in the repo); the rows the replay uses are extracted to `data/rerank187/` with the replay outputs.

## 1. Candidates, predictions, targets

- **Candidate** = a `submit` event with kind in {solve, partial, pair, block, thompson, explore} (incumbents, monitor
  reads and init points are not candidates). Its reference is the submit's `ref` (the size's incumbent at submit).
- **Prediction** of an arm = log(KL_model(cand) / KL_model(ref)) on the same model (the driver's `pred_logr` is this
  quantity for the live fit).
- **Measured target** = `decision.logr` (the driver's statistic: the mean over parts of the cap-weighted log ratio,
  RULES-picks). Events with decision `stale` carry no logr and are excluded.
- **Whole-slice row** = the candidate's result has `stopped` = null. These are the metric rows.
- **Early-stopped row** = `stopped` is a dict. Treatment, fixed here: **excluded** from Spearman, calibration slope and
  overstatement. In the hit rate it is **censored**: its decision was "reject" on the prefix, so it counts as measured
  not better (a predicted-better stopped candidate is a miss).
- **Audit rows** (`audit` true: a random 10–30 % of candidates run without early stop) are an unselected sample. They are
  reported as their own line in the table and are not used for the winner rule.

## 2. Replay protocol (time-ordered)

- Results are ordered by their `result` event (ingestion order). N = number of result events so far, all kinds (the
  driver's `n_results`).
- Refit points at N = 20, 40, … (`refit_every` = 20). At each point every arm fits on the rows of results with index
  ≤ N plus `prior_obs` (as the driver does), built with the driver's own `add_obs` logic (same labels, anchors, SEs).
- The candidates whose results have index in (N, N+20] are predicted by that fit. The fit saw no result after N. A
  candidate's prediction uses only its assignments and its ref's assignments, which the driver also has at submit.
- Replay starts at the first refit point (N = 20). Metrics cover all replayed chunks (about N = 20 … 320).

## 3. Metrics (per band)

Bands = the driver's `fit_groups`: **b1** = 1.25–2.5, **b2** = 3–4, **b3** = 4.5–6, **b4** = 7 (reported; not in the winner rule).

- **ρ** = Spearman(pred, measured) over whole-slice rows, pooled over all replayed chunks.
- **hit** = P(measured < 0 | pred < 0), over all decisions with early-stopped rows censored as above; base rate
  P(measured < 0) beside it.
- **slope** = OLS slope of measured on pred over whole-slice rows (1 = calibrated).
- **overstatement** = mean(pred | pred < 0) / mean(measured | pred < 0), whole-slice rows.
- **Uncertainty**: paired bootstrap over whole-slice rows (2000 resamples, seed 187) for every difference reported; per-chunk ρ
  (mean and spread) beside the pooled value.
- **Reproduction check**: arm (a)'s replay predictions against the driver's live `pred_logr` on the same candidates. They
  can differ, because the live fits finish later than the replay's exact refit points. Reported, not a gate.

## 4. Arms

- **(a) baseline:** the driver's per-band hard fits (`fit_groups` b1–b4, rows = sizes of the group), fit.py's SModel.
- **(b) joint:** one fit on every size (one group "all", as the driver's first fit).
- **(c) partial pooling** (`pool187.py`, a subclass of fit.SModel): the shared parameters are fit.py's (theta, c0,
  embedding, q, kappa, family factors, slice offsets, xscale). Each band has deviations from them: dtheta (per group × depth knot), dq
  (log q) and dkappa (log kappa), with a Gaussian shrinkage prior of sd σ_dev. **Three strengths: σ_dev = 0.1, 0.3, 1.0**
  (arms c1, c2, c3). Everything else as (a).
- **(d) reranker:** built on the best of (a)–(c) (chosen by the winner rule below, before (d) is scored). An xgboost
  regressor (xgboost 3.4.1, `reg:squarederror`, max_depth 2, 60 trees, eta 0.1, min_child_weight 5, subsample 0.8,
  colsample_bytree 0.8, seed 187). Target = measured log r. Features: the base prediction; move kind (one-hot); bytes moved
  (cand − ref) / the size's byte target; per tensor family and per group, the number of changed tensors and their
  mean depth; the mean K of the changed tensors before and after; the number of changed tensors; the band index; the
  size; embedding changed (0/1); head changed (0/1). Training rows: whole-slice rows with result index ≤ N, using their
  out-of-sample base predictions (the replay's own predictions made before their results). With fewer than 30 training
  rows, the reranker returns the base prediction.

## 5. Winner rule (fixed now)

- **S** = mean(ρ_b2, ρ_b3) over whole-slice rows.
- The winner is the arm with the highest S among (a), (b), (c1–c3) and (d), subject to:
  (i) S(arm) − S(a) ≥ **0.10**;
  (ii) ρ_b1(arm) ≥ ρ_b1(a) − **0.05** ("b1 must not get worse", tolerance for noise);
  (iii) the 80 % paired-bootstrap CI of S(arm) − S(a) has a lower bound > 0.
- If no arm passes (i)–(iii), the winner is (a): no change to the live search. (d) is adopted only if it passes the
  same three conditions against its base arm.
- The replay selects among 5–6 arms on the same snapshot, so the winner is a candidate for the live run, not proof.
  The live test (driver run on a fresh slice, the lead's call) is the confirmation.

## 6. Limits (stated before the run)

Whole-slice rows number 102 on the snapshot (b1 52, b2 29, b3 21), so the replay has ~60–80 scored rows in b2+b3.
The standard error of ρ at that size is ~0.12–0.20: the margin of 0.10 is about one SE, and the bootstrap CI is the
real gate. Results below the margin are reported as "no detectable difference", not as a null.

The lead's quoted b1 0.72 does not reproduce with these definitions (0.53 on the 52 whole-slice b1 rows; 0.74 on the
44 non-audit ones). b2 (0.31) and b3 (0.17) do reproduce within noise. The table reports the definitions above; the
discrepancy is noted in LOG #187.

## 7. Outputs and what is not touched

- No edit to `driver.py`, `fit.py`, `moves.py`, `kfit3.py`, the driver config or the live run. The winner's driver change
  is a `.patch` under `search/v3/rerank187/` with the reason, for the lead to apply.
- All code: `search/v3/rerank187/`. Raw rows, predictions, bootstrap outputs, the table and `winner.json`: `data/rerank187/`.
- Compute: colo only (CPU, single-threaded jobs, `nice -n 19`), `/data/work/w165/`. No GPU, no ds8 / box2.
