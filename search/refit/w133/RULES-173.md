# #173 rules (w133): does a better sensitivity model on the existing measurements beat kfit3?

Written 2026-11-13 ~08:55 AM CST, before any candidate model is fitted or scored, and committed alone. Data: the
live V3 search (#171, `local/workers/w131/run/drv/state.jsonl` + `assign/`, registry, menu129h) and w129's
calibration rows (fit129 = the driver's prior_obs, s129 held-out swaps). The ver3 monitor reads are **verify data**:
not a fit row, and not used for any selection here. They appear at most as a report-only column. CPU only, colo.

## Data units (`ds133.py`, a replay of driver.py's on_result / add_obs; checked: per-part log r == the driver's, max diff 0)
- **Fit rows:** exactly the driver's obs (label `S|p|full` / `S|p|pre<k>`, the incumbent's anchor on the same
  documents, paired SE) + the 90 fit129 prior rows. Each row carries the result index `ri` it became available at.
- **Events:** every accept / reject decision (stale dropped) with the incumbent it was judged against (driver's rule),
  per part p ∈ {c, x}: measured log r_p on the scored documents and its paired SE, the size offset `off` (b·Δbpw of
  the submit) and the call. A part with < 3 scored documents is dropped from that event.
- **Prediction of an event** by a model: per part, pred_p = the model's log ratio candidate / incumbent for that
  part (label offsets cancel; e.g. kfit3 with xscale: c = log(P_c / P_i), x = λ·log(P_c / P_i)).

## Splits (fixed now)
- **Snapshot S0:** state lines with time stamp ≤ the commit time of this file; N0 = its result count.
- **D, development:** fit on rows with ri ≤ 0.4·N0 (+ prior rows), predict events with 0.4·N0 < ri ≤ 0.6·N0. All
  term choices and hyperparameters (ridge α, shrinkage σ, which kfit3 extensions to keep) are made on D only, plus
  inner CV inside D's training rows. D's numbers are reported but never decide.
- **Freeze:** the candidate set and every hyperparameter are written to `candidates173.json` and committed
  (rules amendment) before the first T / P run. After that no model may change.
- **T, test by time (rolling origin):** origins 0.6·N0 and 0.8·N0: fit on rows with ri ≤ origin (+ prior rows), predict
  the events in (origin, origin + 0.2·N0]. Run once.
- **P, prospective:** fit on all S0 rows (+ prior), predict every decision that the search makes after S0 up to the
  cut **11-13 3:30 PM CST** (the first state line after it ends P). Reported separately by act mode: **P-f16**
  (slices scored before w132's A8 switch) and **P-A8** (slices after it; the models are f16-fitted, no A8 term is
  fitted). P-A8 is the transfer test: do f16-fitted ratios predict A8 ratios? The A8 hook (per-mode additive /
  multiplicative offset) is built but not fitted on A8 rows here, since that would need A8 training rows.
- **S, test by slice (report + guard):** fit on all S0 search rows **without** the prior rows, predict (i) fit129:
  within-label centred log error of the 90 rows (kfit3's within-slice RMSE), (ii) s129: the 8 held-out swaps' per-part
  log r vs their base.

## Metrics (per split, per part c / x, and the part mean)
1. **Ratio error:** weighted MSE of pred_p − meas_p, weight 1 / (SE_p² + 0.02²) (inverse variance with kfit3's misfit
   floor τ; weights do not depend on the model, so differences between models are unbiased). Also reported:
   unweighted log-RMSE and the noise-corrected RMSE √max(0, MSE − mean SE²).
2. **Calls:** predicted call = accept iff mean_p(pred_p) + off < 0 and every part pred_p + off < 0 (seqrule.final on
   the predictions). Balanced accuracy (mean of accept recall and reject recall) vs the search's actual calls (primary),
   plain accuracy, AUC of −(mean_p pred_p + off) for accept; sign agreement with the measured statistic on events whose
   |mean_p meas_p + off| > 2 SE (the clear-cut ones).
3. **Accepted moves:** over the held-out accepts, mean predicted vs mean measured log r (pmean), their ratio; over all
   held-out events, the calibration slope of meas on pred. (Accepted measured gains carry a winner's-curse bias; reported, not
   decided on.)
CIs: paired bootstrap over events (4,000 resamples, seed 173), 95 % percentile intervals, for every metric and for
every model − kfit3 difference.

## Candidates (pre-registered; details settled on D)
- **M0 kfit3** as the search runs it: `search/v3/fit.py`, opts tau 0.02, beta 0, xscale true (the baseline).
- **M1 kfit3 + richer terms:** per-group K-curve shape (q per group), per-part non-linear compression (x: λ(P) =
  λ0 + λ1·log P instead of one λ; c: its own λ_c with prior at 1), head × trunk (the head's own compounding factor),
  sibling / adjacent interactions (within-layer attention-or-SSM × FFN product term; neighbour-layer term). Each term
  is kept only if it lowers D's weighted MSE.
- **M2 residual model on M0:** ridge over tensor-level features summed as candidate − incumbent differences (group ×
  depth band × K band indicators, log err ratio, log h̄, proxy error), one per part; α by inner CV on D's training
  events. Linear in per-tensor terms, so the knapsack stays decomposable. (A GP variant only if time allows; then it
  is a 5th candidate frozen in the same file.)
- **M3 hierarchical shrinkage:** M0 + per-tensor random effects in log s_w, u_w ~ N(0, σ_g²) per group, σ_g from a
  grid on D.
- **M4 combination:** at most one combination of terms that individually helped on D.
Parameter counts are reported with every result (rows ≈ 1.3k, events ≈ 400 at S0).

## Decision: "wins clearly"
Pooled test set = T (both origins) ∪ P-f16 (events pooled). A candidate **wins** over M0 iff all of:
(a) the part-mean weighted MSE difference vs M0 has a 95 % CI entirely below 0 **and** a point reduction ≥ 10 %;
(b) neither part's weighted MSE is worse than M0's by more than 5 % (point);
(c) balanced accuracy of the calls ≥ M0's − 0.02 (point).
Several winners: the one with the lowest pooled weighted MSE, unless a simpler one (fewer parameters) is within its CI,
in which case the simpler one. **Guards** (flag in the LOG, a winner that fails one needs the lead's look before use):
S split (fit129 within-label RMSE or s129 part-mean RMSE) worse than M0 by > 20 %; P-A8 part-mean weighted MSE worse
than M0's with a CI above 0. No winner → kfit3 stays; report it as a null with the CIs.

## If there is a winner (step 3)
Re-solve the knapsack at every search size with it (the driver's `solve` move: moves.py's Lagrangian MCKP at the
size's byte cap) and with M0 on the same rows, and report vs the current incumbents and w129's fit129h2 picks: bytes
moved (sum of |Δbytes| over tensors), tensors changed, predicted gain of the new pick vs the incumbent under both
models. These picks are candidates for the lead's ver4 + xcorp test. Nothing is submitted to the search from here.

## Amendment 1 (2026-11-13 ~9:45 AM CST, after D, before any T / P / S run): frozen candidates
`candidates173.json`: M0 (kfit3 as run), **M1** = kfit3 + per-(label, size) offsets profiled out + per-part curvature
λ1 / μ1 (the "per-part non-linear compression" term; the head, sibling and adjacent terms did nothing on D and are
dropped), **M2** = ridge residual on M0 (gb8 + gk + h̄ + total features, α by CV), **M2b** = the same ridge (gb8) on
M1, **M3** = per-tensor shrinkage (sd 0.1) on kfit3 + per-(label, size) offsets, **M4** = M1 + per-group q. The
per-(label, size) offsets were added on D after D's in-sample residuals showed kfit3's search labels pooling every
size (cross-size x misfit 0.27–0.39 log vs 0.06 / 0.10 within a size); they fall under "per-part non-linear
compression". A ds133 fix found on D (an incumbent's anchor row kept the result index of its last re-creation instead
of its first) was applied before the D numbers that chose these candidates. The decision rule is unchanged.
