# #171 per-part fit labels + equal-weight per-part accept: rules (w131, written 2026-11-13 ~1:35 AM CST, before any run; git commit time is the attestation)

Why (w129, #169): kfit3 predicts held-out calib swap ratios to ~3 % but xcorp ratios to ~14 % (xcorp ratios are
compressed), and xcorp carries ~90 % of the pooled KL of a two-part slice, so pooled rows and a pooled accept are
xcorp-driven. Changes (lead's task, decided): (1) fit.py labels each slice part (x / c) on its own (own offset, own
anchor); (2) the accept / early-stop statistic is the equal-weight mean of the per-part log ratios (size-adjusted as
now), "both parts better" stays. Two checks below decide only what the task leaves open.

## Test A: a per-part ratio scale for x (optional fit term; adopt or not)
- Term: rows whose label's part is x are predicted as exp(soff_label) · P^λ (P = kfit3's KLD form, λ one fitted
  scalar, prior log λ ~ N(0, 0.5)); every other row as now. Candidate generation keeps predicting P (the calib scale).
- Data: w129's committed rows (`data/refit/w129/`), β 0, menu129 (V2 3.00 head): train = the fit129 U + G rows (one
  calib label) + the two s129 anchors u-k2.5 / u-k4 on each part (labels s129|x, s129|c); held out = the 8 s129 H swaps
  (group swaps vs their base) on each part, scored as predicted vs measured log ratio to their base (floor129.py's way).
- Arms: A pooled s129 label (reference, = floor129 A); C parts as own labels, no scale (before); E parts + λ (after).
- Report: log-RMSE per part (x, c) and pooled over both, for A / C / E, and the fitted λ.
- **Adopt the scale** (fit_opts `xscale` true in the launch config) iff E's x log-RMSE ≤ 0.9 × C's x log-RMSE **and**
  E's calib log-RMSE ≤ C's + 0.005. Else keep it off (the code stays, default off). Caveat recorded either way: λ is
  identified here by one anchor pair on s129; in the search every x label holds many rows.

## Test B: the early-stop boundary under the new statistic (boundary pick unchanged?)
- boundsim.py with `--stat pmean`: z = (mean of the two parts' log ratios) / its SE (each part's delta-method SE over
  its scored documents, SE = ½ √(SE_x² + SE_c²); no stop until each part has ≥ 3 documents scored, the hosts' rule);
  normal model: the same paths with weight ½ per part; real documents: pairs filtered and scaled on the per-part mean
  log ratio, accept = mean < 0 and both parts < 0. Everything else as RULES-boundary.md (data, priors, M 4000, costs,
  metrics, the pick procedure).
- **Pass** = the pick procedure still picks **B p2 w5**. If it picks another rule, report both and keep B p2 w5 (the
  user's boundary) until the lead / user decide.

## Also (code checks, not decisions)
- standin.py with w126's si6 settings (`--sizes 1.5,2.5,4.0,6.0 --results 400`, seed 0) passes 7 / 7 with the new
  statistic and per-part labels (also one `--real --layers 64` run, w127's realistic stand-in).
- test_hostd_es.py passes with the new seqrule copied into rxrun/earlystop.py.
