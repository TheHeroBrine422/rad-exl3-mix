# RULES-picks — statistics for final picks and in-search decisions (user, 2026-11-13 12:30 PM CST)

Committed before any ver4 read and before any search restart. Supersedes the statistic parts of
`docs/design/eval-data.md` "Final picks" and the driver's accept statistic where they differ.

Per candidate A vs B, per part p (calib c, xcorp x), from per-document KL sums (long traces cut into ≤ 32k chunks;
each chunk is a document for the statistics, a whole trace is one unit for resampling):
- (a) pooled: log(Σ_d KL_A,d / Σ_d KL_B,d) — the board's estimator.
- (b) capped: as (a), but a document contributes at most 20 % of its part's total (larger documents scaled down to
  the cap in both A and B).
- (c) per-document mean: mean_d log(KL_A,d / KL_B,d).
CIs: paired bootstrap over documents (traces resampled as whole traces), 10,000 draws, 95 %.

## Final picks
- Slice: ~500k–1M scored positions (target 750k): ver4 + a fresh xcorp part + reused verify docs (ledger rows,
  role final), two parts as before.
- Decision: (a), size-adjusted as frozen per band; the best (a) wins; report its CI vs the runner-up and vs the
  default pick (the re-solve's).
- Guard: the winner's (c) must not be significantly worse (its CI not wholly on the wrong side) in either part.
- Check: if any document holds > 20 % of a part's KL, flag it and look before trusting the result (expected never).

## In-search accept / reject (small slices, sequential)
- Decision: (b) with the existing sequential rule (size-adjusted, equal-weight per-part mean, both parts better,
  early stop).
- Guard: (c) must have the same sign as (b) in each part (no CI requirement).
- Held-out monitor: (b) and (c), both with CIs; never the raw pooled (a) alone (11-13: one document, calib 4681,
  held a median 34 % of ver3's KL).
