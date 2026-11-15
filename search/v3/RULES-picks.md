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

## Clarifications (lead, 2026-11-13 1:05 PM CST, before any data under this rule; w145's questions)
- The (c) guard is computed size-adjusted, like (b) and (a).
- Final picks: (a) is pooled over both parts (all documents of the slice).
- The final (c) guard compares the pick with the runner-up and the default pick only (not every finalist).
- Final-pick slice: the long trace enters as ≤ 32k chunks, capped near calib's trace share (w145 plan), not whole.

## Finalists per size (user, 2026-11-13 1:10 PM CST, before any data)
3–5 candidates per size go to the ~750k final slice, chosen for diversity (different sources, not the top of one
noisy ranking): (1) the restarted search's incumbent; (2) the M1 re-solve on the final menu with no search moves;
(3) the re-solve from the alternate fit (V2-prior-free, w143's ablation) if its picks differ; (4) the search's runner-up
by (b) on its last slices; (5) a near-tie variant where the re-solve flags one (embedding / head choice). Identical
candidates are merged. Cost ~0.35 card-h each (≈ 3–4 h on ds8 for all sizes).

## Amendment 2026-11-14 7:20 PM CST (user): no capped statistic in the search
From the switch to the stratified slices (s186a, RULES-strata) on, the in-search accept / early-stop statistic is the
uncapped one: (b) is dropped, driver CONF `cap: null` (RULES-strata §4 then sums each (part, stratum) without weights).
Why: since the board gen mask (11-14) the prompt-side rows that made single documents dominate are gone, the strata
weighting now controls the non-Latin share explicitly, and the cap was down-weighting exactly the high-KL (non-Latin)
documents (#186). The per-document guard (c) stays for now (re-checked after ~200 s186 decisions: how often it alone
rejects). Final picks are unchanged ((a) with the guard, finalpick.py). Decisions on s173 stand.

## Amendment 2026-11-14 8:25 PM CST: per-document guard (c) dropped from the search
Test stated to the user before its data (11-14 ~7:30 PM CST): score the 20 largest guard-only rejects of #183 and their
incumbents on t183 (never decides); mostly better on t183 → drop the guard; mostly not → keep; mixed → log-only.
Result (`search/v3/run183/guardtest183.py`, size-adjusted with V2's t183 curve, per-document bootstrap): 13 / 20 better
pooled, 7 with the 95 % interval below 0, none above; mean −1.4 % pooled (active sizes 2–6: 9 / 13, −2.5 %), and the
per-document mean on t183 also improves (−2.4 %). The guard mostly blocked real held-out gains → driver CONF
`guard: null` from the next restart. The per-document mean stays in the decision events (logged, not used). Final picks
keep their own checks (finalpick.py, golden-final750).
