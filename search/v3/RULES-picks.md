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

## Amendment 2026-11-16 ~1:45 PM CST (user): final picks in two phases (+ 1.5); supersedes "Final picks" slice / finalists above
Written before any phase-1 data; the search (#183) ends at ≥ 4,929 results with no confirmation pending (`stopat.sh`).
1. **Candidates per size (release sizes + 1.5 / 1.75):** every confirmed accept / join since eval step ~1000, the s186e
   re-selection top 3, the live incumbent and pool at stop time, and at 6.5 / 7.0 the stage-1 tentatives whose
   confirmation missed narrowly (stage-2 |z| ≤ 0.5). Dedup by assignment hash.
2. **Phase 1:** a fresh ~1.5M-scored-position xcorp slice (never used, stratified as RULES-strata with ~10 % N, 8k cap,
   ledger as non-calib), built on ds8 / box2. Every candidate is scored on it. Statistic: (a) pooled, size-adjusted
   with **b frozen per size from the measured curve** (slope of ln KLD vs bpw between neighbouring sizes' incumbents on
   this slice; ends one-sided), not the search fit's b. Keep the top 30–50 % per size (at least 3), plus any candidate
   whose CI vs the size's best overlaps 0. N stratum: veto only (a candidate significantly worse on N alone is dropped,
   never picked for N).
3. **Phase 1.5:** consensus and xavg models built from phase 1's leaders (consensus192 / own193 xavg with phase-1 scores
   as sources: best + those within the phase-1 CI per size; xavg neighbour window |Δsize| ≤ 0.75), scored on the
   phase-1 slice; they join phase 2 if they qualify under §2's keep rule.
4. **Low-noise search round (optional, decided after phase 1):** if phase 1 shows real differences at a size, i.e. its
   best beats the search incumbent with the 95 % CI wholly below 0 and by ≥ 1.5 %, or any search model ranks the phase-1
   candidates with Spearman ρ ≥ 0.3 (80 % CI above 0.1) when replayed on the phase-1 reads, the lead proposes to the user
   a short search round at those sizes on large fresh xcorp slices (phase-1 size or ≥ 2× s195; phase-1 reads may feed
   the fit; accept on the new slice, confirm on a second large fresh slice). Its confirmed results join phase 2.
   Never on ver4 or any phase-2 data.
5. **Phase 2:** the survivors on golden-final750 / ver4 (calib part) + a fresh xcorp part (as "Final picks" above). The
   pick and every published number come from phase 2 only; the calib part vetoes (pick must not be significantly worse
   on it). Report each pick's CI vs runner-up, vs the search incumbent, and vs IQR-V4 (interpolated t183 line).
6. **Data:** phase-1 reads are kept (data/picks202/) as the low-noise benchmark for replaying search models.
   **§1 addendum (user, 2026-11-16 ~2:40 PM CST, before any phase-1 read):** the narrow-miss rule (stage-1 tentatives
   whose stage-2 |z| ≤ 0.5) applies at every size with fewer than 6 candidates after the other sources (not only
   6.5 / 7.0). Thin sizes are further served by phase 1.5 (xavg window |Δsize| ≤ 0.75 draws on neighbour leaders) and,
   if still thin, by §4.

## Amendment 2026-11-16 ~5:15 PM CST (user), before any phase-2 data; changes §3–§5 of the 11-16 amendment
1. **Phase 1.5, targeted:** consensus192 / xavg (window |Δsize| ≤ 0.75, sources = phase-1 leaders of the size and its
   neighbours) only at the thin sizes (2.75, 3.25, 3.75, 5.5, 6.5) and at any size whose phase-1 top 5 spans > 1 %;
   2–4 models per targeted size, scored on the phase-1 slices; they compete for phase 2 like any candidate.
2. **Phase 2 = top 5 per size** (size-adjusted phase-1 score, frozen measured b) on ~3M positions: golden-ver4 /
   golden-final750 (calib verify part) + final-x + s195c6–c10. The pick = best size-adjusted pooled KLD; the calib-verify
   part vetoes (pick not significantly worse there than the runner-up).
3. **§4 low-noise search round**, if triggered, uses the unused staged search slices s195s2–s5 (never c6–c10).
4. **Published numbers:** the **public** corpus after the picks are frozen (true holdout, report only, never selects),
   scored as the board does, plus the 2 public long-agentic traces whole at full context (TP2 where needed) as the
   long-context check; RULER as planned. Calib-verify numbers are reported too (secondary).
   Clarification (user, same day): public is scored on the **16 final picks only** (one per size), never on candidates.

## Amendment 2026-11-16 ~6:15 PM CST (user: "more 3.75 candidates, also more 3.25 and 6.5")
At 3.25, 3.75 and 6.5 (thin: the search made 1 / 1 / 4 accepts there in ~234 decisions each) phase 1 adds, before
their phase-1 ranking is read: (a) fresh knapsack solves from the search's final fit of the size's group (b2: fit0591,
b3: fit0588): the optimum + 3 Thompson draws from the fit's posterior draws, each from the size's final incumbent, window
[size − 0.02, size]; (b) the 3 best stage-1 whole-slice reads (size-adjusted log r, any decision) not already in the set;
(c) at 6.5 every past incumbent / pool member and every stage-1 tentative not already in. Phase 1.5 (consensus / xavg
from neighbour leaders) still follows for them. All are scored on s195c2–c5 like every candidate.

## Amendment 2026-11-16 ~7:50 PM CST (lead, after the 6.5 phase-1 reads; user: "holy shit 6.5")
Observation (phase 1, s195c2–c5): every 6.5 candidate ~4 % better than the incumbent uses a **mul1** head (u6.5m /
u6.375m) and mostly a mul1 embedding (t6.5m); the incumbent has a mad head and a mul1i8 embedding at the same K (same
bytes). Confounder test, before phase 2: codebook-only swaps of the search incumbents, nothing else changed:
6.0 head u6 → u6m; 6.5 head u6.5 → u6.5m, embedding t6.5 → t6.5m, both; 7.0 embedding t6.5 → t6.5m (its head is
already mul1). Scored on s195c2–c5 like every candidate (tags swap-<size>-<H|E|HE>); they compete for phase 2.
Phase 1.5 at 6.0–7.0 then uses the mul1 forms. Not a selection on this result: the swaps only isolate the cause.
