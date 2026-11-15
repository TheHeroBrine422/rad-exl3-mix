# RULES-confirm — two-stage accept: confirm every search accept on fresh documents (#189)

Written 2026-11-14 9:05 PM CST, before any confirmation data (user: "do that confirmation step").

## Why
Held out on t183, the #183 search's accepted gains are mostly not real above 2.5 bpw: share of the training estimate that
shows up on t183, first read → latest: 2.0 ~75 %, 2.5 ~45 %, 3.0 ~35 %, 3.5 ~15 %, 4.0 negative (+12 % worse), 4.5 ~5 %,
5.0 ~35 %, 6.0 ~25 %. Cause: the winner's curse — a candidate is accepted when its measured gain (true effect + noise) is
negative, so among many near-null candidates the accepted ones are those the noise favoured, and the chained sum of
accepted gains is biased. A second, independent measurement of only the accepted candidates removes the selection bias.

## Rule
1. **Stage 1 (unchanged):** the search slice decides as now (stratified pooled statistic, cap null, guard null, early stop,
   pool join rule). A stage-1 accept or pool join is **tentative**: nothing changes in the incumbent or the pool yet.
2. **Stage 2 (confirmation):** a tentative candidate is scored whole (no early stop) on the **confirmation slice**, and so is
   its reference: the size's *current* incumbent (for a pool join that would replace the worst member: that member). Each
   model is scored once per confirmation slice (cached). Statistic: the same as stage 1 (stratified pooled, size-adjusted,
   board weights). **Confirmed iff z < −1** on the confirmation slice (z = statistic / its per-document bootstrap or delta
   SE). Confirmed → applied as a normal accept / join against the current state; not confirmed → rejected (logged as
   `reject (confirm)` with both stages' numbers).
3. If the incumbent changed between the tentative accept and the end of its confirmation, the candidate is compared with
   the **new** incumbent on the confirmation slice (its cached score, or one more eval).
4. **The training estimate and the plot chain the stage-2 log r**, never stage 1's. Both stages' rows feed the fit (they
   are honest measurements; labels by slice as now).
5. **Confirmation slices are fresh and rotate with the search slice.** Pairs (search, confirm): (s186a, s186b),
   (s186c, s186d), (s186e, s186f). A confirmation slice never becomes a search slice later. Confirmation evals do not
   count towards `rotate_after`. New pairs are built with w164's slice builder when these run out (same RULES-strata mix).
6. **All active sizes (2.0–6.0).** Paused sizes (< 2, 7.0) are untouched.
7. **At go-live:** each active size's current incumbent is confirmed against the incumbent it had at the s186 switch
   (state line 3271) on the first confirmation slice, under rule 2. Not confirmed → the size goes back to the switch-time
   incumbent (and its pool to that model alone, re-filled by joins).
8. Final picks are unchanged (golden-final750, finalpick.py).

## What to watch (not gates)
- Confirmation pass rate per size; held-out (t183) gain per confirmed accept vs the stage-2 estimate (should agree within
  noise; if stage 2 still overstates, raise the threshold — by a new amendment, before looking at more data).
- Cost: one or two whole evals per tentative accept (~10–15 % more evals at the current accept rate).
