# RULES-strata — script-stratified search slices and accept statistic (#186, user 11-14)

Written and committed before any stratified slice is built or scored.

**Why.** Our allocations (V2's search, the #173-03 M1 picks, the #183 search) are 10–25× worse on non-Latin-script text
than on Latin text, against 3.0× for IQR-V4 and 4.4× for a uniform V3 mix (LOG #186). Calib, which the board is drawn
like, is 3.9 % non-Latin by gen rows. The slices hold too few of these documents for the search to see them, and the
RULES-picks cap of 0.2 per document plus the docmean guard down-weight exactly these high-KL documents.

1. **Stratum of a document.** Non-Latin (N) when more than 50 % of the alphabetic characters of its decoded text are not
   Latin script (`unicodedata.name` without "LATIN"; `search/v3/run183/ml186.py`); otherwise Latin (L). Computed once per
   document from its tokens and stored in the slice's kld.json docs as `"stratum": "L" | "N"` plus `"script"` (the
   commonest non-Latin script name, else "Latin").
2. **Board weight.** w_N = 0.039, w_L = 0.961 (calib gen rows, `calibnl186.py`, 11-14). Fixed for V3; not refitted.
3. **Slices (s186*).** Each part (c from the merged fit + search pool, least-used first; x from xcorp, unused pieces)
   holds 15–20 % N gen positions, from at least 6 scripts per part, at most 1/3 of a part's N positions from one script,
   documents capped at 4,096 tokens as before, gen-masked (board rows). The rest of the draw is as `draw171.py` /
   `mkslices171.sh`.
4. **Accept statistic.** For each part p and stratum s: d_ps = log(Σ_s w_d B_d / Σ_s w_d A_d), with the RULES-picks cap
   (0.2) applied within (p, s): a document is capped at 0.2 of its stratum-part's incumbent total. d_p = w_L d_pL + w_N d_pN
   (a stratum absent from a part: the other alone). Parts combine with equal weight (pmean) as now; size adjustment as
   now. Accept iff d + off < 0 and every part's d_p + off < 0. Early stop on the same statistic (hosts get the stratum of
   each document in the early-stop block).
5. **Guard.** docmean per part = w_L · mean_L log(B_d/A_d) + w_N · mean_N log(B_d/A_d); same sign as d_p (as before).
6. **Fit rows.** One row per (slice, part, stratum) with its own label and offset ("S|c|N|full" etc.), so the model sees the
   N-stratum ratios separately.
7. **Tracking.** t183 stays (its references stay valid) and gains a non-Latin supplement t186n (~30 k gen positions,
   ≥ 6 scripts, fit + search pool + xcorp, never decides). The status plot shows L, N and the board-weighted combination.
   V2 / IQR-V4 references are scored on t186n.
8. **Transition.** The driver rotates to the first s186 slice as soon as it is built (incumbents rescored there); s173
   decisions stand. Nothing else in RULES-picks changes.
