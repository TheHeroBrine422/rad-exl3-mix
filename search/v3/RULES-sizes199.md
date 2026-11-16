# RULES-sizes199 — in-between release sizes 2.75 / 3.25 / 3.75 and 5.5 / 6.5 join the search (#199)

Written 2026-11-16 1:05 AM CST before any decision at these sizes (user 11-15/16: finer sizes between 2.5 and 4.0
"because those are the sizes that are really usable"; 5.5 / 6.5 per the w184 review item 5; 2.25 not added — user and
lead: even with QAD it stays above 0.05). w181 (#193-02) stopped at the same time (3 % confirmed on s195).
1. CONF `sizes` += 2.75, 3.25, 3.75, 5.5, 6.5. Fit groups: 2.75 → b1, 3.25 / 3.75 → b2, 5.5 / 6.5 → b3.
   `audit_frac_by_size` 0.3 for 3.25 / 3.75 / 5.5 / 6.5 (as their neighbours); 2.75 none (as 2.5).
2. Initial incumbent = the #192 consensus probe of that size (`drv/inbox/done/<size>-probe.json`, menu-v2, built 11-15
   7:10 AM CST from the s186e re-selection; 6.5: `lead-consensus-6.5-probe.json`). Its t183 reads (report only):
   2.75 0.04061, 3.25 0.01843, 3.75 0.00962, 5.5 0.001305, 6.5 0.000616.
3. Everything else unchanged: same moves, two-stage confirmation (z < −1), pool K = 3, slices, rotation. The driver's
   size choice (fewest decisions first) gives the new sizes the moves until they catch up; no size_weights.
4. ~4 h after the start the lead may submit one consensus re-run per new size (consensus192 sources + the new sizes'
   confirmed models) through the inbox as `decide` — judged like any candidate.
5. The new sizes enter the final picks like the others (RULES-picks amendment to come), with their confirmed models
   and pool as candidates.

## Amendment 1 — 2026-11-16 1:00 AM CST (user: "i question if rotation is even necessary given at this point we are just trying to improve the new targets")
No further rotation: s195s1 stays the search slice and s195c1 the confirm slice until the final picks (`rotate_after`
100000). The new sizes start fresh on s195s1 (0 decisions there); the final picks re-score every candidate on two
fresh large slices, so slice reuse here cannot carry into the published numbers. s195s2–5 / c2–10 stay unused (spare
for the final picks or V3.1).

## Amendment 2 — 2026-11-16 3:10 AM CST (user: "add 7.0 bpw to the search based on consensus")
7.0 resumes (STOP-7 removed). Fit group: 7.0 moves into b3 (with 4.5–6.5; b4 had 200 rows / 55 labels) and b4 is
dropped. The #192 consensus probe `lead-consensus-k7cap-7-probe.json` (K ≤ 7 cap; t183 0.000501 at 6.987 bpw vs the
stopped incumbent's 0.00051) is submitted through the inbox as `decide` for 7.0, judged like any candidate (stage 1 on
s195s1 vs the incumbent, confirmation on s195c1). 7.0 joins the final picks like the other sizes.

## Amendment 3 — 2026-11-16 3:30 AM CST (user: "add 1.5 and 1.75 bpw targets"; "do xavg in addition to consensus")
1. 1.5 and 1.75 resume (research points, not release sizes; fit group b1; confirmation as every size). 1.25 stays stopped.
2. Lead inbox loop (`rad-local/bin/allocloop.sh`, every 3 h until the user stops the search): for every active size,
   (a) a consensus refresh (consensus192 `--whole --slice s195s1`: sources = every whole result on the live search
   slice, best + within 3 %, at most 5 per size) and (b) the #193 `xavg` allocator (own193.py variants "", "w", "p":
   the shifted mean of all sizes' incumbents / pool members), submitted as `decide` files, each judged like any candidate
   (stage 1 on s195s1, confirmation on s195c1). A file whose assignment equals one already submitted is skipped.
3. No rerankers (xgb regressor / learn-to-rank failed the offline gate and confirmed 0 of 60 on s195).

## Amendment 4 — 2026-11-16 4:15 AM CST (user: "yea try it. also you might as well kill the automated consensus/xavg")
1. allocloop.sh stopped (round 1 decided as logged; round 2's unprocessed files held in `drv/inbox-held/`, not submitted).
2. One round of **partial pulls** (`search/v3/inbox192/pull199.py`): per size, the live incumbent with its N = 12 / 24
   trunk tensors of largest |K − K_consensus| set to the consensus option (consensus = allocloop round r11161001's
   consensus192 output, s195s1 whole results), bytes rebalanced into the size window by menu err per byte among the
   other trunk tensors; head / embedding unchanged. 32 decide files, judged like any candidate (stage 1 + confirmation).
