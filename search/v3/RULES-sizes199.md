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
