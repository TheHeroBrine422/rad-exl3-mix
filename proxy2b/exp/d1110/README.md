# d1110 · Overfitting check: replicate the adopted 0.8B stack on untouched calib-verify text (#93)

Concern (user, 2026-11-05): 31 experiment scripts judged on one 0.8B slice (`/data/g08-search1`: 173 search-pool docs,
197k scored rows). Pass rules, seeds and paired SEs guard each test, but not the selection across many tests on the
same slice (winner's curse, garden of forking paths). Statistics were fit on qad / fit pools, so the risk is in what
we *adopted*, mostly the small effects.

Check: a fresh 0.8B golden **`g08-verify1`** from the calib **verify** pool (pools-v3; never used for any 0.8B
decision), ~180 docs, seed 1110, each capped at 2,048 tokens like g08-search1, bf16 0.8B reference (`mkgold.py`).
Re-score the existing quants (no new quants), seeds 0 and 1, K3 and K2, paired:
base → edy (Σdy² H_out) → edy+swj → +writers asym ρ0 → +healing (d1104); plus the rejected ones as a control where
their quants exist (i8c on edy, #78-03; A4 all-MLP, d1106).

Rule (written 2026-11-05 ~4:50 AM CST, before the golden exists): for each adopted step, the log ratio vs the
previous step on verify must be < 0 with both seeds, and keep **≥ 50% of its search-slice log gain**; the cumulative
stack must keep **≥ 80%** of its search gain. A step that fails is re-labelled **inconclusive** and re-tested on the
27B before it ships. Afterwards g08-verify1 is read once and frozen for finals; new 0.8B decisions use a fresh
search-pool slice (`g08-search2`, drawn when needed), rotated every ~10 experiments.
Runtime cost: none.

**Spare goldens (user, 2026-11-05; `more.sh`):** g08-search2 / -search3 (170 search-pool docs each), g08-verify2 (180
verify docs), g08-xcorp2–4 (180 xcorp docs each); disjoint from each other and from g08-search1 / -verify1 / -xcorp1.
Built up front, not scored against until a decision needs one; each first use gets a LOG line.

**Addition (user, before data): `g08-xcorp1`**, 180 xcorp v1 documents (outside calib's documents, same upstream
sources; stratified by category, seed 1110, cut at 2,048 tokens, every position scored; `xgold.py`). The same
re-scoring runs on it (tags `xc-…`). Same thresholds, reported separately: a step that passes on verify but loses
most of its gain on xcorp is calib-specific and is flagged for the 27B check. (xcorp served as healing data in
#89-05's mix arm only; a few of these documents may overlap it.) xcorp's 96M tokens allow more such goldens later.
