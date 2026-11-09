# w96 rules: #127 head Fisher, rare-row smoothing and more data, scored on held-out rows (CPU; user 11-09)

Written 2026-11-08 ~8:10 PM CST, before any Fisher pass, plan or score of this task. Committed alone before launch.
Data file fixed with this rule (sha256): `split96.json` f6a7269f9700a138892c78dcd7d710eff3b016484da2fc68d03524242d895463
(`mksplit96.py`: qad pool of pools-v3 in two halves, B = the 85 documents of w94's qad head dump + random documents,
seed 96; A = 1,400 documents, B = 1,399).

**Why.** w94 (#124): slab plans look ~2× better on the qad rows their Fisher was fit on than on held-out search-gen
rows. The qad-gen Fisher Σp(1−p) (`headsens.npz`, `fisher_qad_gen`) is noisy or zero for rare vocab rows. Only the plans
change here: today's K3–K7 donors (`/data/headslab3`, saved wq, corrected rowerr) and `perm.npy` stay fixed. No GPU.

## Common

- **Planner:** `headslab.py plan` unchanged (non-curve mode: u4 / u5 / u6, s4-3 / s5-3 / s6-3, ...), with HS_TRAIN set
  to the arm's Fisher (written as `fisher_<arm>_gen` into a copy of headsens.npz). Plans compared: **s4-3, s5-3, s6-3**;
  the twins u4 / u5 / u6 are the same in every arm (uniform plans do not depend on the Fisher).
- **Scorer:** `score96.py` (CPU): headsim94's exact head-only KL computed per 128-row block and donor
  (KL = Σ_b d_b + log1p(Σ_b δ_b), d_b = Σ_{v∈b} p(z_ref − z_k), δ_b = Σ_{v∈b} p·expm1(z_k − z_ref); same inputs: bf16
  reference head in f32, donor f16 rows with f16 inputs, f32 accumulation), so any plan is scored from the stored
  block sums without matmuls.
- **Held-out set:** w94's search-gen rows (`dump/search`, gen rows: 38,960 rows, 94 documents; never fit on).
  **Tune set:** w94's qad-dump gen rows (33,172 rows, 85 documents, all in split half B).
- **G0 (code):** my Fisher pass over the calib golden with pools.json's qad pool reproduces `headsens.npz`
  `fisher_qad_gen` (max relative difference ≤ 1e-6 on rows with mass). **G1 (scorer):** on the base plans (u3–u7,
  s4-3, s5-3, s6-3), my search-gen KL matches w94's `sim-search.npz` per row: mean ratio within ±1% per plan, per-row
  correlation ≥ 0.99. Any failure → stop, nothing reported as a result.

## 1. Census (report only)

Gen positions behind today's Fisher; per vocab row the count n_v of qad-gen positions with p > 1e-4 (golden top-512);
rows with Fisher 0, with n_v < 10, with n_v < 100; the share of held-out Fisher mass in each of those row sets (exact
Fisher Σp(1−p) on the search-gen dump rows, full softmax of the bf16 head; and the golden's search-pool gen Fisher).

## 2. Arms

Notation per row v: F_v = Σ p(1−p), N = positions, n_v as above, rate r̂_v = F_v / N.
- **base:** today's `fisher_qad_gen` (headsens.npz, pools.json qad: 2,810 documents). Control base-v3: the same from
  my pass on pools-v3's qad (2,799 documents), report only.
- **smooth:** empirical-Bayes shrinkage r̃_v = (n_v r̂_v + κ m_v) / (n_v + κ), prior m_v = the mean r̂ over v's bin:
  - prior U: 64 equal-count bins of the log unigram frequency of v (token counts, +0.5, over calib fit + qad
    documents of pools-v3 and xcorp v1 `meta/tokens.i32`; counts only, no logits; search and verify documents excluded);
  - prior E: 10 deciles of the bf16 `lm_head` row norm;
  - κ ∈ {1, 3, 10, 30, 100, 300, 1000} (κ in positions with p > 1e-4).
- **more:** more positions from existing goldens (no new forwards; the calib golden already holds every qad position,
  so no qad-pool gen position is left): F = F_qad + w F_x, N and n likewise, w ∈ {0.25, 0.5, 1}:
  - more-q: x = the qad pool's non-gen (prompt) positions of the calib golden;
  - more-x: x = night1105's xcorp golden `xc8k-b1` (all positions; gen mask all 1), minus the 65 pieces of
    golden-search27b-x. Not used: the fit pool (it judges older head plans), xc8k-b2-* (c2 is golden-search27c-x;
    c0 / c1 stay fresh for future search slices), the trace goldens.
- **both:** smoothing on top of the more-data arm with the better tune score, κ and prior re-tuned. Counts for the
  verdict only if smooth and more are each "useful"; otherwise report only.
- **Tuning (never on search rows):** every grid point is built from half A only (A's qad-gen positions, plus x for
  more, priors' bins from A) and scored on the tune set (B rows). Criterion: geometric mean over s4-3, s5-3, s6-3 of
  KL / KL(plan from A's unsmoothed Fisher). The arm's setting = the argmin; within 0.5% of it, the smaller κ / smaller w
  wins. Fisher-cost proxy on B's Fisher (Σ F_B · rowerr) reported beside it, report only. The chosen setting is then
  applied to the full qad pool (pools-v3, A ∪ B; bins from it) and scored once on the held-out set.
- **Verdict (held-out search-gen, 2,000-draw document bootstrap of the paired ratio of means):** an arm is
  **useful** if KL(arm's plan) / KL(base's plan) − 1 ≤ −3% with the 95% CI excluding 0, for at least two of s4-3,
  s5-3, s6-3, at equal or fewer bytes (avg bits ≤ base's + 0.002, else reported as unequal). Also reported: each
  plan vs its uniform twin, top-1, and rare-token KL (rows whose bf16 argmax has base-perm rank ≥ 20,000, w94's guard
  rows), report only. Borderline (CI touching 0): the number of held-out documents / rows that would put the CI's
  edge at 0 at the observed effect (n × (1.96 SE / |effect|)²), proposal only.
- **Best arm:** the useful arm with the lowest geometric mean of the three ratios (if none is useful, the arm with
  the lowest point estimate, labelled "not useful"). For it: w94's knap5 re-solve (curve HS_BUDGETS 3.125–7 by 1/8,
  HS_FLOOR_ALL=1, HS_FLOOR 3, scored on search-gen, `simjson94`-format JSON, knap5 KN_HEADGRID with KN_HG_DONORS
  3,4,5,6,7 at 2.0 / 2.5 / 3.0 / 4.0) vs w94's base grid. Report only.
- **Report-only extra:** the best arm with blocks re-ordered by the arm's block Fisher before the slab DP (a block
  permutation of the existing donors, no requant).

## 3. Not run

Two-sided LDLQ with a smoothed Fisher as H_out needs a requant (GPU): proposal only. New data: cost estimate only.

## Machines

Thor (CPU, restartable: per-block outputs, finished blocks skipped) for the Fisher passes and the scorer: donors and
the calib golden copied there from gpubox (LAN, streamed like `backup-gpubox.sh`); colo for knap5. No GPU, no 4070.
