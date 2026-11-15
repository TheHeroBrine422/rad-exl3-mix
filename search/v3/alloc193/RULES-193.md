# RULES-193 — alternative allocators for the #183 search (w172, 11-15)

Written 2026-11-15 ~2:05 AM CST, before any #193 candidate, model fit or replay exists. Agenda: BACKLOG "Pooled
allocation model + anomaly finder" parts 1–4 (user 11-15 2:00 AM CST). Committed alone; amended only by dated sections
committed before the data they affect.

## 0. Common rules

- **Judge:** only the live search's normal rule. `decide` files → stage 1 on the active slice (early stop / audit as
  generated) → tentative → confirmation on its pair, z < −1. Success = a confirmed accept (new incumbent) or a confirmed
  pool join; reported separately. `probe` files → fit rows only (no decision). Nothing else counts as a win.
- **Training / selection / tuning data (allowed):** rows and results on the search slices s173a–e, s186a, s186c, s190a;
  the s186e re-selection; the retired confirm slices s186b, s186d; the prior rows (w158 waves 1–2, label `fit129`);
  the state's decision / submit events on those slices; the assignments.
  **Never used** for training, selection, tuning or choosing what to submit: s190b (live confirm slice; this includes
  stage-2 decisions and confirm evals on s190b, and the live fit outputs, which contain s190b rows), t183, t186n,
  golden-final750, any verify slice, public, any monitor read. Every data loader in `alloc193/` filters by an explicit
  allow-list of slice names (rows whose slice is not on the list are dropped and counted). Where fit.py's model is needed
  (curvature, sensitivity-optimal K, byte compensation, ranker baseline features), it is **refitted by me** on the
  allowed rows (the driver's latest fit inputs b1/b2/b3 with every non-allowed label dropped), never read from the live
  `fits/*.out.json`.
- t183 / t186n are reported for accepted candidates only, as the tracker does; I do not request extra reads on them for
  `decide` files and do not look at probe t183 reads.
- **Inbox budget:** ≤ 6 files of mine in `drv/inbox/` + outstanding at any time (and never more than the driver's
  inbox_max allows together with others' files); ≤ **40 decide** files and ≤ **15 probe** files in total. Every file's
  `note` = `#193 <allocator> | <hypothesis> | code <commit>`. Bytes are checked against the driver's `window()` /
  probe rule with the live menu before a file is written (write to a temp name in `inbox/`'s parent, then `mv`).
- Sizes for decides: the active sizes 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0 (no stopped size).
- Per-allocator decide budgets below (sum 40); a budget not used by one allocator is not moved to another without an
  amendment.

## 1. `curve` — pooled curve model (agenda part 1) · decide ≤ 16, probe ≤ 3

- **Hypothesis:** the per-tensor allocation is a smooth function of size; fitting that curve to *every* allowed eval at
  once (each size's evals inform the others) gives allocations at least as good as each size's incumbent, at sizes where
  the size's own search is noisy.
- **Model:** per tensor t (338 incl. head / embedding), C_t(s) = s + o_t + β_t (s − 4) + γ_t · min(0, s − 2.5) (kink
  below 2.5). log KL_i = a_{label(i), size(i)} + κ_{part(i)} · Σ_t c_t(s_i) (K_ti − C_t(s_i))², with c_t(s) the
  curvature of log KL in K_t from my fit.py refit at the size's incumbent (second difference over menu steps, floored
  at a small positive value), κ_c = 1 and κ_x one fitted part scale. Floor censoring: a tensor at its lowest menu option
  carries the one-sided information through the quadratic (K − C > 0 there); the sub-1 options of menu-v2 are ordinary
  options. Parameters (o, β, γ) fitted by weighted least squares (weights 1 / (se² + 0.02²)) with a ridge prior centred on
  the refreshed consensus curve (§2's local-linear K_t(s), linearized to (o, β, γ)); the ridge strength and κ_x by 5-fold
  cross-validation grouped by eval tag, on allowed rows only.
- **Allocation:** per size, min Σ_t c_t (K_t − C_t(s))² over the menu under the size's byte window (Lagrangian bisection,
  then top-up to the window's top, consensus192's rule). One decide per active size (8); a second round (≤ 8) only after
  a refit that includes round 1's stage-1 rows and only at sizes where round 1 was not level-or-worse at stage 1 by more
  than +1 % (no repeat of a clear loser). Probes (≤ 3): in-between sizes where #192 had no probe (from 2.125, 3.125, 4.375, 5.25), as fit rows
  for the curve between the decided sizes.

## 2. `cons` — refreshed consensus (as #192) · decide ≤ 8

- **Hypothesis:** #192's consensus helped at 3.0 and tied at 2.5; incumbents changed since (2.5 thompson-01480, 3.0
  manual-01487, 3.5, 4.0, 4.5, 5.0 newer). Re-pooling with the newer models moves the curve to them.
- **Method:** consensus192's local-linear fit (bw 1.0, rounding unchanged) with sources = s186e re-selection models within
  3 % of the size's best (weights as #192) **plus** the current pool members of each size scored on s190a (weight
  e^(−Δ / 1 %) vs the best member on s190a, Δ = log KL ratio). Only s186e and s190a scores; no s190b. Decides only at
  sizes whose assignment differs from the current incumbent in ≥ 10 tensors.

## 3. `anom` — anomaly finder (agenda part 2) · decide ≤ 12

- **Data:** every decision with a measured log r on allowed slices: stage-1 decisions (incl. early-stopped, with their
  se), stage-2 decisions on s186b / s186d, and the s186e re-selection rows (paired vs the size's incumbent). Never s190b.
- **Features:** groups g = 9 tensor types × 8 depth bands (8 layers each) + head + embedding (74). x_jg = (bytes-weighted
  K of group g in the candidate − in its reference) × (g's share of trunk bytes): the bits the candidate moved into g.
- **Model:** weighted ridge (weights 1 / (se² + 0.005²)), y_j = log r_j; intercept per size; β_g pooled over sizes plus
  per-band deviations δ_{g,band} (bands b1 = 2.0–2.5, b2 = 3–4, b3 = 4.5–6). Two versions: (i) y = measured log r
  (marginal value of bits in g), (ii) y = measured − my fit.py refit's predicted log r (where fit.py is wrong). Ridge
  strength by grouped CV. SE by bootstrap over decisions (500).
- **Flags:** a group-band whose version-(ii) coefficient has |β / se| ≥ 3, or whose version-(i) sign is opposite to
  fit.py's predicted marginal value with |β / se| ≥ 3. At most 6 flags tested, largest |β / se| first.
- **Test per flag:** at the flagged band's sizes (the size with the most decision rows first), the current incumbent with
  every tensor of g moved one menu step in the direction the coefficient favours, bytes rebalanced by my fit.py refit's
  cheapest opposite moves outside g, filled to the window. ≤ 2 decides per flag (two sizes of the band).

## 4. `rank` — learn-to-rank (agenda part 3) · offline only, decide 0

- XGBRanker (xgboost 3.4.1, rank:pairwise and rank:ndcg, max_depth 2–3, ≤ 100 trees, eta 0.1, seed 193), groups =
  (size, slice, reference). Labels: measured stage-1 log r (early-stopped included as their prefix log r; ndcg uses
  within-group rank bins). Features: RULES-187 arm d's features + the live pre-submit `pred_logr` (the driver's own
  out-of-sample prediction) + the candidate's distance to the §2 consensus curve (Σ bytes-weighted |K − curve|, and by
  group type).
- **Replay:** time-ordered as RULES-187 §2 (train on candidates whose results are ≤ N, predict (N, N + 20]), on (a) w165's
  snapshot with arm a's replay predictions as the baseline, and (b) the full live state up to my snapshot with the live
  `pred_logr` as the baseline (stage-1 decisions on allowed slices only). Metrics and winner rule: RULES-187 §3 / §5
  unchanged (S = mean ρ over b2, b3; S gain ≥ 0.10, ρ_b1 not worse by > 0.05, 80 % bootstrap CI > 0), plus within-group
  pairwise accuracy (reported, not gating).
- No inbox file comes from the ranker. Outcome = recommend / not as a driver move ranker.

## 5. `boot` — uncertainty-targeted probes (agenda part 4) · decide ≤ 4, probe ≤ 12

- Bootstrap (200) the §2 consensus over its source models (resample models within size) and over slices (s186e vs
  s190a weights): per tensor and size a K band (10–90 %). Tensors with a band ≥ 0.5 wide, or whose consensus K differs
  from my fit.py refit's sensitivity-optimal K by ≥ 0.5 with the band excluding fit.py's K, are "uncertain".
- Single-tensor ± moves are below the noise (one tensor ≈ 0.3 % of bytes), so a probe moves **a set**: at one size, the
  incumbent with the uncertain tensors moved toward fit.py's K (probe "+"), or toward the consensus' far side ("−"), bytes
  balanced as in §3. Probes: ≤ 12 (≤ 6 sizes × ±). My reading of a probe = its paired log r vs the size's incumbent on
  the same active slice (s190a; allowed). If a probe set shows z < −2 there, its allocation may be re-submitted as a
  `decide` (≤ 4).

## 6. Offline comparison and report

- Every allocator's predicted-vs-measured stage-1 log r on its own candidates is reported; the curve model and the
  ranker are also compared against fit.py in the replay of §4 (ρ per band).
- Report per allocator: files submitted, stage-1 / stage-2 outcomes from `drv/driver.log`, replay metrics, and a
  recommendation (adopt as a driver move kind / keep as a tool / drop).

## Amendment 1 — 2026-11-15 2:35 AM CST (before any curve fit, curvature computation or curve candidate)
- §1 curve: the bowl scale is fitted, not fixed: κ_{band} (b1 = sizes < 2.75, b2 = 2.75–4.25, b3 > 4.25), log κ ~ N(0, 1),
  times κ_x for x-part rows. Reason: fit.py's predicted gains are overstated 1.5–7× by band (RULES-187), so its
  curvature cannot set the scale. The size level is a piecewise-linear h(s) (knots at the menu's sizes 1.25 … 6 plus
  the label offsets), since one label holds several sizes. Rows: the fit rows of every allowed label, all sizes 1.25–6
  (curvature at sub-2 sizes from those sizes' incumbents, constant beyond the end sizes); 7.0 rows dropped.
- §3 anom: the CV grid for (λ, λ_dev) is {0.1 … 1e3} × {0.01 … 1e4} (the committed values were on the wrong scale for
  x in bpw units; seen on the first CV run, no flag or candidate made from it).

## Amendment 2 — 2026-11-15 2:03 AM CST by the system clock (UTC−6) (before any fit.py-optimizer candidate exists)
- **Time correction:** the times written above ("~2:05 AM", "2:35 AM") are wrong; by the system clock the rules were
  committed 1:51 AM CST (2bf24b03) and amendment 1 at 1:55 AM CST (d2e13776). Git times are the record.
- **Lead note 2:25 AM CST (agenda item 5): cautious optimizers on fit.py.** Four allocators, all on **my** fit.py refit on
  the allowed rows (never the live fit), all starting from the size's current incumbent, all at the sizes 2.5, 3.0, 4.0,
  5.0 (one decide per allocator per size), bytes in the window as usual:
  - `tr` (trust region): fit.py's mckp optimum with each tensor restricted to options within ±0.25 K of the incumbent
    (head / embedding: ±0.25 too). Hypothesis: the `solve` move's few accepts are the optimizer's curse; a bounded step
    keeps fit.py's direction and drops its extrapolation.
  - `rob` (robust quantile): from a candidate set (trust-region optima at radius 0.125 / 0.25 / 0.5 under the posterior
    mean and under each of 16 posterior draws of the refit), pick the one with the best 80th percentile of predicted
    log r vs the incumbent over the draws. Hypothesis: the pessimistic quantile avoids directions the model is unsure of.
  - `recal` (recalibrated): objective = s_band · fit.py log r + Σ_g β_g x_g with s_band and β_g from §3's version-(ii)
    regression (the slope of measured on fit.py-predicted log r per band, then the per-group residual coefficients),
    trained on the allowed paired outcomes (stage-2 rows only from s186b / s186d; never s190b), optimized in the same
    ±0.25 trust region (linearized, mckp). Hypothesis: correcting fit.py's known group biases makes its optimum real.
  - `ens` (ensemble): objective = mean of fit.py's log r and the §1 curve model's log r (κ_band Σ c (K − C)² difference),
    ±0.25 trust region. Hypothesis: two models with different errors cancel part of each other's exploitation.
- **Budget re-split (total unchanged: 40 decide, 15 probe):** `cons` 5 (all used), `curve` 10, `anom` 8, `boot` 1,
  `tr` 4, `rob` 4, `recal` 4, `ens` 4. Probes: `curve` 3, `boot` 12.

## Amendment 3 — 2026-11-15 2:16 AM CST (system clock; before any `recal` file is submitted, before any anom result)
- `recal`: β_g enters the objective only where its version-(ii) |β / se| ≥ 2 (else 0). Seen before submitting: with all
  222 group × band coefficients the optimum moved ~310 tensors to the trust-region edge on noise (fit.py itself
  predicted +2.6…+4.7 %); with |z| ≥ 2 (14–20 groups per band) 223–280 tensors move. Env `RECAL_ZMIN=2`.
- `anom`: direction of a test = −sign(β) of the flagging version (ii: fit.py overvalues the group → fewer bits). The flag
  attn_output@5 b3 is not tested: its group is 2 small tensors (a one-step move shifts ~0.0002 bpw; expected effect
  ~0.03 %, far below the stage-1 SE). 7 anom decides instead of 8; the spare one is not reassigned.
