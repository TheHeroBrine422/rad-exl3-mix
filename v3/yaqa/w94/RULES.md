# w94 rules: #124 gen-only head inputs (merge 11-08 A2b), #125 head-setting batch (A6)

Written 2026-11-08 ~5:05 PM CST (time corrected 5:12 PM CST; first commit 650436c), before any dump, headsim run or proxy arm. Committed alone before launch.
Data files fixed with this rule (sha256): `docs94.json` 1afe236b0179a4464b3551862ae1cd98668cb1fd1af69b81d0f04e33b1f9d3b5,
`blocks94.json` 42f0ab2a00bc03c7b76122eaf4b0b6ccbcbc23cf850f450cf3c53bd571834d25.
Smoke tests before this rule (no arm compared): headx94 on the CPU, 3 short documents, KL vs golden top-512 1.6e-4 /
2.6e-4, top-1 1.00 / 0.99; hq94 prep raw 237 s, K3 scale g 0.7364 (the real K3 donor's: 0.7369), C0 block 0 K3 rel
4.61e-4 (real donor's rowerr on that block: 4.81e-4), 27 s a block.

## #124 A2b — head inputs at scored positions only

**Why.** `headsim.py` priced every head plan (w88's knap5 grid, #34) on `336.x`: consecutive prefill rows of the calib-A
slice (fit pool), unscored positions included. The metric only scores gen rows. The old dump is gone (11-05 crash).

**Data (fixed now, `docs94.json` from `mkdocs94.py`):**
- one window of ≤ 512 consecutive golden rows per document, inside the first 8,192 tokens, seeded start;
- **search (held out):** the 103 calib documents of the retired golden-search27 (all search pool, so no fresh search
  document is spent): 46,712 rows, 38,960 gen;
- **qad (fit / pricing; the pool behind headslab's Fisher ranking):** 85 qad documents (seed 94): 36,828 rows, 33,172 gen;
- every row is stored with its gen flag. "search-gen" is the held-out set, "search-all" (same documents, consecutive,
  unscored rows included) stands in for the old prefill dump, "qad-gen" prices.

**Method.** `headx94.py`: bf16 forward of the checkpoint (HF modules, layer-streamed on card 0), final-norm output at
the window rows. `headsim94.py`: headsim's exact head-only KL, every row kept, document bootstrap in `report94.py`.
Donors: today's K3–K7 (`/data/headslab3`, corrected rowerr). Plans: `plan.json` (u4/u5/u6, s*-3/-4, b*) and a curve
(`headslab.py plan`, HS_BUDGETS 3.125–7 by 1/8, HS_FLOOR_ALL=1, HS_FLOOR=3).

**G1 (validity, before anything is read):** `gate94.py` on each dump, gen rows: mean KL vs the golden's top-512
≤ 0.01 and top-1 agreement ≥ 0.97. Fail → stop, nothing is reported as a result.

**Readout (reported, no pass / fail):**
1. Uniform u3–u7 and s5-3 / s6-3 / u5 / u6 head-only KL on search-gen, search-all and qad-gen; the move =
   search-gen / search-all (gen mask) and qad-gen / search-gen (pool).
2. Predicted gains s5-3 vs u5, s6-3 vs u6 (paired, 95% document-bootstrap CI) on each set.
3. **Does it change which plans win?** Per budget of the curve, the plan of lowest KL (the floor choice) and the lower
   hull, on each set. "Changes" = a different winner at any budget where the two winners differ by more than their
   paired 95% CI on search-gen.
4. w88's grid: knap5 prices a head option at r(K) × headsim KL with r = model head error / headsim uniform KL, so a
   uniform rescale cancels; the grid moves only through slab / uniform ratios. Reported: the slab / uniform ratio at
   each budget, search-gen vs search-all; budgets 2.0 / 2.5 need K2 / K2.5 donors that do not exist (said so).
- Pricing set from here on: qad-gen (fit / pricing), checked on search-gen (held out).

## #125 A6 — head quantizer settings, CPU proxy on the new rows (colo)

**Proxy (`hq94.py`, fixed now):** the head's rows in sensitivity order (`perm.npy`, as the donors), exllamav3's
regularize, LDLQ last-to-first under the calib `lm_head.input` H, σ 0.025 (today's donors), the CPU bitshift trellis
of v3/e3 (mul1x_i8, free start), exllamav3's global scale search with ourq's drift, per-row scale refit. Only 64
aligned 128-row blocks are quantized (`blocks94.json`: stratum A = blocks 0–15, all; B = 24 of 16–255; C = 24 of
256–1939; systematic). Head-only KL with only those rows changed is exact per stratum
(KL = −Σ p dz + log(1 + Σ p (e^dz − 1)), lse of the full bf16 head from headsim94); estimate = KL_A + 10 KL_B +
70.17 KL_C. K3 and K4, seed 94.

**Arms:**
- C0: one-sided, no mean-row (today's donor recipe; for the validity check)
- B: one-sided, mean-row (the baseline: mean-row was adopted for every donor on 11-02)
- T(σo): two-sided with H_out = the softmax-Fisher diagonal `fisher_qad_gen` (perm order), block-diagonal (patch 0008's
  objective), mean-row, σo ∈ {0.01, 0.1, 1}
- R4, R8: mean-row, then the top-4 / top-8 principal components of the rows removed (factors stored bf16), one-sided
- real: today's GPU donors' rows (headslab3 K3 / K4) on the same blocks (validity)

**G2 (proxy validity, before any arm is read):** (a) estimator: KL_est(real) within ±15% of headsim94's full-head
u3 / u4 KL on search-gen; (b) quantizer: KL_est(C0) / KL_est(real) in [0.80, 1.10] at K3 and K4 on search-gen (the CPU
trellis is 4–6% optimistic). Fail → arms are reported as "proxy unverified" and nothing is adopted.

**Selection of σo (one value, before judging):** the σo with the lowest geometric mean of KL_est over K3 / K4 on
qad-gen rows of the **odd-numbered documents** (selection half).

**Adoption rule per setting (T(σo*), R4, R8), vs B, paired on the same rows:**
- KL_est ratio ≤ 0.97 at **K3 and K4**, on **search-gen** and on **qad-gen even-numbered documents** (judging half);
- and the rare-token guard (rows whose bf16 argmax is outside the 20k most sensitive vocab rows) ratio ≤ 1.3 on
  search-gen at both K.
- If R4 and R8 both pass: R4 unless R8 is ≥ 3% lower than R4 at both K on search-gen (bytes and an extra tensor).
- If T and an R arm both pass: a combined arm is run after the verdict and reported; it is recommended only if it is
  not worse than the better single arm at both K on search-gen.
- Mean-row (B vs C0) is reported only (already adopted).

**τ (logit temperature) — on headsim94 (exact, GPU, real donors; no 27B eval):** τ ∈ {0.96, 0.98, 0.99, 1.01, 1.02,
1.04} on u3 / u4. τ* = the lowest KL on qad-gen (odd documents) per K. Adopt (fold into the head's stored output
scales: no format change) only if KL(τ*) / KL(1) ≤ 0.97 at K3 and K4 on search-gen and on qad-gen even documents.

**Runtime class:** two-sided LDLQ and τ: none (quantizer / folded scale). Rank-r: P+D (an extra bf16 r × 5120 and
248,320 × r tensor and two skinny GEMMs: a plugin / format addition, post-freeze at the earliest). DPCM / affine bias:
out of scope (no format field).

## Amendment 1 (#125-01, written 6:25 PM CST 2026-11-08, after the CPU proxy's verdict, before any GPU arm)

**What happened:** #125's G2 failed (K3: the additive stratified estimate is 1.32× headsim's full-head u3 for the real
donor's rows; the quantizer half passed: C0 / real 0.94 at K3, 1.00 at K4), so by the rule nothing is adopted from the
CPU proxy. The proxy's per-stratum ratios (exact for the rows they change; report-only) show two-sided σo 0.1 and rank-4
lowering the top 16 blocks' KL by 20–28% at both K, and mean-row *raising* it by ~15%. These are hypotheses now; this
amendment tests the two that need no format change on the real quantizer.

**Arms (gpubox card 0, repo `quantize/ourq.py` staged as is, fork with patch 0008 first on the path; all with today's
donor flags `--only-head --save-wq --head-perm perm.npy --qtf2 --frac-ext SRC --scaled-hessian --sigma-reg 0.025
--codebook i8`, same seed), K3 and K4:**
- C0': no mean-row (today's donors, re-made with the same code; reported against /data/headslab3 K3 / K4 as a check)
- B: `--head-mean-row`
- T: B + `--hout HD --hout-blocks group --sigma-reg-out 0.1`, HD/lm_head.safetensors `hout_diag` = `fisher_qad_gen`
  in sensitivity order (σo 0.1 = the proxy's selection on qad-odd)

**Score:** headsim94 exact full-head KL (uniform heads) on both dumps; document bootstrap as before.

**Rules:**
- T adopted for the donors iff T / B ≤ 0.97 at K3 and K4 on search-gen and on qad-even, and guard ratio ≤ 1.3 on
  search-gen at both K. Reported also: T / C0'.
- Mean-row: recommended for the donors iff B / C0' < 1.00 at K3 and K4 on search-gen; if B / C0' > 1.00 at both K on
  search-gen and qad-even, recommend dropping it (lead decides; it was adopted 11-02 on other rows / donors).
- If T passes, the donors take T's flags on top of the mean-row decision (T is always run with mean-row: if mean-row
  is dropped, T without it is untested and the recommendation says so).
- Rank-r: CPU proxy evidence only, reported; it needs an ourq change and a format addition (post-freeze).

## Amendment 2 (#125-02, written 7:00 PM CST 2026-11-08, after amendment 1's verdict, before this arm)

Amendment 1 read (report94b): T / B 0.880 / 0.885 (K3 / K4, search-gen), 0.863 / 0.806 (qad-even), guard 0.97 / 1.02
→ T passes; but B / C0' 1.074 / 1.076 (search-gen), 1.077 / 1.123 (qad-even) → by amendment 1, recommend dropping
mean-row. T was only run with mean-row, so the donor recipe "two-sided without mean-row" is untested.

**Arm T0** (same code, fork, flags and seed): `--hout HD --hout-blocks group --sigma-reg-out 0.1`, no `--head-mean-row`,
K3 and K4; headsim94 on both dumps.

**Rule:** the donors take T0 iff T0 / C0' ≤ 0.97 at K3 and K4 on search-gen and qad-even, guard ratio ≤ 1.3 on
search-gen, and T0 / T ≤ 1.00 at both K on search-gen. Otherwise the donors take T (amendment 1).
