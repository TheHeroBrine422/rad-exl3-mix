# Red-team review of the V3 pre-freeze stack (lens: redteam, 2026-11-09)

This is an adversarial review of the adopted stack: H_out at every level on the block-diagonal path, swj, C9 on the writers, tile CD on attn_qg, sub-1 input damping 0.05, σ 0.2 at K0.125, and #117 block order. It covers only items that change the per-level library, quantizer settings, codebooks, level menu or knapsack inputs. Each item states the evidence, the cheapest check that would expose the risk, and what to do if the check fires. GPU tests are proposals for the lead (gpubox card 0, or the 4070 for 0.8B work). I ran only CPU work on colo.

**What I ran and read (new evidence in this report):**
- **H_out concentration** (`/data/work/ideas-redteam/houtstat.py`, `houtperm.py`, output `houtstat.json` on colo): over the 400 block-normalized Σdy² diagonals I computed the participation ratio (PR) and an in-128-block spread. The spread is 0.5·log2(AM/GM) per 128 rows, with a floor at 0.01·mean.

  | family | PR (median) | in-128 spread (bits) | notes |
  |---|---|---|---|
  | writers (down / o / out_proj) | 0.82–0.94 | 0.01–0.03 | nearly flat |
  | gate | 0.52–0.68 | ~0.08 | |
  | in_proj_qkv | 0.16–0.23 | 0.20–0.28 | up to 5.1% of rows below 1% of the mean (L0–7) |
  | in_proj_z | 0.08–0.30 | 0.25–0.29 | |
  | q_proj (L0–7) | 0.09 | 0.32 | |

  Whole-tensor spread for qkv / z / q is about twice the in-128 spread (qkv 0.47 vs 0.21 bits). Sorting rows makes blocks almost homogeneous (≈0.002 bits).
- **Scale polish is H_out-blind** (`exllamav3-frac/.../quantize.py` L1730–1800, L1985–1990): `refit_scales` runs after every quantization, including the two-sided path, and receives only H_in. The per-output-column `svh` refit is unaffected by a diagonal H_out. The per-input-row `suh` refit solves ((QQᵀ)∘H)r = rowsum(Q∘HW), which sums over output columns without H_out weights.
- **C9 has no K schedule** (`rxq/post.c9_rows`): c = (1−e)^(−α/2), with e clipped at 0.5. At gdn_out α = 3, e = 0.35–0.5 gives c = 1.9–2.8. The w80 log at K0.5 shows o_proj e_mean 0.348.
- **suh underflow is tolerated** (`rad-quant-at-home/TESTS.md` L771): one of 5120 `suh` entries underflowed to f16 zero on a real 27B q_proj at K0.5, and the validity rule was relaxed to allow ≤1% exact zeros.
- `out_scales = always` (config assert, #48) forces `force_out_scales=True`, so the fork's H_out-derived output-scale path is never taken.

**One reinterpretation:** the brief's "27B diagonal run wasting 8.5 h" refers to a collection that is finished (#33-24, 1,923 sequences). I read that topic as "what would waste the ladder build and the remaining 27B runs" (theme I).

---

## A. H_out at every level

**R1. H_out has never been tested inside a mixed-K mix**
- **Mechanism:** every H_out pair (#33-28) swapped a uniform-K trunk. H_out shapes each tensor's error toward directions that are insensitive downstream, with the downstream taken as the bf16 model. In a knapsack mix the neighbours sit at other K values. Per-level writer asym won every uniform test and then lost by 8–24% in mixed-K mixes (#111). H_out is less sibling-specific than asym, but nobody has checked.
- **Source:** new. Analogue of #111; GEMQ/MixQuant (2605.23078, 2607.23047) on operating-point mismatch.
- **Evidence:** all K2.5 and K4 pairs are uniform swaps (0.972 / 0.992 / 0.976; 0.961 / 0.967). There are zero mixed-K rows.
- **Fit:** per-level setting, applied to every level.
- **Test (27B, card 0, ~2–2.5 card-h, no new quants):** splice existing levels into two 50/50 mixes, even layers at K4 and odd at K2.5 (or split by family). Arm 1 uses bn + bn4; arm 2 uses base + base4. Repeat with the seed-1 twins (bns1 / bn4s1 vs s1 / base4s1). Score on a fresh slice.
  - Holds if geo ≤ 0.985 and both pairs < 1.
  - **If it fires (geo ≥ 1.00):** limit H_out to the K bands the knapsack uses as whole blocks, or price an H_out mix penalty. Do not ship "every level" blind.
- **Gain / risk:** P(fire) ≈ 15%. If it fires it avoids roughly +2–3% at 2.5 bpw, so the expected value is about −0.4%. This item is decisive for every level.
- **Conflicts:** #33-28, #111.

**R2. No KLD evidence for H_out below K2.5**
- **Mechanism:** the i4 sub-1 levels, i8 K1.125–2.375 and the 1.0–2.0 release sizes all use H_out on the strength of f2 alone. f2 is the objective H_out minimizes, so it is circular evidence (i4: −2.6% at K0.125 up to −13% at K2, #115).
- **Source:** new.
- **Fit:** per-level.
- **Test (27B, ~3 card-h):** one K1.5 seed pair, each level quantized with blocks at ~37 min plus pack and scores.
  - Criterion: ratio < 1 in both pairs.
  - **If it fires:** set apply_k_min to the lowest passing K.
- **Gain / risk:** ≈0 at 2.5 bpw; 0–2% at 1.5–2.0. P(fire) ≈ 15%.
- **Conflicts:** #33-28, #115.

**R3. No KLD evidence for H_out above K4 (K5–K7 i8, mul1 K6.5–8)**
- **Mechanism:** near-lossless levels, where the i8 grid floor binds near mul1 K6.85 (#94). Two-sided feedback changes the target statistics the int8 grid has to represent.
- **Source:** new.
- **Fit:** per-level.
- **Test:** first the CPU check in R29. Then, only if R29 shows a shift, one K6 KLD pair (~3 card-h). **If it fires:** turn H_out off above K5.
- **Gain / risk:** 0 at 2.5 bpw; protects the 5.0–7.0 sizes. P ≈ 10%.
- **Conflicts:** #33-28, #94, #109.

**R4. "Every level" rests on a missed rule, read post hoc**
- **Mechanism:** the K2.5 rule missed (geo 0.9801 vs ≤ 0.98). The decision to adopt was taken before the verify read, and the verify rule only asked for direction.
- **Evidence:** with document error included (#116) the combined SE is 0.0116, so the K2.5 z is −1.74. Verify gave geo 0.9751, but a true zero effect passes "geo < 1" about half the time.
- **Source:** #116; the audit in LOG 11-08 6:20 AM.
- **Fit:** evaluation rule that decides a per-level setting.
- **Test (CPU, minutes, stored rows):** fixed-effect meta-analysis of the search, verify and K4 rows with seed and document terms.
  - Expect the pooled effect to be about −2.5% with a 95% CI excluding 1. This is likely real.
  - **If the CI crosses 1:** keep H_out only where it costs nothing (it is now +2.5 min a level). In that case the cost-based decision stands anyway.
- **Gain / risk:** this is about evidence quality, not KL. Low risk.
- **Conflicts:** #33-28, #116.

**R5. σ_out = 0.01 was never swept at 27B**
- **Mechanism:** the output damping comes from the 0.8B (#33). On 27B the q / z / qkv diagonals are very concentrated (PR 0.08–0.30), and up to 5.1% of qkv rows sit below 1% of the mean. With σ_out = 0.01 those rows get almost no weight, so their quantization error is effectively free. If their Σdy² estimate is low only because of noise, the loss is real.
- **Source:** new. Damping logic from #114-03 (which swept σ_in only).
- **Fit:** per-level setting (`hout.sigma_out`).
- **Test (colo, ~4 CPU-h):** w87's two-sided held-out proxy with H_out_A built from even 128-sequence blocks and scored on H_out_B from odd blocks. Use the halves the collector saved (#33-17). Sweep σ_out ∈ {0.003, 0.01, 0.03, 0.1, 0.3} at K2.5 / K4 on 27 tensors.
  - Adopt if the gain is ≥2% held out and agrees on ≥⅔ of tensors.
- **Gain / risk:** 0–0.5% KLD (proxy gains shrink about 5× to KLD, per #117). Low risk.
- **Conflicts:** #114-03, #33-28.

**R6. H_out floor rows may be estimation noise**
- **Mechanism:** gen-masked Σdy² puts some rows below 1% of the mean. If the two halves disagree on these rows, the floor reflects sampling, not sensitivity.
- **Source:** my houtstat; heavy-tail finding #33-27.
- **Fit:** per-level (an H_out floor clamp).
- **Test (colo, minutes, CPU):** for rows below 1% of the mean, per-row ratio of even and odd halves.
  - **If the median |log ratio| > 0.5:** clamp the diagonal to max(d, 0.05·mean) and run R5 with the clamp as an arm.
- **Gain / risk:** this is a diagnostic. If it fires, expect about 0.2% KLD.
- **Conflicts:** #33-27.

**R7. H_out was measured at the bf16 operating point**
- **Mechanism:** Σdy² comes from bf16 forward and backward passes. At K ≤ 2 the quantized model's sensitivities differ. GEMQ gained 9.66 → 9.01 ppl by re-estimating on a 2-bit model, and Q-Strata saw 13.72 → 12.14.
- **Source:** 2605.23078, 2608.30564.
- **Fit:** per-level.
- **Test (0.8B, 4070, ~3 h):** collect Σdy² on a K2 quantized model and on bf16, quantize K2 with each, compare KL.
  - **If the quantized-point H_out wins by ≥3% at 0.8B:** note it for V4. A 27B recollection costs about 36 card-h and is not worth it pre-freeze.
- **Gain / risk:** 0 pre-freeze; this informs V4 only.
- **Conflicts:** #33-24.

**R8. The collection domain (qad 2048-token gen-masked windows) may differ from private agentic text**
- **Mechanism:** rows that matter for tool-call or JSON tokens, or for long-range GDN writes, may sit in the floor.
- **Evidence:** H_out collected on pisess agentic text was +3.5% worse on calib (#33-19), but nobody measured it on agentic text itself. Calib-fitted statistics keep only 52–91% of their gain on xcorp (#93).
- **Source:** #33-19, #91, merge B4.
- **Fit:** per-level.
- **Test:** the readout in R65.
  - **If agentic docs show a ratio > 1:** apply the floor clamp from R6 or raise σ_out on GDN qkv / z.
- **Gain / risk:** P ≈ 15%. If it fires, ~1% on the agentic share.
- **Conflicts:** #33-19.

**R9. Per-family attribution is unknown at 27B**
- **Mechanism:** at 0.8B the split was gdn_in 39%, mlp_in 44%, attn_in 13%, writers 3% (#82). If 27B has the same split, H_out on writers is near zero gain and adds only noise.
- **Fit:** per-level.
- **Test:** none needed. Writers' H_out is flat (PR 0.82–0.94, in-block 0.01–0.03 bits), so codes barely change (#118 writer ratio 0.99).
- **Gain / risk:** 0. This is a null finding; keep as is.
- **Conflicts:** #82.

**R10. The between-block H_out spread cannot be exploited (closed)**
- **Mechanism:** about half the qkv / z / q spread lies between 128-row blocks. The block-diagonal path is exact for a diagonal H_out, but it cannot move error budget between blocks. Interleaving rows across blocks would raise in-block spread from 0.21 to 0.47 bits. However, for qkv / z / q the 128-row blocks coincide with heads, and permutations within a head cannot change block membership.
- **Source:** my houtperm; consistent with #96 (per-head K at about 0.04 bpw, closed).
- **Fit:** format change. Not actionable.
- **Gain / risk:** none. Do not re-propose.
- **Conflicts:** #96.

**R11. The block path has not been checked at quarter masks**
- **Mechanism:** the #115 gate ran at K4 and at i4 integer K. Quarter masks (0x8888 / 0xEEEE) under ldlq_2hess_blocks were never compared against the dense path.
- **Source:** #115.
- **Fit:** per-level (correctness).
- **Test (card 0, ~20 min):** f2calc blocks vs dense at K2.25 / 2.75 / 3.25 on L10.
  - Criterion: every tensor within ±0.5%.
  - **If it fails:** fix 0008 before the ladder.
- **Gain / risk:** correctness check. P ≈ 3%.
- **Conflicts:** #115.

**R12. H_out × C9: low risk, by evidence**
- **Mechanism:** C9 was only tested on non-H_out swj levels (#112-02). But writers' H_out is flat (R9), so the codes C9 sees are essentially unchanged.
- **Fit:** per-level.
- **Test:** rides on the w92 full-stack pair.
- **Gain / risk:** 0. Null; no action.
- **Conflicts:** #112.

## B. Post-quantization scale fits (polish, C9, swj)

**R13. The scale polish (`refit_scales`) undoes part of the H_out shaping**
- **Mechanism (verified in code):** after two-sided LDLQ, the fork's polish refits `suh` / `svh` by alternation with H_in only. The `svh` column step is unaffected by a diagonal H_out. The `suh` row step weights every output column equally, so for q / z / qkv (PR 0.1–0.3) it is dominated by low-sensitivity columns and pulls input scales toward the one-sided optimum. The #118 finding that H_out levels show ×1.076 one-sided proxy (GDN qkv 1.17, z 1.22, attn_qg 1.13) fits a polish that serves the wrong metric.
- **Source:** new. JARQ (2609.38599) and Schur Replay (2609.36654) on refitting scales under the right metric.
- **Fit:** per-level (fork patch: h-weighted rowsum in the row step, plus a stamp change).
- **Test (card 0, ~0.3 card-h, then 1.5 card-h):** rxq on L10 and L39 at K2.5 and K4 with three variants: polish as now, `no_refit`, and h-weighted polish. Compare f2calc results.
  - **If the h-weighted polish lowers f2 by ≥1% on qkv / z / q with no family worse:** run one spliced 27B reader pair at K2.5.
  - **If `no_refit` beats the current polish:** the polish is actively harmful with H_out.
- **Gain / risk:** 0.2–0.8% KLD at 2.5 bpw (±0.4%). Low risk, small code change. Top item.
- **Conflicts:** #115, #33-28, #39.

**R14. Scale underflow leaves dead input channels**
- **Mechanism:** the refit underflowed 1 of 5120 `suh` entries to f16 zero (K0.5 q_proj). The validator now accepts up to 1% zeros, so a dead input column can ship silently in any level.
- **Source:** rqah TESTS.md L771.
- **Fit:** per-level. Clamp refit scales to ≥ max(f16 min normal, 1e-3·median), or keep the pre-refit value if it underflows.
- **Test (colo, minutes):** scan `suh` / `svh` of v3ladder2 K2.5, the bn levels, and the w80 / w86 outputs for zeros and subnormals.
  - **If any appear at K ≥ 1:** apply the clamp before the freeze.
- **Gain / risk:** tiny KL; this is a correctness fix that changes the stamp.
- **Conflicts:** none.

**R15. C9 has no K schedule (c up to 2.8 at sub-1)**
- **Mechanism:** c = (1−e)^(−α/2) with α_gdn_out = 3, tuned at K2 / K3 on the 0.8B. At K0.125–0.75, e approaches the 0.5 clip, so every gdn_out row is inflated by about 2–2.8×. The sign also looks wrong: Wiener-style optimal scaling shrinks when error is high.
- **Source:** new, from post.c9_rows and the w80 K0.5 log (o_proj e_mean 0.348, c 0.72–0.996 at α −1).
- **Fit:** per-level.
- **Test (colo, minutes):** C9 statistics from w86's dry-run K0.5 out_proj (or a CPU stand-in job on L10 out_proj at K0.125 / 0.5 / 1), then a 0.8B KL at K0.5 / K1 with α vs 0 (4070, ~1 h).
  - **If c_mean > 1.3 at any K < 1.5, or KL is worse:** use α(K) = α·clip((K−1)/1, 0, 1), or α = 0 below K1.5.
- **Gain / risk:** 0 at 2.5 bpw. At 1.0–1.5 sizes it could avoid several % of harm. P ≈ 50%. Very cheap.
- **Conflicts:** #112.

**R16. C9 α values came from the 0.8B and sit at the grid edge**
- **Mechanism:** the 0.8B searched gdn_out α = 3 at the edge of its grid (#112-01: α 4–4.75 was only 0.1–0.4% better). At 27B only one α triple was tested (geo 0.9925, PARTIAL). The 0.8B has tied embedding / head (about ⅓ of its parameters), so its writer factor differs.
- **Source:** #112.
- **Fit:** per-level.
- **Test (27B, fixed codes, no requant, ~1 card-h per arm):** gdn_out α ∈ {1.5, 3, 4.5} on swj K2.5 levels.
  - **If a neighbour wins with z ≤ −2:** change α.
- **Gain / risk:** ±0.2%.
- **Conflicts:** #112.

**R17. swj at K ≥ 5 costs card time for about zero gain**
- **Mechanism:** gains are −6.5% at K2.5 and −1.1% at K4 and fall with K. The swj passes cost about 53 card-h for the whole ladder (w86). The 19 levels above K4.5 (i8 K4.625–7, mul1) are about 35% of that.
- **Source:** #104.
- **Fit:** per-level (swj k_max = 4.5). The knapsack already prices the swj factor as 1.0 from K5.
- **Test:** none needed.
  - **If anyone doubts it:** run one K6 swj fixed-code pair (~1 card-h).
- **Gain / risk:** saves ~18 card-h; KL change about 0.
- **Conflicts:** #104 ("stays adopted, never hurts").

**R18. swj has never been tested at sub-1**
- **Mechanism:** a joint gate / up refit through SwiGLU on very coarse codes (K0.125–0.875) is an extrapolation. The fit can amplify scales much as C9 does.
- **Fit:** per-level.
- **Test (0.8B, K0.5 / K1, ~1 h):** swj vs no swj.
  - **If it is worse:** set swj k_min = 1.
- **Gain / risk:** 0 at 2.5 bpw; protects the 1.0–1.25 sizes.
- **Conflicts:** #104.

**R19. swj may depend on the mix**
- **Mechanism (unverified):** if swj's whole-model fit uses activations from the uniform-K quantized trunk, its scales are tuned to upstream K = level K. In a mix the upstream differs. The effect is smaller than #111 but has the same failure mode.
- **Source:** #111.
- **Fit:** per-level.
- **Test:** read swj.py's forward (minutes).
  - **If it uses the quantized trunk:** run one uniform K2.5 pair with a bf16-input swj fit vs the current fit (1.5 card-h), and keep the mix-independent variant unless it loses by more than 1%.
- **Gain / risk:** protects about half of a 6.5% gain. P ≈ 30% that it uses the quantized trunk.
- **Conflicts:** #104, #111.

**R20. swj assumes gate and up share K**
- **Mechanism:** swj is fitted jointly at one K. If the knapsack or the mix builder ever unties gate and up K, the scales are mismatched.
- **Fit:** knapsack input.
- **Test (code review, minutes):** confirm knap5 prices gate_up as one item (V2's fused entry), and add an assertion in the mix builder.
- **Gain / risk:** guardrail.
- **Conflicts:** §1d "fused GEMMs force K ties".

**R21. Hessian-derived output scales are unused (null)**
- **Mechanism:** `out_scales = always` disables the fork's `sensitivity_out_scales` even when H_out is present. Diagonal output scales lost before (#33-06: K2 +10–15%).
- **Gain / risk:** none. Do not re-propose.
- **Conflicts:** #48, #33-06.

## C. Sub-1 and low-K settings

**R22. Some K0.125 options are worse than a zero tensor**
- **Mechanism:** the GPU search on unit-scale gate tiles has rel_mse 1.0141, worse than all-zero (w77). The CPU proxy is about 13% optimistic at K0.125. The knapsack may buy K0.125 for tensors where zero bytes (or bf16 zero plus bias) is better.
- **Source:** w77, #114.
- **Fit:** knapsack input plus the level menu.
- **Test (card 0, ~10 min):** real rxq K0.125 on 9 tensors, true H-weighted error relative to zero.
  - **If any tensor's error is ≥ 0.97:** price that option at the zero-tensor error, or drop it.
- **Gain / risk:** 0 at 2.5 bpw; 0.5–2% at 1.0–1.25.
- **Conflicts:** #90, #114-02.

**R23. The scale search may be pinned at its lower grid edge at sub-1**
- **Mechanism:** the scale grid starts at 0.5. At K0.125 the search needs to shrink below that (w77). If many tensors pick the grid floor, the level is under-shrunk.
- **Source:** #94-style smoke-vs-real lesson.
- **Fit:** per-level.
- **Test (log scan, minutes):** histogram of chosen g_scale at K0.125–0.5 in the w80 / w86 outputs.
  - **If more than 10% sit at the floor:** extend the grid to 0.25.
- **Gain / risk:** 0 at 2.5 bpw; up to 1% at 1.0 bpw. P ≈ 30%.
- **Conflicts:** #114-02.

**R24. σ 0.2 at K0.125 per family (near-null)**
- **Mechanism:** the w79 family optima vary (attn_v prefers 0.05, ssm_z 0.3). No family gains more than 1% over 0.2 (#114-02), and the proxy is unreliable at this level.
- **Gain / risk:** none. Keep.
- **Conflicts:** #114-02, #114-03.

**R25. One sub-1 input damping value of 0.05 (null)**
- **Mechanism:** a single value for all input families. #114 found at most +0.63% from per-family values.
- **Gain / risk:** none. Keep.
- **Conflicts:** #114, #38.

**R26. The full sub-1 stack has never been KLD-tested**
- **Mechanism:** H_out, swj, C9, tile CD and σ settings all apply at once at K0.25–0.875. Each was tested only separately and only by proxy.
- **Fit:** per-level.
- **Test (27B, ~1.5 card-h):** one spliced layer band at K0.5, full stack vs plain, KL.
  - **If the full stack is worse:** bisect, starting with R15 and R18.
- **Gain / risk:** 0 at 2.5 bpw; relevant to 1.0–1.25.
- **Conflicts:** #90.

**R27. Tile CD on the i4 / salted path is untested**
- **Mechanism:** tests and gpucheck disable tile CD at i4. It is unclear whether v3.toml applies tile CD at i4 sub-1 levels.
- **Fit:** per-level.
- **Test (CPU stand-in, minutes):** q_proj at K0.5 i4 with tile CD on, decode via rxq.deq, compare with the job's wq.
  - **If they differ, or the run errors:** restrict tile CD to i8 levels.
- **Gain / risk:** correctness check.
- **Conflicts:** #77.

**R28. The i4 K1 / i8 K1.125 boundary is priced from different curves**
- **Mechanism:** a knapsack discontinuity: i4 K1 and i8 K1.125 come from different error curves. #90-02 found i4 within 0.3% of i8 up to K1.5, so the effect is small.
- **Gain / risk:** below 0.1%. No action.
- **Conflicts:** #90-02.

## D. Codebook × fractional K × H_out

**R29. The i8 ceiling under two-sided LDLQ is unmeasured**
- **Mechanism:** #94 and #109 measured i8 / mul1 one-sided (×1.018 at K6.5, ×1.27 at K7). With H_out feedback the targets change, so the int8 grid floor may bind earlier.
- **Source:** #94, #109.
- **Fit:** codebook and level menu.
- **Test (colo, ~3 CPU-h):** w87 settle (it has ldlq_2hess) at K6 / 6.5 / 7, i8 vs mul1, with and without H_out.
  - **If the ratio moves by more than 2%:** drop i8 K7 (use mul1) or turn H_out off at K ≥ 6.
- **Gain / risk:** 0 at 2.5 bpw; protects the fast 7.0.
- **Conflicts:** #94, #109.

**R30. "i8 is free to K6" is a one-sided claim**
- **Mechanism:** same as R29 at K4–K6.
- **Test:** rides with R29 (add K5 and K6).
- **Gain / risk:** below 0.2%.
- **Conflicts:** #43.

**R31. Fractional-K drift under two-sided LDLQ**
- **Mechanism:** #107 found no effect, but it was one-sided, on the 0.8B, n = 3 and noisy (K1.25 0.964 ± 0.028).
- **Fit:** per-level.
- **Test:** none until R11 runs. Low priority.
- **Gain / risk:** below 0.2%.
- **Conflicts:** #107.

**R32. Fractional positions in the knapsack come from one-sided proxies of H_out levels**
- **Mechanism:** the knapsack reads level positions from the levels' one-sided proxy. #118 found the H_out/plain ratio equal at K2.5 and K4 (1.076 / 1.078), so positions are unbiased there. Outside that range nobody knows, and the ladder will have no plain twins to check against.
- **Fit:** knapsack input.
- **Test:** keep the K1.5 and K6 plain twins that R2 and R3 produce and compute the ratio.
  - **If the ratios differ by more than 3%:** apply K-dependent corrections.
- **Gain / risk:** below 0.3%.
- **Conflicts:** #118.

**R33. i8c on top of H_out (null)**
- **Mechanism:** the cubic int8 codebook added nothing on top of H_out at 0.8B over 6 seeds (#78-04).
- **Gain / risk:** none. Do not re-propose.
- **Conflicts:** #78-04.

## E. Knapsack inputs and additivity

**R34. Per-family attention offsets from #118 are not applied**
- **Mechanism:** knap5 overprices attn_k error (measured/priced 0.74 / 0.72), attn_v (0.83 / 0.80) and attn_qg (0.93 / 0.92), so attention tensors get bits they don't need.
- **Source:** #118 (left to the lead).
- **Fit:** knapsack input.
- **Test (colo, minutes):** re-solve knap5 with the offsets and record allocation shifts and predicted gain. Validate in the stream.
- **Gain / risk:** 0.3–1% at 2.5 bpw (±0.5%). Low risk.
- **Conflicts:** #118.

**R35. H_out is not priced per family**
- **Mechanism:** knap5 has no H_out factor. If the 27B gain is concentrated in GDN and MLP inputs as at 0.8B (39% / 44%), those families are now cheaper per KL than priced, and writers are relatively overpaid.
- **Source:** #82, #118.
- **Fit:** knapsack input.
- **Test:** add per-family KL factors scaled so the model total matches −2.5% at K2.5 and −3.6% at K4, re-solve, and compare.
- **Gain / risk:** 0.2–0.5%.
- **Conflicts:** #118.

**R36. Stacked gain predictions double count**
- **Mechanism:** the w63-style predictions multiply separately measured factors: swj 0.935, H_out 0.975, C9 0.9925, #117 0.997, plus tile CD. Interactions shrink stacks (#108: GDN refits vanished in the full stack). The only full-stack measurement is w92.
- **Fit:** knapsack input (prediction).
- **Test:** compare the w92 full-stack ratio with the product of the factors.
  - **If it shrinks by more than 30%:** rescale the factors before knapsack predictions are compared with V2.
- **Gain / risk:** accounting only; avoids over-promising.
- **Conflicts:** #108.

**R37. Gain factors are diluted by the head's share**
- **Mechanism:** the K4 H_out ratio was 0.936 with V2's 4.00 head and 0.961 with the 3.00 head. Measured ratios include head and embedding KL, which the trunk setting does not touch. Applied per tensor, they under-price trunk gains.
- **Source:** #116 readout.
- **Fit:** knapsack input.
- **Test (CPU):** compute the trunk-only factor = (r·KL − KL_head-share) / (KL − KL_head-share) from the head-swap pairs (#99, #34-02).
- **Gain / risk:** below 0.3%.
- **Conflicts:** #104, #33-28.

**R38. The swj factor at K2 and K3 is a guess**
- **Mechanism:** the curve ×0.91 at K2 and interpolation to K3 drive gate/up K2 vs K2.5 choices. None of those points were measured at 27B.
- **Fit:** knapsack input.
- **Test:** one 27B swj K3 fixed-code pair on the v3ladder2 K3 level (~1 card-h).
- **Gain / risk:** below 0.3%.
- **Conflicts:** #104, #118.

**R39. Additivity has no 27B check with V3 levels**
- **Mechanism:** knap5 is additive per tensor plus kfit2's compounding term, which was fit on V2 quants. With H_out shaping errors relative to the bf16 downstream, κ may differ. In 2609.09867, composition error was 26–42× the pricing error.
- **Source:** 2609.09867, 2608.30564.
- **Fit:** knapsack input.
- **Test (27B, ~2 card-h, existing levels):** measure two family swaps alone and jointly (for example all gate/up K2.5→K4 and all GDN qkv K2.5→K4 on a bn base).
  - Criterion: |joint − (a + b)| / joint < 15%.
  - **If not:** add pair terms for those families or cap the number of moves per refit.
- **Gain / risk:** 0–1%.
- **Conflicts:** #45, #68.

**R40. The compounding term needs a V3 refit**
- **Mechanism:** kfit2's κ was fit on V2. The first knapsack mixes will be predicted with stale compounding.
- **Fit:** knapsack input.
- **Test:** in the stream, refit kfit2 on the first 20 V3 mixes and re-solve. Plan this now; no freeze impact.
- **Gain / risk:** handled post-freeze.
- **Conflicts:** #68.

**R41. Sensitivities are not re-measured at the operating point**
- **Mechanism:** GEMQ, Q-Strata and CASA all gain by re-measuring per-tensor costs in the current mix context and re-solving: build → measure → re-solve.
- **Source:** 2608.30564, 2609.25916, 2610.31931.
- **Fit:** knapsack input (procedure).
- **Test:** for the 2.5 bpw band, budget-neutral one-up / one-down swaps accepted on true search-pool KL. This is the existing local search; make sure the screening proxy includes pair terms.
- **Gain / risk:** 0.5–1% (it is already planned).
- **Conflicts:** #70.

**R42. The head grid was built on plans chosen with the broken rowerr**
- **Mechanism:** w88's knap5 head grid (−2.4 / −3.8 / −2.7 / −1.4% predicted at 2.0 / 2.5 / 3.0 / 4.0) uses 11-01 plans probably selected on err ≈ ref. The corrected plans are very different (s5-3 −85.6% vs −25.8%).
- **Source:** LOG 11-08 12:50 PM.
- **Fit:** knapsack input plus the head level menu.
- **Test:** rebuild the plans from corrected rowerr (CPU seconds), then run headsim on the new donors after the trunk.
  - **Until then:** do not enable KN_HEADGRID.
- **Gain / risk:** ±1% at 2.5 bpw.
- **Conflicts:** #34.

**R43. Rare-token guard 1.0 for slab heads**
- **Mechanism:** the 2.0 pick (K2 floor) has rare-token KL ×3.2 of uniform K3. The private corpus has multilingual text. Under guard 1.0 the 2.5 pick is unchanged (−3.8% either way), so the guard is free at 2.5 and protects 2.0.
- **Source:** w88.
- **Fit:** knapsack input.
- **Test:** none. Adopt KN_HG_GUARD = 1.0.
- **Gain / risk:** 0 at 2.5 bpw; it gives up 0.8% predicted at 2.0 for robustness.
- **Conflicts:** #34.

**R44. Embedding prices come from a 6.00 base**
- **Mechanism:** trellis-embedding costs were measured on V2 6.00 (golden-ml, scaled by 0.333). At 2.5 the embedding sits at t1.5–t2, where interactions with a noisier trunk differ.
- **Fit:** knapsack input.
- **Test (27B, ~1 card-h):** t2 vs t2.5 swap on a 2.5 bpw base, read on the calib part.
- **Gain / risk:** ±0.3%.
- **Conflicts:** #35.

**R45. V3 knapsack factors still come from the V2 fit**
- **Mechanism:** every V3 option is a new quantizer family (i8, H_out, swj).
- **Fit:** knapsack input.
- **Test:** the stream refits (post-freeze). Make sure the prior's family factors are widened, not fixed.
- **Gain / risk:** small.
- **Conflicts:** #68.

**R46. Two-sided pricing proxy was never done (#118 2(c))**
- **Mechanism:** within a family the H_out/plain ratio is constant, so pricing is consistent.
- **Gain / risk:** low priority; skip.
- **Conflicts:** #118.

**R47. Sub-1 options are priced from optimistic CPU curves**
- **Mechanism:** #90 priced sub-1 options from settle curves that are about 13% optimistic at K0.125 (w77). The knapsack over-buys sub-1 at 1.0–1.25.
- **Fit:** knapsack input.
- **Test:** recompute the sub-1 ratios from the real GPU levels once built.
- **Gain / risk:** 0 at 2.5 bpw; ~1% at 1.0.
- **Conflicts:** #90.

## F. Proxy-to-27B transfer

**R48. 0.8B head/embedding share distorts transferred factors**
- **Mechanism:** about 33% of the 0.8B's parameters are a tied embedding/head. Factors involving head-adjacent effects (C9 α, swj's K curve, healing) are mis-scaled.
- **Test:** covered by R16 and R38.
- **Gain / risk:** below 0.3%.
- **Conflicts:** #112.

**R49. Some settings still rest only on 0.8B or CPU evidence**
- **Mechanism:** sub-1 damping, σ 0.2 at K0.125, C9 α, swj and C9 outside K2.5 / K4, and tile CD outside K2.5 / K4 have no 27B KLD pair. Proxy effects shrink to about ⅓ at 27B.
- **Fit:** meta.
- **Test:** none. Rank these on cost: all are cheap except swj above K4.5 (R17).
- **Gain / risk:** none.

**R50. The #39 gain verdict used MSE on the biased (pre-order-fix) settle proxy**
- **Mechanism:** w43's settle.py ran LDLQ in the wrong order, and w64 showed order-dependent results were biased (#106: −7.3% → −3%).
- **Source:** merge B3.
- **Fit:** per-level (scale gain).
- **Test:** colo settle64 with H_out at gain {0.96, 1.0, 1.04, 1.08}, K2.5 / K4 (~2 CPU-h).
  - **If any gain is ≥1% better held out:** run a 0.8B KL check.
- **Gain / risk:** 0–0.5%.
- **Conflicts:** #39.

**R51. Proxy gains below 3% do not show in 27B KLD**
- **Mechanism:** #117's proxy gain of −1.5% became −0.3% KLD; tile CD's 0.5–1.2% became 0.9995. Rule: no pre-freeze adoption on a CPU proxy gain below 3% unless the change is free.
- **Fit:** evaluation rule.
- **Gain / risk:** prevents noise adoptions.
- **Conflicts:** #117, #77.

**R52. H_out heterogeneity is similar at 0.8B and 27B (null)**
- **Mechanism:** 0.8B input-diagonal CV was 1.5–4; 27B PR is 0.08–0.3. The transfer of H_out's mechanism is plausible.
- **Gain / risk:** none.
- **Conflicts:** #33-09.

## G. Slices, reuse, selection bias, z

**R53. The adoption rules have a high false-positive rate**
- **Mechanism:** "both pooled r < 1 and parts agree in sign" with correlated same-seed pairs is weak. #117 passed at z −0.62 with the calib part of p0 at 1.0083.
- **Fit:** evaluation rule.
- **Test (CPU, minutes):** null calibration by per-document sign-flip permutation of stored pair rows.
  - **If P(pass | null) > 0.2:** future rules need z_combined ≤ −2. Keep #117 because it is free, but count its gain as 0.
- **Gain / risk:** rule quality.
- **Conflicts:** #117.

**R54. SE must combine seed and document error**
- **Mechanism:** the document SE (0.010 for a 3-pair mean) is as large as the seed term. Per-document bootstrap z ignores seed noise; seed-spread z ignores the slice.
- **Source:** #116.
- **Fit:** evaluation rule for w92, R13, R5 and R1.
- **Test:** write that into every remaining rule. Zero cost.
- **Gain / risk:** none.
- **Conflicts:** #116.

**R55. golden-search27b is already reused**
- **Mechanism:** it has served #117 (w84), #34-02 / #99 (w90) and #120-02 (w91), and probably w92. It was meant to be fresh per round.
- **Fit:** evaluation.
- **Test:** cut golden-search27c (CPU) for R1, R13 and the w92 follow-ups.
- **Gain / risk:** reduces selection bias.
- **Conflicts:** eval-data.md.

**R56. The verify rule only checks direction**
- **Mechanism:** "geo < 1 on verify" passes a true zero effect about 50% of the time.
- **Fit:** evaluation.
- **Test:** report verify CIs instead of a pass/fail. golden-verify27 is spent (5 uses).
- **Gain / risk:** none.
- **Conflicts:** #33-28.

**R57. Most remaining candidates are below the slice's resolution**
- **Mechanism:** with ~101 documents and 2 seed pairs, the minimum detectable effect at 80% power is about 2.5–3%. Candidates R5, R13, R16 and R34 are 0.3–1%.
- **Fit:** evaluation.
- **Test:** decide these on proxy plus cost, or on a pooled meta-analysis across slices. Do not spend card time on single-slice pairs for them.
- **Gain / risk:** saves card hours.
- **Conflicts:** none.

**R58. xcorp dominates pooled slices**
- **Mechanism:** golden-search27b is 65% xcorp. The board is drawn like calib, head effects are 15× diluted, and calib-fitted settings keep only 52–91% of their gain on xcorp (#93).
- **Fit:** evaluation.
- **Test:** require the calib-part sign, and weight the decision 50/50 by part rather than by document count.
  - **If a setting is < 1 pooled but > 1 on calib:** treat it as unproven.
- **Gain / risk:** none.
- **Conflicts:** #34-02.

**R59. Trace and source clustering in the bootstrap**
- **Mechanism:** chunks of one agent trace, or xcorp pieces from one source document, are treated as independent. The cluster SE is up to 5× wider (merge B4).
- **Fit:** evaluation.
- **Test (CPU, minutes):** re-bootstrap the stored pairs with the source document as the unit.
- **Gain / risk:** none.
- **Conflicts:** #116.

**R60. Pair arms must differ only in the intended setting**
- **Mechanism:** the swj K4 head mismatch produced a fake −22%, and the w80 `--only` group-head seed produced divergent codes. PAIR OK checks which tensors differ, not why.
- **Fit:** evaluation.
- **Test:** add a stamp-diff assertion in the pair tool.
- **Gain / risk:** guardrail.
- **Conflicts:** #104.

**R61. Seed luck in controls**
- **Mechanism:** s1/base was 0.975 on search and 0.991 on verify. Any pair with a single control seed carries about 1.3% seed luck.
- **Fit:** evaluation.
- **Test:** covered by R54.
- **Gain / risk:** none.
- **Conflicts:** #116.

**R62. Pick-on-B1 / judge-on-B2 for σ is fine (null)**
- **Gain / risk:** none.
- **Conflicts:** #114.

## H. Private-corpus mismatch

**R63. Long context: all 27B decisions use positions ≤ 8k**
- **Mechanism:** the private traces run to ~250k tokens. H_out was collected on 2048-token windows, so long-range GDN writes may be under-weighted.
- **Counter-evidence:** on the same model under NVFP4 the NLL gap shrinks with position (+0.081 nats in the first 16k, +0.011 in the second; 2609.04098), and V2's edge held to 250k (#73).
- **Fit:** per-level.
- **Test (card 0, ~1 card-h):** score the full-stack and plain arms on eval/traces chunks at 8k–32k.
  - Criterion: ratio within ±1% of the ratio below 8k.
  - **If H_out is worse at long range:** raise σ_out on GDN qkv / z.
- **Gain / risk:** P ≈ 10%.
- **Conflicts:** #73, #52.

**R64. Agentic text is barely measured**
- **Mechanism:** agent traces are about 25% of scored calib rows. At 0.8B the stack was +3% on agentic text (merge B4), and the xcorp H_in blend was +4% on agentic at K2 (#101).
- **Fit:** per-level.
- **Test (CPU, minutes, stored rows):** per-category readout of the H_out (bn vs base), swj and C9 pairs on the agentic docs in the calib part, with trace-clustered SE.
  - **If agentic is > 1 with a CI excluding 1:** see R8.
- **Gain / risk:** diagnostic.
- **Conflicts:** #101, #91.

**R65. Multilingual text**
- **Mechanism:** calib is about 94% English / code. H_in and H_out under-weight multilingual-specific channels.
- **Fit:** per-level.
- **Test (CPU):** multilingual-docs readout of the stored pairs (verify-pool docs).
  - **If the ratio is > 1:** add multilingual windows to any future statistics. This is not a freeze blocker.
- **Gain / risk:** below 0.3%.
- **Conflicts:** #32.

**R66. Board scoring mask vs gen mask**
- **Mechanism:** H_out (YAQA_GENMASK), C9, swj and all goldens target generated positions. If the board scores all positions, the targeting is off.
- **Fit:** per-level.
- **Test (docs plus a free readout):** confirm the challenge scorer's mask. The stored rows already contain all positions (for example #105: "all 1.2833 → 1.2673"), so re-read H_out / swj on all positions.
  - **If the board scores all positions and H_out is > 1 there:** revisit gen-masking.
- **Gain / risk:** P ≈ 5%.
- **Conflicts:** #83-02.

**R67. The calib Hessian includes verify documents**
- **Mechanism:** verify reads flatter calib-fitted settings (C9, swj, H_in) relative to free settings (#117, tile CD).
- **Fit:** evaluation.
- **Test:** none. Note the bias when comparing those classes.
- **Gain / risk:** none.
- **Conflicts:** v3.md §5 taint.

## I. Things that would waste the ladder (~90–115 card-h quant + ~53 swj + 24–32 CPU-h pack)

**R68. No Hessian / H_out integrity check at load**
- **Mechanism:** a corrupt span (2.1e18) already occurred once (L43 o_proj xcorp H_in). Nothing checks the 257 H_in or 400 H_out files before hashes are pinned.
- **Source:** merge B2.
- **Fit:** per-level (datapack).
- **Test (colo, ~10 min):** finite, positive diagonal and Cauchy–Schwarz checks for H_in; finite, positive and within a plausible range for the H_out diagonals.
  - **If anything fails:** recollect that file before enqueue.
- **Gain / risk:** avoids re-quantizing tensors across 55 levels.
- **Conflicts:** none.

**R69. H_out coverage**
- **Mechanism:** ourq silently skipped missing H_out files. There are 400 diagonals for 400 quantized trunk linears.
- **Fit:** per-level.
- **Test:** add an assertion in rxq datapack that every quantized key has a file, and refuse otherwise (seconds).
- **Gain / risk:** guardrail.
- **Conflicts:** none.

**R70. Settle every candidate switch before the stamp**
- **Mechanism:** R13, R14, R15, R17, R18, R23 and R5 each change the settings stamp. A change after enqueue invalidates donor work.
- **Fit:** per-level.
- **Test:** add the switches now, with defaults decided by the checks above.
- **Gain / risk:** process.
- **Conflicts:** none.

**R71. expandable_segments in worker environments**
- **Mechanism:** it caused 7 faulted runs in stock kernels, and some gpubox scripts still export it (merge A1).
- **Fit:** process.
- **Test:** grep the rqah worker images and entrypoints for PYTORCH_HIP_ALLOC_CONF and refuse it as sibmom27 does (minutes).
- **Gain / risk:** guardrail.
- **Conflicts:** #105 debugging.

**R72. Fork patch 0008 must be pinned in donor images**
- **Mechanism:** the fork working copy lacked the 0008 commit (w86). A donor without it runs the dense path: the same objective but different codes and 5× the time.
- **Fit:** process.
- **Test:** check that the image pin includes 0008 and that the stamp records the fork tree hash.
- **Gain / risk:** guardrail.
- **Conflicts:** #115.

**R73. The donor scorer is one-sided**
- **Mechanism:** an H_out level that accidentally runs one-sided scores *better* on the H_in band (by about ×1.076) and passes.
- **Source:** merge B1.
- **Fit:** process (verification).
- **Test:** make the scorer check the stamp's H_out flag and flag results more than 5% below the trusted minimum, or score f2 directly.
- **Gain / risk:** guardrail.
- **Conflicts:** #118.

**R74. Audit for reuse of the in-place-regularized weight**
- **Mechanism:** this bug class already broke head rowerr. C9 in-job matched a fresh-weight pass bit for bit (w76 / w80), so it is probably safe.
- **Fit:** process.
- **Test:** grep for every use of `w` after quantize_exl3 in rxq / ourq (minutes).
- **Gain / risk:** guardrail.
- **Conflicts:** #34-02 fix.

**R75. Full-stack rule for w92**
- **Mechanism:** the whole adopted stack (H_out blocks, #117, tile CD, in-job C9, swj, σ settings) has never been scored together.
- **Fit:** per-level.
- **Test:** pre-register before scoring: full stack ≤ bn+swj × 1.000 and full stack vs base ≤ 0.93 at K2.5, using combined SE (R54).
  - **If it fails:** bisect before enqueue.
- **Gain / risk:** decisive.
- **Conflicts:** #33-28, #104, #112, #117.

**R76. Defer ladder pass 2 until pass 1 is used**
- **Mechanism:** pass 1's 25 levels came within 0.2–0.8% of all 73 levels (w58 / w63).
- **Fit:** level menu.
- **Test:** quantize pass 1, run knap5 on it, and queue only the pass-2 levels whose neighbours are picked.
- **Gain / risk:** saves about 30–40 card-h at ≤0.8% predicted cost.
- **Conflicts:** v3 ladder plan.

**R77. Pin the block-order file**
- **Mechanism:** route B depends on order82.json.
- **Fit:** process.
- **Test:** check that its hash is in the stamp and the manifest.
- **Gain / risk:** guardrail.
- **Conflicts:** #117.

**R78. Tile CD refuses a dense H_out**
- **Mechanism:** `out_blocks` refuses any mass outside the blocks. If levelargs maps an old level to dense H_out, the job fails on a donor.
- **Fit:** process.
- **Test:** add a config-time refusal of tile CD + dense H_out instead of a job-time failure.
- **Gain / risk:** guardrail.
- **Conflicts:** #115.

**R79. Seeds are shared across levels (null)**
- **Mechanism:** seed spread is 0.5% (#41).
- **Gain / risk:** none.

## J. Block order, tile CD, embedding

**R80. #117 is kept as free but counted as 0**
- **Evidence:** geo 0.9968, z −0.62, one calib part at 1.0083.
- **Gain / risk:** no action beyond R53.
- **Conflicts:** #117.

**R81. Block order was chosen without H_out**
- **Mechanism:** the visit order came from calib H_in block traces, and the w87 damping check ran without block order. With two-sided LDLQ the best order may differ.
- **Fit:** per-level.
- **Test (colo, ~2 CPU-h):** w87 proxy at K2.5 / K4 with and without block order, H_out on.
  - **If block order is worse with H_out:** turn it off (free).
- **Gain / risk:** ±0.2%.
- **Conflicts:** #117, #114-03.

**R82. Tile CD costs 3.6 card-h for about 0.1%**
- **Evidence:** with H_out, the proxy gain is 0.9989 and KLD is 0.9995 / 1.0000.
- **Gain / risk:** keep for no harm, or drop to save time. The lead's call; low stakes.
- **Conflicts:** #77.

**R83. Tile CD outside K2.5 / K4**
- **Mechanism:** no-harm was only checked at K2.5 and K4.
- **Test:** an f2 check at K6 and K1.5 on L39 q_proj (card 0, ~10 min).
- **Gain / risk:** guardrail.
- **Conflicts:** #77.

**R84. Embedding quantizer metric (null)**
- **Mechanism:** #119 found M0 and F disagree; H = I is the safe default.
- **Gain / risk:** none. Keep.
- **Conflicts:** #119.

**R85. H_out at i4 gives little at the lowest levels (null)**
- **Mechanism:** f2 gain is only −2.6% at K0.125, but the cost is now +2.5 min a level. Keep it, unless R26 fires.
- **Gain / risk:** none.
- **Conflicts:** #115.

---

## INDEX experiments resting on the thinnest evidence

| # | conclusion | why thin | cheapest firming step |
|---|---|---|---|
| #33-28 (every level) | H_out at all 55 levels | KLD only at K2.5 (3 pairs; combined z −1.74) and K4 (2 pairs); adopted after the rule missed; never tested in a mixed-K mix; no KLD below K2.5 or above K4 | R1, R2, R3 |
| #117 | residual block order | geo 0.9968, z −0.62; calib part 1.0083 on p0; proxy-to-KLD about ⅕ | R53, R81 |
| #112 | C9 on writers | 27B PARTIAL 0.9925 on non-H_out levels; α from the 0.8B at the grid edge; no K schedule | R15, R16 |
| #104 beyond K2.5 / K4 | swj at every level | no data at K ≤ 2, K ≥ 5 or sub-1; possible mix dependence | R17, R18, R19 |
| #114-02 | σ 0.2 at K0.125 | CPU proxy is ~13% optimistic there; the GPU search loses to all-zero (w77) | R22, R23 |
| #77 | tile CD on attn_qg | adopted on 0.5–1.2% per-tensor proxy gain; KLD 0.9994–1.0000 is no-harm only | R82 |
| #105 | per-level asym negative | one pair (1.0067, CI ~0.985–1.03); base arm stopped | none needed pre-freeze |
| #101 | xcorp H_in dropped | two 27B pairs pooled 0.993; could be a real −0.7% | none (cost decision) |
| #120-02 | every channel-3994 fix dropped | oracle covered L0–L3 writers only; depth arm not triggered | none pre-freeze |
| #39 | scale gain flat (1.0) | MSE on the pre-order-fix settle proxy | R50 |
| #34-02 | slab head build | one slice; calib part ~35 docs; head grid built on broken-rowerr plans | R42 |
| #90 | sub-1 pricing | CPU curves only (optimistic at K0.125), no KLD | R47 |
| #83-02 | gen-masked Σdy² | 0.8B only (K3 0.979, K2 0.96); board mask unconfirmed | R66 |
| #33-27 | block-normalized estimator | chosen on a 3-layer proxy; 27B bn vs dy −1.1%, z −1.4 | R6 |
| #118 | pricing ratios constant across K | measured only at K2.5 and K4 | R32 |
| #116 | pooled pair SD 0.0086 | df 4 (95% CI 0.0052–0.0248) | R54 |
| #107 | fractional drift null | n = 3, noisy (K1.25 0.964 ± 0.028) | R31 |

## Ranked top 20 (expected gain or avoided loss per unit cost)

| rank | item | cost | expected effect at 2.5 bpw |
|---|---|---|---|
| 1 | R15 C9 K schedule | minutes CPU (+1 h on the 0.8B) | 0 at 2.5; avoids several % at 1.0–1.5 |
| 2 | R13 H_out-weighted scale polish | 0.3 card-h, then 1.5 | −0.2 to −0.8% |
| 3 | R14 suh/svh underflow clamp | minutes CPU | correctness |
| 4 | R68 Hessian / H_out integrity check | 10 min CPU | protects the ladder |
| 5 | R69 H_out coverage assertion | seconds | protects the ladder |
| 6 | R43 rare-token guard 1.0 | 0 | 0 at 2.5; robustness at 2.0 |
| 7 | R34 attention pricing offsets | minutes CPU | −0.3 to −1% |
| 8 | R53 + R54 rule audit and combined SE | minutes CPU | re-ranks what to revisit |
| 9 | R66 all-position readout | free | P 5% of a large miss |
| 10 | R64 agentic readout | minutes CPU | diagnostic |
| 11 | R76 defer pass 2 | 0 | saves 30–40 card-h |
| 12 | R17 swj k_max 4.5 | 0 | saves ~18 card-h |
| 13 | R1 mixed-K H_out splice pair | 2–2.5 card-h | avoids +2–3% (P ≈ 15%) |
| 14 | R19 swj input source | minutes, maybe 1.5 card-h | protects about half of 6.5% (P ≈ 30%) |
| 15 | R23 g_scale floor scan | minutes | 0 at 2.5; ~1% at 1.0 |
| 16 | R35 per-family H_out pricing | minutes CPU | −0.2 to −0.5% |
| 17 | R42 corrected head plans | seconds CPU (headsim later) | ±1% |
| 18 | R5 σ_out held-out sweep | 4 CPU-h | 0 to −0.5% |
| 19 | R81 block order × H_out | 2 CPU-h | ±0.2% |
| 20 | R75 full-stack rule for w92 | 0 extra | decisive for the freeze |

## The 5 cheapest diagnostics that would most change this ranking

1. **f2 with polish vs no-refit vs h-weighted polish** on L10 / L39 at K2.5 and K4 (card 0, ~20 min). If the polish is H_out-neutral, R13 drops out of the top 5. If `no_refit` wins, it becomes the largest pre-freeze item.
2. **C9 statistics at K0.125 / 0.5 / 1** for gdn_out and down (colo CPU stand-in, minutes). This settles whether R15 is a real harm or a non-issue.
3. **Null-calibrated rule audit and combined-SE re-read** of every adopted 27B decision from stored rows (CPU, minutes). This changes which adopted items deserve card time (R1–R4, R80).
4. **Per-category, all-position and trace-clustered readout** of the stored H_out / swj / C9 pairs (CPU, minutes). If agentic or unscored positions disagree, R8, R63 and R66 move to the top.
5. **Split-half stability of the H_out floor rows** (colo CPU, minutes). This decides whether R5 / R6 (floor clamp, σ_out) matter or can be dropped.

These five rank on cheapness. The most decisive single test that is not cheap is R1, the mixed-K splice pair (~2 card-h with existing levels). I would run it before the ladder enqueue whatever the five diagnostics show.
