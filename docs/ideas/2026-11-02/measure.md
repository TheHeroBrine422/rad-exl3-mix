# Research ideas, lens: **measure** (2026-11-02)

## Repo facts behind these ideas (checked this run)

- **What the H_out collector sums over.** In `v3/yaqa/collect.py` the loss is −Σ_t log p(y_t) over **every** position of 2048-token windows. Windows are cut only from documents of at least 2048 tokens. There is no gen mask. The diagonal accumulators keep only the running mean, not per-sequence values.
- **What the board scores.** Our evals and the board score **generated rows only** (`gen.u8` mask). On the 2B search golden that is 166k gen rows out of ~300k tokens.
- **How kfit2 prices tensors.** kfit2 maps the **one-sided** hproxy error tr(ΔW H ΔWᵀ)/rows to KLD. It uses per-group depth curves (knots every 8 layers), per-family factors and the directional compounding term. Of V2's 1,450 results, 256 are `cal` group swaps, **all on the 4.00 base** (`sc4.00-e4`, plain-2.00 donors).
- **Where the score lives.** Per `docs/research/quant-sota.md`, the two ~256K traces are ~512K of ~1.45M scored positions on public (**~35%**). In calib, the 8 traces are ~2M of 10.26M tokens. Search slices cap documents at 2k positions; ver3 caps at 8k.
- **KL-per-byte conversion used throughout.** Near 2.5 bpw, d ln KL / d bpw ≈ −1.6 to −1.8. The board points 0.164 → 0.0666 → 0.0306 at 2.0 / 2.5 / 3.0 give this, so **0.01 bpw ≈ 1.7% KL** at 2.5 bpw. `local/TODO.md` uses the same rate (0.007 bpw ≈ 1.25%).
- **Already in `local/TODO.md`** (marked "TODO" below, not counted as new): GPU-side scoring, head only on gen rows, gen-row headsim, two evals per card, GDN a/b to int8, separate q vs k/v sensitivities.

Fit labels: **per-level** (library + knapsack), **final** (shipped mixes), **per-mix**, **format**, **training**, **infra/diagnostic**.

---

## O. Output-side (Fisher) information for measurement and allocation

**O1. Gen-masked output Fisher for the 27B diagonal run.**
Restrict the sampled-label loss to generated (scored) positions. Prompts, system blocks and tool outputs still flow through as context, but their own predictions get zero weight. diag(H_out) then is the curvature of the metric we are actually scored on. The change is largest on agentic traces, whose unscored tool outputs are long.
- **Source:** new for the trunk. TODO already plans the same fix for headsim ("Head judged on unscored rows"). 2609.36828 weights the states whose decisions matter.
- **Why it should help:** the Fisher equals the KL Hessian only over the positions the KL sums. Unscored positions add curvature (e.g. copying tool output) the board never sees.
- **Fit:** per-level (input to the ladder requant).
- **Test:** 0.8B on the 4070. Run yaqaB diag with and without the mask over the same 1K chat-rendered windows (role masks). Quantize with base settings + diag at K3 and K2, and compare paired on g08-search1 gen rows. Pass: masked better with z ≤ −2 at one K and not worse at the other. ~2 h. If it passes, the 27B run uses the masked loss at no extra cost.
- **Gain / risk:** 0–3% at 2.5 bpw. Fewer effective positions make the diagonal noisier; if so, use 1.5K sequences.
- **Conflicts:** #33-13. This changes the loss, not the estimator; decide before the run starts.

**O2. Keep per-sequence, per-tensor Fisher traces (plus window metadata) in the 27B run.**
Per sequence, log 496 floats ‖G_b‖²_F (= Σ_r diag_r), per-128-row-block sums for the large tensors, and the window's document id, category, offset and gen-token count. That is ~0.1–2 MB per sequence. With these logs we get bootstrap CIs, split halves, dose curves and category/position splits without a second run.
- **Source:** new. G_b already exists per sequence in the collector.
- **Why it should help:** it turns the 8.5 h run into an allocation dataset (O3, O4, O7, O15, O17, O20, L6). Today only the mean survives.
- **Fit:** infra.
- **Test:** gate that Σ of the logged traces equals Σ diag to fp32 rounding.
- **Gain / risk:** enabling; no risk.
- **Conflicts:** #33-13 (augment it before it starts).

**O3. Fisher-predicted swap KL: an absolute, parameter-free price.**
Let d_r be the sampled-label diag of E[G_bG_bᵀ] per 2048-token sequence, and Ĥ the calib input Hessian normalized to unit trace. For each option, predict per-token ΔKL ≈ (1/2T)·Σ_r d_r·e_r Ĥ e_rᵀ, with e_r the rows of ΔW decoded by hproxy. This is the second-order expansion of exactly our metric.
- **Source:** YAQA 2505.22988 (Kronecker Fisher). 2609.28270 prices ½Tr(HΣ) with log-Pearson 0.93–0.95 vs measured KL on ≤0.6B. 2610.41825: local second-order tracks typical sequences but misses a few confident rows.
- **Why it should help:** today's price is one-sided proxy × a smooth, group-level depth curve anchored on a 4.00 base. A per-tensor absolute prior would price fractional, sub-2 and H_out options with no swaps. If measured/predicted is near-constant per family, the 8-knot depth curves can go.
- **Fit:** per-level.
- **Test:** 0.8B first (the 8K diag exists): ~48 single-layer group swaps at K2.5/K3 (fake-quant, ~20 s each, ~1 h). Pass: Spearman ≥ 0.8 and log-RMSE ≤ 0.3. Then 27B on CPU against the 256 V2 cal swaps once the diag exists.
- **Gain / risk:** 1–3% via allocation, more at sub-2 options. At K2 the second-order expansion undershoots; fit an exponent.
- **Conflicts:** extends #67/#68. #15 (noise injection) was dropped because it needed bf16 forwards; this needs only the planned collection.

**O4. Two-sided proxy error as kfit2's input.**
Have hproxy emit per-row errors e_r H e_rᵀ. Define e2 = Σ_r d_r e_r H e_rᵀ / Σ_r d_r w_r H w_rᵀ, and refit kfit2 on e2 (or on e and e2 with a learned mixing exponent).
- **Source:** 2609.33923, on Qwen3.8-27B: allocating by the layer objective gave 6.087 PPL vs 6.041 for a downstream-weighted block cosine. KronQ 2607.07964.
- **Why it should help:** the ±12% residual may partly be output-side structure the one-sided proxy cannot see.
- **Fit:** per-level.
- **Test:** CPU on V2 data once the 27B diag exists (~1 h). Pass: CV log-RMSE ≤ 0.10 (now 0.118) or pair ranking ≥ 82% (now 79.9%).
- **Gain / risk:** 1–3%; low risk.
- **Conflicts:** #19, #20 failed because they added free parameters; this adds data.

**O5. Re-price H_out-quantized options before the stream sees them.**
Two-sided LDLQ deliberately trades one-sided error for two-sided error (objcheck: full Sketch B raised one-sided error 1.4–3.5×). kfit2 maps one-sided error to KL, so it will rate H_out quants worse than they are and move bytes away from them. Fix with O4's proxy, or give H_out quants their own family factor seeded by a few swaps.
- **Source:** LOG #33-08 objcheck.
- **Why it should help:** protects the −8% gain from being partly undone by mispricing.
- **Fit:** per-level.
- **Test:** CPU, 30 min: per-tensor one-sided proxy of 0.8B `k3bdiag`/`k2bdiag` vs `base`. If the ratio is > 1.05 where KL fell, the fit must switch.
- **Gain / risk:** avoids losing an estimated 2–4%; no risk.
- **Conflicts:** #33-12/13.

**O6. Does the output side explain the ours-vs-turboderp anomaly at high K?**
Compute O4's proxy for his and our K5–K8 tensors (all are already in hproxy). V2 took 104 of his tensors at 5.5 bpw, although ours have 0.74–0.80× his one-sided error. If our edge disappears on the two-sided proxy, his wins are output-side, and the 1.58× vs ~1.0× family factors collapse into one.
- **Source:** v3 §1d; #50 (D2: ~9% of the edge is in-sample).
- **Fit:** per-level.
- **Test:** CPU, 1 h after the diag.
- **Gain / risk:** matters mainly ≥ 4 bpw (<0.5% at 2.5); it simplifies the fit.
- **Conflicts:** #65, #50.

**O7. Fisher prior for per-tensor sensitivities (partial pooling with an informative mean).**
Set log s_w = log(Fisher-predicted scale) + a shrunk group/depth residual. The prior variances come from O2's bootstrap and feed the planned uncertainty-aware solve (B11).
- **Source:** KronQ (s = tr H_G · tr H_X); ThinkBit 2610.34886; BACKLOG "partial pooling".
- **Why it should help:** the new V3 options (quarter steps, sub-2, i8, H_out quants) need priors per tensor.
- **Fit:** per-level.
- **Test:** CPU CV, including leave-band-out (S12).
- **Gain / risk:** 0.5–2%; low risk.
- **Conflicts:** complements the BACKLOG item.

**O8. Out-of-sample test of the Kronecker model: why the gradient-weighted H_in hurts.**
Split the 8K snapshots into independent halves (H(4096) and 2H(8192) − H(4096)). Quantize with half A's Sketch-B factors; score tr(E H_in,B Eᵀ H_out,B) and the calib one-sided error on half B. Add KL-Shampoo token weights (inverse quadratic forms) as a third H_in.
- If the held-out two-sided objective also improves while KL worsens, the Kronecker objective itself is wrong for H_in.
- If the held-out objective worsens, the factor is overfit.
- **Source:** objcheck (#33-08, in-sample only). ShamAN-Q 2609.38521: NKP's quartic token weights are dominated by massive-activation tokens; tempering gave 19.21 → 17.46 PPL.
- **Fit:** diagnostic.
- **Test:** CPU on the 4070 box with existing snapshots, 1–2 h.
- **Gain / risk:** none direct. It decides whether a tempered H_in deserves GPU time.
- **Conflicts:** explains #33-07/#33-10; does not repeat them.

**O9. Tempered output factor d_r^p.**
Replace d_r by d_r^p (trace-preserving), p ∈ {0.25, 0.5, 1}, in the quantizer and in O4's proxy (p by CV).
- **Source:** ShamAN-Q (p = ½ best); KBBQ 2609.08135 (interior λ optimum); NatScale 2610.20353 (β ≈ 0.2).
- **Why it should help:** at 2–3 bpw the error is far outside the quadratic regime, so fully trusting the largest curvature rows over-weights a few of them.
- **Fit:** per-level.
- **Test:** 0.8B, 2 extra quants per K, ~30 min. Pass: z ≤ −2 vs p = 1.
- **Gain / risk:** 0–2% on top of diag; low risk.
- **Conflicts:** #33-12 measured the dose only at p = 1.

**O10. Per-token output curvature as the diagonal.**
One backward gives two diagonals: Sketch B's sequence-level rowsum(G∘G), and per-token Σ_t dy_t² (unweighted by inputs). Test both in the same two-sided LDLQ path.
- **Source:** 2610.40345, where output-feature curvature beat YAQA's factor (Llama-3-8B W4-g32: 6.39 → 6.32) and YAQA was unstable in other settings.
- **Fit:** per-level.
- **Test:** 0.8B, collector change + 2 quants per K, ~2 h.
- **Gain / risk:** 0–2%; moderate chance of a null.
- **Conflicts:** #33-06 used dy-based diagonals in exllamav3's out_scales path (hurt); here they enter two-sided LDLQ the way #33-12's `bdiag` did.

**O11. Output-side row-slab ceiling (E3c redone with diag H_out).**
Per tensor, compute ½log₂(AM/GM) over 128-row slabs of (slab Fisher weight × slab input-side error).
- **Source:** E3c (#29-02); SoftWater 2608.12026 (class-aware head rates: 6.5–8.3× lower head KL); AlignQuant 2610.07457 (within-tensor CV 0.47–2.10).
- **Why it should help:** E3c's 0.05-bit ceiling used input-side norms, which cannot tell rows apart. On the 0.8B, diag H_out has CV 1.5–4 on input projections.
- **Fit:** diagnostic → format change.
- **Test:** CPU now on the 0.8B, then on the 27B diag. Rule: ≥ 0.15 bit on the large tensors reopens trunk row-slab K. §6b dropped it because output weighting "hurt"; it now helps.
- **Gain / risk:** if built, 2–6%; the diagnostic itself is free.
- **Conflicts:** #29-02.

**O12. Fisher-weighted activation-quant proxy.**
Replace E1c's unweighted rel_out by Σ_r d_r (W e_x)²_r / Σ_r d_r (W x)²_r on the existing 512-row dumps, then refit on the 45 E1b runs.
- **Source:** #27 (log-RMSE 0.25; late layers under-predicted 1.7×); AlignQuant.
- **Fit:** per-level (A8 mode choice).
- **Test:** CPU, 1 h after the diag. Pass: log-RMSE ≤ 0.18.
- **Gain / risk:** ≤ 0.5% KL at 2.5 bpw, plus better speed choices.
- **Conflicts:** #27.

**O13. Fisher traces at bf16/x2 precision, for allocation only.**
Per-tensor sums may be fine where per-row diagonals were not, since bf16's bad rows were a minority. On the 0.8B, compare traces under fp32 / x2 / bf16.
- **Why it should help:** if they agree within 5%, a bf16 run (5 s vs 30 s per sequence) of 4–8K sequences gives tight category- and position-resolved sensitivities.
- **Source:** #33-03, #33-14.
- **Fit:** per-level.
- **Test:** extend `prectest.py`, 30 min on the 4070.
- **Gain / risk:** enabling; early layers may be biased.
- **Conflicts:** #33-14 failed a per-row bound needed for quantizer use; this needs a much weaker one.

**O14. Midpoint gradient attribution: one pass prices every tensor in context.**
Run the fp32 two-card collector at W + ½ΔW_mix (a precomputed bf16 checkpoint) on search-pool documents. The loss is KL to the golden top-512 on gen rows. Each linear also computes ΔW x in the forward pass and accumulates ⟨dy, ΔW x⟩ = ⟨∇_W KL, ΔW⟩ at the midpoint. For a quadratic loss the midpoint prices sum exactly to the total ΔKL (Aumann–Shapley), giving every tensor's in-context share of a 2.5 mix with document-level CIs.
- **Source:** 2609.09867 (midpoint price signs 97/98 vs 85/98 for current-state gradients); 2607.12266 (fit-free Aumann–Shapley); the RCO spike in research-rco.md.
- **Why it should help:** V2's prices are isolated K2 swaps on a 4.00 base. GEMQ shows re-estimation at the operating point helps (9.66 → 9.33 PPL).
- **Fit:** per-mix (a new kfit2 measurement type).
- **Test:** 0.8B first: attribution vs measured leave-one-group-out swaps. Pass: Spearman ≥ 0.8 and the sum within 15% of the total. Then 27B: ~200 docs × ~45 s ≈ 2.5 h on TP2.
- **Gain / risk:** 2–5%. Risks: non-quadratic at K2; ΔW has to be streamed to the GPU.
- **Conflicts:** #17 (RCO was dropped for its 800–3,200-pass optimization loop; this is one attribution pass), #15.

**O15. Pairwise interactions from the Fisher run.**
For one option set, log per-sequence scalars s_i = ⟨G_i,b, ΔW_i⟩ (one extra matmul per linear). E[s_i s_j] across sequences estimates the cross-curvature β_ij of every tensor pair.
- **Source:** CLADO; 2609.09867 (β_ik = d_iᵀHd_k; pair-quadratic models cut NLL MAE 8×); 2609.25916.
- **Why it should help:** tests the additivity behind the knapsack without measuring pairs.
- **Fit:** per-level.
- **Test:** 0.8B, compare predictions with 10 measured pair swaps (40 evals). Pass: ≥ 8/10 sign agreement.
- **Gain / risk:** 0.5–2%; the cross term may miss saturation.
- **Conflicts:** #16 (Shapley at 15–18 h per 8B; here nearly free).

**O16. Bounded scored-span weighting of the input Hessian.**
Use H_in = Σ w_t x x ᵀ with w_t = 1 except λ ∈ {0.5, 0.75} on long unscored tool outputs. No gradient magnitudes are involved.
- **Source:** new.
- **Fit:** per-level.
- **Test:** 0.8B, re-collect calib H_in with masks (~15 min) + 2 quants per K.
- **Gain / risk:** 0–2%; high risk of a null.
- **Conflicts:** #33-07 (gradient-weighted, +31–44%). This differs: bounded weights, no labels.

**O17. Category- and position-resolved Fisher shares.**
From O2's logs, compute each family's share by category and by window origin (deep in long traces vs short documents). Shares that differ by > 1.3× argue for weighting the allocation by calib token shares rather than the slice's.
- **Source:** MODE 2606.17118; mix-allocation.md (ii).
- **Fit:** diagnostic.
- **Test:** CPU after the run.
- **Gain / risk:** 0–2%.

**O18. Triangulate the sensitivity sources: Fisher vs kfit2 vs turboderp's S.**
Within-family Spearman of the three (his S vs our fit: 0.71 overall, weak within families). Where Fisher sides with one, trust it.
- **Fit:** diagnostic.
- **Test:** CPU, 30 min.

**O19. Fisher-disagreement–driven measurement.**
Spend a fixed share of stream evals where the O3 prior and kfit2 disagree by > 2×, not only where bootstrap refits disagree.
- **Source:** BACKLOG "measure where unsure". The Fisher run is an independent instrument; bootstrap spread only reflects data noise.
- **Fit:** stream.
- **Gain / risk:** 0.5–1%.

**O20. Sequences needed for allocation-grade traces.**
From O2, the bootstrap CV of each tensor's trace vs N. Allocation may need fewer sequences than the quantizer does, or more for small tensors.
- **Fit:** diagnostic.
- **Test:** CPU.

---

## S. The sensitivity model and the knapsack

**S1. Re-anchor group swaps at the 2.5 operating point.**
Measure designed ±¼K group swaps around the best 2.5 mix (and the 2.0 and 3.0 mixes), weighted up in the fit.
- **Source:** GEMQ 2605.23078 (FP / 4-bit / 2-bit / 2-bit+FT-anchored sensitivities: 9.66 / 9.39 / 9.33 / 9.01 PPL); Q-Strata 2608.30564 (separable 13.72 vs in-context 12.14); MixQuant 2607.23047. Counter-evidence: MixFrag 2607.28589 (re-estimation hurt on DeiT-T).
- **Why it should help:** all 256 cal swaps sit on the 4.00 base, so compounding at 2.5 is extrapolated.
- **Fit:** per-level.
- **Test:** 27B, ~40 evals (~1.5 card-h). Pass: kfit2 error on held-out 2.5 measurements down ≥ 25%.
- **Gain / risk:** 2–5%; low risk.
- **Conflicts:** #70 measured whole mixes, not local slopes.

**S2. Fractional-factorial swap designs.**
Each eval toggles a balanced half of k candidate swaps, with every up-move paired with a down-move so bytes stay fixed. Main effects get SE ≈ 2σ/√N: 24 swaps in 32 evals give σ/2.8 per effect, vs σ per effect for 24 single-swap evals. Resolution IV keeps pair interactions out of the main effects.
- **Source:** deadends ("use group swaps, whole mixes, or random multi-tensor perturbation designs"); classical design of experiments.
- **Fit:** per-level.
- **Test:** 27B, one 32-run design at 2.5 (~1.1 card-h); compare effect SEs with one-at-a-time data.
- **Gain / risk:** 2–3× information per eval, indirectly 1–3%. Higher-order interactions are the risk.
- **Conflicts:** #14. This is the remedy it named, never run as a formal design.

**S3. Merged-pair knapsack items for within-layer interactions.**
Merge gate_up×down, GDN qkv/z×out and attn qkv×o into joint items (400–600 options each, plus an interaction term). The per-layer MCKP stays exact.
- **Source:** TWLA 2606.13054 (pair costs 15.03 → 12.83 PPL); LoopQuant 2610.16368; CLADO. Counter-evidence: pair ≈ sum 0.91–1.10 (2609.01587); weight-only is near-additive (2607.12266).
- **Fit:** per-level.
- **Test:** CPU on V2 + S2/O15 estimates; check DP runtime.
- **Gain / risk:** 0.5–2%; may be ~0.
- **Conflicts:** #11 had the wrong (symmetric) form.

**S4. Coverage (saturation) link vs compounding.**
Fit −log(1 − f/c) as an alternative or addition to the compounding term; compare CV on ≤ 2.5 bpw mixes.
- **Source:** 2607.12266.
- **Fit:** per-level.
- **Test:** CPU, 1–2 h.
- **Gain / risk:** 0.5–2% at ≤ 2.5.
- **Conflicts:** #11, #68.

**S5. Tempering exponent on fitted sensitivities.**
Solve with s_w^β, β ∈ {0.7, 1, 1.3}; pick β by search KL of the solutions.
- **Source:** NatScale (weighting contrast: 0.194 → 0.292 KLD when removed).
- **Fit:** per-level.
- **Test:** 6 evals at 2.5 and 3.0.
- **Gain / risk:** 0–2%; slice noise (use paired CIs).

**S6. Marginal-equalization (KKT) audit at the optimum.**
At an optimal allocation, measured ΔKL per byte for a quarter step up and a quarter step down should be equal across groups. Measure 8 groups at 2.5 with antithetic pairs. Unequal marginals show where the model misprices.
- **Source:** Lagrangian optimality; SACQ 2610.41408.
- **Fit:** per-mix.
- **Test:** 16 evals (~0.6 card-h).
- **Gain / risk:** 1–3%.
- **Conflicts:** #70.

**S7. SPSA on the allocation.**
Perturb all groups by Rademacher ±¼K (bytes balanced) and evaluate antithetic pairs. Each pair gives a full-gradient estimate; step the allocation.
- **Source:** new (Spall).
- **Fit:** final.
- **Test:** 27B, 20 pairs (40 evals); accept only on search + ver3.
- **Gain / risk:** 1–3%; noisy unless E1/E2 make evals cheap.

**S8. Realization noise of sensitivities.**
Requant 4 groups with a second rotation seed at K2.5 and swap both. The difference is noise the document bootstrap doesn't see, and part of kfit2's residual floor.
- **Source:** 2609.18005 (null conditions); #41 (0.5% seed spread).
- **Fit:** diagnostic.
- **Test:** 8 evals.

**S9. A/A null and cross-card determinism.**
Run the same mix on card 0 and card 1, plus once at another chunk/step setting. Per-row scores must be byte-identical; otherwise this noise goes into the bootstrap.
- **Source:** 2609.18005; v3 §1c.4 (a chunk change once moved KLD +1.5%).
- **Fit:** diagnostic.
- **Test:** 3 evals.

**S10. Counteraction map: do isolated swaps overstate in-context cost?**
Dump block errors for 20 docs of a 2.5 mix and compute 2⟨Δh_{l−1}, Δu_l⟩/‖Δu_l‖² per layer.
- **Source:** 2609.11716 (cancels 50–63%; removing it raises final error 2.94×).
- **Why it should help:** decides how much S1/O14 matter.
- **Fit:** diagnostic.
- **Test:** needs bf16 hidden states (TP2) + one dump run, ~2 h.

**S11. Swap transfer across K levels.**
For 6 groups, swaps at K2, K2.5 and K3 on the same base check the proxy-ratio extrapolation (the fit under-predicts K1-dominated swaps by 20–36%).
- **Fit:** per-level.
- **Test:** 18 evals.
- **Gain / risk:** 1–2% at 2.5, where K2–K3 dominate.
- **Conflicts:** #44.

**S12. Leave-band-out validation.**
Fit on ≥ 3 bpw and predict 2.0–2.5, and the reverse. This is the decision-relevant extrapolation error, not random folds.
- **Fit:** diagnostic.
- **Test:** CPU.

**S13. Locally weighted kfit2 per size band.**
Kernel-weight measurements by bpw distance (σ ≈ 0.75) and fit per band.
- **Why it should help:** one global model trades accuracy at 2.5 for fit at 6–9.
- **Fit:** per-level.
- **Test:** CPU, judged by S12.
- **Gain / risk:** 0.5–2%; higher variance.
- **Conflicts:** #19 worked on a different axis.

**S14. Inversion rate near the optimum.**
How often does kfit2 misorder measured mix pairs within 0.15 bpw in each band's top quartile?
- **Source:** 2610.09227 (15.3% inversions in the top quartile vs 9.9% overall; compressive slope ≈ 0.53).
- **Fit:** diagnostic.
- **Test:** CPU.

**S15. Clip negative measured sensitivities.**
Negative swap effects (noise or slice-specific "regularization") must not buy bytes.
- **Source:** 2609.34765.
- **Fit:** per-level.
- **Test:** CPU. Minor.

**S16. Separate q vs k/v and qkv vs z sensitivities (TODO K4) via the Fisher prior.**
O7 separates them without new swaps.
- **Fit:** per-level.
- **Test:** CPU.

**S17. Sub-2 levels: one measured anchor swap per family, the rest Fisher-priced.**
- **Fit:** per-level.
- **Test:** ~6 evals.
- **Gain / risk:** matters at ≤ 2.25 (nothing below K1.5 is picked at 2.5).

**S18. Head and embedding additivity with the trunk.**
Head K4↔K5 and embedding t2↔t3 swaps on a 2.5 base vs a 4.0 base. Equal ΔKL validates kfit2's entry and exit terms.
- **Test:** 4–8 evals.

**S19. Robust allocation over category-weight uncertainty.**
Minimize the max (or CVaR) KL over plausible category weightings of a finite private draw.
- **Fit:** per-level.
- **Test:** CPU.
- **Gain / risk:** ~0 expected; insurance.

**S20. Re-price per family after each ladder change.**
An H_out requant improves families unevenly (diag CV is high on input projections). Measure 1–2 swaps per family before trusting V2-derived factors.
- **Fit:** per-level.
- **Test:** ~10 evals.
- **Conflicts:** pairs with O5.

---

## E. Cheaper evaluations

**E1. Layer-boundary hidden-state caching (suffix-only evals).**
Dump the incumbent's residual stream entering layers {16, 32, 40, 48, 56}: 481k positions × 5120 × bf16 ≈ 4.9 GB per boundary, on NVMe. A candidate identical below layer L loads the stream at L and runs L..63 + head. This is exact: each layer recomputes its own GDN state and KV from its inputs.
- **Source:** new.
- **Why it should help:** local-search candidates differ in a few layers, so this saves ~50% of trunk time on average (more for the L60–63 spikes).
- **Fit:** infra.
- **Test:** `--dump-hidden`/`--inject-hidden` in the radiance fork. Gate: per-row scores byte-identical to full evals for 10 swaps at various L.
- **Gain / risk:** 1.5–2× evals/h in local search (indirectly 1–3% KL). Cost: 1–2 days of engine work.
- **Conflicts:** complements the resident-base proposal (that cuts fixed cost; this cuts per-position cost).

**E2. Group-sequential early rejection.**
Fix a random document order. After each 20% of positions, compare the running paired difference to the stored per-doc scores of the incumbent; abort when P(better) < 2% (O'Brien–Fleming bounds). Accepted candidates always finish, so their estimates stay unbiased.
- **Source:** new.
- **Why it should help:** most candidates lose; expect 30–50% time saved.
- **Fit:** stream.
- **Test:** replay on V2's stored per-row scores (CPU, 2 h). Pass: < 2% false rejections and ≥ 30% saved.
- **Gain / risk:** indirectly 1–3%; low risk.

**E3. Information-first document order for E2.**
Order documents by historical paired-difference variance per token. Precommit the order.
- **Fit:** stream.
- **Test:** CPU.

**E4. Neyman-optimal search slice.**
From the stored scores, take per-document paired-difference variance per GPU-second by category × length. Choose documents and caps to minimize var(mean D) per second, with Horvitz–Thompson weights so the estimate stays unbiased for the corpus mean.
- **Source:** capstudy (2k cap: −27% noise per GPU-second); survey sampling.
- **Fit:** eval design.
- **Test:** CPU, 2–4 h. Pass: projected SE per GPU-second −20%.

**E5. GPU-side KL scoring + head on gen rows only (TODO).**
- **Why now:** card 1 is ~25% slower just from the f32 full-vocab copy; the gen-row head gives another +4–8%. Both compound with E1/E2.
- **Fit:** infra, ~1 day.

**E6. Two evals per card at ≤ 3 bpw (TODO I6).**
At 2.5 bpw the weights are ~8.4 GB and the GEMMs run at ~40% of peak.
- **Fit:** infra.
- **Test:** 2 evals.

**E7. Last-layer suffix simulators (generalized headsim).**
Dump the residual stream entering layer 63 (or 60–63) once, then price layer-63 options exactly in torch (layer + norm + head vs golden). Layer-63 ffn_gate_up is the spike (K2 = +80% of base KLD).
- **Source:** headsim, validated against whole-model swaps.
- **Fit:** per-level.
- **Test:** 1 dump run + a torch layer; validate with 3 whole-model swaps.
- **Gain / risk:** 1–2% from exact prices on L60–63; low risk.
- **Conflicts:** #21.

**E8. Paired half-slice exploration evals.**
Run exploration kinds on half the slice, paired against the incumbent's per-doc scores; promote only promising ones.
- **Source:** Loopy 2610.05265.
- **Fit:** stream. Needs E1 or the resident base, since the fixed 80 s dominates.

**E9. Long-probe slice economics.**
Prefill costs ~0.4 s per 1k positions. Scoring 8k–64k of the two search-pool traces is ~112k rows (~45 s plus fixed cost), far cheaper than whole 250k traces. KV memory fits.
- **Fit:** eval design (used by L1).

**E10. Fan-out of sibling candidates.**
Load a cached boundary once and run several suffixes in one process.
- **Fit:** infra, building on E1.

---

## U. Statistics and uncertainty

**U1. Posterior-mean selection (winner's-curse shrinkage).**
Rank candidates by the posterior mean of their true KL. The prior is the kfit2 prediction (log-sd ≈ 0.12); the likelihood is the measured search KL (paired SE ≈ 1.8%).
- **Source:** empirical Bayes; #72 (V2's fit-slice picks were ~3% optimistic).
- **Fit:** search and final.
- **Test:** CPU replay on V2: would posterior picks have scored better on ver2 than the measured best?
- **Gain / risk:** 0.5–2%; low risk.

**U2. Post-stratified estimates.**
Report Σ_c w_c·mean_c with calib token shares by category × length band. This corrects slice drift and makes the unmeasured bands visible (positions beyond the cap).
- **Fit:** analysis.
- **Test:** CPU.

**U3. Stratified per-document bootstrap.**
Resample within category strata, with long traces as their own stratum, matching how the private set is drawn.
- **Fit:** analysis.
- **Test:** CPU.

**U4. Robust screening statistic.**
Use a Huber or trimmed mean of paired per-doc differences for accept/reject screening only. Decisions that ship use the plain mean, because the board metric is the mean and tails count.
- **Source:** 2610.20826 (mean ≈ 20× median).
- **Fit:** stream.
- **Test:** CPU replay; check whether the final best changes.

**U5. Prediction-powered inference for finalists.**
Combine predictions over many candidates with a few measured ver3 reads to get valid CIs for "A beats B" (Angelopoulos et al. 2023).
- **Fit:** final.
- **Gain / risk:** fewer ver3 reads per decision.

**U6. Reusable holdout (Thresholdout) for ver3.**
The stream sees only noisy threshold answers, which bounds adaptive overfitting over ~150–300 reads. Add FDR control across bands.
- **Fit:** process; zero cost.

**U7. Power analysis per decision type.**
Compute the MDE for swap acceptance, band stop and finalist decisions; fix N and slice sizes beforehand, as the stop rule requires.
- **Fit:** process.
- **Test:** CPU.

**U8. Interval-coverage audit of kfit2.**
Do 80% bootstrap prediction intervals cover 80% of later measurements? Under-coverage means Thompson explores too little (ties to "score past forecasts").
- **Fit:** diagnostic.
- **Test:** CPU.

**U9. Variance decomposition of paired differences.**
Which documents and categories carry var(D)? The answer steers E4 and U4.
- **Fit:** diagnostic.
- **Test:** CPU.

**U10. Exact sign-flip permutation tests on small slices.**
Use them for ver3 (104 docs) near decision thresholds.
- **Source:** ThinkBit 2610.34886.
- **Fit:** process.

**U11. Slice × bpw-band offsets in kfit2.**
Slice composition interacts with bits (e.g. multilingual-heavy slices are more sensitive at low bpw).
- **Fit:** per-level.
- **Test:** CPU CV.

**U12. Fit-pool vs search-pool agreement.**
Refit kfit2 on search-pool measurements alone and compare the family factors.
- **Fit:** diagnostic.
- **Test:** CPU.

**U13. Realized gain per stream kind.**
Track each kind's improvement size with a CI, not only its beat rate, and drop useless kinds sooner.
- **Fit:** stream.

---

## R. Search

**R1. Lazy-greedy measured descent.**
From the 3.0 mix, repeatedly demote the group whose one-step demotion raises measured KL least per byte saved. A lazy heap refreshes stale marginals (~2.5 evals per move); one sweep yields 3.0 → 2.0 with in-context marginals.
- **Source:** Q-Strata 2608.30564. Mixtral JSD 823 (one-shot ILP) → 749; it matched AMQ with 1,378 vs 10,474 queries.
- **Fit:** per-mix.
- **Test:** 27B, ~56 groups, ~500 evals (~17 card-h; ~8 with E1/E2).
- **Gain / risk:** 2–5% (uncertain). A 1.8% SE against ~0.5% marginals is the risk; pool with S2.
- **Conflicts:** #70 (model-proposed mutations; this is measurement-driven).

**R2. Budget-neutral swaps pre-screened by an interaction proxy.**
Rank one-up/one-down candidates by a prediction that includes adjacent same-type cross terms (O15 or a diagonal bound); evaluate the top-K.
- **Source:** CASA 2609.25916 (screening by cross terms 22.41 vs self-terms 28.15 PPL).
- **Fit:** stream.
- **Test:** ~50 evals.
- **Gain / risk:** 1–3%.

**R3. Successive halving on nested document subsets.**
Score N candidates on ⅛ of the slice, keep half, extend, and reuse per-doc scores.
- **Source:** Loopy (4.2× cheaper).
- **Fit:** stream. Needs E1 or the resident base.
- **Gain / risk:** indirectly 1–2%.

**R4. Best-arm identification for finalists.**
LUCB over ver3/ver4 documents among the top 2–4.
- **Fit:** final.

**R5. Cross-band move transfer.**
An accepted 2.5 swap is proposed at 2.25 and 2.75 (frontier mixes nearly nest).
- **Fit:** stream.
- **Gain / risk:** 0.5–1%.

**R6. Uncertainty-directed exploration.**
Tilt along the top eigenvectors of the bootstrap covariance of group sensitivities, not random directions (Bayesian optimal design).
- **Fit:** stream.

**R7. Layer-band crossover.**
Combine two good mixes at the same budget by swapping depth bands, with byte repair.
- **Fit:** stream.
- **Gain / risk:** 0.5–2%.
- **Conflicts:** #18 concerned surrogates; this is an operator.

**R8. Contiguous-block mutations.**
Move adjacent layers of one family together (a structural prior that resists noise-chasing).
- **Fit:** stream.

**R9. Antithetic exploration pairs.**
Evaluate +δ and −δ together for shared-noise cancellation and a curvature estimate (pairs with S7).
- **Fit:** stream.

**R10. Recenter after every ~16 accepted moves.**
Re-measure local prices around the new incumbent.
- **Source:** 2609.09867 (after 16 executed changes, 117/192 prices reversed).
- **Fit:** stream.

**R11. Cap the number of simultaneous changes per proposal (2–4).**
- **Source:** 2609.09867 (large-support additive selection: +0.066 NLL; capped: no harm). KFIT_MAXSTEP caps bits, not counts.
- **Fit:** stream.
- **Gain / risk:** 0.5–1%.

**R12. Stop on predicted headroom.**
Stop a band when ver3 is flat **and** the best untested candidate's predicted gain is below the MDE.
- **Fit:** process.

---

## L. Long-context behaviour

**L1. Long-position group-sensitivity probe.**
On the two search-pool traces (ripgrep, cobra), score positions 8k–64k (E9) for ~12 designed group swaps (GDN in/out, attention in/out, FFN, two depth halves) on a 2.5 base. Compare each family's long/short ΔKL ratio.
- **Source:** #52/B4 (ratios flat across quants, but those quants did not isolate families; 2 docs); STEPQuant 2609.38169 and LeapQuant 2609.38166 (GDN error grows with horizon).
- **Why it should help:** ~35% of public scored positions are in the long traces, while fitting and selection see ≤ 2k positions.
- **Fit:** per-level.
- **Test:** 27B, 13 long evals (~1 card-h). Rule: a family whose ratio CI excludes 1 by > 25% gets a long-context term.
- **Gain / risk:** 0–5% (wide; the weight is large).
- **Conflicts:** #52, #73.

**L2. Position-weighted objective.**
J = (1 − w_L)·KL_short + w_L·KL_long, with w_L from calib (8 traces ≈ 20% of tokens; gen share to be checked). KL_long comes from L1's family × position multipliers in kfit2.
- **Fit:** per-level; depends on L1.

**L3. Mine the 17 existing long-trace evals.**
Regress per-position-band KL of the golden-long2 V2 mixes on their per-family bits. Do attention or GDN bits matter more at long positions?
- **Fit:** diagnostic.
- **Test:** CPU, 2 h, no GPU.

**L4. Engine floor by position.**
Score V2 8.0 by position band on the long traces. If its floor rises faster than lossy mixes do, part of the long-context KL is engine numerics (GDN state carry dtype, chunking), fixable for all sizes at once. Check radiance's prefill state dtype.
- **Source:** LeapQuant (bf16 per-step state: AIME 87.9 → 72.1); LinQuant 2610.39418 (FP32 carry ≈ lossless).
- **Fit:** engine.
- **Test:** 1–2 long evals.
- **Gain / risk:** < 1% at 2.5; possibly several % at ≥ 4 bpw.
- **Conflicts:** #49, #74.

**L5. 0.8B long-context sensitivity study.**
Build 0.8B goldens of the search-pool traces; run family swaps by position band (~20 s evals).
- **Fit:** diagnostic.
- **Test:** 4070, ~3 h.

**L6. Fisher at long context (0.8B).**
Family Fisher shares from fresh 2k windows vs windows deep in long traces with carried state. Tells whether the 27B collection (2k windows, zero initial state) misses long-context curvature.
- **Fit:** diagnostic.
- **Test:** 4070, ~2 h.

**L7. Carried-state windows in the collector.**
For long traces, carry GDN state and KV across consecutive windows, with backprop truncated to the current window. Same cost.
- **Fit:** per-level.
- **Gain / risk:** 0–2%; depends on L6.

**L8. Collector window-composition audit.**
Windows come only from documents ≥ 2048 tokens, so short chat, math and multilingual documents are excluded. The 4 qad traces (~1M tokens) may supply a large share of the first 1K windows. Count windows by category against corpus shares (and gen-row shares); pack short documents or reweight.
- **Fit:** per-level.
- **Test:** CPU, minutes.
- **Gain / risk:** 0–2%; prevents a silent skew.
- **Conflicts:** #33-13 (do before the run).

**L9. Trace-level leave-one-out robustness for finalists.**
A pick that wins only because of one trace is fragile; the private set holds different traces.
- **Fit:** final.
- **Test:** CPU.

**L10. Protocol invariance on long traces.**
Chunk 2048 vs 4096 and different checkpoint intervals must agree.
- **Fit:** diagnostic.
- **Test:** 2 evals.

**L11. Agentic guard for embedding and head.**
Tool outputs carry rare tokens (paths, hashes). Run embedding and head swaps on long-trace windows; a multilingual guard exists, an agentic one doesn't.
- **Fit:** diagnostic.
- **Test:** 2–4 evals.

---

## P. Pools, taint, overfitting

**P1. Untainted diagnostic slice (xcorp + a 27B golden).**
The published Hessians contain every calib document, verify included. A small xcorp slice with a TP2 bf16 golden measures how much the "ours vs turboderp" and H-fitting advantages shrink out of sample. Diagnostics only; check the challenge rules.
- **Source:** #50.
- **Fit:** diagnostic.
- **Test:** TP2, ~2 h for 300k tokens.

**P2. Leave-verify-out Hessians for a few layers.**
Recompute H for 4 layers without verify documents (HSPLIT path), requant, splice, and compare on verify. This measures how optimistic verify is for H-fitted quants.
- **Fit:** diagnostic.
- **Test:** ~2 h.

**P3. Slice rotation.**
Redraw the search slice every N accepted moves (re-measure incumbents).
- **Source:** #55 found no flattering, so this is insurance.
- **Fit:** stream.

**P4. Ledger use-count weighting.**
Prefer fresh documents for decisive comparisons.
- **Fit:** process.

**P5. Reserve one qad long trace as a never-trained final long check.**
Do this now, before QAD consumes the traces. All verify long documents are spent (17 uses).
- **Fit:** process; zero cost now, irreversible later.

---

## A. Activations (int8)

**A1. Factorial A8 family masks at 2.5.**
A 16-run two-level design over 8 families × {f16, A8-g128} gives each family's cost and pair interactions.
- **Fit:** per-level.
- **Gain / risk:** ≤ 0.5% KL; better speed choices.
- **Conflicts:** extends TODO C3.4.

**A2. Weight × activation additivity.**
A 2×2 design (K vs K+½) × (A8 on/off) for 3 groups. kfit2's V3 plan adds the two errors.
- **Fit:** per-level.
- **Test:** 12 evals.

**A3. A8 cost by position on long traces.**
Activation ranges drift with position (sinks, massive activations).
- **Fit:** diagnostic.
- **Test:** 2 long evals.

**A4. GDN-operand int8 check by position.**
If any int8 path touches GDN prefill operands, keep per-channel value scales and an FP32 state carry.
- **Source:** LinQuant (that pairing is near-lossless; per-tensor scales are not).
- **Fit:** diagnostic.

---

## H. Head, embedding, small tensors

**H1. GDN in_proj_a/b as priced options (TODO).**
They take 47 MB bf16 = 0.014 bpw. 2609.04098, on this model, finds a/b the least sensitive GDN projections (2.1%/2.6% layer-output error vs 10–13% for qkv/z/out under NVFP4). Offer int8 and trellis K4–K6 options.
- **Fit:** per-level (+ an int8 weight op).
- **Test:** TODO's 4 evals.
- **Gain / risk:** ~1–1.5% at 2.5; low risk.

**H2. headsim inputs dumped from a 2.5 trunk.**
TODO already redumps gen rows. Dump them from a 2.5 mix, not K6, and check that slab-plan rankings and floors hold at the product's input distribution.
- **Fit:** per-level.
- **Test:** 1 eval + CPU.

**H3. Split head KL into top-k vs leaked-tail components.**
The K3 tail floor is about leaked mass; price slab plans on both components.
- **Fit:** diagnostic.

**H4. Audit the small bf16 leftovers.**
List every non-trellis tensor's payload bytes (norms, conv1d, A_log, dt_bias, a/b). Anything > 0.002 bpw gets a priced option.
- **Test:** CPU, 30 min.

---

## X. Proxy testbed for measurement methods

**X1. Allocation-method testbed on the 0.8B.**
Brute-force ground truth: ~150 linears × 3 levels of single-tensor swaps (~450 × 20 s ≈ 2.5 h) plus a few hundred random mixes. Compare allocations at a 2.5-bpw-equivalent budget built from:
- (a) one-sided proxy + kfit2,
- (b) the Fisher prior (O3),
- (c) midpoint attribution (O14),
- (d) measured single swaps,
- (e) measured singles + pairs.

Score them on a held-out golden. This ranks measurement methods, which should transfer even when the numbers don't.
- **Source:** ROADMAP.
- **Fit:** diagnostic.
- **Test:** 4070, ~6 h.
- **Gain / risk:** indirectly 2–5%. The 0.8B is tied and has 24 layers.
- **Conflicts:** #60-04 (that study replicates known results; this ranks methods).

**X2. Simulate the stream policies on X1's ground truth.**
E2 thresholds, U1, R1, with realistic noise from document subsampling.
- **Fit:** diagnostic.
- **Test:** CPU.

**X3. Precondition for X1: mixed vs uniform at equal bytes on the 0.8B.**
If mixing gains < 5%, X1 cannot separate methods.
- **Test:** 1 h.

---

## Ranked top 20 (expected gain per unit cost)

| # | Idea | Expected KL at 2.5 bpw | Cost |
|---|---|---|---|
| 1 | O2 per-sequence Fisher logging | enabling (O3/O4/O7/O15/O17) | ~0 (code before the run) |
| 2 | L8 collector window audit | 0–2%, prevents skew | CPU, minutes |
| 3 | O1 gen-masked Fisher | 0–3% | 2 h on the 4070 |
| 4 | O5 re-price H_out options | protects 2–4% | CPU, 30 min |
| 5 | L3 mine the 17 long-trace evals | decides L1/L2 (~35% of score) | CPU, 2 h |
| 6 | H1 GDN a/b options (TODO) | 1–1.5% | 4 evals |
| 7 | O4 two-sided proxy in kfit2 | 1–3% | CPU, 1 h after diag |
| 8 | E2 sequential early rejection | 1.3–2× search throughput | CPU replay + patch |
| 9 | U1 posterior-mean selection | 0.5–2% | CPU |
| 10 | O3 Fisher-predicted swap KL | 1–3% | 1 h on the 4070 + CPU |
| 11 | L1 long-position probe | 0–5% (wide) | ~1 card-h |
| 12 | S1 + S2 re-anchored factorial design at 2.5 | 2–5% | ~1.5 card-h |
| 13 | E7 last-layer suffix simulator | 1–2% | small engineering |
| 14 | E5 GPU scoring + gen-row head (TODO) | +4–8% throughput (card 1 +25%) | ~1 day |
| 15 | O9 tempered d_r^p | 0–2% | 30 min on the 4070 |
| 16 | X1 0.8B method testbed | indirectly 2–5% | 6 h on the 4070 |
| 17 | E1 hidden-state caching | 1.5–2× local search | 1–2 days engine work |
| 18 | O14 midpoint attribution | 2–5% | 0.8B, then a TP2 night |
| 19 | L4 engine floor by position | < 1% (more at ≥ 4 bpw) | 2 evals |
| 20 | O11 output-side row-slab ceiling | gates a 2–6% format change | CPU |

## The 5 cheapest diagnostics that would most change this ranking

1. **L3:** regress the 17 long-trace V2 evals' position-band KL on per-family bits (CPU, existing data). It decides whether L1, L2 and L7 move to the top or drop out.
2. **L8 + gen-share count:** collector window composition and unscored-token share (CPU, minutes). It decides O1's priority before the 8.5 h run.
3. **O5's check:** one-sided vs two-sided proxy on the existing 0.8B `bdiag` quants (CPU, 30 min). It decides whether kfit2 must change proxies before the ladder requant.
4. **E2 replay** on V2's stored per-row scores (CPU, 2 h). It sizes the payoff of all eval-infrastructure ideas (E1, R1, S7).
5. **O3-lite on the 0.8B:** ~24 group swaps against Fisher predictions (1 h on the 4070). If they track, O4/O7/O14/O15 rise; if not, they fall.

(Also worth 3 evals at any time: **S9**, the A/A and cross-card byte-identity check. Every later paired SE assumes it.)
