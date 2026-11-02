# Low-bpw research ideas: the 2.0–2.5 bpw regime (lens **lowbpw**)

I found 97 ideas. The six with the best return per unit of cost:

1. **Fix the 27B H_out loss mask before the 8.5 h run.** Both collectors (`v3/yaqa/collect.py`, `proxy2b/yaqaB.py`) put the loss on every position, but the board scores only gen rows. Changing this costs nothing now (A1).
2. **Offer low-budget slab heads at 2.0 and 2.5.** knap4 only offers slab heads at K5–6, so the 2.5 product is stuck with a uniform K3 head. That head carries about 9% of the predicted KL (C1).
3. **Add a Hessian-aware input scale.** EXL3's `su` is plain column RMS, while the output side already has a Hessian-aware rule. Testing this takes about an hour on the 0.8B (B1).
4. **Test scale-only KL healing, with the published golden as a free teacher.** It is the largest possible gain but the most uncertain, and it would also test whether healing can live in the per-level library (D1–D3).
5. **Revisit row-slab K for the trunk using diag H_out.** It was dropped before H_out worked. A CPU ceiling check on the 0.8B decides it in minutes (A10).
6. **Test activation-denoising Hessians.** This is a sequential-Hessian substitute that does not depend on the mix, so it fits the library (B6).

## 0. What carries the error at 2.0–2.5 bpw

I applied the fitted kfit2 model (`data/v3/kfit2-model-final.json`) to the shipped V2 mixes (`v2/mixes/…-2.50bpw.mix`, `…-2.00bpw.mix`). It predicts 0.0683 at 2.5 (board 0.0666) and 0.174 at 2.0 (board 0.164), so the breakdown below is credible. Shares are of the summed per-tensor costs before compounding.

| family | bpw of total (2.5) | KL share 2.5 | KL share 2.0 |
|---|---|---|---|
| ffn_gate_up | 0.98 | **45.0%** | 45.3% |
| ffn_down | 0.51 | **21.6%** | 19.4% |
| GDN in (ssm_qkv + ssm_z) | 0.38 | 13.2% | 17.3% |
| LM head (hd-K3) | 0.142 | **8.9%** | 4.1% |
| attn_qg | 0.084 | 4.7% | 5.6% |
| ssm_out | 0.165 | 4.0% | 5.7% |
| attn_output, attn_k, attn_v | 0.07 | 2.7% | 2.5% |
| embedding (e3) + unmodelled c0 | 0.148 | ~1.7% (additional) | — |

- **Compounding across 64 layers:** it multiplies the additive sum by ×1.19 at 2.5 and ×1.38 at 2.0. kfit2's κ is about 52 at layer 0 and falls to about 2 late, so early-layer error is amplified by everything after it.
- **What V3 is expected to give:** knap4 predicts 0.0583 at 2.5. Its mix is 29% K2.25, 25% K2.5, 15% K2.75, 14% K2 and 10% K3, with a t2 embedding (0.095 bpw).
- **Bytes to KL:** from the board curve near 2.5, 0.01 bpw is worth about 1.6–1.9% KL. A half-K step of the head is 0.024 bpw, about 4% KL. I use this rate to price ideas that cost bytes.
- **Implications:**
  - Anything that improves the K2–K2.75 quantizer on gate_up and down is the main lever.
  - The head is the most KL-dense item per byte.
  - The embedding barely matters at 2.5 (about 1% of KL).
  - Compounding is about a sixth of the KL at 2.5 and over a quarter at 2.0, so cross-layer error methods are worth more at 2.0.

Two facts from the code shape several ideas below:
- In `regularize()`, the input scales are `su = block_rms(weight, dim=1)`, with no Hessian term. The output side has the Hessian-aware t ∝ (σ²/s)^{1/4} rule (`sensitivity_out_scales`, the one used in gate 2d).
- `prepare_H_out` rotates H_out with the output-side 128-Hadamard. A diagonal H_out therefore becomes dense inside each 128-block, so `ldlq_2hess` gets real error feedback across output rows within blocks. That is the mechanism behind #33-12's gain.

Notation used throughout:
- **Hardware:** "4070" is the 0.8B/2B proxy box, "the GPU box" is the 2× R9700 27B box, and "CPU" is second CPU box or CPU box.
- **Gains:** relative change in mean KL at 2.5 bpw unless stated.

---

## A. Output-side information (building on #33)

**A1 · Gen-masked loss for the H_out collection.**
Neither collector masks the loss; both put sampled-label cross-entropy on every position. The board scores only gen-masked rows. Keep the full forward context, but put the sampled-label loss only on rows the metric scores (gen.u8 for calib windows; assistant-token masks for xcorp). H_out then measures the sensitivity that is actually scored.
- *Source:* new (metric matching).
- *Why it should help:* agentic traces carry long tool outputs and prompt prefixes whose gradient mass may differ from that of scored tokens. The diagonal is cheap to get wrong in a direction the metric never sees.
- *Fit:* per-level.
- *Hardware and cheapest test:*
  - On the 4070, collect 1K-sequence diagonals masked and unmasked (about 26 min each, scaled from the 8K run's 207 min).
  - Quantize K2 and K3 with each and score on g08-search1.
  - Rule: use the mask if its diagonal has cosine < 0.99 to the unmasked one, or if it wins with z ≤ −2.
  - About 3 h total.
- *Gain:* 0 to −2%. *Risk:* none, but decide **before** the 27B 8.5 h run.
- *Conflicts:* none.

**A2 · Family ablation of diag H_out.**
Turn the diagonal on for one family at a time: MLP gate/up, down, GDN in (qkv/z), GDN out, attention in, attention out, plus all together.
- *Source:* extends #33-12.
- *Why it should help:*
  - #33-09 shows the diagonal is flat for residual writers (CV 0.17–0.2), so their gain should be about 0.
  - It is very uneven for input projections (CV 1.0 gate/up, 1.7 qkv, 4.0 z).
  - The full gate/up H_out *hurt* at K2.
  - Dropping families that hurt, or that are inert, de-risks the 27B swap.
- *Fit:* per-level.
- *Test:* 0.8B, 7 arms × K2/K3, about 2 h on the 4070.
- *Gain:* 0 to −2%, plus scoping information for the 27B run.
- *Conflicts:* #33-12 (only "all" was tested).

**A3 · Tempered diagonal.**
Use diag(H_out)^p with trace normalisation, p ∈ {0.25, 0.5, 0.75, 1, 1.5}.
- *Source:* 2609.38521 (ShamAN-Q: p = ½ beat p = 1, PPL 17.46 vs 19.21 untempered on 1.7B at 1 bpw); 2609.08135 (KBBQ, interior λ < 1 optimum); 2610.40345.
- *Why it should help:* at K2 the relative error is large (about 0.07–0.09), well outside the quadratic regime, so a flatter metric may generalise better.
- *Fit:* per-level.
- *Test:* 0.8B K2/K3 × 4 exponents, about 1.5 h on the 4070.
- *Gain:* 0 to −3% on top of p = 1. *Risk:* low.
- *Conflicts:* #33-12 tested p = 1 only.

**A4 · Estimator granularity: why per-token H_out failed at 27B.**
Feed four estimators of the diagonal into `ldlq_2hess` at matched data:
- (a) per-token Σ‖x_t‖² dy_t² (the gate 2b–2d estimator);
- (b) per-token Σ dy_t² (H_Z's E[G_ZᵀG_Z]);
- (c) per-chunk summed weight gradient (256 tokens);
- (d) per-sequence Sketch B.

Also add (e) Sketch B on 1K × 4096-token windows vs 2K × 2048 at equal tokens.
- *Source:* 2610.40345; #33; new analysis.
- *Why it should help:* with sampled labels, E[G_bG_bᵀ] keeps the cross-position terms ∂ℓ_s/∂y_t · ∂ℓ_s/∂y_u that come from the same loss s. Per-token estimators drop them, and attention and GDN make them large. Gates 2b–2d never tested a per-token diagonal *inside ldlq_2hess* (2d used it as output scales). The test settles whether the estimator or the use site failed, and whether longer windows capture more GDN memory.
- *Fit:* per-level.
- *Test:* 0.8B, five 1K collections plus K2/K3 quants, about 4 h on the 4070.
- *Gain:* information, and possibly a cheaper or better 27B collection.
- *Conflicts:* #33-02 to #33-06 (different use site).

**A5 · Operating-point H_out.**
Collect the 1K diagonal on a uniform-K2.5-quantized 0.8B instead of bf16. This stays per-level compatible because each level's library entry can use the gradients of its own level.
- *Source:* 2605.23078 (GEMQ: sensitivities from the quantized model beat FP, 9.66 → 9.01 PPL); 2609.38521 (fresh statistics).
- *Why it should help:* at K2 the bf16 curvature is far from the curvature along the actual error path.
- *Fit:* per-level.
- *Test:* 0.8B, 1 collection plus 4 quants, about 1.5 h.
- *Gain:* 0 to −3%. *Risk:* the KL expansion point is bf16, so it could also hurt.
- *Conflicts:* none.

**A6 · Output scales re-tested with the Sketch B diagonal.**
Use `sensitivity_out_scales` (t = (σ²/s)^{1/4}) with the good estimator. Run it alone and stacked on two-sided diag LDLQ, and sweep the ¼ exponent.
- *Source:* 2607.07964 (KronQ S_G: removing S_X and S_G cost 11.92 → 12.94 PPL at W2).
- *Why it should help:* gate 2d failed with the noisier per-token diagonal, and the output-scale route is free at inference.
- *Fit:* per-level.
- *Test:* 0.8B, 4 arms, about 1 h.
- *Gain:* 0 to −3%.
- *Conflicts:* #33-06 (different estimator).

**A7 · Out-scales on/off per tensor decided by the Sketch B diagonal.**
Apply `out_scales_ratio` with s = diag H_out, replacing upstream's calibration-statistic rule.
- *Source:* exllamav3 auto rule.
- *Why it should help:* #48 failed because the auto rule misjudged sensitivity, not because switching scales off can never pay.
- *Fit:* per-level.
- *Test:* 0.8B, about 30 min.
- *Gain:* 0 to −1%.
- *Conflicts:* #48.

**A8 · Output damping per family.**
Sweep sigma_reg_out ∈ {0.003, 0.01, 0.03, 0.1} by family. For a diagonal, damping is shrinkage toward flat.
- *Source:* #33-04 (0.3 vs 0.025 mattered for the full H_out).
- *Fit:* per-level.
- *Test:* 0.8B K2, about 1 h.
- *Gain:* 0 to −2%.
- *Conflicts:* #38 (input side only).

**A9 · Reliability shrinkage of the 27B diagonal.**
Split the planned 256/512/1024 snapshots into halves. Per tensor, shrink the diagonal toward its mean by split-half reliability (James–Stein).
- *Source:* #33-03 (bf16 noise grows toward early layers); D2 split-half method.
- *Why it should help:* 27B mid-depth gradients are noisier than the 0.8B's (diagonal halves cos ≥ 0.997 there). This protects the 27B swap test.
- *Fit:* per-level.
- *Test:* CPU on the 27B snapshots, then the K2 swap.
- *Gain:* risk reduction; 0 to −1%.
- *Conflicts:* none.

**A10 · Row-slab K for the trunk driven by diag H_out (revisit).**
- *Mechanism:*
  - Sort output rows by diag(H_out). For the MLP, permute intermediate channels and fold the permutation into down's input order, which is free at inference.
  - Split gate/up (and GDN z/qkv) into 2–3 row slabs, each a separate library item quantized at every K.
- *Source:* 2610.07457 (within-tensor sensitivity CV 0.47–2.10); 2606.15652 (error is block-concentrated); design §6b.
- *Why it should help:*
  - Diagonal CV is about 1.0 for gate/up and 4.0 for z (#33-09).
  - The high-rate ceiling ½·log₂(AM/GM) for a CV-1 spread is about 0.2 bit/weight at row granularity. Even 0.05 bit on gate/up is about 4% KL.
  - E3c's 0.05-bit ceiling used input-side norms, which cannot tell rows apart.
- *Fit:* per-level. Slabs are independent sub-tensors, but they need H4's encoding (split GEMMs or per-row-block K).
- *Test:*
  - First, the CPU ceiling per family on the 0.8B diagonal.
  - Then a 0.8B fake-quant test: gate/up top 25% of rows at K3, the rest at K2.33, at equal bits, vs uniform K2.5. Rule: z ≤ −2 and ≥ 3%. About 2 h on the 4070.
  - 27B plumbing: 2–4 days.
- *Gain:* −2 to −10% (speculative; two-sided LDLQ already captures some of it). *Risk:* the diagonal may not predict KL at row level.
- *Conflicts:* design §6b ("dropped"), #29-02. Both predate #33-12.

**A11 · Sensitivity-sorted intermediate channels without slabs.**
Permute the MLP intermediate channels by diag H_out of gate+up, so each 128-Hadamard block is homogeneous.
- *Why it should help:* it is the prerequisite for A10. On its own it is ambiguous: it removes within-block diversity that `ldlq_2hess` exploits, but it also changes down's input blocking.
- *Fit:* per-level, free.
- *Test:* 0.8B, about 30 min.
- *Gain:* ±2%.
- *Conflicts:* none.

**A12 · Per-head K for GDN projections.**
GDN heads are exactly 128 rows (z, the v part of qkv) or 128 columns (ssm_out). Price each head by its share of diag H_out and allocate per head.
- *Source:* 2609.38169 (STEPQuant: gate half-life vs accumulated state error, ρ ≈ 0.80 over 2,304 Qwen3.8 heads).
- *Fit:* per-level, with H4.
- *Test:* CPU correlation of per-head H_out mass vs half-life on the 0.8B, then a fake-quant test with 2 head groups.
- *Gain:* −1 to −3%.
- *Conflicts:* as A10.

**A13 · Analytic SwiGLU output weights (forward-only diagonal for gate/up).**
Use w_gate,r = ‖W_d[:,r]‖²·E[(SiLU′(g_r)u_r)²] and w_up,r = ‖W_d[:,r]‖²·E[SiLU(g_r)²].
- *Source:* 2610.41408 (SACQ).
- *Why it should help:*
  - If it gets ≥ 80% of the Sketch B gain, it allows cheap refreshes (operating-point, healed or sequential models) without any backward pass.
  - It is also a 27B sanity check on the collected diagonal.
  - It targets gate/up, the family with 45% of the KL.
- *Fit:* per-level.
- *Test:* 0.8B: Spearman against Sketch B plus 2 quants, about 1 h.
- *Gain:* mainly cost saving; possibly a less noisy estimator.
- *Conflicts:* none.

**A14 · Analytic q/k output metrics for the 16 attention layers.**
Weight q_proj per head by the key Gram and k_proj by the query Gram, forward-only.
- *Source:* 2605.17757 (OSCAR); 2609.24298 (KV-COBRA bound F(p) ⪯ ½I).
- *Fit:* per-level.
- *Gain:* ≤ −0.5% (attention is about 6% of KL). Low priority.

**A15 · H_out-weighted scale refit.**
`refit_scales` refits su by least squares with all rows weighted equally. Weight rows by diag H_out; the sv refit is per-row, so it is unaffected.
- *Fit:* per-level.
- *Test:* 0.8B, about 30 min.
- *Gain:* 0 to −1.5%.

**A16 · Two-sided proxy in kfit2.**
Recompute hproxy as tr(E·H_in·Eᵀ·diag H_out) for every library option once the 27B diagonal exists, then refit kfit2. Compare CV log-RMSE and pair ranking against the one-sided proxy.
- *Source:* 2609.33923 (RAM, on Qwen3.8-27B: a downstream-weighted signal allocates better than the layer objective, 5.99 vs 6.09 PPL at IQ3_M size); 2609.02451.
- *Why it should help:* a free check of whether H_out predicts KL at 27B before the swap test, and possibly better allocation.
- *Fit:* allocation.
- *Test:* CPU, hours.
- *Gain:* 0 to −3%.
- *Conflicts:* #18/#20 (re-rankers null), but this is a physical feature, not a black box.

**A17 · KL fixed-point (matrix-normal ML) estimator for the output factor.**
Use inverse-quadratic token weights (bounded leverage) instead of Sketch B weighting. Keep the calib H_in, since H_in replacements failed.
- *Source:* 2609.38521 (condition number about 40 vs several hundred; suppresses massive-activation tokens).
- *Fit:* per-level.
- *Test:* 0.8B, one 1K collection plus quants, about 2 h.
- *Gain:* 0 to −2%.
- *Conflicts:* #33-10 (H_in swap) only if done wrongly.

**A18 · K-dependent full vs diagonal H_out.**
bmix (full except gate/up) beat the diagonal at K3 by 2% but lost at K2. The 2.5 mix sits at K2.25–2.75, so measure the crossover at K2.5 on the 0.8B before paying for a 51 GB full collection at 27B. If full H_out is used anywhere, use FastKron's recursive anti-diagonal schedule.
- *Source:* 2608.06291 (FastKron).
- *Fit:* per-level.
- *Gain:* ±1%.

**A19 · Weighted Viterbi metric inside tiles.**
Make the in-tile cost Σ ω_ij·(t−q)² with ω_ij = D_in,j · D_out,i (the diagonals of the block LDLs). Today's Viterbi is Euclidean inside a tile.
- *Source:* 2610.09212 (CurveTQ: +2.44 accuracy points native, inert after full RHT since Γ_b drops from 0.038 to 0.00013 bit).
- *Why it should help:* with a two-sided diagonal, the output-side pivots vary within 16-row tiles.
- *Fit:* per-level. Quantizer kernel only (qtf2 cost function); the decoder is unchanged.
- *Test:* first, CPU Γ_b on 0.8B tensors with the diagonal.
- *Gain:* 0 to −2% if Γ_b > 0.01 bit.
- *Conflicts:* none.

**A20 · Piggyback statistics on the 27B backward run.**
Since the 27B run pays for a full fp32 backward anyway, also accumulate:
- the gradient covariance at the embedding output (5120², for C7);
- per-head H_out mass (for A12);
- diag(E[dy²]) per token (for A4's comparison).
- *Fit:* collection.
- *Cost:* about 0 extra.
- *Gain:* enables other ideas.

---

## B. Quantizer changes (per-level library)

**B1 · Hessian-aware input channel scales.**
Set t_j ∝ (σ_j²/H_jj)^{β/2}, sweeping β ∈ {0, 0.25, 0.5}. β = 0 is today's column-RMS rule; β = ½ is the QuIP/KronQ fixed-rate optimum and the mirror of the existing output-side rule. Keep the scaled Hessian on.
- *Source:* 2607.07964 (KronQ: removing S_X and S_G cost +8.6% PPL at W2 with Hadamard + GPTQ); QuIP diagonal rescale.
- *Why it should help:* massive-activation channels make H_jj span orders of magnitude (mean diagonal about 9× the median), and the trellis spends equal resolution on every column.
- *Fit:* per-level (requant).
- *Test:* 0.8B K2/K3 × 3 values of β, about 1 h on the 4070. Rule: z ≤ −2.
- *Gain:* 0 to −5%. *Risk:* LDLQ feedback may already compensate. su also scales the activations before rotation, so check the A8-g128 error with the E1c proxy.
- *Conflicts:* #42 concerns sharing su, not this.

**B2 · Block coordinate-descent polish after LDLQ.**
After LDLQ, revisit each 16-column block: compute its target from the errors of *all* other blocks (earlier and later), rerun Viterbi, and accept only if tr(EHEᵀ) falls. Run 1–3 sweeps.
- *Source:* 2610.32127 (MPQ, 2-bit D4: objective 359.9 → 340.2 (−5.5%), WikiText-2 PPL 20.3 → 14.6; E8P12 on Llama-3-8B 15.01 → 12.76; 12.7% of blocks change at 2 bit).
- *Fit:* per-level; quantizer time ×2–4.
- *Test:* 0.8B K2; a Python loop around `quantize_tiles` in ourq (about 1 day of engineering) plus 1 h of runs. Rule: proxy −2% and KL z ≤ −2.
- *Gain:* −1 to −4%. *Risk:* overfitting the Hessian sample (use B12).
- *Conflicts:* none.

**B3 · Alternate scales and codes.**
After `refit_scales`, rerun LDLQ and Viterbi with the refit su/sv/g_scale, then refit again (2–3 rounds), accepting on the objective.
- *Source:* 2610.37336 (CoS-VQ on Qwen3.8-27B at 2 bpw: WikiText-2 7.84 vs QTIP 8.29; same-start scale-only → joint 21.0 → 13.7 from a weak start); 2609.38599 (JARQ: ≤ 1% on strong GPTQ hosts).
- *Fit:* per-level.
- *Test:* 0.8B, about 1 h.
- *Gain:* −0.5 to −2%.

**B4 · LDLQ-aware g_scale search.**
Score global-scale candidates with a replayed LDLQ objective on a column subset (Schur complement δᵀS_{G|P}δ, or proxy-static 1/U_qq² weights) instead of tile MSE.
- *Source:* 2609.36654 (Euclidean instead of Schur scoring: −90 of 12,047 correct answers); 2609.37416 (proxy-static lowered PPL in all 6 W2 runs).
- *Fit:* per-level.
- *Test:* 0.8B, about 1 h.
- *Gain:* −0.5 to −2%.
- *Conflicts:* #39 (blocked); this redoes it with the right metric.

**B5 · Block act-order.**
Run LDLQ over 16-column (or 128-column) blocks in order of tr(H_bb), largest first. Storage order is unchanged, because tiles do not care about quantization order.
- *Source:* 2610.00432 (energy ordering: 8.06 → 8.00 PPL at 2 bit).
- *Fit:* per-level; free.
- *Test:* 0.8B, about 30 min.
- *Gain:* −0.3 to −1%.

**B6 · Activation-denoising Hessians.**
- *Mechanism:*
  - Run one forward pass that injects Gaussian noise σ_c = γ·max|x_c| at every linear input and propagates it downstream.
  - Accumulate C = E[xηᵀ] and N = E[ηηᵀ].
  - Quantize W·F under the metric G = H + αI + λ²N + λ(C+Cᵀ). Defaults γ = 0.05, α = 0.01, (λ, ρ) ≈ (0.4, 0.3).
  - The result does not depend on the mix, so it fits the library.
- *Source:* 2610.07522. It closes 18–31% of the parallel-to-sequential gap at W3 GPTQ and 38% at W2 QuIP# on 1B, and beats sequential calibration on 8B/13B. It also shrinks ‖WF‖, which matters for a trellis whose error scales with row RMS.
- *Fit:* per-level (requant).
- *Test:*
  - On the 4070: a 0.8B noise pass (forward-only, about 30 min) plus a 3-point (λ, ρ) grid × K2/K3, about 2.5 h. Rule: K2 ≥ 3% better with z ≤ −2.
  - 27B: one forward-only night on both cards. C and N take about twice the Hessian storage, roughly 100 GB.
- *Gain:* −2 to −8%. *Risk:* knobs need tuning per model; interaction with H_out is unknown (I4).
- *Conflicts:* ROADMAP "sequential Hessians" (V3.5+); this is the library-compatible stand-in.

**B7 · Per-level sequential (asymmetric) target.**
For level K, collect X̃Xᵀ and X̃X̃ᵀ from a uniform-K prefix on 0.5–1M tokens. Blend the correction onto the published 10M-token H, and use the GPTAQ target W·XX̃ᵀ(X̃X̃ᵀ)⁻¹.
- *Source:* 2504.02692 (GPTAQ); 2610.07522.
- *Why it should help:* about 90% of 2.5-mix weights are within K2–2.75, so a level-matched prefix is representative. Blending with the published H avoids the token-noise caveat in the ROADMAP.
- *Fit:* per-level; about 2× a level's quant time.
- *Test:* 0.8B K2/K3, half a day.
- *Gain:* −2 to −6%.
- *Conflicts:* ROADMAP item; B6 is cheaper.

**B8 · Intra-layer sequential quantization only.**
Quantize q/k/v/gate_up first. Then quantize down, ssm_out and attn_output with inputs X̃ from their already-quantized siblings in the same layer at the same level. This needs only a one-layer forward.
- *Source:* GPTAQ, local variant.
- *Why it should help:* down's input is the product of two quantized projections.
- *Fit:* per-level.
- *Test:* 0.8B, 2 h.
- *Gain:* −1 to −3%.

**B9 · First-order term with a trust region.**
Add G²PTQ's β[gH⁻¹…] correction, with g taken from the quantized-prefix block-output loss.
- *Source:* 2609.31009 (Qwen3.5-2B W3 KL 0.228 → 0.177; Qwen3-0.6B 1.50 → 0.333; without the trust region, KL 25.3).
- *Fit:* per-level with a uniform-K prefix.
- *Test:* 0.8B, about 1 day of engineering.
- *Gain:* −2 to −6%; overlaps B6 and B7.

**B10 · Depth-scheduled denoising strength.**
Make (λ, ρ) in B6 depth-dependent, either stronger late (error accumulates) or front-loaded (early error is amplified, κ ≈ 52 early).
- *Source:* 2610.07522 (sequential calibration of only the first quarter closes 40–60% of the gap).
- *Fit:* per-level.
- *Test:* 0.8B after B6.
- *Gain:* −0.5 to −2% over B6.

**B11 · Null-space ridge shrink of the target.**
Quantize W′ = W·H(H + μ·mean(diag)·I)⁻¹ instead of W. This removes weight energy along directions the activations barely excite, lowering row RMS at a small output cost. Choose μ per tensor on split-half H.
- *Source:* 2605.04738 (OSAQ: on top of QuIP at W3, 6.50 → 6.37).
- *Fit:* per-level.
- *Test:* first, the CPU share of row energy in eigen-directions below 10⁻⁴·mean.
- *Gain:* 0 to −2%.

**B12 · Split-half acceptance for every refinement.**
Gate B2, B3, B11 and A19 on a held-out-half Hessian, using the D2 infrastructure.
- *Source:* 2610.11226 (DRQ: refining for lower reconstruction loss raised held-out PPL in 73.8% of AWQ layers).
- *Fit:* process.
- *Gain:* risk control.

**B13 · Amplitude de-biasing at low K.**
MSE-optimal quantizers shrink outputs: ‖ŵ‖² ≈ (1−D)‖w‖², about 3.5% amplitude at K2. For each level and family, scale sv by (1−D_K)^{−γ}, with γ ∈ {0, 0.5, 1}.
- *Source:* 2610.07121 (SoloQ's unbiased-projection scale ‖z‖²/⟨q,z⟩); new.
- *Why it should help:* residual writers then under-update the residual stream in every layer, a systematic effect that compounds.
- *Fit:* per-level, zero bytes.
- *Test:* 0.8B fake-quant by rescaling stored sv (no requant), about 30 min.
- *Gain:* 0 to −2%; also a probe of healing headroom.
- *Conflicts:* #39 concerns the search scale before rounding, not output amplitude.

**B14 · Per-survivor intra-tile feedback in Viterbi.**
The Euclidean in-tile Viterbi ignores coupling inside each 16-column block (D_bb). Per-survivor processing would carry each survivor's running feedback.
- *Source:* new (classic PSP).
- *Fit:* per-level; heavy kernel work.
- *Test:* CPU free-start Viterbi on sample tiles to bound the gain first.
- *Gain:* 0 to −3% (speculative).

**B15 · Extra fp feedback columns, only for outlier tensors.**
For ssm_out L0 and attn_output L3 (E3c outliers), append r unquantized directions X·V_r to LDLQ so early rounding error has somewhere to go.
- *Source:* 2606.01412.
- *Fit:* per-level; costs r(m+n)·16 bits.
- *Gain:* ≤ −0.5%. Low priority.

**B16 · Best of two LDLQ orders.**
Run natural and act-order LDLQ and keep the lower objective per tensor.
- *Source:* new.
- *Fit:* per-level; 2× quant time.
- *Gain:* 0 to −1%.
- *Conflicts:* #41 (seed spread 0.5%) suggests the spread may be small.

---

## C. Head and embedding at low budgets

**C1 · Low-budget slab heads in the 2.0 and 2.5 menus.**
knap4 offers slab heads only at K5/5.5/6 (`KN_HEADSLAB`), so at 2.5 it takes a uniform hd-K3 head (0.142 bpw, about 9% of KL).
- *Mechanism:* offer slab plans at budgets 2.25–3.75 in 1/8 steps, with mean-row donors at K2/2.25/2.5/2.75/3/3.5 and a K2 floor, priced by headsim including the rare-token guard.
- *Source:* #34's `lcurve`: K2-floor slab at budget 3.0 gives 0.0060 vs uniform K3 0.0079 (−24%); at budget 2.125, −48% vs uniform K2.
- *Fit:* per-level knapsack option.
- *Test:* donor quants on the GPU box (1–2 h each, about 22 GB host RAM), headsim on CPU, knap4 at 2.0/2.5, then one whole-model check.
- *Gain:* −1.5 to −3% (either −24% head KL, or about 0.02 bpw moved to the trunk). *Risk:* rare-token KL at low floors (×2.5 already seen at a K4 budget).
- *Conflicts:* #34 (running); this extends its menu downward.

**C2 · Two-sided LDLQ for the head with its exact output factor.**
Use diag H_out = mean p(1−p) per vocabulary row (already computed for slab ranking) in `ldlq_2hess`. The output Hadamard over vocabulary rows turns this into feedback within 128-row blocks.
- *Source:* 2608.12026 (SoftWater: 6.5–8.3× lower head KL at 2 bits with variable rate).
- *Why it should help:* the head's output curvature is exact, not estimated, and extremely concentrated (400 rows hold 50% of the Fisher mass).
- *Fit:* per-level (head donors).
- *Test:* headsim, K2.5/K3 donors, both unsorted and after slab sorting (sorting flattens the within-block diagonal).
- *Gain:* −15 to −30% head KL, about −1.5 to −3% total.
- *Conflicts:* gate 2d (trunk estimator), which does not apply here.

**C3 · Low-rank head correction in the whitened metric.**
Fit rank 4–16 int8/bf16 factors to (W − Ŵ)·T_p with T_p = C_λ^p, optionally with Fisher row weights. At r = 16 this costs ≤ 0.002 bpw.
- *Source:* 2608.02703 (ARCHead vs a GPTQ-INT4 head: relative PPL 1.0104 → 1.0058, −45% of the excess, at equal size).
- *Fit:* per-level head option; the plugin needs two thin GEMMs.
- *Test:* first, a CPU randomized SVD of E·C^{1/2} (and Fisher-weighted) for the K3 head. Rule: > 5% of the weighted error in the top 16 directions.
- *Gain:* −0.5 to −3%. *Risk:* the LDLQ residual may be white, giving no gain.

**C4 · Softmax shift search beyond mean-row.**
Try W_t = W − tμ1ᵀ with t ∈ {0, 0.5, 1, 2, 4} per donor, priced by headsim.
- *Source:* 2609.31291 (Phi selected t = 4; no gain on Qwen3.5 with GPTQ).
- *Fit:* per-level.
- *Gain:* 0 to −1%.
- *Conflicts:* #34 (mean-row is t = 1).

**C5 · Final-norm and head-row-scale healing from dumped head inputs (no trunk backward).**
- *Mechanism:*
  - Dump the head inputs of the shipped 2.5 mix on qad-pool windows (EXL3_AQ_DUMP hook), with golden top-512 targets.
  - Fit the final RMSNorm gain (5120 values) and the head's row scales svh (248k values, shrunk toward 1) by KL.
  - Both fold into existing tensors, so it costs zero bytes. It corrects head quantization and the trunk-induced output shift together.
- *Fit:* final pass on shipped mixes (per-mix, but cheap).
- *Test:* on the GPU box, one dump eval (about 1 h, roughly 5 GB per 500k positions), a GPU fit (minutes), then search and ver3 evals. Rule: −1% with z ≤ −2.
- *Gain:* −1 to −4%. *Risk:* overfitting rare rows; use strong shrinkage and a held-out check.
- *Conflicts:* none (a no-backward subset of V5 QAD).

**C6 · Global logit temperature τ.**
Fold τ into the final norm gain and sweep τ ∈ {0.97, 0.985, 1, 1.015, 1.03} on the 2.5 mix (search slice, 5 evals, about 20 min).
- *Source:* new.
- *Why it should help:* it is the cheapest probe of systematic sharpening or flattening, and so of healing headroom.
- *Fit:* final pass.
- *Gain:* 0 to −1%.

**C7 · Embedding with a downstream column metric.**
LDLQ the embedding rows against H_col = E[g₀g₀ᵀ] (gradient at the embedding output, collected via A20), uniform across rows.
- *Source:* new; ARCHead-like.
- *Why it should help:* it does not weight by token frequency (which is what failed in #32), so rare and multilingual tokens are not starved. The embedding is about 1% of KL, so the value is mostly bytes: t2 → t1.75 at equal KL is about 0.012 bpw, about 2%.
- *Fit:* per-level.
- *Test:* CPU embtrellis plus one swap on golden-ml.
- *Gain:* −0.5 to −2%.
- *Conflicts:* #32, #35 (both row-weighted).

**C8 · Quarter-step heads at the 2.5 margin.**
A half step of the head is 0.024 bpw (about 4% KL); quarter steps let the knapsack land on the knee. This is redundant if C1's slab plans already give near-continuous budgets.
- *Fit:* per-level.
- *Gain:* −0.5 to −1%.
- *Conflicts:* design §6b (quarter heads only as a fallback).

**C9 · Rare-token and multilingual guard in low-budget head pricing.**
Price C1 and C2 options with headsim's guard KL plus golden-ml. Low floors were 3–12× worse without a floor.
- *Fit:* process. Prevents a ver4 surprise on multilingual text.

**C10 · Combine slab sorting with two-sided LDLQ.**
Test sorted-slab and two-sided LDLQ head variants against each other. Sorting groups similar Fisher values into 128-blocks, which kills the within-block diversity that C2 uses.
- *Fit:* per-level.
- *Test:* headsim only.
- *Gain:* picks the better of C1 and C2, or a hybrid.

**C11 · Head input Hessian from scored positions only.**
Logits at unscored positions have **zero** effect on the metric. So the head's H_in (and Fisher) should come only from gen-masked rows. This is unlike the trunk, where every position feeds later ones.
- *Source:* new.
- *Fit:* per-level.
- *Test:* recompute the head H_in from gen rows, requant the K3 donor, headsim.
- *Gain:* 0 to −1.5%. *Risk:* fewer tokens.

---

## D. Healing and QAD (final pass on shipped 2.0/2.5; D3 is per-level)

**D1 · Scale-only KL healing with frozen codes.**
- *Mechanism:*
  - Train su, sv, all RMSNorm gains (including the GDN norm and q/k norms), A_log/dt_bias and conv1d.
  - Loss: KL to the golden top-512 (renormalised, plus a residual bucket), on gen-masked rows of qad-pool windows.
  - The teacher is free if the published golden covers the qad pool: no bf16 forward needed.
- *Source:* 2610.34886 (ThinkBit: scales-only KL 0.641 → 0.282 on 1.7B at 2 bit); 2609.35465 (Tetra: row-scale KL retraining +1.7–3.3 MMLU at 2.7 bpp in 1.1–1.8 GPU-h); 2610.21808 (RoleQ scale distillation: 400 steps).
- *Fit:* final pass on shipped mixes.
- *Test:*
  - On the 4070 first: 0.8B uniform K2/K2.5 plus one knap mix (fp32, a few hours). Rule: ≥ 5% on g08 held-out.
  - 27B: torch EXL3 decode reference (already a backlog item) plus training on both cards. At about 5 s/sequence bf16 forward+backward, 3.7M tokens is about 2.5 h per epoch.
- *Gain:* −8 to −20% (the largest and most uncertain). *Risk:* bf16 backward noise in early layers (#33-03); overfit at 3.7M tokens.
- *Conflicts:* ROADMAP V5. I recommend pulling it forward for 2.5 only.

**D2 · Late-layer healing only (layers 48–63).**
Truncate the backward at layer 48.
- *Source:* field report in ROADMAP (last 16 layers about 22% vs 6–13% for all layers); #33-03 (bf16 down_proj cosine 0.999 at L63).
- *Why it should help:* gradients are reliable there and the backward costs about a quarter.
- *Fit:* final pass.
- *Test:* 0.8B (last 6 of 24 layers vs all).
- *Gain:* −5 to −15%.

**D3 · Library-level healing.**
- *Mechanism:*
  - Heal uniform-K models (K2 to K3 in quarter steps) separately.
  - Store healed su/sv per (tensor, level).
  - Mixes assemble healed tensors, and the knapsack prices healed options.
- *Source:* new.
- *Fit:* per-level (would make healing allocation-aware and avoid per-mix runs).
- *Test:* 0.8B: a 50/50 K2/K3 mix using level-healed scales vs mix-specific healing vs none. Rule: level-healed recovers ≥ 60% of the mix-specific gain.
- *Gain:* turns D1's gain into a library property.
- *Conflicts:* none.

**D4 · Block-wise healing, then short end-to-end KL.**
Train scales per block to match cached teacher block outputs, with quantized-prefix inputs and RMSE loss. One block in memory at a time, then a short end-to-end KL stage.
- *Source:* 2606.10520 (UniSVQ); 2609.33601 (JustQuant stage 1); 2610.00983 (RMSE loss).
- *Fit:* final pass; fits one card; needs G4's cache.
- *Gain:* −3 to −10% (a warm start for D1).

**D5 · Closed-form asymmetric scale refit.**
Sequentially per linear, set sv_i = ⟨ŵ_iX̃, w_iX⟩ / ‖ŵ_iX̃‖² (and fit su by least squares), with X from bf16 and X̃ from the quantized prefix.
- *Source:* GPTAQ target applied to scales.
- *Fit:* final pass; no backward, one lockstep stream.
- *Test:* 0.8B, about 1 h.
- *Gain:* −1 to −4%.

**D6 · KL-trained low-rank adapters, priced in bytes.**
Rank-8 bf16 adapters on all linears cost about 0.026 bpw (about 4.5% KL-equivalent at 2.5). Initialise B = 0, train by KL with frozen codes and healed scales, and keep them unmerged.
- *Source:* 2606.04238 (Recover-LoRA; OLoRA init diverges); 2609.39223 (ranks 4–256 within 0.004 KL of each other).
- *Fit:* final pass plus format (two thin GEMMs per linear).
- *Test:* 0.8B: D1 vs D1 + rank 4/8 at equal total bytes.
- *Gain:* 0 to −5% net. *Risk:* may not beat spending the bytes on K.

**D7 · Tail-KL data selection.**
Rank qad windows by the mean of the top-25% token KL of the current quant, and train on the top half.
- *Source:* 2610.14934 (FidelityQ: worst decile carries 45% of KL; 17.05 → 16.69 vs random selection).
- *Fit:* recipe for D1.
- *Gain:* −0.5 to −2%.

**D8 · Long windows for healing.**
Use 8k–16k windows at equal tokens, since the board scores 250k agentic traces.
- *Source:* 2609.39223 (32K vs 4K: +1.9 points at equal tokens).
- *Fit:* recipe.

**D9 · Optimiser recipe.**
Parameterise scales as s = s₀·e^r with an r² trust region, use a learning rate about 1000× the latent learning rate, and keep the best held-out checkpoint.
- *Source:* 2610.31412 (SQuAT: 3e-2 vs 3e-5); 2610.34886.
- *Fit:* recipe; avoids blind tuning.

**D10 · Heal with int8 activations on.**
Run the A8 hook in the healing forward so scales adapt to the deployed compute.
- *Source:* 2610.09202 (CanonQ).
- *Fit:* recipe; no extra cost.

**D11 · STE latent healing with periodic re-Viterbi, for the most sensitive tensors.**
Only for gate_up/down in layers 56–63 (the L63 spike). At qtf2's 50–80k tiles/s, a gate_up re-encode takes about 10–15 s.
- *Source:* 2610.41822 (TVQ: 2-bit Llama-3-8B 7.33 → 7.13 at 8× cost); 2610.27699; 2605.10655 (drift-budget rule: learning rate × steps must exceed the code's Voronoi radius).
- *Fit:* final pass.
- *Gain:* −1 to −4% beyond D1.

**D12 · Bias correction for residual writers.**
Add a per-output fp16 bias, −E[(Ŵ−W)x], to down, ssm_out and attn_output; about zero bytes, plus a kernel epilogue add.
- *Source:* 2608.21952 (only +0.2 points; the uncentered H already penalises mean error).
- *Fit:* final pass.
- *Test:* 0.8B fake-quant, 30 min.
- *Gain:* 0 to −1%.

**D13 · Re-rank after healing.**
Heal the top 3 candidate mixes for 2.5 and pick on ver3.
- *Source:* 2609.22238 (static ranking ≠ post-recovery ranking).
- *Fit:* process; costs 3 healing runs.

**D14 · On-policy distillation (low priority).**
Our metric is teacher-forced forward KL on fixed text, so off-policy forward KL is the matched objective. Use on-policy only as long-context augmentation, and only if the rules allow.
- *Source:* 2610.09346, 2610.29378.

**D15 · Code-flip (Ising) healing (low priority).**
- *Source:* 2610.31931 (scale tuning beat code moves, KL 0.325 vs 0.371). Only if D1 saturates.

---

## E. Allocation at low budgets

**E1 · Sensitivity probes measured at 2.5.**
V2 measured group swaps on a 4 bpw base. Measure group swaps (+0.5 K for one family × depth band) in a 2.5 context instead, and refit the low-bpw κ and family factors.
- *Source:* 2607.23047, 2605.23078, 2608.30564 (in-context marginals: 12.14 vs 13.72 PPL at 1.75 b).
- *Fit:* allocation.
- *Test:* about 24 evals on the GPU box.
- *Gain:* −1 to −3%.
- *Conflicts:* #14 (single tensors too noisy); group swaps avoid that.

**E2 · Pairwise interaction probes within layers.**
Measure gate_up × down and qkv × ssm_out swaps at low K, jointly vs the sum of singles.
- *Source:* 2606.13054 (pairwise DP 11.92 → 8.31); 2609.09867 (composition error 26–42× the pricing error).
- *Fit:* allocation; add an interaction term if it is above noise.
- *Test:* about 12 evals.
- *Gain:* 0 to −2%.

**E3 · Separate K for gate and up.**
Gate and up have different downstream paths (SiLU′(g)·u vs SiLU(g)), and gate_up is 45% of KL.
- *Fit:* format (stacked-seam K) or speed trade-off (§1d K ties).
- *Test:* gate-only vs up-only K2 → K3 swaps in 4 depth bands, 8 evals.
- *Gain:* −0.5 to −2%.

**E4 · Separate K for attention q and output gate (attn_qg).**
- *Gain:* ≤ −0.5% (attn_qg is about 5% of KL).

**E5 · Lazy-greedy measured-KL descent as a stream kind.**
From the band's best 2.75 mix, repeatedly demote the group with the smallest measured KL increase per byte, using a stale-marginal heap.
- *Source:* 2608.30564 (about 2.5 evals per move; JSD 749 vs one-shot ILP 823).
- *Fit:* stream.
- *Gain:* −0.5 to −2%.
- *Conflicts:* #70 (local search accepted by the model).

**E6 · 1/8-step levels K2.125–2.875.**
Run knap4 interpolation first; build only the levels it picks.
- *Why it should help:* at K3–5 for 4.0 the gain was ≤ 0.8%, but the KL curve is steeper at 2.5.
- *Fit:* per-level.
- *Test:* CPU, then 1–4 levels.
- *Gain:* −0.5 to −1.5%.
- *Conflicts:* #36/#37 (extra levels "don't pay" at higher K).

**E7 · Depth-tilt check against RAM.**
RAM gave layers 32–63 4.3–4.5 bits and 0–31 2.7–2.8 on Qwen3.8-27B, and beat the layer-objective oracle. Compare V3's 2.5 depth profile; if it is flatter, add a tilt exploration kind.
- *Source:* 2609.33923.
- *Test:* CPU.
- *Gain:* 0 to −2%.

**E8 · Tail-aware pricing of K1 at 2.0.**
Mean KL is dominated by the tail (mean about 20× median), K1 behaves like a convex cliff, and kfit2 under-predicts it by 20–36%. Add a tail penalty and cap K1 tensors per layer.
- *Source:* 2610.20826.
- *Fit:* allocation.
- *Gain:* −1 to −4% at 2.0, about 0 at 2.5.

**E9 · Priors for new options.**
Give slabs (A10), healed options (D3) and low-budget slab heads (C1) priors and uncertainty penalties under the planned B11 scheme.
- *Fit:* process.

**E10 · Re-measure after healing: help or hurt?**
- *Source:* 2607.28589 (MixFrag: re-estimation hurt); 2605.23078 (GEMQ: helped).
- *Test:* decide on the 0.8B (one healed re-measure).

---

## F. Gated DeltaNet and the linear-attention state

**F1 · Separate K for the q, k and v blocks inside ssm_qkv.**
in_proj_qkv stacks q (2048), k (2048) and v (6144) rows, all 128-aligned. v errors enter the persistent state; q errors only affect the readout. These are natural slab boundaries.
- *Fit:* per-level plus a stacked-seam K.
- *Test:* CPU per-block H_out mass, then a 0.8B fake-quant test.
- *Gain:* −0.5 to −2%.

**F2 · Decay half-life as a per-head prior.**
Compute half-lives from A_log/dt_bias for free and use them as a prior for A12, or as row weights where H_out is missing.
- *Source:* 2609.38169 (ρ ≈ 0.80).
- *Fit:* CPU.

**F3 · Check GDN state precision in our eval path.**
Confirm the state stays fp32 across 2048-token chunks and checkpoints in both the reference and our runs.
- *Source:* 2609.38166 (LeapQuant: bf16 per-step state cost 16 AIME points); 2609.38169.
- *Test:* one eval with forced fp32.
- *Gain:* 0 to −1%; mainly a floor check for long traces.

**F4 · Normalization-aware q/k metric (low priority).**
L2-norm and RMSNorm remove only the radial 1/head_dim of isotropic error, so the gain is negligible. The per-head 1/‖k‖² weights already enter H_out.

---

## G. Error accumulation across layers

**G1 · Error-cancelling targets for residual writers (final pass).**
Requantize down, ssm_out and attn_output sequentially, targeting W·X plus a damped least-squares correction ΔW = −δX̃ᵀ(X̃X̃ᵀ)⁻¹ that cancels the accumulated residual error δ_l. Writers are about 27% of the 2.5 bytes.
- *Source:* 2609.11716 (blocks already cancel 50% of Σ‖Δu‖² in Qwen3-32B); new.
- *Fit:* per-mix; about one night.
- *Gain:* −2 to −6% (speculative).

**G2 · Per-layer residual mean drift.**
Measure E[h_q − h_bf16] per layer on the 0.8B. If the mean is ≥ 10% of the RMS error, add a writer bias.
- *Fit:* diagnostic first.
- *Gain:* 0 to −1%.

**G3 · Spend quantizer effort where κ is large.**
Run CD sweeps (B2) and alternation (B3) mainly on early layers (κ ≈ 52 vs about 2 late).
- *Fit:* per-level scheduling; makes B2/B3 cheaper.

**G4 · Teacher hidden-state cache for the 27B.**
One streamed bf16 forward over about 200k qad tokens: 64 layers × 5120 × 2 bytes per token, about 130 GB. This is infrastructure for D4, D5, G1 and G2.
- *Fit:* infrastructure; one night.

---

## H. Format and codebook (low K)

**H1 · Chained (non-tail-biting) trellis across tile pairs or rows.**
A tile's last positions read the next tile's first L−K bits instead of wrapping around. Decoding stays random-access; Viterbi runs over chains of 2–8 tiles.
- *Source:* #57 (free start vs tail-biting 0.948 at K2, 0.962 at K4); 2406.11235.
- *Fit:* format (quantizer plus every decode kernel).
- *Test:* CPU chain-of-2 vs tail-biting on real LDLQ tiles with the existing Viterbi tool.
- *Gain:* −1.5 to −3%.
- *Conflicts:* #57 ("not worth a kernel change" referred to 16-candidate search, which recovered only 8% of the gap).

**H2 · Longer state L = 18/20 at K ≤ 2.5.**
- *Source:* QTIP's L ablation (quality improves with L); 2-bit Gaussian MSE 0.069 vs the bound 0.063 (about 9% headroom, 0.066 bit).
- *Fit:* format (codebook variant; the computed code means no lookup-table growth; Viterbi 4–16× slower).
- *Test:* CPU MSE at L = 16/18/20 on Gaussian and real tiles.
- *Gain:* −1 to −3%. *Risk:* decode VALU cost.

**H3 · Codebook matched to LDLQ-fed targets.**
Measure the kurtosis of fed tile targets at K2. If they are clearly non-Gaussian, refit the i8 codebook shape per level.
- *Test:* CPU tsweep.
- *Gain:* 0 to −2%.

**H4 · Row-block K as a first-class encoding.**
Generalise the head slab to "one tensor, several 128-row blocks at different K" (importer, kernels, knapsack items). This is the prerequisite for A10, A12, F1 and E3.
- *Cost:* 2–4 days.

---

## I. Diagnostics and process for low bpw

**I1 · Validate the KL attribution with real swaps.**
Raise one family (gate_up, down, GDN in, head) by +0.5 K in the 2.5 mix and check kfit2's 45/22/13/9% shares. 4–6 evals.

**I2 · KL tail profile, 2.5 vs 4.0.**
Measure the share of KL in the top 10% of tokens and documents, by category, from existing per-row scores (CPU). This guides D7 and C9.

**I3 · 2B proxy calibration at low K.**
Resume CALIBRATION.md with K2/K2.5 checks, or require a 2B confirmation of 0.8B wins before 27B. The 0.8B is only 1024 wide.

**I4 · Stack test on the 0.8B before the single Phase-B requant.**
Combine diag H_out, B1, B2 and B6, and check whether the gains add up.

**I5 · Low-bpw-first requant.**
Apply every new quantizer change to K1.75–3.75 first, since the 2.0 and 2.5 products live there; then extend.

**I6 · Per-category KL at 2.5.**
Check whether code, multilingual or agentic text dominates; this informs data selection and the guards.

**I7 · Healing gain vs bpw on the 0.8B.**
Run D1 at K2, K2.5, K3 and K4 to see where healing pays.
- *Source:* 2606.09012 (QAT gains grow as bits fall).

---

## J. 2.0-specific

**J1 · Measured K1.25/1.5/1.75 levels** with the new quantizer settings. At 2.0, 11% of parameters sit at K1, and K1 is the worst-fitted level. (ROADMAP V3.5.)

**J2 · Heal 2.0 first if D1 works.**
Gains grow as bits fall (2610.29378: +10 points at 2-bit vs +1.4 at 3-bit). Expected −10 to −25% at 2.0.

**J3 · Embedding floor at 2.0.**
At 2.0 the embedding sits at t1.25. Test C7 or t1 to free bytes (run5/6 showed K1 is trellis-limited, not scale-limited).

**J4 · Sub-1 options, priced by proxy in knap4 first** (V3.5 plan). The 2.0 slab head is already covered by C1.

---

## Considered and rejected (with numbers)

- **Stored start state per tile:** 16 bits per 256 weights is 0.0625 bpw (about 11% KL at 2.5) for ≤ 5% proxy error. CurveTQ's +0.22 bpw start state paid only because it uses no rotation.
- **HARP learned rotation:** +0.05 bpw for 3.5% PPL at 2 bit, and 0.7% at 3 bit on QTIP. That is less than spending the same bytes on K.
- **Trace-optimal joint rotation and feedback (GMD):** loses to RHT + feedback in 16 of 16 comparisons (2610.18139).
- **Larger input Hadamard or a cross-block mixer:** spreads massive channels and hurts A8-g128 (#47).
- **Codebook swaps (1MAD/3INST/HYB, E8P, Leech):** all within 0.068–0.071 MSE at 2 bits, with slower kernels.
- **Entropy-coded weights:** lose fixed-rate random access.
- **BCJR-QAT:** hours per layer even at 1B.
- **Seed or sign best-of-N (#41), per-group scales (#40), gradient-weighted H_in as a replacement (#33-07).** A small-α *blend* of the last remains untested (2610.18139 found fixed α unsafe).

---

## Ranked top 20 by expected gain per unit of cost

| rank | idea | expected KL at 2.5 | cost |
|---|---|---|---|
| 1 | A1 gen-masked H_out loss | 0 to −2% | ~0 (before the 27B run) |
| 2 | C1 low-budget slab heads in knap4 | −1.5 to −3% | head donors + headsim, ~1 day |
| 3 | A2 + A3 family ablation + tempering | −1 to −3% | 3 h on the 4070 |
| 4 | B1 Hessian-aware su exponent | 0 to −5% | 1 h on the 4070 (+ Phase B requant) |
| 5 | C5 final-norm + head-svh healing from dumps | −1 to −4% | 1 dump eval + minutes |
| 6 | C2 two-sided Fisher-diagonal head LDLQ | −1.5 to −3% | head quants + headsim |
| 7 | B13 amplitude de-biasing | 0 to −2% | 30 min on the 4070, zero bytes |
| 8 | D1 scale-only KL healing | −8 to −20% | 3–5 days engineering + nights on the GPU box |
| 9 | B6 activation-denoising Hessians | −2 to −8% | 2.5 h on the 4070; 1 night + requant on 27B |
| 10 | A10 diag-H_out row-slab K | −2 to −10% (speculative) | CPU check minutes; H4 2–4 days |
| 11 | D2 late-layer healing | −5 to −15% | cheaper variant of D1 |
| 12 | D3 library-level healing (0.8B test) | makes D1 per-level | 0.8B half day |
| 13 | A6 output scales with the Sketch B diagonal | 0 to −3% | 1 h on the 4070 |
| 14 | B2 block-CD polish | −1 to −4% | 1 day engineering, ×2–4 quant time |
| 15 | A16 two-sided proxy in kfit2 | 0 to −3% | CPU, after the 27B diagonal |
| 16 | C3 low-rank head correction | −0.5 to −3% | CPU SVD first |
| 17 | B3 scale/code alternation | −0.5 to −2% | 1 h on the 4070 |
| 18 | B8 intra-layer sequential | −1 to −3% | 2 h on the 4070 |
| 19 | E6 1/8-step levels via interpolation | −0.5 to −1.5% | CPU, then 1–4 levels |
| 20 | F1 q/k/v sub-block slabs | −0.5 to −2% | CPU + 0.8B, needs H4 |

If the larger items hold (D1/D2, B6, A10, C1/C2), together with the planned 27B diag H_out, gains of 15–30% at 2.5 are plausible, so 0.0583 could become about 0.045. But they overlap: B6, B7 and B9 all target compounding, and A10 competes with two-sided LDLQ for the same anisotropy.

## The five cheapest diagnostics that would most change this ranking

1. **Row-sensitivity ceiling on the 0.8B** (CPU, minutes): ½·log₂(AM/GM) of diag H_out over sorted 128-row slabs per family, then one 2-slab fake-quant test (2 h). This decides A10, A12, F1 and H4, the largest speculative cluster. Above 0.05 bit for gate/up promotes them to top 5; below 0.02 drops them.
2. **Healing headroom probe on the 27B 2.5 mix:** the τ sweep (C6), B13 amplitude scalars and the C5 final-norm/head-svh fit on dumped head inputs. If these zero-backward fits give −2% or more, push D1–D3 to the front at 27B; if about 0, keep healing at V5.
3. **Scale-only healing plus library transfer on the 0.8B** (D1 + D3, half a day on the 4070). This tells whether healing belongs in the per-level library, which changes the whole pipeline order.
4. **headsim battery on the 27B head** (CPU/GPU, a few head quants): the low-budget slab curve with mean-row and the guard (C1), the Fisher two-sided head (C2), and the residual SVD spectrum (C3). Head ideas are the cheapest exact-to-score wins.
5. **One-night 0.8B quantizer battery:** A4's estimator 2×2 plus the A1 mask (run before the 27B collection), A2/A3 ablation and tempering, B1's su exponent, and B6 denoising. This reorders ranks 3–14, and A1/A4 decide how the 27B 8.5 h run should collect its diagonal.
