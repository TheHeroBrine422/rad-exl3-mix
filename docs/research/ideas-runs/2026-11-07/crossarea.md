# Cross-area lens: research ideas for the V3 quantizer freeze

**Bottom line.** Of the 85 ideas, the strongest cross-area thread is that the coherent errors you found (the LS refit shrinking gains) have three more closed-form siblings. The cheapest one to test is replacing the shrinking LS scale refit with an unbiased (instrumental-variable) refit, on fixed codes. Before investing in any of these, run diagnostic 1 at the end: it measures how much of today's KL is seed-independent drift, which is the most these ideas can win.

## Two new measurements from colo

I ran two read-only diagnostics on colo (CPU, about 15 minutes; scripts and results in `/data/work/ideas-crossarea/`).

**1. Calib input Hessians after a 128-point Hadamard with random signs (one draw, seed 0).** The 0.025 and 0.005 columns show how much the damping `σ·mean(diag)` adds relative to each channel's own diagonal.

| tensor input | raw max diag / mean | top-1 eigenvalue share | top 128-block trace / median block | damping 0.025: median, share of channels > 0.1 | damping 0.005: median, share > 0.1 |
|---|---|---|---|---|---|
| L3 layer input (residual readers) | 4765 (ch 3994) | **0.944** | 640 | **0.44, 94%** (27% of channels > 0.5) | 0.087, 27% |
| L10 layer input | 3423 (ch 3994) | 0.692 | 86 | 0.079, 21% | 0.016, 0% |
| L11 layer input | 4487 (ch 3994) | **0.897** | 336 | **0.24, 95%** | 0.048, 0% |
| L31 layer input | 3055 (ch 3994) | 0.612 | 64 | 0.065, 9% | 0.013 |
| L50 layer input | 1979 (ch 3994) | 0.424 | 28 | 0.043 | 0.009 |
| L63 layer input | 552 (ch 3994) | 0.175 | 6 | 0.028 | 0.006 |
| L31 MLP input | 1901 (ch 3994) | 0.384 | 25 | 0.039 | 0.008 |
| L3 o_proj input | 1265 | 0.431 | 17 | 0.042 | 0.008 |
| L10/L11 MLP input, L11 o_proj input | 107–430 | 0.04–0.11 | 2–5 | ≈0.026 | ≈0.005 |

In plain terms:
- **Residual channel 3994 is the massive channel at every depth.** From about L3 to L50 it is close to a rank-1 spike in the residual-reader Hessians.
- **The scalar damping over-damps the bulk spectrum.** It is scaled by a mean diagonal that the spike dominates, so it adds 24–44% of the median channel's own curvature at L3/L11 with σ = 0.025, and 5–9% with the inputs' σ = 0.005.

**2. Super-weight scan of down_proj (bf16 checkpoint, 64 layers).**
- **Output row 3994 is the largest-norm row in almost every layer**: 5–26× the median row (L2: 25.7×; largest entry 108× RMS).
- **L0 has 40 entries above 50× RMS.** The other entry labelled "layer 0", with outlier row 3281, is probably the MTP layer.

So the writers of the massive channel and its readers are the same few rows and columns throughout the model.

## A framing that organizes most ideas

Your evidence says coherent per-channel error compounds with depth and grows with model size (swj −6.5% at 27B). The cross-area literature names four mechanisms that produce coherent error from random rounding, and each has a closed-form fix:

1. **Errors-in-variables shrink.** An MMSE / least-squares fit (refit_scales; Lloyd centroids) returns a reconstruction with less energy than the target. With θ_H the angle between w and q in the H-metric:
   - LS gain = 1
   - unbiased (instrumental-variable) gain = 1/cos²θ_H
   - energy-matched gain = 1/cosθ_H

   At K2.5 the relative H-weighted error is about 0.04–0.08, so the unbiased gain is about 1.04–1.09. That fits swj's c_g ≈ 1.07 ± 0.10. Sources: the unbiased-MMSE decision-feedback-equalizer literature, and RaBitQ / SoloQ's unbiased inner-product scaling (2606.31519, 2610.07121).
2. **Angular shrink from normalizing noisy vectors.** RMSNorm, q_norm/k_norm and the final norm all compute ŷ = (y+e)/‖y+e‖. The clean component shrinks by 1/√(1+σ²), coherently, at every norm. This gives flatter attention, under-confident logits, and smaller block contributions relative to the skip path.
3. **Jensen bias through curved gates.** E[f(g+ε)] − f(g) ≈ ½f''σ² for silu and sigmoid. This partly explains why swj pays.
4. **Drift on persistent inputs.** Error E·μ on a non-zero-mean or temporally persistent input repeats at every position. Attention and GDN integrate over positions, so random parts average out and drift does not (2610.01889: round-to-nearest drift grows as O(n), random error as O(√n)).

Fixes for #1–#4 are closed-form or one-pass. Most fit per level on bf16 inputs; a few need one forward pass per mix. Noise-shaping ideas (temporal Hessians, robust H estimation) get the default ⅓ shrink from 0.8B to 27B. Coherent fixes are priced at about 1× (swj precedent).

Sizes used below:
- **0.8B fixed-code pair:** about 0.3 h on the 4070, no rounding noise.
- **27B fixed-code pair:** a staged forward pass like swjfit27 plus two evals, about 1.5–2 card-h on gpubox.
- **CPU proxy:** settle64 order-fixed, minutes to 1 h on colo or Thor.

---

## Theme 1 — Diagnostics that measure coherence (run first)

### 1. Seed-ensemble drift/variance split of the KL
Take n seeds of the same level (0.8B has six seeds for some arms; 27B K2.5 has base/s1/s2). Per position, split the logit perturbation Δ = z_q − z_bf16 into drift b = E_seed[Δ] and fluctuation ξ. Then KL ≈ ½bᵀFb + ½tr(FΣ), with F = diag(p) − ppᵀ from the golden, and a bias-corrected b̂ (E|b̂|² = |b|² + tr Σ/n).
- **Source:** 2610.01889 (drift / curvature / Fisher-variance decomposition, eq. 9), applied across quantizer seeds instead of rounding modes.
- **Why it should help:** it measures directly how much KL any coherent fix could remove. If the drift share is ≥30%, theme 2–4 ideas are worth the cards; if it is ≤10%, drop most of them.
- **Fit:** diagnostic.
- **Test:** 0.8B with 6 seeds on g08-search1, storing top-k quantized logits (about 0.5 h on the 4070). At 27B, three evals on golden-search27 with a logit recorder (about 0.5 card-h).
- **Expected gain / risk:** none directly. It decides whether ideas 11–40 are worth the cards.
- **Conflicts:** none; it reuses the #33-28 seed arms.

### 2. Layerwise coherence growth exponent
Do the same split as #1 on the residual stream at each layer end (hidden states of three seeds versus bf16). Fit ‖b_l‖ ∝ l^α_b and ‖ξ_l‖ ∝ l^α_ξ. Coherent error should grow with α ≈ 1, random with α ≈ 0.5, with partial cancellation (2609.11716). Also measure the "coherent gain" g = ‖Σ_l e_l‖²/Σ_l‖e_l‖² of per-layer error injections.
- **Source:** 2610.15709 (coherent gain of per-step errors in recurrent states), 2609.11716 (counteraction).
- **Why it should help:** shows where drift is born (families, depths) and whether it compounds. It ranks the junctions for ideas 11–35.
- **Fit:** diagnostic.
- **Test:** 0.8B, one forward per seed with hooks (about 0.3 h on the 4070).
- **Expected gain / risk:** none directly.
- **Conflicts:** none.

### 3. Junction drift ledger
Per tensor family, at bf16 inputs, compute the seed-averaged output error E_seed[ΔY] and split it into a per-channel gain part, a mean part, and a residual. This tells, per junction, which fix applies: refit (gain), bias (mean), or nothing.
- **Source:** new; it combines the 2610.01889 decomposition with your fixed-code refit tooling.
- **Why it should help:** #108 found GDN junctions pay and attention junctions do not. A ledger says why, before more fits are run.
- **Fit:** diagnostic.
- **Test:** 0.8B, three seeds, per-family hooks (0.3 h).
- **Expected gain / risk:** none directly.
- **Conflicts:** none.

### 4. Attention logit-scale and entropy shift
Per full-attention layer and head, compare with bf16: the mean attention entropy, and the IV slope of quantized against bf16 logits (Σss/Σsŝ, per query, centered). Theme-2 theory predicts flatter attention (slope < 1) by about σ_q²/2 + σ_k²/2.
- **Source:** angular shrink of normalized noisy vectors (JL / RaBitQ inner-product bias, 2606.31519).
- **Why it should help:** decides #21–22 in one pass.
- **Fit:** diagnostic.
- **Test:** 0.8B (it has full-attention layers), forward-only (0.1 h).
- **Expected gain / risk:** none directly.
- **Conflicts:** none.

### 5. Position-0 (attention sink) oracle
In the quantized model, replace position 0's per-layer K/V (full attention) and post-token-0 GDN state with bf16 values. Both depend only on token 0, so one bf16 one-token forward provides them. Score the KL change at all other positions.
- **Source:** 2609.09085 (sinks and massive activations form at position 0 regardless of the token); KV-cache practice of keeping sinks in bf16 (OSCAR 2605.17757, KVarN).
- **Why it should help:** every later query reads the sink, so sink error is coherent across the whole document. This bounds #33 and #42.
- **Fit:** diagnostic.
- **Test:** 0.8B; patch the eval harness's first-position states (about 0.3 h).
- **Expected gain / risk:** none directly.
- **Conflicts:** none.

### 6. Massive-row / massive-column oracle at 27B
Swap the bf16 values of down_proj, o_proj and out_proj output row 3994 into a K2.5 mix, in L0–L5, then in all layers (variant A). Separately, swap the bf16 input column 3994 of all residual readers (variant B).
- **Source:** our scan above; super weights (2607.10803); D-SLR "rows verbatim" (2610.10636).
- **Why it should help:** row 3994 is 5–26× the median row and the column dominates the readers' H. This bounds #41–45.
- **Fit:** diagnostic (weight-only swap, no requant).
- **Test:** 27B, two arms plus a base eval (about 1 card-h, gpubox).
- **Expected gain / risk:** none directly. If an oracle moves KL by less than 1%, drop theme 6.
- **Conflicts:** faraway M5 (massive channel inefficiency) motivated this; it is untested.

### 7. Low-rank structure of the drift
H^{1/2}-whitened SVD of the per-tensor quantization error E, and of the seed-averaged E_drift. Random error is white after the Hadamard; drift may be low-rank.
- **Source:** IO-SVD (2605.15626), AIR (2606.19993), D-SLR (2610.10636).
- **Why it should help:** decides whether #79 (rank-r drift correction) can ever pay.
- **Fit:** diagnostic.
- **Test:** CPU on colo, a few 0.8B tensors from 3 seeds (minutes).
- **Expected gain / risk:** none directly.
- **Conflicts:** none.

### 8. Temporal persistence of tensor inputs
Per tensor input, compute the share of H_in energy in (a) the global mean μμᵀ, (b) the per-document mean scatter, and (c) the lag-1..3 autocovariance C_k = Σ x_t x_{t−k}ᵀ.
- **Source:** sigma-delta noise-shaping view (2609.17276; error diffusion 2610.34566); down_proj positive-input drift (2610.01889).
- **Why it should help:** sizes #28–30 and #36–39. It needs only a forward pass (no Σdy²).
- **Fit:** diagnostic.
- **Test:** 0.8B forward hooks (0.2 h). The 27B variant reuses mkhin27 (about 1–2 card-h).
- **Expected gain / risk:** none directly.
- **Conflicts:** not a Σdy² collection, so it is allowed under the 11-05 rule.

### 9. Additivity before and after coherent fixes
For about 20 measured pair swaps, compare the joint ΔKL with the sum of singles, on the base and on the swj (later IV) stack. Coherent drifts add as vectors, so pairs should be super-additive before the fixes and closer to additive after them.
- **Source:** 2609.22131 (correlation-aware pruning: additivity fails through correlated units); 2608.30564 (Q-Strata).
- **Why it should help:** tells w58's knapsack repricing whether swj changed the sensitivity model's structure, not just its scale.
- **Fit:** diagnostic for the knapsack.
- **Test:** 27B, 20 pair swaps on the search slice (about 3 card-h). A 0.8B version costs 0.5 h.
- **Expected gain / risk:** none directly.
- **Conflicts:** #14 showed single-tensor swaps are too small to measure; use group pairs.

### 10. LDL precision check on the spike-dominated tensors
Compare LDLQ proxy error with fp32 versus fp64 factorization (and with randomized-preconditioned Cholesky) for L3/L10/L11 attn_qg and ssm_qkv. Raw diagonals span 4765:1 and the top eigenvector holds 94% of the trace.
- **Source:** 2606.18411 (mixed-precision randomized Cholesky-QR: Cholesky breaks near κ ≳ 1e8).
- **Why it should help:** rules out a numerical cause for faraway M5's 1.74× inefficiency.
- **Fit:** diagnostic.
- **Test:** CPU on colo, settle64 with dtype switched (minutes).
- **Expected gain / risk:** none directly. Probably null; it is cheap insurance before the freeze.
- **Conflicts:** none.

---

## Theme 2 — Unbiased rescaling (errors-in-variables, unbiased-MMSE equalizers, unbiased inner products)

### 11. Instrumental-variable (IV) output-scale refit
Keep the codes and replace refit_scales' LS column gain c_n = (q_nᵀHw_n)/(q_nᵀHq_n) with the IV gain (w_nᵀHw_n)/(q_nᵀHw_n). This uses the bf16 weight as the instrument, as econometrics does to undo attenuation bias. It is per channel and closed-form, with no grid and no KL fit.
- **Source:** errors-in-variables / IV regression (classic); unbiased-MMSE equalizers; RaBitQ's unbiased estimator (2606.31519); SoloQ's scale correction c = ‖z‖²/⟨q(z),z⟩ (2610.07121). Difference: those apply it to KV vectors or inner products, not to Hessian-metric weight refits.
- **Why it should help:** it attacks the LS shrink that your mechanism note blames for swj's success, and it covers every family, not just gate/up. The expected inflation, 1/cos²θ_H ≈ 1.04–1.09 at K2.5, matches swj's fitted c_g 1.07 ± 0.10.
- **Fit:** per level (bf16 inputs, H only); library-safe; zero runtime.
- **Test:** 0.8B fixed-code pairs at K3/K2, with 3 seeds of one family at a time: o, out, down, q, k, v, qkv, z. Then all families except gate/up on top of swj. Pass if ≤ 0.99 at K2.5-equivalent. Then a 27B fixed-code pair (2 card-h).
- **Expected gain / risk:** −1…−3% at 2.5 bpw, uncertain. Risk: on linear-into-residual paths, MMSE (no inflation) is already optimal. C9 found down and attn_o at α ≈ 0 while gdn_out wanted α ≈ 3, so gains will be family-specific.
- **Conflicts:** #85 C9 and #112 C9 α (per-family exponent, grid-tuned). IV is the parameter-free special case and also applies to the input side (#12). Run it next to #112's 27B pair, not instead of it.

### 12. IV input-side (suh) refit
The input-side solve ((QQᵀ)∘H)r = rowsum(Q∘HW) is also LS. Use instead ((WQᵀ)∘H)r = rowsum(W∘HW), with W as the instrument.
- **Source:** as #11.
- **Why it should help:** the per-input-channel gains shrink the same way, coherently, for every token.
- **Fit:** per level; zero runtime.
- **Test:** 0.8B fixed-code pair, all families (0.3 h).
- **Expected gain / risk:** −0…−1%. Risk: the system can be ill-conditioned; clamp r to [0.8, 1.25].
- **Conflicts:** none measured.

### 13. Energy-matched (gain-shape) refit and a λ-ladder
Set c_n = √(w_nᵀHw_n / q_nᵀHq_n), which restores each channel's H-energy, as Opus/CELT preserves band energy explicitly. Parameterize c = c_LS·(c_IV/c_LS)^λ; λ = 0, ½, 1 give LS, energy-matched, IV. Fit one λ per family by KL on 0.8B.
- **Source:** CELT band-energy preservation (classic); gain-shape VQ.
- **Why it should help:** brackets the family-specific optimum seen in C9. gdn_out's α ≈ 3–4.75 suggests λ > 1 there (see #23 for why).
- **Fit:** per level.
- **Test:** 0.8B fixed codes, λ ∈ {0, 0.5, 1, 2, 3} × family (1 h).
- **Expected gain / risk:** same envelope as #11; this is the tuning version.
- **Conflicts:** #112. λ is a principled reparameterization of α with a closed-form anchor at λ = 1.

### 14. Deming / total-least-squares refit
Fit c with errors in both variables. The noise ratio comes from the known trellis distortion D(K) of the level, so c sits between LS and IV with no free parameter.
- **Source:** Deming regression (classic).
- **Why it should help:** gives a single default for families where #13's λ is unclear.
- **Fit:** per level.
- **Test:** folded into #13's sweep (0 extra).
- **Expected gain / risk:** ≤ #11.
- **Conflicts:** none.

### 15. IV-swj
swj fits (c_g, c_u) by LS on SwiGLU outputs, which also attenuates. Replace c_u's closed form Σap/Σp² with the IV form Σa²/Σa·p, with the bf16 activation a as instrument, and keep the c_g grid. Variant: also pick c_g by the IV moment Σa(a − p(c)) = 0.
- **Source:** IV regression applied to swiglufit.
- **Why it should help:** swj is the one big coherent win; if LS shrink remains inside it, the gain is left on the table.
- **Fit:** per level (bf16 inputs); zero runtime.
- **Test:** 0.8B fixed codes, swj versus IV-swj, 3 seeds at K3/K2 (0.3 h). Then a 27B pair reusing swjfit27 (it is one extra moment per channel).
- **Expected gain / risk:** −0.5…−2% at K2.5. Risk: the grid optimum may already compensate.
- **Conflicts:** #104 is the baseline; this is a strict variant.

### 16. Quantization-time slicer de-biasing
Scale the LDLQ target by g(K) = 1/(1 − D_rel(K)) before Viterbi and undo the scale in svh. The trellis then picks codes for an unbiased target, rather than inflating scales after codes are fixed. This is the unbiased-MMSE decision-feedback-equalizer correction.
- **Source:** unbiased-MMSE DFE theory (Cioffi; classic).
- **Why it should help:** post-hoc inflation also inflates the noise. Choosing codes for the inflated target may keep more of the gain.
- **Fit:** quantizer setting (must precede the freeze).
- **Test:** 0.8B requant, gain ∈ {1.00, 1.04, 1.08} at K3/K2, three seeds, judged by KL (2 h on the 4070).
- **Expected gain / risk:** −0…−2%. Risk: overlaps #11.
- **Conflicts:** #39 found gain flat, but on proxy error. Proxy error is the MMSE metric, so it cannot see this effect by construction. Re-test by KL.

### 17. Analytic per-K gain prior
Tabulate c(K) = 1/cos²θ_H(K) from the per-tensor proxy error each level already logs (proxy_err ≈ 1 − cos²θ). Compare it with swj's and C9's fitted gains. If they line up, every level gets a zero-fit default and the knapsack's swj factor (KN_SWJ) gets a model, not just two anchor points.
- **Source:** new (follows from #11).
- **Why it should help:** w63's corrected swj curve rests on K2.5 and K4 points only.
- **Fit:** knapsack pricing.
- **Test:** CPU, with the existing proxy jsonl and swj fit logs (minutes).
- **Expected gain / risk:** none directly; improves pricing.
- **Conflicts:** w58 / w63 repricing.

### 18. Embedding row IV gain
An embedding row's error repeats at every occurrence of that token, so it is coherent per token. Refit each row's scale by IV (‖e‖²/⟨ê,e⟩) instead of the MSE scale search, which shrinks (with H = I, as now).
- **Source:** #11 applied to a lookup table; the embedding's per-row scale is already stored.
- **Why it should help:** at ≤ 2.5 bpw the embedding sits at its t1.5–2 floor (knap4), where shrink is largest (D_rel ≈ 0.2 at t1.5 implies about 25% inflation).
- **Fit:** per level (embedding menu); zero runtime.
- **Test:** 27B embedding-only re-pack at t2 / t2.5, paired with the current embedding, inside a K2.5 mix (about 1 card-h).
- **Expected gain / risk:** −0.3…−1.5% at 2.5 bpw; more at 2.0.
- **Conflicts:** #35 (embedding menu) tuned scales for MSE (e.g. run6 searched 0.5..1.0, i.e. shrink). This tests the opposite direction.

### 19. Head row IV gain
Apply #11 per vocab row of the head in the head's H, on top of the mean-row correction. It is the closed-form, per-level cousin of the shortlisted KL-side head gain refit.
- **Source:** #11.
- **Why it should help:** LS shrink of head rows is a systematic under-confidence (a temperature).
- **Fit:** per level (head menu); zero runtime.
- **Test:** headsim (exact head-only KL, 32k held-out inputs), on CPU or GPU in minutes. Then fold into a 27B pair with #26.
- **Expected gain / risk:** −0.2…−1% total.
- **Conflicts:** shortlisted "head KL-side gain refit". This is the free, mix-independent half of it.

### 20. Estimator choice by downstream averaging
Use IV/energy gains for tensors that feed temporal integrators: attention V (via o_proj), GDN qkv. Keep LS for purely per-token paths. Attention dispersion and GDN decay average random error but not gain bias, so the optimal estimator differs by family.
- **Source:** 2610.01889 (drift versus √n random growth along reductions).
- **Why it should help:** it explains C9's family split and gives a default before fitting.
- **Fit:** per level.
- **Test:** part of #13's sweep.
- **Expected gain / risk:** none beyond #11–13.
- **Conflicts:** #108 (attention refits never paid). Its LS loss may be why.

---

## Theme 3 — Shrink from normalizing noisy vectors

### 21. Attention temperature via q_norm.weight (one scalar per layer)
Per full-attention layer, fit τ_l by IV on logits (Σss/Σsŝ) or by 1-D KL on attention probabilities, with bf16 inputs and the level's quantized q/k. Fold τ_l into q_norm.weight. q_norm is shared across heads, so per-layer is the free granularity.
- **Source:** angular-shrink bias of normalized noisy vectors (JL / RaBitQ 2606.31519); temperature scaling from calibration (classic).
- **Why it should help:** q_norm and k_norm remove the *magnitude* of W_q/W_k error but keep its *angular* part. That shrinks q̂·k̂ by about 1 − σ²/2 in every head and query, so attention is flatter in all 16 layers. #108 refit svh scales, which q_norm absorbs; a scale on the norm weight is not absorbed.
- **Fit:** per level (bf16 inputs); zero runtime; no format change.
- **Test:** run #4 first. Then a 0.8B fixed-code pair (0.2 h), then a 27B pair (1.5 card-h).
- **Expected gain / risk:** −0.3…−1.5% at 2.5 bpw. Risk: heads differ; per-layer averaging may cancel.
- **Conflicts:** #108's attention arms. It differs in mechanism: logit temperature, not projection scales.

### 22. Per-head attention temperature (format: one scale per head)
If #4 shows a spread across heads, add a per-head logit scale (16 layers × 24 heads, fp16, applied in the attention kernel).
- **Source:** as #21.
- **Why it should help:** captures head-specific sharpness loss.
- **Fit:** format change (tiny); must precede the freeze if wanted in V3.
- **Test:** 0.8B arm next to #21 (0.2 h).
- **Expected gain / risk:** an extra −0…−0.5% over #21. Risk: kernel and plugin change.
- **Conflicts:** none.

### 23. Residual RMSNorm gain re-inflation (the LLM analogue of BatchNorm recalibration)
Every reader's RMSNorm divides by an rms inflated by accumulated noise, so the clean signal shrinks by 1/√(1+σ_l²). Measure σ_l² (relative residual error per layer) for the mix in one forward pass. Multiply each input-norm and post-attention-norm weight by √(1+σ_l²). That is 128 scalars folded into norm weights.
- **Source:** BN statistics re-estimation after quantization or pruning (classic CNN practice); scale-state feedback (2610.15139).
- **Why it should help:** a coherent, depth-compounding shrink of each block's contribution relative to the skip path. It may be why gdn_out wants α ≫ 1 (C9), since its output shrinks against a noisier skip.
- **Fit:** per mix (one forward pass, zero runtime). A per-level approximation predicts σ_l from the level's proxy errors.
- **Test:** 0.8B mix at K2/K3: measure σ_l, apply, evaluate (0.2 h). Then a 27B K2.5 mix (1 card-h).
- **Expected gain / risk:** −0…−2%. Risk: also inflates noise; may overlap swj and C9.
- **Conflicts:** #89-02 found norm gains worth ≤ 0.2% on top of healing, but healing's scales absorb the same degree of freedom. Without healing this is open.

### 24. GDN gated o_norm re-inflation
Same as #23 for the gated RMSNorm on the GDN readout (head_dim-shared weight).
- **Source:** as #23.
- **Why it should help:** noisy readouts shrink before out_proj, which may be why gdn_out's C9 α is high.
- **Fit:** per mix or per level.
- **Test:** in #23's 0.8B arm.
- **Expected gain / risk:** −0…−1%.
- **Conflicts:** #108's z-gate refit (different parameter: the gate, not the norm weight).

### 25. Final-norm temperature, predicted closed-form
The theory predicts head under-confidence T > 1, with a fix of logits × √(1+σ_L²). Compute σ_L² from the mix's final hidden-state error and compare with the fitted temperature.
- **Source:** as #23; logit-scale growth and quantization error in the head (2610.38277).
- **Why it should help:** gives the shortlisted KL-side head refit a free initial value and a sanity check.
- **Fit:** per mix.
- **Test:** an eval post-pass on stored logits (minutes).
- **Expected gain / risk:** −0.5…−2% at 2.5 bpw (shortlisted value).
- **Conflicts:** the shortlisted head KL-side gain refit (this is its analytic prior).

### 26. Head vector scaling plus logit bias (Dirichlet / vector calibration)
Per mix, fit (a) a temperature, (b) per-vocab-row gains folded into the head rows, and (c) a per-vocab logit bias (248k fp16, 0.5 MB, one add at decode). Minimize KL(p_bf16 ‖ softmax(z_q)) on calib fit slices; the problem is convex in (b, c). The final RMSNorm per-channel gain can be fitted by backprop through the head only.
- **Source:** vector / Dirichlet scaling in calibration (classic).
- **Why it should help:** removes systematic frequency drift (the quantized model under- or over-predicting common tokens), which mean KL weighs heavily.
- **Fit:** per mix final pass. Runtime: one vector add at decode, ~0. Format: logit bias field.
- **Test:** stored top-512 golden logits plus the mix's final hidden states; fit on CPU, evaluate on the search slice (about 0.5 card-h at 27B).
- **Expected gain / risk:** −0.5…−2%. Risk: overfitting to the fit slice; verify on ver3.
- **Conflicts:** #34 (mean-row correction is the exact logit-shift part); shortlisted head gain refit.

### 27. Radial discount in writers' output metric
To first order, writer error along the current residual direction ĥ_t is invisible to the next readers' RMSNorm. Weight writer output rows by E_t[1 − ĥ_{t,r}²]. In the massive-channel layers this mainly discounts row 3994.
- **Source:** RMSNorm invariance (classic).
- **Why it should help:** a cheap, forward-only partial H_out for writers.
- **Fit:** per level (quantizer, needs the out_weight path that refit_scales already has).
- **Test:** CPU proxy plus a 0.8B pair (0.5 h).
- **Expected gain / risk:** −0…−0.5%. Writers carry only 3% of H_out's gain (#82).
- **Conflicts:** #33 (H_out). This needs no Σdy².

---

## Theme 4 — Drift on mean and persistent inputs

### 28. Centered LDLQ plus exact bias
Run LDLQ with the centered covariance Σ = H − μμᵀ, and cancel the mean-direction error exactly with a bias b = −Eμ (DFQ-style). LDLQ then spends no shaping on μ, and the drift on μ is zero by construction.
- **Source:** DFQ bias correction (Nagel 2019, classic); 2610.01889 (round-to-nearest drift on positive inputs, down-projection worst); 2608.06630 (difference Hessian, approximately centered).
- **Why it should help:** removes the mean-input drift for all tensors with no tuning. SwiGLU outputs and gated GDN outputs have non-zero channel means (sizes from #8).
- **Fit:** per level (μ from a bf16 forward pass). **Format: an optional per-output bias field.** If wanted in V3, it must be reserved before the freeze; runtime ~0 (epilogue add).
- **Test:** 0.8B requant of all tensors at K3/K2, 3 seeds (2 h on the 4070).
- **Expected gain / risk:** −0.5…−2%. Risk: μ is tiny for post-RMSNorm readers; the gain sits in writers.
- **Conflicts:** shortlisted "writers' mean-shift bias" (#31). This is the general, all-tensor form with a modified H.

### 29. Drift-amplified Hessian H + γμμᵀ
No format change: just over-weight the mean direction in LDLQ, since drift compounds with depth and random error does not.
- **Source:** 2610.01889 (drift terms D + Q versus variance V).
- **Why it should help:** a bias-free version of #28.
- **Fit:** per level (quantizer setting).
- **Test:** 0.8B requant with γ ∈ {0, 4, 16} (normalized to tr H/n), K3/K2, 3 seeds (2 h).
- **Expected gain / risk:** −0…−1% at 27B after the ⅓ shrink.
- **Conflicts:** #33-07 (gradient-weighted H_in, negative) and #86 (gen-weighted). This is rank-1, not token reweighting.

### 30. Document-scatter Hessian
H + γ·Σ_doc n_d(μ_d − μ)(μ_d − μ)ᵀ. Errors on directions that are constant within a document repeat at every position of that document and are integrated by attention and GDN.
- **Source:** Debias-SparseGPT's contrastive second-order term (2609.02496), recast for persistence; sigma-delta shaping toward high frequency.
- **Why it should help:** protects category and language identity directions. It is a forward-only statistic (per-document means come from the H_in pass).
- **Fit:** per level (needs per-document means: a 27B mkhin27-style pass, forward-only).
- **Test:** 0.8B, one forward pass for μ_d, then requant at γ ∈ {1, 4}, 3 seeds (2.5 h).
- **Expected gain / risk:** −0…−1% at 27B.
- **Conflicts:** #101 and #91 (statistics blends). This reweights calib's own structure; it does not add data.

### 31. Writers' mean-shift bias, fitted in the logit Fisher metric
Fit writer biases to cancel drift as measured through the head's Fisher (backprop from logits to the residual at each writer, from stored goldens), not plain L2.
- **Source:** 2610.01889 (logit-space drift D + Q terms); DFQ.
- **Why it should help:** only the drift directions that move logits are corrected.
- **Fit:** per level (bf16 inputs).
- **Test:** 0.8B (0.5 h).
- **Expected gain / risk:** −0.5…−1.5%.
- **Conflicts:** this is the shortlisted item; the Fisher weighting is the addition.

### 32. Position-dependent bias for early positions
A tiny additive table b[pos, layer] for positions 0..P (P ≈ 4) at writers, because sink and early positions accumulate error differently (DARTS: bias grows with position under causal attention).
- **Source:** DARTS (2608.28547).
- **Why it should help:** only if #5 shows position-0 error matters.
- **Fit:** per mix or per level; format change (small).
- **Test:** 0.8B after #5.
- **Expected gain / risk:** −0…−0.5%. Speculative.
- **Conflicts:** none.

### 33. Sink-weighted H_in for full-attention k_proj and v_proj
Up-weight position-0 (and other massive-activation) rows in H_in for k/v, because every later query reads those keys and values. Test the opposite sign too (#51).
- **Source:** KV-cache practice of sink tokens in bf16 (2605.17757, 2608.27875 HyQuant); 2609.09085.
- **Why it should help:** a coherent error source (sink K/V error) shaped away at no format cost.
- **Fit:** per level (quantizer; H_in recollection on 0.8B, forward-only).
- **Test:** 0.8B, weights ∈ {1, 10, 100} on position-0 rows, 3 seeds (1.5 h).
- **Expected gain / risk:** −0…−1% at 27B.
- **Conflicts:** #33-07 / #86 (token reweighting, gradient or gen-based). The mechanism here is structural.

### 34. Band-offset correction on SwiGLU outputs (video-codec SAO analogue)
Correct E[error | a_j in magnitude band b] with a small per-layer LUT applied in the down_proj input prep pass, which already touches every activation in A8.
- **Source:** HEVC/VVC sample-adaptive offset (classic).
- **Why it should help:** captures Jensen bias that depends on activation magnitude, which a single gain cannot.
- **Fit:** per level; runtime: a LUT in the prep kernel (small prefill cost; needs a kernel change).
- **Test:** 0.8B fake-quant hook (0.5 h).
- **Expected gain / risk:** −0…−0.5%. Speculative; low priority.
- **Conflicts:** none.

### 35. Jensen-bias offsets on sigmoid/silu gates
Add a per-channel bias on the GDN z-gate pre-activation and the attention output-gate pre-activation equal to −½ f''/f' × σ²_noise (or fitted). A gain cannot express this offset.
- **Source:** dropout inference bias and noise-aware inference (classic second-order correction).
- **Why it should help:** gates sit near curvature regions; noise biases them toward 0.5.
- **Fit:** per level; format: bias field (#28).
- **Test:** 0.8B (0.3 h).
- **Expected gain / risk:** −0…−0.5%.
- **Conflicts:** #108 (gate gain refits). This adds the missing offset degree of freedom.

---

## Theme 5 — Temporal noise shaping (attention and GDN act as integrators)

### 36. Attention-filtered H_in for v_proj
The output error of V through attention is Σ_s a_ts E x_s, so the right metric for v_proj rows of KV head h is H̃ = Σ_t Σ_{q∈group} x̃^q_t x̃^qᵀ_t with x̃^q_t = Σ_s a^q_ts x_s. Compute it forward-only by running each layer's attention over the inputs x instead of values. Random error that attention averages out is down-weighted; persistent error is up-weighted.
- **Source:** sigma-delta noise shaping (shape noise toward what the downstream filter removes); score-weighted value covariance in KV quantization (OSCAR 2605.17757, eq. 2); SACQ's v-metric (2610.41408).
- **Why it should help:** a forward-only, exact-structure output metric for V, which needs no Σdy².
- **Fit:** per level (quantizer; H̃ from one forward pass at 27B, about 1–2 card-h).
- **Test:** 0.8B, collect H̃ and requant v_proj only, 3 seeds (1 h).
- **Expected gain / risk:** −0.2…−0.8% at 27B (⅓ shrink; v is small).
- **Conflicts:** #33 (H_out); #101 (H blends). Different object: a mechanism-derived H.

### 37. Query-covariance metric for k_proj (and key covariance for q)
Logit error = qᵀ(E_k x). Use two-sided LDLQ with H_out = E[q qᵀ] per head (q_norm-normalized) for k_proj, and E[k kᵀ] for the q part of attn_qg.
- **Source:** OSCAR / NOVA-KV logit-error metric (2605.17757, 2608.04074); SACQ (2610.41408).
- **Why it should help:** a forward-only analytic H_out for 16 layers' q/k. k/v are tiny tensors, so the extra quant time is negligible.
- **Fit:** per level (two-sided LDLQ path exists).
- **Test:** 0.8B (1 h).
- **Expected gain / risk:** −0…−0.5%.
- **Conflicts:** #108 (attention refits null); #33 (H_out). This shapes the error; it does not rescale.

### 38. GDN decay-filtered H for the v/k rows of in_proj_qkv
The GDN state integrates β_s v_s k_sᵀ with per-head decay. The error a reader sees is Σ_s (∏α)(q_t·k̂_s)β_s E_v x_s. Build H̃_v = Σ_t x̃_t x̃_tᵀ with x̃_t the decay- and overlap-weighted input sum, from a bf16 forward pass with the actual gates (an observability-Gramian view).
- **Source:** observability Gramians and balanced truncation for recurrent states (2609.30950, 2610.36617); 2609.38169 (gate half-life predicts error persistence, ρ ≈ 0.8 across 2,304 heads).
- **Why it should help:** gdn_in carries 39% of the H_out gain on 0.8B (#82). This is a forward-only proxy for that output metric.
- **Fit:** per level (quantizer).
- **Test:** 0.8B, compute H̃_v for 3 GDN layers and compare the CPU proxy error in H̃ against the KL-based ranking. Then a full requant (2 h).
- **Expected gain / risk:** −0.3…−1% at 27B (⅓ shrink).
- **Conflicts:** #33 / #82 (Σdy² H_out); shortlisted conv-aware H_in (#39 is its short-filter cousin).

### 39. Conv-aware H_in, row-clustered
The depthwise conv after in_proj_qkv makes the input metric row-dependent: H_c = Σ_{j,j'} f_cj f_cj' C_{j−j'} from lag covariances C_0..C_3. Cluster rows by filter shape, k-means on f_c with k ≈ 4–8, and run LDLQ per cluster with the cluster's H.
- **Source:** FIR channel-equalizer theory (classic).
- **Why it should help:** makes the shortlisted conv-aware item tractable with LDLQ's one-H-per-tensor structure.
- **Fit:** per level (quantizer; a lag-covariance collection, forward-only).
- **Test:** CPU proxy on 3 GDN layers once lag covariances exist (#8), then 0.8B.
- **Expected gain / risk:** −0.2…−1%.
- **Conflicts:** shortlisted "conv-aware GDN qkv H_in". The clustering is the addition.

### 40. Temporal-kernel check on o_proj and out_proj inputs
o_proj and out_proj act per token after mixing, so theory predicts no temporal weighting is needed. Verify with #8's lag-covariance share and skip these families if it is negligible. This keeps theme 5 to the v/k/qkv rows.
- **Source:** as #36.
- **Why it should help:** prevents wasted runs.
- **Fit:** diagnostic.
- **Test:** inside #8.
- **Expected gain / risk:** none directly.
- **Conflicts:** none.

---

## Theme 6 — Massive channel 3994 and outlier structure

### 41. Massive-column side-car for residual readers
For readers whose H_in has a top-1 eigenvalue share above about 0.5 (L3–L31 layer inputs, per the colo table), store input column 3994 in bf16. Quantize the remaining columns by LDLQ on H with row and column 3994 removed, so the problem is no longer spike-dominated. Runtime: y += x_3994 · w_col, a rank-1 epilogue add (~0 prefill). Bytes: about 70 KB per tensor.
- **Source:** D-SLR (2610.10636: storing some rows verbatim is free under squared error); OCGQuant outlier handling (2609.00066); SpQR / OWQ outlier columns.
- **Why it should help:** removing a 94% spike makes the remaining H well-conditioned, and both the over-damping (table) and the faraway M5 inefficiency (L11 attn_qg 1.74×, L10 ssm_qkv 1.35×) should fall.
- **Fit:** format change (side-car column) plus quantizer; must precede the freeze; per level.
- **Test:** CPU proxy on settle64 for L3/L10/L11 attn_qg and ssm_qkv. Pass if the proxy error ratio to the ideal coder drops from 1.74 to ≤ 1.3. Then the 27B oracle (#6, variant B).
- **Expected gain / risk:** −0.5…−2% at 2.5 bpw. Risk: kernel work; also, the massive channel's column may be cheap already in KL terms (#6 decides).
- **Conflicts:** shortlisted "massive-channel handling on residual readers". This is the concrete design plus the data that motivates it.

### 42. Spike-excluded damping
Scale damping by the bulk spectrum, not by the mean diagonal: λ = σ·(tr H − λ₁)/(n − 1), or the median rotated diagonal. Today 0.025·mean(diag) adds 44% of the median channel's curvature at L3.
- **Source:** spike-and-flat preconditioner structure (Pro-KLShampoo 2605.06316; subspace-aware Kronecker 2606.00542); Levenberg–Marquardt scaling (classic).
- **Why it should help:** stops the massive channel from setting the bulk damping. The input groups' per-tensor σ sweep mitigates this partly (0.005 at L3 still adds 8.7%); MLP inputs at fixed 0.025 are mild (≤ 4%).
- **Fit:** quantizer setting (must precede the freeze); per level.
- **Test:** CPU proxy on colo/Thor, settle64 with proxy error measured against the undamped H (as #38 did), over 20 tensors including L3/L10/L11/L31 readers (about 1 h).
- **Expected gain / risk:** −0…−1%. Concentrated in about 10 early-to-mid reader tensors.
- **Conflicts:** #38 (scalar damping grid). The sweep is over magnitudes, not over what the damping is normalized to.

### 43. Marquardt diagonal damping
Damp with H + λ·diag(H) in the rotated basis, which damps each channel relative to its own curvature (per-block adaptive after the Hadamard).
- **Source:** Levenberg–Marquardt (classic); FOAM damping analysis (2606.02365).
- **Why it should help:** a variant of #42 that also handles the block-trace spread (max/median up to 640).
- **Fit:** quantizer setting.
- **Test:** in #42's CPU run.
- **Expected gain / risk:** same as #42.
- **Conflicts:** #38.

### 44. Massive-row exact writers
Store output row 3994 of down_proj, o_proj and out_proj in bf16 (17,408 values per down_proj). Quantize the rest. Runtime: one extra dot product per token (epilogue), ~0.
- **Source:** super weights (2607.10803: < 0.5 ppm of parameters can collapse a model); our scan (row 3994 is the top row in about 60 of 64 down_proj tensors).
- **Why it should help:** these rows write the massive activation that the sink mechanism relies on. Their error is coherent: the same channel at every massive-activation token.
- **Fit:** format change (side-car row); per level.
- **Test:** #6 oracle variant A first (1 card-h).
- **Expected gain / risk:** −0…−1%. Risk: the error on this channel may be harmless (radial; see #27).
- **Conflicts:** #96 / #29-02 (row-slab K, negative). This is a single-row escape, not a slab.

### 45. Super-weight sparse escapes
Store the k largest |w| entries of early down_proj tensors exactly: L0 has 40 entries above 50× RMS and 644 above 20×. Quantize W − S. Runtime: a sparse add (k ≤ 100 per tensor), negligible.
- **Source:** 2607.10803; Seed-Q's 1e3-ratio outlier columns (2609.38477).
- **Why it should help:** the 128-point Hadamard spreads a 92× entry over 128 coordinates in its block, which raises that block's error.
- **Fit:** format change; per level.
- **Test:** CPU proxy on L0/L2 down_proj (minutes), then fold into #6.
- **Expected gain / risk:** −0…−0.5%.
- **Conflicts:** #40 (per-group scales, negative). This is sparse and exact.

### 46. Sink tokens on the f16 path in A8 GEMMs (fast variant)
Route the few rows (tokens) with massive activations, detected at runtime by norm, through the f16 path inside A8 GEMMs. Their per-token int8 scale is otherwise crushed, and every later query reads them.
- **Source:** HyQuant (2608.27875), LampAttention look-ahead mixed precision (2609.39361), KV-sink protection.
- **Why it should help:** A8's error on sink tokens is coherent across the document.
- **Fit:** kernel feature (row split); quality setting for the fast variant.
- **Test:** E1b fake-quant hook with the sink rows exempted, on a K2.5 base (about 1 card-h).
- **Expected gain / risk:** reduces the fast variant's A8 cost by perhaps 10–30% of its increment; small in absolute terms at ≤ 3 bpw. Risk: kernel complexity (flagged; skip if speed-only).
- **Conflicts:** #47 (R1 negative); w42 (per-block A8 exceptions negative). Rows, not blocks.

### 47. Spike-and-flat H_in
Keep the exact top-r eigenspace of H_in and replace the tail by its mean eigenvalue: H_r = U_rΛ_rU_rᵀ + λ̄(I − U_rU_rᵀ). This is a structured regularizer that matches the model's actual spectrum shape.
- **Source:** Pro-KLShampoo (2605.06316); 2606.00542 (the top eigenspace aligns with the Hessian, the tail is unreliable).
- **Why it should help:** robustness to the calib/private mismatch in tail directions.
- **Fit:** per level (quantizer setting).
- **Test:** 0.8B requant, r ∈ {64, 256}, 3 seeds (2 h).
- **Expected gain / risk:** −0…−0.5%. Risk: the tail matters for LDLQ shaping.
- **Conflicts:** #38 (damping, a crude version of this).

### 48. Spectral tempering H^p (trace-preserving)
Use H_in^p with p ∈ {0.8, 0.9}, trace-normalized. ShamAN-Q found p = ½ best at sub-1-bit, where error is large; our K2.5 regime suggests p near 0.9.
- **Source:** ShamAN-Q (2609.38521); Bregman divergence choice (2610.08534).
- **Why it should help:** distributional robustness of the quadratic proxy (DRQ 2610.11226 shows reconstruction-optimal can hurt).
- **Fit:** per level (quantizer setting).
- **Test:** 0.8B, 3 seeds (2 h).
- **Expected gain / risk:** −0…−1% at 27B (⅓ shrink). Risk: likely null at our error levels.
- **Conflicts:** #38.

### 49. Global residual-stream permutation by companion grouping
Permute the 5120 residual channels globally, so that channel 3994 shares its 128-block with the 127 lowest-variance channels (or the most massive channels share one block). This is exact and free at runtime: reindex the embedding columns, every reader's input columns, every writer's output rows and the norm weights. It changes only which channels the Hadamard mixes.
- **Source:** OCGQuant outlier-companion grouping (2609.00066); MosaicQuant block hot spots (2606.15652).
- **Why it should help:** today the spike's block mixes it with 127 ordinary channels, and their coding is coupled to the spike.
- **Fit:** quantizer and container (permutation is folded); zero runtime; must precede the freeze since it touches every tensor.
- **Test:** CPU proxy, permute H and W columns before the Hadamard for L3/L11 readers and compare proxy error (minutes on colo). Check writers' rows too.
- **Expected gain / risk:** −0…−1%. Risk: the gain may be nil, since LDLQ handles the spike direction.
- **Conflicts:** #41 (alternative). #95 / #103 / #106 act order: those were within-tensor column orders; this is a residual-stream gauge choice.

### 50. Rotation-seed choice for the spike block only
Pick the Hadamard sign pattern for block 31 (channels 3968–4095) by proxy error. The spike makes this one block seed-sensitive.
- **Source:** 2609.21450 (sign-pattern sampling helps W4A4).
- **Why it should help:** a targeted version of best-of-N seeds.
- **Fit:** per level; zero runtime.
- **Test:** CPU proxy, 8 seeds for that block on L3/L11 (minutes).
- **Expected gain / risk:** −0…−0.3%.
- **Conflicts:** #41 (best-of-N rotation seeds: null for whole tensors). One spike block may behave differently. Low priority.

---

## Theme 7 — Hessian estimation and robustness

### 51. Robust (Tyler / leverage-weighted) H_in
H_robust = Σ_t x_txᵀ_t / (x_tᵀ Σ⁻¹ x_t). This down-weights massive-activation tokens, as KL-Shampoo's inverse-quadratic token weights do. Run it as a pair with #33 (opposite sign).
- **Source:** Tyler's M-estimator (classic); KL-Shampoo token weights via ShamAN-Q (2609.38521: condition number ~40 versus several hundred).
- **Why it should help:** stops a few sink tokens from setting LDLQ's shaping for every other token.
- **Fit:** per level (0.8B recollection, forward-only).
- **Test:** 0.8B, 3 seeds (2.5 h).
- **Expected gain / risk:** −0…−1% at 27B (⅓ shrink).
- **Conflicts:** #33-07 (gradient-weighted, negative); #86 (gen-weighted, inconclusive).

### 52. Teacher-entropy-weighted H_in
Weight tokens by the bf16 next-token entropy, taken from the goldens (no backward needed).
- **Source:** EGAD (2605.01732), DARTS (2608.28547), CaRE-KD (2609.36734): uncertain positions dominate the downstream damage.
- **Why it should help:** mean KL concentrates at high-entropy positions.
- **Fit:** per level (recollection, forward-only).
- **Test:** 0.8B (2.5 h).
- **Expected gain / risk:** −0…−1% (⅓ shrink).
- **Conflicts:** #86 (gen-weighted H_in: K2 0.976, every seed, missed its bar). Entropy weighting is a sharper version; #33-07.

### 53. Category-worst-case (distributionally robust) H_in
H_DR = H + α·(top generalized eigen-direction of per-category H_c deviations), or minimize the maximum over categories of the proxy error.
- **Source:** DRQ (2610.11226), robust optimization (classic).
- **Why it should help:** the private corpus is "drawn like calib" but the per-category mix varies; this is insurance for the agentic and multilingual tails.
- **Fit:** per level.
- **Test:** 0.8B with per-category H_c (forward-only), 3 seeds (3 h).
- **Expected gain / risk:** ≈0 on the mean, protects the tails. Low priority.
- **Conflicts:** #101 / #91.

### 54. Symmetry-averaged head-wise Σdy² smoothing (K ≥ 4 levels only)
Average the existing Σdy² H_out diagonal across GQA heads that share structure, or across equivalent positions, to cut estimation noise. No new collection.
- **Source:** curvature from weight-space symmetries (2606.00442).
- **Why it should help:** H_out is adopted only at K ≥ 4; denoising the existing diagonal is free.
- **Fit:** per level (K ≥ 4).
- **Test:** CPU proxy plus one 27B K4 pair (3 card-h, since it is the slow path).
- **Expected gain / risk:** −0…−0.5% at K4; ~0 at 2.5 bpw.
- **Conflicts:** #33-27 (block normalization, already a robustness step).

### 55. Kronecker-sum head Hessian by entropy cluster
H_head ≈ Σ_c X_cᵀX_c ⊗ diag(F̄_c), clustering positions by entropy (2–4 clusters) instead of one Fisher diagonal.
- **Source:** Kronecker-sum Fisher (2609.02451), SoftWater class-aware head allocation (2608.12026).
- **Why it should help:** the head's row importance depends on context type.
- **Fit:** per level (head quantizer; needs a non-Kronecker LDLQ variant).
- **Test:** headsim (minutes to 1 h).
- **Expected gain / risk:** −0…−0.5% total.
- **Conflicts:** #34 (slab head, Fisher diagonal).

---

## Theme 8 — The LDLQ solver (MIMO detection, halftoning, numerical linear algebra)

### 56. Feedback damping and error clamping in LDLQ
Decay the feedback by γ_i = exp(−λ·[H⁻¹]_ii/mean·i/B), and clamp per-column feedback to ±kσ. This is the halftoning fix for runaway error-diffusion "worms" at the end of the scan, where the last-processed block has absorbed all accumulated corrections.
- **Source:** error-diffusion clamping (classic; 2609.17276); QTEA error decay (2609.00224: +0.46 avg).
- **Why it should help:** reduces end-of-scan pile-up, which explains why order matters (#106: reversed order cost +5%).
- **Fit:** quantizer setting.
- **Test:** CPU proxy settle64 (exllamav3 order), λ ∈ {0, 0.5, 1} (1 h on colo).
- **Expected gain / risk:** −0…−0.5%.
- **Conflicts:** #95 / #103 / #106 (order effects; different lever).

### 57. K-best LDLQ across column blocks
Keep M ≤ 4 candidate trellis solutions per 16-column block and pick jointly with the next blocks.
- **Source:** K-best / M-algorithm MIMO detection versus SIC (classic); MPQ (2610.32127).
- **Why it should help:** SIC is weakest exactly where the H spectrum is spiky (theme 6 tensors).
- **Fit:** quantizer (×M quant time on selected tensors).
- **Test:** CPU proxy on L3/L11 attn_qg (1 h).
- **Expected gain / risk:** −0…−0.3% (tile CD gave 0.5–1.2% per tensor).
- **Conflicts:** #77 (tile CD, adopted on attn_qg); #113 (beam over trellis states, a different axis).

### 58. Tile CD on the spike-dominated readers
Extend #77's cd3 from all attn_qg to L3/L10/L11/L31 ssm_qkv as well (in GDN layers): the tensors with ≥ 0.6 top-1 share in the colo table.
- **Source:** MIMO local search pays most where SIC is worst.
- **Why it should help:** targets faraway M5's 1.35–1.74× inefficiency.
- **Fit:** quantizer setting (a few minutes per level).
- **Test:** GPU quantizer ratio on those tensors (#77's harness, about 1 card-h).
- **Expected gain / risk:** −0…−0.3%.
- **Conflicts:** #77 (extension).

### 59. K-dependent MMSE regularization
Set damping ∝ D(K)·tr H/n, so the regularizer equals the quantization-noise power, as MMSE-SIC theory prescribes.
- **Source:** MMSE-SIC detection (classic).
- **Why it should help:** principled K-scaling of damping.
- **Fit:** quantizer setting.
- **Test:** CPU proxy (#38 harness), K1.5–5.5 (1 h).
- **Expected gain / risk:** −0…−0.5%. Low confidence.
- **Conflicts:** #38 re-checked at fractional K and found no alternative ≥ 1%. Run only together with #42.

### 60. fp64 LDL with iterative refinement on ill-conditioned tensors
Only if #10 shows a gap: use fp64 factorization plus one refinement step for the about 20 spike tensors.
- **Source:** 2606.18411, 2607.04828.
- **Why it should help:** numerical hygiene.
- **Fit:** quantizer setting.
- **Test:** #10.
- **Expected gain / risk:** ≈0 expected.
- **Conflicts:** none.

### 61. Hessian write-time certification
Cauchy–Schwarz check plus an LDL pivot-growth check plus a Gershgorin bound at collection and at load, refusing the job on failure.
- **Source:** WitCert-style runtime soundness checks (2607.28699); numerical LA (classic).
- **Why it should help:** the corrupt L43 o_proj H_in (LOG 11-06) was caught only by a stalled job.
- **Fit:** pipeline hygiene; zero quality cost.
- **Test:** unit test over all 257 published files (CPU, minutes).
- **Expected gain / risk:** none to KL; prevents silent damage in the ladder.
- **Conflicts:** already listed as "Open" in LOG 11-06. This specifies the checks.

---

## Theme 9 — Library versus mix (decision feedback, precoding, Wyner–Ziv)

### 62. Drift-only ("expected-decision") asym writers
Fit the writers' asym target to the *seed-averaged* sibling error E_seed[ΔA] (2–3 seeds of the sibling level) instead of one realization. That is the deterministic, K-dependent part of what the writer sees. It should transfer across sibling seeds, and with a per-K table, across sibling K.
- **Source:** decision-directed versus expected-decision equalization (classic DFE); #87-07 diagnosis (the realization-specific part does not transfer).
- **Why it should help:** keeps the transferable part of asym (#87: K3 −14% on the 0.8B) in a library-safe form.
- **Fit:** per level (needs 2–3 sibling seeds per level; the library has one).
- **Test:** 0.8B, writers fitted to the mean over seeds 1–2 and evaluated on seed 0's siblings, at matched and ±1 K (#111's design; 2 h on the 4070).
- **Expected gain / risk:** keeps perhaps 30–60% of asym's gain library-safe. If #105 finds asym at 27B is −5%, that is −1.5…−3%. Risk: drift may be a small part of what asym exploits.
- **Conflicts:** #88 (statistical noise target, negative: Gaussian noise, not realized drift); #87-07; #111.

### 63. Closed-loop sequential refit of continuous parameters per mix
For a finished mix, one staged forward pass. Layer by layer, refit svh, suh and biases (and swj's (c_g, c_u)) using the mix's own quantized-prefix inputs against bf16 targets: block outputs or the residual stream at block ends. This is GPTAQ's asymmetric target applied to scales only, i.e. an LQR-style closed-loop controller. No requant.
- **Source:** layerwise LQR (2605.04230); SHIFT-LLM closed-form corrections (2608.25068); LACE-SVD propagation-aware correction (2607.03057); calibration correction layer by layer (2608.08506).
- **Why it should help:** captures the cross-tensor compensation asym gets from exact siblings, at minutes to hours rather than 4–5 card-h per mix, with few parameters and so little room to overfit.
- **Fit:** per-mix final pass (one forward, about 1–2 card-h at 27B); zero runtime.
- **Test:** 0.8B mixes at K2/K3 against open-loop swj, 3 seeds; pass ≤ 0.97 on search with verify agreeing (1 h). Then 27B with the swjfit27 staging.
- **Expected gain / risk:** −1…−4% at 2.5 bpw, uncertain. Risk: #88-02 (asym target on all 150 tensors with quantized upstream: 1.53× worse). That refit codes; scales-only has about 1000× fewer degrees of freedom.
- **Conflicts:** #108 (open-loop joint refits, fail in the full stack); #89 (healing, deferred). This is one pass and closed-form.

### 64. Writers' asym as a fixed-code refit per mix
The subset of #63 for writers only: refit down, o and out scales against the sibling-quantized inputs of the mix.
- **Source:** as #63.
- **Why it should help:** the cheap rollout form if #105 passes but per-mix requant is too slow.
- **Fit:** per mix.
- **Test:** 0.8B next to #111's per-mix writer pass (xm/xc 0.81–0.90) (0.5 h).
- **Expected gain / risk:** keeps perhaps 30–50% of the per-mix pass gain.
- **Conflicts:** BACKLOG review idea 4 (writers' asym as a fixed-code refit). This prioritizes it.

### 65. Sparse re-encode for the per-mix writer pass
If a per-mix asym requant is chosen, re-run the trellis only on writer tiles whose asym target moved more than a threshold from the per-level codes (most tiles barely move). Warm-start the others.
- **Source:** conditional replenishment and RDO partial re-encode in video coding (classic).
- **Why it should help:** cuts the 4–5 card-h per mix (#111) by several-fold.
- **Fit:** per mix (quantizer).
- **Test:** 0.8B: measure the share of tiles whose optimal code changes, and the KL at 10/30/100% re-encoded (1 h).
- **Expected gain / risk:** cost reduction only; ~0 KL loss at 30%.
- **Conflicts:** #111.

### 66. Narrow-band product library for writers
Ship asym writers for sibling K ∈ {K_w, K_w ± ⅛} only, and constrain the knapsack to pick within the band.
- **Source:** successive-refinement / Wyner–Ziv style side-information codebooks (classic).
- **Why it should help:** #111 shows a one-K-step mismatch costs 8–24%; a ±⅛ band may be close enough.
- **Fit:** per level (3× writer levels).
- **Test:** 0.8B at mismatch ⅛ versus ½ (1 h).
- **Expected gain / risk:** the knapsack keeps asym at small flexibility cost. Risk: ⅛ may still mismatch.
- **Conflicts:** #111 (writer K tied to siblings is one of the user's options; this relaxes it).

### 67. Interaction-aware knapsack terms for writer–sibling pairs
Add pairwise costs for (writer K, sibling K) under asym or closed-loop refits, and solve with greedy interaction using exact marginal costs.
- **Source:** CASP greedy interaction (2609.22131); ILA-AMP pair DP (2606.13054).
- **Why it should help:** asym's value depends on the pair.
- **Fit:** knapsack.
- **Test:** CPU, from #111 data plus about 10 new pair measurements.
- **Expected gain / risk:** enables #62–66; no direct gain.
- **Conflicts:** w58 repricing.

---

## Theme 10 — Knapsack and allocation

### 68. Drift-plus-variance sensitivity model
Model KL ≈ ½‖Σ_l b_l‖²_F + Σ_l v_l: drift adds as vectors (coherently), variance as scalars. Fit the b_l scale from the seed-ensemble data (#1/#2); the knapsack becomes quadratic in drift.
- **Source:** 2610.01889 decomposition; WitCert's finding that aggressive schemes survive on cross-layer error *cancellation* (2607.28699).
- **Why it should help:** predicts the super-additivity at low bpw that kfit2's κ partly absorbs. Once coherent fixes remove drift, the knapsack should reallocate.
- **Fit:** knapsack.
- **Test:** CPU refit of kfit2 with the extra term on V2 data plus seed data (2 h CPU).
- **Expected gain / risk:** −0…−2% at ≤ 2.5 bpw via better allocation. Risk: data-hungry.
- **Conflicts:** #11 (symmetric compounding, negative); kfit2's κ. This has a different functional form.

### 69. CVaR or category-robust knapsack objective
Minimize mean KL + β·CVaR over document categories, or cap any category's ratio.
- **Source:** robust optimization (classic); DRQ (2610.11226).
- **Why it should help:** insurance for the private agentic and multilingual mix.
- **Fit:** knapsack.
- **Test:** CPU replay on V2 slices.
- **Expected gain / risk:** ~0 on mean KL; protects tails.
- **Conflicts:** none.

### 70. Explicit engine-floor term at high bpw
Add an additive non-weight floor to the fit (head precision, bf16 stores, attention numerics), so the 7.0 sensitivities are not inflated by the floor.
- **Source:** noise-floor-aware allocation (NANQ 2608.02700).
- **Why it should help:** the 7.0 fast-variant decision rests on 0.00045 versus 0.00052 differences.
- **Fit:** knapsack.
- **Test:** CPU, after #99's floor split.
- **Expected gain / risk:** none at 2.5; it improves 7.0 and 8.0 decisions.
- **Conflicts:** #99 (planned; this consumes it).

### 71. Analytic swj and IV factor per family and K in pricing
Feed #17's c(K) model, and the families fixed by #11/#13, into KN_SWJ-style factors for all families, not just gate_up.
- **Source:** #17.
- **Why it should help:** keeps knap5 consistent once coherent fixes go beyond the MLP.
- **Fit:** knapsack.
- **Test:** CPU.
- **Expected gain / risk:** pricing accuracy only.
- **Conflicts:** w58 / w63.

---

## Theme 11 — Distillation ideas usable in one-pass fits

### 72. Entropy- or Fisher-weighted swj and bias fits
Weight positions in the closed-form fits by the bf16 next-token entropy, or by the logit-Fisher mass Σp(1−p), both from the goldens. Uncertain positions dominate mean KL.
- **Source:** DARTS entropy-weighted L1 (2608.28547); EGAD (2605.01732); AED (2608.14685: the reverse-KL optimum is a tempered teacher).
- **Why it should help:** spends the fit's few degrees of freedom where KL lives.
- **Fit:** per level (bf16 inputs); zero runtime.
- **Test:** 0.8B fixed codes, swj versus weighted swj, 3 seeds (0.3 h).
- **Expected gain / risk:** −0…−1%.
- **Conflicts:** #104 (baseline).

### 73. Disagreement-weighted closed-loop fits (per mix)
In #63, weight positions by |p_t − q_t| (top-k disagreement) from the mix's own eval pass.
- **Source:** 2609.33791 (only tokens with large teacher–student disagreement matter for on-policy distillation).
- **Why it should help:** focuses #63 on the positions that carry the KL.
- **Fit:** per mix.
- **Test:** an arm inside #63 (0.3 h).
- **Expected gain / risk:** −0…−0.5% over #63.
- **Conflicts:** #63.

### 74. Block-end hidden-state targets versus local targets
For #63, compare targets: (a) per-module output, (b) residual stream at block end, (c) residual stream 2 blocks ahead ("lookahead"). LAQuant found the one-block lookahead best (2605.08755).
- **Source:** LAQuant (2605.08755); hidden-state distillation (2605.11513).
- **Why it should help:** the target choice decides how much propagation the fit compensates.
- **Fit:** per mix.
- **Test:** three arms in #63 (0.5 h).
- **Expected gain / risk:** decides #63's design.
- **Conflicts:** #63.

---

## Theme 12 — Numeric floors (high-bpw sizes; engine side)

### 75. Engine numeric floor audit for 7.0 and 8.0
Measure KL change from fp32 GDN state carry between chunks, fp32 residual accumulation, and f32 output stores, on V2 7.00 and 8.00. LeapQuant: per-step bf16 state cost 16 AIME points on Qwen3.5-9B (2609.38166); HEAL: downcasts at kernel boundaries (2606.21023).
- **Source:** 2609.38166, 2606.21023.
- **Why it should help:** a non-weight floor near 0.0003 would decide whether 8.0 is built and how 7.0 is priced.
- **Fit:** engine setting.
- **Test:** extend #99's floor split with two engine switches (about 1 card-h).
- **Expected gain / risk:** none at 2.5; it may decide 8.0.
- **Conflicts:** #99, #33-26 (chunk floor 0.0007 at 512 on the 0.8B).

### 76. Row-local invariance guard for A8 kernels
Assert that per-token, per-128-group A8 quantization and its scales are row-local and independent of chunking (prefix invariance).
- **Source:** 2609.39816 (non-row-local fast matmul changes scored likelihoods).
- **Why it should help:** a chunk-size-dependent KL shift was seen once (+1.5%); this is a correctness check, not a speed item.
- **Fit:** kernel test.
- **Test:** selftest matrix entry (minutes).
- **Expected gain / risk:** none to KL; prevents measurement drift.
- **Conflicts:** v3.md §1c.4 selftest list (adds one case).

---

## Theme 13 — Low-rank side channels

### 77. Rank-r drift correction
If #7 shows E_drift concentrated (top-16 singular values carry ≥ 20% of its H-weighted energy), add a rank-16 bf16 correction for the drift part only, on the 5–10 worst tensors.
- **Source:** QERA / LQER-type corrections; IO-SVD whitening (2605.15626); BiLoCo (2610.32243).
- **Why it should help:** fixes structured error that scales cannot.
- **Fit:** per level; runtime: a small extra GEMM (prefill about +1% on those tensors).
- **Test:** after #7; 0.8B (1 h).
- **Expected gain / risk:** −0…−1%. Risk: the A1 lesson (#81-04): extra GEMMs cost more than about 3% KL is worth.
- **Conflicts:** #81-04.

### 78. Drift-aware SVD for the head's coherent error
A whitened SVD of the head error over frequent-token rows only. Rank-8 corrections are cheap at decode relative to the 1.27 GB head.
- **Source:** ARCHead (2608.02703: rank-6 residual correction gave relative PPL 1.007 versus 1.151 group INT4).
- **Why it should help:** the head error is coherent per token class.
- **Fit:** per level; runtime: decode +1–2%.
- **Test:** headsim (minutes).
- **Expected gain / risk:** −0.2…−0.8% total at ≤ 3 bpw; small against a slab head.
- **Conflicts:** #34 (slab head); #24 (head above K6).

---

## Theme 14 — Format and V4 notes (not for V3 unless marked)

### 79. Reserve an optional bias vector in the exl3i encoding (V3, decide before the freeze)
#28, #31, #35 and #26 all need a per-output (or per-vocab) bias add. Reserve the field and a fused epilogue add now (bytes ~0, runtime ~0), even if V3 ships it unused.
- **Source:** new (a dependency of this report's bias ideas).
- **Why it should help:** avoids a requant or re-encode later.
- **Fit:** format; must precede the freeze.
- **Test:** plugin selftest (1 day of work, no cards).
- **Expected gain / risk:** enables −0.5…−2% (bias ideas).
- **Conflicts:** none.

### 80. Reserve a column/row side-car field (V3, decide before the freeze)
Same for #41, #44 and #45: a sparse bf16 column/row/entry escape list per tensor.
- **Source:** D-SLR (2610.10636); super weights (2607.10803).
- **Why it should help:** keeps theme 6 possible after the ladder.
- **Fit:** format.
- **Test:** plugin selftest.
- **Expected gain / risk:** enables up to −1…−2% if #6's oracles pay.
- **Conflicts:** none.

### 81. VVC-style dependent quantization (V4)
VVC's 4-state "dependent quantization" uses two interleaved scalar quantizers selected by parity. It decodes with a few ALU ops (no hash) at about 1.3–1.5 dB below ideal TCQ.
- **Source:** H.266/VVC (classic); XOR-Trellis (2610.00432).
- **Why it should help:** an ALU-cheap alternative for decode speed.
- **Fit:** format (V4).
- **Test:** CPU proxy versus mul1i8 at K2–4.
- **Expected gain / risk:** speed only; likely KL loss.
- **Conflicts:** #100 (hash codebook, dropped), #113.

### 82. Leverage-weighted Σdy² diagonal (V4 collections only)
If H_out collections ever resume (another model, V4), weight tokens by 1/(xᵀA⁻¹x) (the KL-Shampoo fixed point) instead of plain or Sketch-A weights.
- **Source:** ShamAN-Q (2609.38521), Bregman Shampoo (2610.08534).
- **Why it should help:** the 27B Σdy² heavy tail (#33-27) is the problem this estimator suppresses.
- **Fit:** V4.
- **Test:** none now (collections closed by the user).
- **Expected gain / risk:** n/a for V3.
- **Conflicts:** #33-20 (Sketch A refined weights up-weight these tokens; this does the opposite); the 11-05 "no more collections" rule.

### 83. Halftoning-style terminal-mismatch fix for tail-biting (V4)
Close the tile chain along a single path and absorb the terminal mismatch with one constant update, so tail-biting's 4–6% cost (free/TB 0.944–0.959) is paid once per row, not per tile.
- **Source:** 2609.17276 (rank-one lattice closed path plus constant update).
- **Why it should help:** recovers part of the tail-biting loss.
- **Fit:** format (V4).
- **Test:** CPU proxy.
- **Expected gain / risk:** −1…−2% proxy error; format change.
- **Conflicts:** #102 (chained tail-biting, negative: chaining only amortizes the wrap guess). This targets the start-state bits differently. Low confidence.

### 84. Position-0 / template-prefix bf16 state table — rules check first
Position-0 states depend only on token 0 (2609.09085). For the few most frequent first tokens or chat-template headers, ship bf16 K/V (16 full-attention layers) and rank-1 GDN states (about 100s of KB each), and substitute them at prefill.
- **Source:** prompt and prefix caching (classic); sink protection (2605.17757).
- **Why it should help:** only if #5 shows large sink-driven coherent error.
- **Fit:** container table plus engine hook.
- **Test:** #5 first.
- **Expected gain / risk:** unknown. **Risk: may be judged as gaming the metric. Do not pursue without the challenge author's explicit OK.**
- **Conflicts:** none.

### 85. Seed-averaged scale fits
Fit swj, IV or C9 gains on statistics pooled over 2–3 seeds of the same level, then apply them to each seed. This separates the deterministic shrink, which transfers, from realization noise, which overfits.
- **Source:** ensemble estimation (classic).
- **Why it should help:** more robust per-level fits.
- **Fit:** per level (needs extra seeds; the library has one, so use for checking only).
- **Test:** 0.8B (0.3 h).
- **Expected gain / risk:** checks robustness; ~0 direct gain.
- **Conflicts:** #104.

---

## Ranked top 20 (expected gain per unit cost)

| # | idea | why it ranks here | first test, cost |
|---|---|---|---|
| 1 | **#11 IV output-scale refit, all families** | closed-form, fixed codes, per level, zero runtime; directly attacks the LS shrink behind swj | 0.8B fixed codes, 0.3 h |
| 2 | **#25 + #26 head temperature and vector/bias calibration** | per-mix post-pass on stored logits; convex; shortlisted item gets an analytic prior | eval post-pass, under 0.5 card-h |
| 3 | **#21 attention temperature via q_norm.weight** | zero format change; mechanism #108 could not reach | #4 diagnostic plus 0.8B pair, 0.3 h |
| 4 | **#15 IV-swj** | refines the one proven 27B win; one extra moment | 0.8B, 0.3 h |
| 5 | **#63 closed-loop sequential refit per mix** | potentially the largest; one forward pass; library-safe alternative to per-mix requant | 0.8B, 1 h |
| 6 | **#23 residual RMSNorm re-inflation** | 128 scalars, one forward pass; may explain C9's gdn_out α ≫ 1 | 0.8B, 0.2 h |
| 7 | **#42 / #43 spike-excluded or Marquardt damping** | colo data: 24–44% over-damping at L3/L11 (σ 0.025), 5–9% at σ 0.005; CPU only | colo proxy, 1 h |
| 8 | **#49 global residual-stream permutation** | zero runtime; CPU proxy decides; must precede the freeze | colo proxy, under 1 h |
| 9 | **#41 massive-column side-car** | top-1 eigenvalue share 0.94/0.90 at L3/L11; but format and kernel work | CPU proxy then the #6 oracle |
| 10 | **#62 drift-only (seed-averaged) asym writers** | library-safe asym if #105 passes | 0.8B, 2 h |
| 11 | **#79 / #80 reserve bias and side-car fields** | cheap insurance before the freeze; enables 8 ideas | plugin work, no cards |
| 12 | **#13 λ-ladder (LS → energy → IV)** | tunes #11 per family; clarifies C9 | 0.8B, 1 h |
| 13 | **#18 embedding row IV gain** | coherent per-token error; largest at ≤ 2.5 bpw | 27B embedding swap, 1 card-h |
| 14 | **#16 quantization-time de-biasing (re-test #39 by KL)** | #39 measured the wrong metric for this question | 0.8B requant, 2 h |
| 15 | **#28 centered LDLQ plus exact bias** | general drift fix; needs #79 | 0.8B requant, 2 h |
| 16 | **#29 drift-amplified Hessian** | bias-free variant of #28; no format change | 0.8B requant, 2 h |
| 17 | **#72 entropy-weighted swj and bias fits** | cheap fit-weight change | 0.8B, 0.3 h |
| 18 | **#36 / #38 attention- and decay-filtered H for V and GDN v/k** | forward-only, structure-exact output metrics; no Σdy² | 0.8B, 1–2 h |
| 19 | **#65 sparse re-encode for the per-mix writer pass** | makes per-mix asym affordable if #105 passes | 0.8B, 1 h |
| 20 | **#33 sink-weighted H_in for k/v** | targets the most coherent attention error; forward-only | 0.8B, 1.5 h |

The data also points to one non-cross-area check: **re-test #108's GDN z-gate and qkv refits under the actual 27B ≤ K3.5 stack.** #108 failed because its gain overlapped H_out and asym in the 0.8B full stack. At 27B, H_out is not adopted below K4 and asym is still unproven (#105), so the overlap that killed it may not exist there. The 0.8B screen was 0.95–0.97 at K2. Use the IV estimator from #11.

## Five cheapest diagnostics that would most change this ranking

1. **#1 seed-ensemble drift/variance split** (0.8B with six seeds on the 4070, about 0.5 h; 27B base/s1/s2 about 0.5 card-h). If drift is under about 10% of KL, themes 2–4 and 9 drop sharply in rank. If it is over about 30%, they become the main V3 quality lever after swj.
2. **#4 attention logit-scale and entropy shift** (0.8B forward-only, 0.1 h). Confirms or kills #21/#22, and tests the normalization-shrink theory behind #23 and #25.
3. **#42/#49 spike-damping and companion-permutation proxy** (colo CPU, about 1 h, settle64 order-fixed on L3/L10/L11/L31 readers). Decides theme 6's zero-runtime half before anything touches the format.
4. **#6 massive-row and massive-column oracle at 27B** (1 card-h). If neither bf16 column 3994 in readers nor row 3994 in writers moves KL by 1%, drop #41, #44, #45 and #80.
5. **#5 position-0 sink oracle** (0.8B, 0.3 h). Bounds every sink-related idea (#32, #33, #46, #84). A large effect would also show that some of the coherent error is document-wide.

Colo artifacts (read-only inputs; scratch only): `/data/work/ideas-crossarea/{hdiag.py, sw.py, h1–h4.jsonl, sw.jsonl}`; logs in `/data/logs/xa-*.log`.
