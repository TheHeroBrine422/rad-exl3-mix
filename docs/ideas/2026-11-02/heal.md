# Healing after quantization: research ideas (lens **heal**)

This report has 108 ideas in 12 themes, then a ranked top 20 and the 5 cheapest diagnostics. I checked these files before writing; the facts below shape most of the ideas:

1. **The bf16 teacher already exists for the qad pool.** The published calib golden (top-512 log p, tail mass, entropy and gen mask per row) covers all four pools (#34's head Fisher used it on each). 27B healing therefore never has to run a bf16 teacher forward. The qad pool's 3.7M rows take about 15 GB of disk.
2. **The format already has the healing parameters.** An exl3 tensor is trellis codes plus fp16 `suh` (input side, before the 128-Hadamard) and `svh` (output side): y = svh ⊙ W0(suh ⊙ x) (`exl3.py`). Training those, plus norm gains and the bf16 small GDN tensors, changes no byte layout.
3. **`refit_scales` already fits suh/svh in closed form** (2 alternating rounds, `quantize.py:1583`). Its docstring says it fits in the *damped* H, and its signature takes no H_out, even in the bdiag path. Anything new has to beat that refit; the refit itself has two free improvements (G1, G2).
4. **Quant-to-quant rounding noise is ±2–3% on the 0.8B (#33-15).** Healed-vs-unhealed comparisons on the *same codes* do not carry this noise, only training-seed noise. Conclusions meant to generalise across quants need seed replicates (#33-16).
5. **A possible explanation for the field report.** The report says training only the last 16 layers gave 22%, against 6–13% for all layers at a 1k window. #33-03 found the bf16 backward wrong on a minority of rows in early layers, with H_out-diagonal cosine 0.36 at layers 0–15 and bad rows 3× too large. That would explain the pattern, and it is testable (A3).

## 0. Shared setup used below

- **P08 pilot (RTX 4070).**
  - Model and quants: Qwen3.5-0.8B; trunk quantized by `ourq` at `base` settings (i8, scaled Hessian, damping 0.025 / input 0.005).
  - Levels: uniform **K3** (KL ≈ 0.099, the closest proxy analogue of the 27B at 2.5 bpw, 0.067) and K2. GDN a/b stay bf16, as on the 27B.
  - Student: dequantized W0 frozen, chosen continuous parameters trainable. Teacher: the bf16 0.8B on the same card, computed on the fly or once with `mkgold.py`.
  - Data and slices: train on qad-pool documents (pools-v3); early-stop on a 0.8B fit-pool slice; score on g08-search1; confirm on a fresh 0.8B verify slice, built once.
  - Cost (estimate): quant ≈1 min; healing ≈4–6 min per 1M tokens at a 2k window; eval <1 min. One arm with 3 seeds × 2 K is under 1 GPU-hour.
- **P2B:** the same on the 2B. It is wider (hidden 2048) and costs about 3×; every recipe that passes P08 is confirmed here.
- **PR(g), the pass-rule template.** Fixed before data, as in YAQA08.md. A recipe passes if:
  - healed/unhealed KL on g08-search1 is ≤ 1−g at both K3 and K2;
  - the ratio is below 1 for every quant seed × training seed (3 × 2);
  - the mean gain exceeds 2× the across-seed SD;
  - the verify-slice ratio is ≤ 1−g/2.
- **PR27(g)**, the 27B rule: same container healed vs unhealed, paired per-document bootstrap on golden-s2k (583 search docs, 481k positions), ratio ≤ 1−g with z ≤ −3, then one ver3-8k read, logged.
- **S27, the 27B gradient-healing stack** (cost estimates; engineering ≈3–5 days):

| | |
|---|---|
| Model and kernels | collect.py's streamed model with fla and checkpointing, plus an exl3i decode torch op for all 64 rates |
| Linear layer | autograd Function y = svh ⊙ W0(suh ⊙ x); W0 decoded per call (≤178 MB); dsvh and dsuh as reductions; dX by one bf16 GEMM |
| Teacher | the golden; no teacher forward |
| Cost per token | about 3 forward-equivalents (forward, recompute, dX; no weight-gradient GEMMs), ≈150 GFLOP |
| Throughput | ≈2–3 h per qad-pool epoch on both cards, data-parallel (gradients ≈7.6M parameters; estimate ±2×) |
| Memory, 2.5 bpw mix on one card | weights ≈8.4 GB + checkpoints 0.65 MB/token + recompute/head chunks ≈4 GB → 4k ≈15 GB, 8k ≈19 GB, 16k ≈25 GB |
| Longer windows | 32k needs a two-card layer pipeline, or a backward truncated at layer k (E3) |

- **Pricing format changes.** Near 2.5 bpw, d ln KL / d bpw ≈ 1.6–1.8 (board: 2.0 → 0.164, 2.5 → 0.0666, 3.0 → 0.0306). So **0.01 bpw ≈ 1.7% KL**; anything that adds bytes must beat that.

---

## A. Gates, infrastructure, diagnostics

**A1 · Oracle healing headroom.** Train every format-free continuous parameter (suh/svh, norms, conv1d, A_log/dt_bias, a/b) to convergence *on g08-search1 itself*, deliberately overfit. The result is a ceiling for fixed-code healing; real recipes are then measured as a fraction of it.
- *Source:* new (ceiling method). ThinkBit 2610.34886 (scales + norms, KL 0.641 → 0.282 on scalar 2-bit codes) and Tetra 2609.35465 show what is reachable on weaker bases.
- *Why:* our LDLQ + refit start far nearer the optimum than those papers' bases, so the ceiling may be small. This decides whether to build S27 at all.
- *Fit:* diagnostic.
- *Test:* P08, 1 seed, K3/K2, ≈1 GPU-h. If the oracle is ≤5% at K3, put forward-only fixes (G) first and defer S27. If it is ≥15%, build S27 now.
- *Gain/risk:* none itself; no risk.
- *Conflicts:* none.

**A2 · Does the 0.8B reproduce the field report's depth pattern?** Heal scales and norms on (a) all 24 layers, (b) the last quarter, (c) the first quarter, (d) the middle half, with a 1k window and bf16 backward as in the report.
- *Source:* field report (ROADMAP V5).
- *Why:* training more parameters and doing worse (22% vs 6–13%) is an optimization or generalization pathology. Fixing it may be worth more than any new parameter family.
- *Fit:* diagnostic.
- *Test:* P08 K3, 3 seeds, ≈2 GPU-h. The pattern reproduces if (b) beats (a) by >3% in every seed.
- *Gain/risk:* unlocks C1–C5 if it reproduces. Risk: the 24-layer model doesn't show it.
- *Conflicts:* none.

**A3 · bf16 vs fp32 healing gradients.** Compute dL/dsvh and dL/dsuh per layer on 64 windows with a bf16, x2 and fp32 backward; report cosine and norm ratio per layer.
- *Source:* #33-03, #33-14 (prectest.py machinery).
- *Why:* the bf16-backward row errors that broke H_out collection would also corrupt early-layer scale gradients, which is a direct candidate cause for A2.
- *Fit:* diagnostic.
- *Test:* 4070, 20 min. If the median cosine is <0.9 in the first third of layers, early-layer healing needs C7 (fp32/x2 backward) or freezing. Repeat on one 27B layer once S27 runs.
- *Gain/risk:* decides C7.
- *Conflicts:* #33-14 failed a matrix bound for x2. Summed over millions of tokens, x2 may still be enough for healing, and this measures exactly that.

**A4 · 27B logit-temperature probe by container edit.** Multiply every head row scale (the head's svh) of a V2 2.5 container by τ ∈ {0.97, 0.99, 1.01, 1.03} and score on golden-s2k.
- *Source:* new; quant-sota.md #14.
- *Why:* systematic blur or sharpening from trunk quantization is one global degree of freedom; its size prices K1/K2.
- *Fit:* diagnostic, and a shippable one-parameter fix.
- *Test:* 4 evals ≈20 min on one card. If the optimum τ ≠ 1 gives ≥1% (z ≤ −3), K1 is a priority.
- *Gain/risk:* 0 to −1.5%; no risk.
- *Conflicts:* none.

**A5 · Residual-norm drift profile.** Per layer, measure ‖h_q‖/‖h_fp‖ and the cosine on 64 docs. The squared relative error splits into a length part and an angle part.
- *Source:* 2609.11716 (on Qwen3 the angular part is 89–99% of the error, with a 3.7% norm change).
- *Why:* if quantization shrinks or inflates residual writes systematically, per-layer scalars (B10) and norms (B4) have an obvious target.
- *Fit:* diagnostic.
- *Test:* 0.8B 15 min; 27B through the collector's bf16 forward vs the quantized forward, ≈1 h. If the length part is ≥20% of the error in any depth band, do B10 first.
- *Gain/risk:* none; none.
- *Conflicts:* none.

**A6 · Build S27, with gates.**
- *Gates:*
  - (i) torch fake-quant KL of a V2 2.5 container on 50 search docs within 0.3% of radiance;
  - (ii) finite differences on 10 svh entries agree to 1%;
  - (iii) peak memory at a 16k window is <30 GB.
- *Source:* quant-sota O8; v3/yaqa/collect.py.
- *Fit:* infrastructure for every 27B gradient idea.
- *Cost:* 3–5 days of engineering plus ≈2 GPU-h of checks.
- *Risk:* fractional-rate decode in torch. The plugin and the qtf2 decode exist, but need wrapping.
- *Conflicts:* none.

**A7 · Healed-container round trip.** Write the healed fp16 suh/svh and norms into the exl3i container, then score in radiance on both the f16 and A8 paths against the torch score.
- *Why:* torch trains with bf16 GEMMs, radiance runs f16/int8 kernels, and the scales are stored fp16; a mismatch would silently eat the gain.
- *Fit:* gate before any release.
- *Test:* agreement within 0.5% relative, and the sha256 diff confined to the scale and norm planes. ≈30 min.
- *Gain/risk:* none; none.
- *Conflicts:* none.

**A8 · Training-noise floor.** Run two data-order seeds of the same recipe on the same quant: 6 runs on P08, one extra run on the 27B.
- *Why:* same-quant recipe comparisons carry this noise, not the rounding noise. It sets the floor for every later recipe-vs-recipe rule.
- *Fit:* diagnostic.
- *Test:* P08 30 min; 27B 2–3 h once.
- *Gain/risk:* none; none.
- *Conflicts:* the companion of #33-15.

**A9 · KL-by-position audit after healing.** Bin KL by position (0–2k, 2–8k, 8–32k, …) on golden-long2, healed vs unhealed.
- *Source:* #52 (B4 method).
- *Why:* healing on ≤16k windows could fit short-range behaviour and hurt long positions, which long agentic traces weight heavily.
- *Fit:* gate.
- *Test:* the healed/unhealed ratio must be ≤1.0 in every bin with ≥10k positions. Two evals.
- *Gain/risk:* none; none.
- *Conflicts:* none.

**A10 · Piggyback on the planned 27B diagonal-H_out collection (time-critical).** Add two outputs to `hdiag27.sh` before it runs:
- (a) per-layer residual-stream gradient second moments E[g gᵀ] at block outputs (5120² fp32 = 105 MB per layer, 6.7 GB total);
- (b) teacher residual snapshots at layers 16/32/48/63 for 256 of the 1,024 windows (≈20 GB).
- *Source:* 2609.00049 (REAL-Q block-output Fisher), 2302.11812 (Teacher Intervention).
- *Why:* G5, G7 and D8 then need no second fp32 night (8.5 h).
- *Fit:* data collection, per level.
- *Cost:* about +5–10% run time and ≈30 GB of disk.
- *Risk:* run-time growth.
- *Conflicts:* rides on #33-13; must not change its H_out output.

**A11 · P2B confirmation.** Rerun every P08 winner on the 2B (width 2048). The 0.8B is width 1024 against 5120 on the 27B, so width effects on scale healing are unknown.
- *Test:* ≈3× P08 cost.
- *Conflicts:* the 2B calibration study (#60-04) is paused; healing results do not depend on it.

**A12 · 9B untied proxy for head and embedding healing.** The 0.8B and 2B tie head and embedding. The 9B (18 GB bf16) fits an R9700 slot for head-only and scale-only work.
- *Test:* golden on a qad subset ≈2 GPU-h, plus healing.
- *When:* only if A4/K1 show the head matters.

---

## B. What to train (format-free first, then cheap format changes)

**B1 · Scale healing: svh only, suh only, both** (the core recipe). Train log-scales (s = s0·e^r) by end-to-end KL with the codes fixed. The KL gradient on svh (Σ_t dy ⊙ y_pre) is the output-side signal that H_out captures, but taken at the deployed point with the true loss.
- *Source:* Tetra 2609.35465 (row scales by KL to FP16, 19.5M tokens: +1.7–3.3 MMLU at ~2.7 bpp, 0 bytes); 2609.04526 (PEQA per-row scales are a strong floor); ThinkBit 2610.34886 (KL-trained scales moved a median 6% and correlated 0.07 with the weight-MSE-optimal scale; KL fell 56% while weight error rose); EfficientQAT E2E-QP.
- *Why:* refit_scales picks scales in the input-side proxy; the KL-optimal scales differ.
- *Fit:* final pass per mix; per-level variants in I1/I6.
- *Test:* P08 with arms svh / suh / both, PR(5%). Then the 27B on S27: one epoch on V2 2.5, PR27(5%).
- *Gain:* −3 to −8% at 2.5 bpw (median guess −5%), probably smaller than the field report's numbers, because our quants are already refit.
- *Risk:* overfitting the qad pool; mitigated by C4.
- *Conflicts:* #48 (output scales off is worse) shows these scales matter. #33-06 (diagonal output scales from H_out by exllamav3's rule: worse) applied a fixed heuristic; this trains the scales on the actual loss.

**B2 · Norm-only healing.** Train the input_layernorm, post_attention_layernorm and final-norm gains (s = e^δ), about 0.65M parameters.
- *Source:* ThinkBit (norms-only KL 0.418 vs scales-only 0.282, from 0.641); ProQuant RoCo 2610.30779 (+2.5 avg points on a Qwen3.5-35B GDN hybrid at 1.5 bit); RoleQ 2610.21808.
- *Why:* each gain is shared by every reader of that norm, so the risk of overfitting is low. It is also the cheapest first 27B run.
- *Fit:* per mix.
- *Test:* P08 arm; first 27B run on S27.
- *Gain/risk:* −1 to −3%; low risk.
- *Conflicts:* none.

**B3 · Attention q/k norms and the GDN output-norm gain.** Train per-head-dim q_norm/k_norm gains in the 16 full-attention layers (re-tunes attention logit temperature after q/k quantization) and the gated-RMSNorm weight inside GDN before out_proj (value/z magnitude errors). About 10k parameters.
- *Source:* new; the KV-quant papers (OSCAR, NOVA-KV) show attention-logit error is what matters.
- *Fit:* per mix.
- *Test:* P08 arm on top of B1.
- *Gain/risk:* 0 to −1%; low risk.
- *Conflicts:* none.

**B4 · GDN conv1d taps.** Train the depthwise 4-tap causal conv after in_proj_qkv (≈41k parameters per GDN layer; bf16 in the container, to be checked). This gives each q/k/v output row a learned temporal correction, a strict superset of a per-row scale (a scale is the conv's tap sum).
- *Source:* new.
- *Why:* in_proj_qkv carries 10.4% of GDN-layer output error on this model (2609.04098).
- *Fit:* per mix.
- *Test:* P08 arm, with L2 to the initial taps.
- *Gain/risk:* −0.5 to −2%; overfitting (2M parameters).
- *Conflicts:* none.

**B5 · GDN decay parameters A_log and dt_bias** (≈96 per layer). Retune each head's memory length to offset quantization-noise accumulation in the state.
- *Source:* new; STEPQuant 2609.38169 (gate half-life correlates ρ ≈ 0.80 with accumulated state error).
- *Fit:* per mix.
- *Test:* P08 arm with a tiny LR, and A9 is mandatory.
- *Gain/risk:* 0 to −1%; small changes have long-range effects.
- *Conflicts:* none.

**B6 · Full-weight training of the bf16 in_proj_a / in_proj_b** (stored bf16 per design §1; ≈0.9M parameters per GDN layer, ≈43M in total). These set β_t and the decay g_t, so full freedom lets the state dynamics compensate upstream error, at zero bit cost.
- *Source:* new. 2609.04098 calls a/b the least sensitive to their *own* quantization; here they are used as compensators.
- *Fit:* per mix.
- *Test:* P08 arm, strong L2-to-init and early stopping.
- *Gain/risk:* −0.5 to −3%; overfitting.
- *Conflicts:* none.

**B7 · Per-tensor global gains only** (496 scalars). The lowest-dimensional healing: it almost cannot overfit, and it is the KL-optimal version of "gain vs K".
- *Source:* #39 (CPU proxy: the best factor rises with K, 1.1 up to K5, 1.3 at K7–8).
- *Fit:* per level (a gain per (tensor, K)) or per mix.
- *Test:* P08 arm. On the 27B without S27, via G8 line searches per (family × depth band).
- *Gain/risk:* −0.5 to −2%; low risk.
- *Conflicts:* #39 (blocked). This unblocks it with KL instead of the proxy.

**B8 · Per-layer residual-writer scalars** (o_proj, ssm_out, down: about 128 scalars).
- *Why:* corrects residual-norm drift if A5 finds any.
- *Fit:* per mix; G8-testable on the 27B with radiance alone.
- *Gain/risk:* −0.5 to −2% if A5 is positive, else 0; low risk.
- *Conflicts:* none.

**B9 · Residual-writer channel biases** (format change). Add a bias in the exl3i epilogue for o_proj, ssm_out and down: 64 × 2 × 5120 fp16 ≈ 1.3 MB, 0.0004 bpw, priced at ≈0.07% KL. Initialize in closed form, b = 0.85·mean(Y* − Y_q) on the G3 stream, then train by KL. Scales cannot correct mean shifts in the residual stream.
- *Source:* SSDi8 2608.21952 (mean correction with damping 0.15); D²Quant (quant-sota #4); QuantForge per-head value affine.
- *Fit:* per mix, plus a kernel selftest.
- *Test:* P08 arm (bias as a torch parameter); 27B closed-form init first.
- *Gain/risk:* −0.5 to −2%; small kernel change.
- *Conflicts:* none.

**B10 · Biases on all linears** (≈3.5 MB, 0.001 bpw, priced ≈0.17%). Reader biases (q/k/v, gate/up) shift nonlinear operating points such as SiLU, which scales cannot do.
- *Fit:* per mix; format change.
- *Test:* P08 arm after B9.
- *Gain/risk:* a further −0.5 to −1.5%; bit price above.
- *Conflicts:* none.

**B11 · Tiny per-layer residual adapters** (format change). A rank-4 to 8 bf16 side path reads the post-norm input and writes to the residual: r = 8 is ≈10.5 MB ≈ 0.003 bpw ≈ 0.5% KL price. Init B = 0.
- *Source:* QATFactory 2609.39223 (LoRA r = 4: KL 0.0778 → 0.0722 vs full QAD 0.0682 on Qwen3.5-9B NVFP4; more rank did not help); 2606.04238 (zero init beats SVD init); 2609.06974.
- *Fit:* per mix; one skinny GEMM per layer (≈0.2% prefill).
- *Test:* P08 after B1/B2 pass; PR: the net gain must beat the price.
- *Gain/risk:* gross −1 to −5%, net −0.5 to −4%; kernel work.
- *Conflicts:* quant-sota calls LoRA low priority (bits are better spent on K). This is ≤0.003 bpw, not 0.06.

**B12 · Side LoRA only on the knapsack's spike tensors**, e.g. L60–63 gate_up and down (single-layer probe: L63 gate_up at K2 costs +0.0065, 80% of the base's KL). r = 8 on 8 tensors ≈ 1.8 MB.
- *Fit:* priced as a knapsack option "K + adapter" on the tensors where a K upgrade is most expensive.
- *Test:* P08, its last layer.
- *Gain/risk:* −1 to −3% if error is concentrated; kernel work.
- *Conflicts:* #21 (the spike is at the end, not layer 0) supports the targeting.

**B13 · Low-rank block-scale grid (research ceiling).** Scales per (row × 128-input-block), stored rank-2 (Scale-QLoRA ΔS = BA, ≈0 bits). Mathematically they commute with the block Hadamard, but the kernel would need a group-scale path (E2: −12 to −29% GEMM speed).
- *Purpose:* only to measure whether freedom finer than rank 1 adds anything over B1. If it adds ≤1%, the direction is closed.
- *Fit:* fake-quant only, on P08.
- *Gain/risk:* diagnostic; the speed cost would kill shipping it anyway.
- *Conflicts:* #40 (group scales lose at equal bits); here the grid is low-rank, ≈0 bits.

**B14 · Proxy-anchored healing.** Loss = KL + λ·Σ_tensors tr(ΔW H ΔWᵀ)/tr(W H Wᵀ), using the 10M-token published H. The Hessian proxy becomes a prior that keeps scales near the refit optimum where the qad data is thin (3.7M tokens against ≈7.6M parameters).
- *Source:* new; DRQ 2610.11226 (refinement overfits its activation sample).
- *Fit:* a regularizer for every B/C recipe.
- *Test:* P08 λ sweep {0, 0.1, 1}× scale-matched.
- *Gain/risk:* 0 to −2% (generalization); λ tuning.
- *Conflicts:* none.

---

## C. Which layers, and how to optimize

**C1 · Back-to-front curriculum.** Heal layers 48–63 to convergence and freeze them, then 32–47 against the healed tail, and so on. Early layers then optimize against a fixed, already-corrected downstream.
- *Source:* JustQuant 2609.33601 (one-shot full-model QAD collapsed, FID 188, vs 6.48 staged local-to-global); 2609.19213 (layer curriculum).
- *Fit:* per mix.
- *Test:* P08 after A2 reproduces the pattern; 4 stages ≈1 GPU-h.
- *Gain/risk:* recovers the all-layer gain without the pathology, −2 to −5% over last-quarter-only; 4× sequential cost.
- *Conflicts:* none.

**C2 · Sensitivity-scaled per-layer LR.** Adam normalizes each step to ≈LR, so an early-layer scale (whose error compounds) moves as far as a late one. Scale the LR per layer by 1/√(kfit2 sensitivity), or by each scale group's measured diagonal Fisher.
- *Source:* REAL-Q 2609.00049 (layer-wise LR schedule 1e-6 to 5e-4); kfit2's compounding term (#68).
- *Fit:* optimizer setting.
- *Test:* P08 arm.
- *Gain/risk:* −1 to −5%; low risk.
- *Conflicts:* none.

**C3 · Fisher-preconditioned scale updates.** Replace Adam's second moment with a fixed diagonal Fisher per scale (e.g. svh_i: Σ_t (∂ℓ/∂y_i · y_i)²), estimated on the first batches; also try the F^{1/2} power.
- *Source:* ShamAN-Q 2609.38521 (tempering the Fisher with p = 1/2 beat p = 1); 2606.20469.
- *Fit:* optimizer.
- *Test:* P08: Adam vs F⁻¹ SGD vs F^{−1/2} SGD.
- *Gain/risk:* overlaps C2; low risk.
- *Conflicts:* none.

**C4 · Trust region plus validation-gated checkpoints** (must-have).
- L2 on the log-scale deltas, as ThinkBit's w_r r².
- Keep the best fit-pool-slice checkpoint; stop after 2 rising evaluations.
- *Source:* ThinkBit; G²PTQ 2609.31009 (without a trust region KL went 0.333 → 25.3); 2609.04526 (lr 2e-4 collapsed competent bases).
- *Fit:* every run.
- *Test:* part of every pilot.
- *Gain/risk:* protects the others; none.
- *Conflicts:* none.

**C5 · Freeze-depth sweep.** Train layers ≥ k for k ∈ {0, 8, 16, 32, 48}; it is cheaper than C1.
- *Fit:* recipe choice.
- *Test:* P08 1 h; then 2–3 values on the 27B (2–3 h each).
- *Gain/risk:* decides the depth policy; none.
- *Conflicts:* none.

**C6 · Layer-type ablation.** Heal GDN layers only, full-attention only, and MLPs only.
- *Source:* Ternary Mamba 2606.18114 (post-hoc output corrections failed for SSMs, because error accumulates through the recurrence).
- *Why:* if GDN-internal scales heal poorly, healing effort (and possibly bits, I2) should move.
- *Test:* P08, 3 arms, 1 h.
- *Gain/risk:* diagnostic; none.
- *Conflicts:* none.

**C7 · Mixed-precision backward.** Run the dX GEMMs in fp32 (or x2) for layers 0–31 and bf16 above, if A3 finds early-layer noise. Cost ≈+40% per step.
- *Source:* #33-03, #33-14.
- *Test:* P08 arm, conditional on A3.
- *Gain/risk:* may be the whole fix for A2; cost as stated.
- *Conflicts:* none.

**C8 · Truncated backward for long windows.** If C5 says layers below k need no healing, stop the backward at layer k: no checkpoints below k, half the backward cost, and about 2× the window on one card (32k at 2.5 bpw).
- *Source:* collect.py already truncates its backward at layer 22 (gate 2c, checked exact).
- *Fit:* infrastructure.
- *Test:* memory check on S27.
- *Gain/risk:* makes E1/E2 cheap; none.
- *Conflicts:* none.

**C9 · EMA / Polyak averaging of scales.** Keep an exponential moving average of the scales during training and ship the average.
- *Fit:* arm in every run.
- *Gain/risk:* 0 to −1%; none.
- *Conflicts:* none.

**C10 · Separate LRs per family.** One LR each for svh, suh, norm gains and conv taps (a 3×3 grid on P08). A norm gain is shared by 3–4 readers, so it has more leverage per step.
- *Source:* SQuAT 2610.31412 (its gain and shape LRs differed by 1000×).
- *Gain/risk:* an enabler; none.
- *Conflicts:* none.

**C11 · Gauge fixing.** (c·svh, suh/c) within a tensor and (norm gain × c, all readers' suh/c) are exact symmetries. Penalize the log-mean along each orbit, or Adam drifts along flat directions and fp16 storage can over- or underflow.
- *Fit:* correctness item, ≈0 cost.
- *Gain/risk:* removes a failure mode; none.
- *Conflicts:* none.

---

## D. Losses

**D1 · Metric-exact differentiable KL.** Use the score's row formula, KL = A/S − log S: the top-512 teacher head, the tail cell (teacher tail mass vs student mass outside the 512), fp64 sums as in `score_rows`, gradient through the student log-softmax.
- *Source:* `common.score_rows`; NVIDIA QAD 2601.20088 (KL beats CE: KL to BF16 0.004 vs 0.311 for QAT).
- *Fit:* the loss for everything.
- *Test:* unit test against score.py; P08 arm vs plain top-512 KL.
- *Gain/risk:* 0 to −2%; none.
- *Conflicts:* none.

**D2 · Gen-row weighting.** Only gen rows are scored, but every row has a teacher target. Arms: gen only / all rows / gen 1 + prompt 0.3.
- *Fit:* loss setting.
- *Test:* P08, 3 arms.
- *Gain/risk:* 0 to −2%; none.
- *Conflicts:* none.

**D3 · Dynamic forward/reverse KL mix**, α_v = sg[p_v/(p_v+q_v)] on top-512 ∪ residual. The metric is forward KL; this tests whether the mix optimizes faster. Adopt only if *forward* KL on search improves.
- *Source:* OnQ 2610.29378 (fixed-response DKL 61.0 vs FKL 59.0 on MATH-500).
- *Test:* P08 arm.
- *Gain/risk:* 0 to −2%; objective mismatch.
- *Conflicts:* none.

**D4 · Tail-KL window selection (KIC).** Rank qad windows by the unhealed model's mean of the top-25% token KL and oversample the top half in epoch 1.
- *Source:* FidelityQ 2610.14934 (the worst decile of tokens carries 45% of KL; KIC vs random ≈2% PPL).
- *Test:* P08; on the 27B one scoring pass of the qad pool (≈30–40 min per card).
- *Gain/risk:* 0 to −2%; none.
- *Conflicts:* none.

**D5 · Robust per-token loss.** Huber/clip per-token KL above the batch p99, or normalize per document, so a few spiky tokens don't dominate the small parameter set. Adopt only if mean KL improves.
- *Source:* new; DRQ.
- *Test:* P08 arm.
- *Gain/risk:* 0 to −1%; none.
- *Conflicts:* none.

**D6 · Position-weighted loss** to match the corpus's token mass by position bucket. With ≤16k windows most trained tokens are early positions, while long traces hold many scored tokens.
- *Source:* quant-sota.md #18; 2609.04098 (the weight-quant NLL gap is +0.081 nats in the first 16K, +0.011 after, so late positions are easy but numerous).
- *Fit:* with E2.
- *Gain/risk:* 0 to −1%; none.
- *Conflicts:* none.

**D7 · Auxiliary block-output Fisher loss** at depths 16/32/48 (needs A10): ℓ_aux = Σ_d ‖Δh_d‖²_{F_d}, auto-balanced as sg(L_KL/L_aux). It gives early layers a short-path target.
- *Source:* 2606.05682 (auto-balancing, +0.5% step time); REAL-Q (block Fisher); LAQuant 2605.08755 (layer-wise 93.95% vs controlled E2E-KL 88.44% teacher-forced top-1).
- *Test:* P08 (teacher residuals are free on the 0.8B).
- *Gain/risk:* 0 to −3%; aux misaligned with KL (CKA-QAD has no KL evidence).
- *Conflicts:* none.

**D8 · Teacher-temperature warm-up.** Distill from p^{1/T} with T = 2 → 1 over the first 20% of steps: softened targets spread gradient over the tail rows, where trunk quantization leaks mass. The final phase stays on the metric (T = 1).
- *Source:* classic KD; new here.
- *Test:* P08 arm.
- *Gain/risk:* 0 to −1%; none.
- *Conflicts:* none.

**D9 · Entropy-stratified gain report** (diagnostic). Split healed gains by teacher-entropy bins (the golden stores entropy per row).
- *Source:* ReQAT 2606.15682 (FP4 damage on low-entropy tokens is tail-mass inflation); EGAD/CaRE-KD.
- *Why:* if gains concentrate in one stratum, a stratum-weighted loss becomes worth testing.
- *Cost:* minutes on any run.
- *Conflicts:* none.

---

## E. Data and windows

**E1 · Window-length sweep at equal tokens.** 1k / 4k / 16k on the 27B; 1k / 2k / 4k on P08.
- *Source:* QATFactory (32K vs 4K sequences: +1.9 avg accuracy at equal ~300M tokens); the field report was limited to 1k.
- *Fit:* recipe setting.
- *Test:* P08 30 min, then 3 runs on the 27B ≈8 h.
- *Gain/risk:* −1 to −4%, mostly on long documents; memory.
- *Conflicts:* none.

**E2 · TBPTT on long traces.** Process agentic traces in 8–16k chunks, carrying the GDN state and the full-attention KV forward detached, so trained positions reach 64k+.
- *Memory:* KV is ≈64 KB/token (16 layers × 4 KV heads × 256 dims × K,V × bf16), so 250k needs 16 GB. Cap at 64k on one card, or use the E3 pipeline.
- *Source:* new; fla state carry.
- *Fit:* recipe setting.
- *Gain/risk:* 0 to −2% on long-trace KL; memory.
- *Conflicts:* none.

**E3 · Two-card layer pipeline vs data parallel.** The pipeline doubles the window; DP doubles tokens per hour. Choose by E1/E2.
- *Fit:* infrastructure.
- *Gain/risk:* enabler; card 1's x4 link adds ≈25%, but only activations at one boundary cross it.
- *Conflicts:* #8.

**E4 · Data dose and epochs.** 0.5 / 1 / 2 / 3.7M tokens and 1–3 epochs on P08, plus 2 checkpoints of one 27B run; track the train-doc vs search KL gap.
- *Source:* RoleQ 2610.21808 (1.6M tokens was enough for scales at 8B); Tetra (19.5M); ThinkBit (268M).
- *Why:* decides whether E6 (more data) is worth building.
- *Test:* P08 ≈1 h.
- *Gain/risk:* diagnostic; none.
- *Conflicts:* none.

**E5 · Category-balanced sampling.** Match the corpus token mix by category (code / chat / agentic / math / multilingual) against file-order sampling.
- *Gain/risk:* 0 to −1%; also protects L2.
- *Conflicts:* none.

**E6 · Self-generated continuations with a surrogate teacher.** Radiance runs V2 9.0 bpw (KL to bf16 ≈3e-4, about 0.5% of the 2.5 bpw KL) to continue qad-pool prompts, and the recorder stores top-512.
- *Gate:* the challenge rules; never on search or verify docs.
- *Source:* NVIDIA QAD (self-generated data ≈ SFT data; random tokens worse); ROADMAP data preference (1).
- *Cost:* ≈10M tokens in 6–10 h of batched generation on one card (estimate) + ≈1 h recording.
- *Gain/risk:* −1 to −3%, only if E4 shows data-limitation; rules.
- *Conflicts:* none.

**E7 · Surrogate-teacher equivalence.** Heal P08 with targets from a K8 0.8B instead of bf16.
- *Why:* if the results are equal, the 27B can use 9.0 bpw logits wherever no golden exists (E6, context beyond the golden).
- *Test:* 30 min.
- *Conflicts:* none.

**E8 · xcorp dose (proxy only).** P08 healed on 3.7M qad tokens vs +30M xcorp tokens.
- *Why:* tells whether same-distribution data would help the 27B. xcorp never trains the 27B without a rule check.
- *Conflicts:* #61 (xcorp is proxy-only).

**E9 · qad-pool audit after the pools-v3 amendment.** 11 qad docs moved to verify, including 7 agentic and 4 long traces. Count the qad tokens by length and category before choosing windows; if few long tokens remain, A9 is a real risk.
- *Cost:* minutes of CPU.
- *Conflicts:* none.

**E10 · Data budget and ledger.** Write which pools may train. The qad pool yes. Fit-pool docs not used by any golden fit slice: only after #55's reuse-bias test is repeated on a healed model. Search and verify: never.
- Append every QAD-trained doc to `ledger.jsonl` with role "qad-trained", and exclude those docs from later slices.
- *Fit:* hygiene.
- *Conflicts:* #55, #72.

---

## F. Fixed codes vs code refresh

**F1 · Scale-conditioned re-encode (EM).** After healing, re-run LDLQ + trellis on each tensor with the healed suh/svh held fixed (so the codes are chosen for the new scales; no refit), then heal briefly. Variant: re-encode with H_out (bdiag).
- *Source:* PV-Tuning 2405.14852 (continuous P step 7.38 → 6.69, adding the discrete V step 5.90 at 2 bit); JARQ 2609.38599 (alternate scale fit and rounding, monotone).
- *Fit:* per mix; 27B ≈35 min requant + 1 h heal.
- *Test:* P08: heal → re-encode → heal against heal ×2 at equal compute, 3 seeds, PR(3%).
- *Gain/risk:* −1 to −4%; the re-encode in H_in may undo the KL-driven changes.
- *Conflicts:* none.

**F2 · PV-lite V-step with KL-gradient targets.** Accumulate G_W per tensor, form the target W* = W_q − η·D_out⁻¹ G D_in⁻¹, and re-Viterbi only the τ% of 16×16 tiles whose target moved most; accept by fit-pool KL.
- *Source:* PV-Tuning; quant-sota O7; TVQ 2610.41822.
- *Cost:* the dW GEMMs make the backward ≈+50%.
- *Test:* P08 first (a 0.8B requant takes 1 min).
- *Gain/risk:* −2 to −8% at 2.5 bpw, largest at low K; high cost.
- *Conflicts:* none.

**F3 · Block-coordinate Viterbi refresh (no training).** After LDLQ, re-solve one tile row at a time with all others fixed, in the exact H_in objective (residual of the whole layer), for 1–3 sweeps. This is "healing at rounding time" and fits the per-level library.
- *Source:* MPQ 2610.32127 (LDLQ + CD3: Llama-3-8B E8P12 2-bit PPL 15.01 → 12.76; ~0.08 at 4 bit).
- *Fit:* per level.
- *Test:* P08 K2/K3, 3 seeds, PR(4%).
- *Gain/risk:* −1 to −4% at K2–3; tail-biting and LDLQ coupling make "others fixed" approximate.
- *Conflicts:* #57 (exllamav3's tail-biting heuristic is near-optimal *per tile*); CD optimizes across tiles.

**F4 · STE latent healing of the last 2–4 layers (TVQ-lite).** Keep a bf16 latent in the rotated domain, forward through the trellis re-encode of the latent (identity STE), refresh the Viterbi every 16–32 steps.
- *Memory:* 4 layers ≈15 GB with Adam; use Adafactor or 2 layers.
- *Source:* TVQ (≤0.2 PPL at 8B, none on Llama2-70B, ≈8× QTIP compute); E8-QAT 2610.27699 (cell-centre init, LR ∝ 1/weight scale, +24 LCB over PTQ at 2 bit).
- *Fit:* per mix.
- *Test:* only if F9 says codes can move; P08 last layer.
- *Gain/risk:* −1 to −4%; high cost.
- *Conflicts:* none.

**F5 · Asymmetric (GPTAQ) requant of late layers inside the shipped mix.** Collect H_in for layers 48–63 on the quantized-and-healed mix's own inputs, with bf16 targets via the cross-covariance X_q X_fpᵀ; requant those 16 layers (≈10 min) and re-heal.
- *Source:* GPTAQ 2504.02692; ROADMAP "sequential Hessians"; 2610.07522 (sequential beats parallel).
- *Fit:* per mix.
- *Test:* P08 (quantized-prefix Hessians cost minutes).
- *Gain/risk:* −1 to −4%; our own collections are much smaller than the 10M-token Hessians, and #50 shows ≈9% in-sample fitting at 1.2M tokens.
- *Conflicts:* none.

**F6 · Activation-denoising LDLQ target per level.** Quantize the pretransformed W·F under a metric G built from synthetic upstream noise (one extra noisy forward, γ = 0.05; small λ and ρ swept on search). It is library-compatible because the noise is generic, not mix-specific.
- *Source:* 2610.07522 (closes 18–31% of the parallel→sequential gap; with QuIP# at W2 it beat sequential on 8B/13B).
- *Cost:* C and N are d×d per linear (down_proj 1.2 GB); needs a noisy bf16 forward of the 27B (collector, a few hours).
- *Test:* P08 first.
- *Gain/risk:* −1 to −3%; risk sits with #33-07's precedent.
- *Conflicts:* #33-07 (gradient-weighted H_in was +31–44% worse). This changes H differently: noise-augmented covariance with partial denoising and small, tuned λ.

**F7 · G²PTQ first-order term at the deployed point.** Requant the late layers of a healed mix with the KL gradient term g H⁻¹ in the LDLQ update, with trust-region β.
- *Source:* 2609.31009 (LLaMA3-8B W3 KL 0.169 → 0.122; exact gradient 0.497 vs approximate 0.710; collapse without the trust region).
- *Fit:* per mix.
- *Test:* P08.
- *Gain/risk:* −1 to −3%; β tuning.
- *Conflicts:* none.

**F8 · Discrete ±1-level code proposals (Ising healing).** Propose moves by KL gradient plus a Kronecker energy and accept them on held-out KL.
- *Source:* TWIG 2610.31931. Its own ablation had scale tuning (KL 0.325) beating discrete healing (0.371).
- *Fit:* only after scales; expected ≤1%.
- *Conflicts:* none.

**F9 · Drift-budget diagnostic.** Per tensor at K2/K3/K4, compare the trellis Voronoi radius in the rotated domain with the expected latent drift lr·N·g_max, and estimate the fraction of codes that would flip.
- *Source:* BCJR-QAT 2605.10655 (codes never change below the radius; per-layer MSE training *raised* PPL 9.09 → 10.44 despite −17–20% MSE).
- *Purpose:* decides F2/F4 before any GPU time.
- *Cost:* CPU, an hour.
- *Conflicts:* none.

---

## G. Forward-only, closed-form and block-wise healing (much of it needs no S27)

**G1 · H_out-weighted `refit_scales`.** refit_scales ignores H_out. The per-output-column scale is unaffected by a diagonal H_out (columns decouple), but the input-side solve should be ((Q D Qᵀ) ∘ H) r = rowsum(Q D ∘ HW) with D = diag(H_out); with a full H_out the output side couples too.
- *Source:* own reading of quantize.py:1583; YAQA.
- *Fit:* per level; seconds per tensor; ships with the 27B bdiag swap test.
- *Test:* P08 bdiag vs bdiag + weighted refit. The codes are identical, so the comparison is free of rounding noise: train-free and paired. Adopt if ≤0.99 at both K across 3 seeds.
- *Gain/risk:* −0.5 to −2% on top of bdiag; none.
- *Conflicts:* #33-06 changed the *pre-quantization* output scales (which rows get precision); this only changes the post-hoc input-scale metric.

**G2 · Refit in the undamped H, with more rounds.** Fits currently use the damped H (0.025 × mean diag, a large share of the median channel's curvature under massive activations), although the reported error uses the undamped one; the scale fit has no stability reason to be damped. Go from 2 to 6 rounds.
- *Fit:* per level, zero cost.
- *Test:* same-codes paired, P08 → 27B.
- *Gain/risk:* 0 to −1%; none.
- *Conflicts:* #38 (damping sweep) concerns LDLQ's damping, not the refit's.

**G3 · Asymmetric sequential scale refit (GPTAQ for scales).** Stream the mix layer by layer. For each linear, inputs X_q come from the already-quantized-and-refit prefix and the targets are Y* = W X_fp from the bf16 stream. Per output row s_i = ⟨w_q,i X_q, w_i X_fp⟩ / ‖w_q,i X_q‖²; input scales by the analogous ridge solve; H_out-weighted per H4.
- *Source:* QuantForge 2609.34680 (sequential closed-form recovery after quantization: Fit-3 damage 0.0650 → 0.0326); GPTAQ; 2609.11716 (blocks oppose inherited error: 50% of Σ‖Δu‖² cancelled on Qwen3-32B).
- *Fit:* per mix.
- *Cost, 27B:* collector bf16 stream + decoded fp16 student layer by layer, ≈256k tokens, activations to NVMe (10 KB/token per stream), ≈2–4 h on one card; host RAM (30 GB) is the binding limit.
- *Test:* P08 first; per layer, accept only if a held-out token subset agrees.
- *Gain/risk:* −1 to −4%; least squares on layer outputs is not KL (DRQ: lower reconstruction loss raised PPL in 47% of GPTQ layers).
- *Conflicts:* none.

**G4 · Nonlinearity-aware MLP fit.** Fit the gate and up svh *jointly through SiLU*: ridge on m* = SiLU(g*) ⊙ u* rather than matching each linear's output separately. The optional gate shift needs B10.
- *Source:* QuantForge HiRes (MLP-only recovery 0.0650 → 0.0391; fitting after the attention fix matters, stale MLP stats 0.0417).
- *Fit:* per mix, inside the G3 stream.
- *Test:* P08.
- *Gain/risk:* −0.5 to −2% over G3; none.
- *Conflicts:* none.

**G5 · Block-wise scale healing with teacher inputs (library-compatible).** For layer l, the input is the bf16 teacher's layer input and the target the teacher's output. Train only that layer's suh/svh and norms to minimize ‖Δh‖²_{F_l} (block Fisher from A10), using normalized RMSE across blocks and a 1-layer lookahead.
- *Why library-compatible:* the inputs are the teacher's, so the result does not depend on the rest of the mix; healed scales can be stored per (tensor, K) and priced by the knapsack.
- *Source:* EfficientQAT Block-AP (scales only here); REAL-Q; 2610.00983 (RMSE: Qwen3.5-27B W2 PPL 37.5 → 22.2 against starvation of early blocks); LAQuant.
- *Fit:* per level.
- *Cost, 27B:* teacher-activation cache ≈42 GB of disk at 64k tokens; ≈1–2 h per level on one card.
- *Test:* P08: block-healed vs E2E-healed vs none.
- *Gain/risk:* −1 to −4%; block loss ≠ KL.
- *Conflicts:* none.

**G6 · Two-stage: block-wise with quantized-prefix inputs, then a short E2E KL pass.**
- *Source:* JustQuant (5–10% of the data suffices for the local stage; geometric step allocation γ = 2); ThinkBit (GSQ → recalibration); OnlineQAT (stage 2 is only 40–150 steps).
- *Fit:* per mix; halves E2E time and may stabilize early layers.
- *Test:* P08, two-stage vs E2E only at equal wall time.
- *Gain/risk:* the same ceiling as B1, cheaper; none.
- *Conflicts:* none.

**G7 · Teacher Intervention, lite.** During early E2E healing, replace the student residual at depths 16/32/48 with the cached teacher residual (A10) with probability p annealed 1 → 0. Later layers learn on clean inputs first.
- *Source:* 2302.11812 (parked for V5 in design/v3.md §7).
- *Test:* P08 (teacher residuals are free).
- *Gain/risk:* 0 to −3%; only cached windows can use it.
- *Conflicts:* none.

**G8 · Forward-only line searches by container edit (27B, no S27).** Golden-section search on a fit-pool slice, with radiance evals of ≈2 min each, over: head τ (A4), per-(family × depth band) gains (B7), per-layer writer scalars (B8), norm-gain scalars.
- *Cost:* ≈30 groups × 4 evals ≈ 4 h on one card.
- *Source:* Q-Strata (measured-loss descent); EvoPress.
- *Test:* use a large slice; effects of 0.2–1% are near the eval noise, so only global knobs are worth it.
- *Gain/risk:* −0.5 to −2%; eval noise.
- *Conflicts:* #14 (single-tensor swaps are too noisy). These are groups, on large slices.

---

## H. Interplay with H_out rounding

**H1 · Does healing absorb the H_out gain?** On P08, heal base and bdiag with the same recipe (B1 + B2), 3 seeds × K3/K2.
- (a) Gains add: keep both.
- (b) Healing erases bdiag's advantage: H_out matters only for unhealed sizes, and healing takes priority.
- *Test:* ≈2 GPU-h. Call it "additive" if healed-bdiag/healed-base ≤0.96 in every seed.
- *Fit:* decision gate.
- *Conflicts:* #33-12, #33-13.

**H2 · H_out as a by-product of healing.** From the healing backward, accumulate per output row r_i = Σ_t dy_{t,i}² ‖x_t‖² (no GEMM), and optionally per-sequence Sketch-B rows (one dW GEMM). Quant with these diagonals and compare against the fp32 Sketch-B diagonal.
- *Why:* if they are equivalent, H_out can be re-collected on the deployed or healed model for free: iterative YAQA.
- *Note:* these are gradients of the true KL at the quantized point, not the sampled-label Fisher at bf16; test which works better.
- *Source:* yaqa08.py's first collector (the ‖x‖² dy dyᵀ sum, stopped in amendment 2); #33-09 (the diagonal is precise: halves agree ≥0.997).
- *Test:* P08 1–2 h; equivalence if KL is within ±2% of the Sketch-B-diagonal quants across 3 seeds.
- *Fit:* per level / per mix.
- *Conflicts:* none.

**H3 · On-model H_out requant of late layers.** After healing a shipped mix, collect the diagonal H_out at the deployed model (H2's method), requant layers 48–63 with it, and re-heal. Sensitivities around bf16 are not the sensitivities at 2.5 bpw.
- *Source:* GEMQ 2605.23078 (importance re-estimated on the 2-bit + FT model: PPL 9.01 vs 9.66 when estimated at FP); MixQuant 2607.23047; ShamAN-Q (fresh statistics).
- *Fit:* per mix; ≈10 min requant + heal.
- *Gain/risk:* −1 to −3%; none.
- *Conflicts:* none.

**H4 · H_out as row weights in output-matching heals.** Use the 27B diagonal (already being collected) as Σ_i h_i‖Δy_i‖² in G3/G5/G6.
- *Fit:* improves alignment of the forward-only heals with KL.
- *Test:* P08 inside G3.
- *Gain/risk:* −0.5 to −1.5% over unweighted; none.
- *Conflicts:* none.

**H5 · Ordering test.** Compare on P08, at equal compute:
- (A) H_out-round, then heal;
- (B) heal base, then H_out-weighted re-encode (F1), then heal.
- *Gain/risk:* decides the final-pass ordering.
- *Conflicts:* none.

---

## I. Interplay with the knapsack, library and search

**I1 · Per-level healing and transfer to mixes.** On P08:
- heal the uniform K2 trunk and the uniform K3 trunk;
- compose a 2.5-equivalent mix (alternating layers or depth bands K2/K3) using the per-level healed scales;
- compare with (i) the unhealed mix, (ii) the mix healed directly, (iii) the mix warm-started from the per-level scales.
- *Rule:* "transfers" if the per-level scales alone keep ≥70% of (ii)'s gain.
- *Why:* if they transfer, healed levels become library options and the knapsack prices healed costs; if not, healing is a per-mix final pass only.
- *Test:* ≈2 GPU-h. The 27B version: heal the K2.5 and K3 trunks (2 × 3 h), then compose.
- *Conflicts:* none.

**I2 · Post-heal sensitivity by depth band.** On a healed K3 0.8B, measure single-band K3 → K2 swaps (with and without a short re-heal) and compare with unhealed band sensitivities.
- *Why:* if late bands get cheaper after healing (the field report hints late layers heal best), the healed optimum moves bits early.
- *Test:* P08 ≈2 h; 27B with 4 band swaps + short re-heals ≈6 h.
- *Gain/risk:* −1 to −4% from reallocation if bands shift; otherwise it confirms "heal the knapsack's pick".
- *Conflicts:* #71 (targeted swaps) — same method, now on a healed base.

**I3 · Healability factors in kfit2.** Fit h(family, K, depth band) = healed cost / unhealed cost from I1/I2, multiply the option costs, re-solve knap4 for the shipped sizes, heal the new picks, and compare with the healed old picks on search.
- *Fit:* knapsack model change; CPU + 2 healings.
- *Gain/risk:* 0 to −3%; none.
- *Conflicts:* #13 (family factors), #18 (generic surrogates) — this is a structured factor, not a black-box model.

**I4 · Heal the finalists, not only the winner.** Per shipped size, heal the top 3 stream candidates, choose by healed search-slice KL, confirm on ver4.
- *Source:* 2610.05504 (advantages can shrink, grow or reverse after recovery).
- *Why:* unhealed differences of 2–3% are near noise, and the healed ranking may differ.
- *Cost:* ≈1 night per size on both cards.
- *Gain/risk:* 0 to −3%; none.
- *Conflicts:* #72 (final picks) — same discipline, one step later.

**I5 · Warm-start per-mix healing from per-level scales.** If I1 transfers ≥50%, per-mix healing becomes a short refinement (≈20% of the steps), which makes I4 affordable for 5–10 finalists.
- *Fit:* recipe.
- *Conflicts:* none.

**I6 · Supernet / multi-mix joint healing.** Keep one set of scales per (tensor, K), trained across mixes sampled around the shipped sizes (knapsack pick + random K perturbations). Card 0 and card 1 run different mixes; the gradients of shared (tensor, K) scales are all-reduced; switch mixes every ≈50 steps to amortize weight loads.
- *Why:* scales robust to context become library options.
- *Source:* new; RCO 2605.00649 (relaxation over precomputed per-option tensors), here learning only scales.
- *Fit:* per level.
- *Test:* P08 first, with 4 mixes.
- *Gain/risk:* −3 to −8% applied to every size with no per-mix healing, if I1 transfers only partially; the scales compromise across contexts.
- *Conflicts:* none.

**I7 · Healing-gradient features for local search.** At a healed mix, the per-tensor norm of the KL gradient on svh (recoverable error that remains) is a candidate feature for upgrade/downgrade proposals.
- *Fit:* by-product of healing.
- *Test:* validate on the search slice.
- *Gain/risk:* 0 to −2%; none.
- *Conflicts:* #20 (the residual re-ranker was null) — this is a new signal, not more summary features.

**I8 · Heal in order of absolute KL.** Order: 2.0 and sub-2 first (KL 0.164; the QAT benefit grows as bits fall, 2606.09012; QTIP: fine-tuning had no effect at 3–4 bit), then 2.5, then 3.0; skip ≥4.0 unless it is nearly free. Report each healed size as an "equivalent unhealed bpw".
- *Fit:* planning; ties to ROADMAP V3.5 (sub-2 quants are the natural QAD targets).
- *Conflicts:* none.

---

## J. Interplay with V3 speed and int8 activations

**J1 · A8-aware healing.** Train with A8-g128 fake-quant (STE) on the inputs of every tensor the V3 solver runs at A8 (suh → Hadamard → int8 per token × 128), so scales absorb activation error too. Score KL at equal `gemm_us`.
- *Source:* E1b (A8-g128 +0.00005 at K6).
- *Fit:* per mix.
- *Gain/risk:* small KL at 2.5 bpw, but the solver may put more tensors on A8 (speed); none.
- *Conflicts:* #26 (activation costs).

**J2 · Tie suh across shared-input siblings, then heal.** gate/up, q/k/v and GDN qkv/z have different suh (gate vs up median ratio 1.015, q vs k/v 1.092). Tying them lets one rotate + int8 prep pass serve all siblings (design §1b.4); healing svh and the shared suh recovers the cost.
- *Source:* #42.
- *Fit:* per level (requant with tied suh) or per mix (geometric-mean tie + heal).
- *Test:* P08; PR: the tie cost after healing is ≤1% KL. Then the time table prices the speed gain.
- *Gain/risk:* a few % of prefill speed at ≈0 KL; none.
- *Conflicts:* #42.

**J3 · Healing toward per-token A8.** Per-token is faster than g128 (iu8 ≈323 vs 265 TOPS). Train candidate tensors with per-token fake-quant; if healed per-token ≈ unhealed g128, the time table gains a faster option.
- *Fit:* per mix; a speed option.
- *Conflicts:* #26-01.

**J4 · Output-range guard.** After healing, record max |y| of down, ssm_out and attn_out over the long verify traces. f16 tiles overflow at 65,504 (design §1c.9), and healing can raise scales.
- *Fit:* gate.

**J5 · Both kernel paths.** A8 triggers on total M ≥ 512 and f16 below it. The healed model must be checked on both paths, since healing on one could hurt the other (A7 runs both).
- *Fit:* gate.

---

## K. Head and embedding

**K1 · Head healing on dumped head inputs (27B, no S27).**
- **Method:** dump head inputs from a 2.5 mix's *quantized* trunk on ≈1M qad-pool tokens (E1b hook `EXL3_AQ_DUMP_ONLY`; ≈10 GB). Train the head's row scales (svh), its input-column scales (suh, which is exactly the final-norm gain), and optionally τ, against the **golden** targets.
- **Why it differs from headsim:** headsim isolates head error against the bf16 head. Training against the golden makes the head compensate trunk error too.
- *Source:* SoftWater 2608.12026 (KL-aware per-row head treatment: 6.5–8.3× lower head KL at 2 bits); 2609.31291.
- *Cost:* decoded head 2.5 GB; ≈7.6 PFLOP per 1M tokens ≈ minutes on one card.
- *Fit:* per mix.
- *Test:* PR27(1%) on the whole model.
- *Gain/risk:* −0.5 to −2% total; overfitting rare rows (handled by K4's freeze).
- *Conflicts:* #34 (mean-row −30% and the slab head) — complementary.

**K2 · Logit bias** (format change: 248k fp16 = 0.5 MB, 0.00015 bpw). Learned on K1's dumps, init 0, L2-regularized, only rows seen in the data. It corrects frequency-level shifts and the tail leakage of rare logits (design §2).
- *Fit:* per mix; a plugin bias add on the logits.
- *Gain/risk:* 0 to −1%; none.
- *Conflicts:* quant-sota #14.

**K3 · Embedding row-scale healing.** A by-product of S27 runs (the gradient reaches the embedding). Only rows seen in the qad pool get gradient; 51% of ids never occur in calib.
- *Gain/risk:* 0 to −0.5%; low priority.
- *Conflicts:* #35 (an MSE scale search was null; this objective is KL in context, but expect little).

**K4 · Re-price the slab head after trunk healing, and guard rare rows.**
- Re-dump the head inputs from the healed trunk and re-run headsim pricing; the head's share of KL may change.
- In any head/embedding healing, freeze rows with fewer than N qad-pool occurrences.
- *Cost:* ≈1 h.
- *Gain/risk:* 0 to −1%; none.
- *Conflicts:* #34, #32 (rare-token rows matter more than frequency predicts).

---

## L. Evaluation hygiene and guards

- **L1 · A HEAL.md with pass rules fixed before data.** Same-quant comparisons need training-seed replicates (A8); cross-recipe generalization needs 3 quant seeds (#33-15/16). The 0.8B K3 is the stand-in for the 27B at 2.5 bpw, and K2 is a stress test.
- **L2 · Multilingual guard.** Score golden-ml (67 fit-pool multilingual docs) for every healed candidate, since the qad pool is ≈94% English and code; adopt only if the ml ratio is ≤1.0 (#32's lesson).
- **L3 · Long-context guard.** Run ver4-long and ver3-long on finalists only, with A9 for diagnosis.
- **L4 · Overfit monitor.** Plot KL on trained qad windows against the search slice every N steps; stop when the gap grows.
- **L5 · Provenance.** Healed containers differ only in the scale and norm planes: record the sha256 and a plane-diff map, and add a selftest that "only these planes changed".
- **L6 · Reproducible healed scales.** Store healed scales as separate files keyed by (mix id, tensor, K) so the release can be rebuilt byte-identically (V2 practice), and record the training data hash.

---

## Ranked top 20 (expected gain per unit cost)

| # | Idea | Cost | Expected gain at 2.5 bpw | Why here |
|---|---|---|---|---|
| 1 | **G1** H_out-weighted refit | ≈0 (code change; same-codes test) | −0.5…−2% | free; rides on the 27B bdiag swap |
| 2 | **G2** undamped refit, more rounds | ≈0 | 0…−1% | free; same-codes test |
| 3 | **A10** piggyback on the 27B H_out run | +5–10% of one run | enables G5/G7/D7 | time-critical |
| 4 | **K1** head healing on dumped inputs (+τ) | hours, no S27 | −0.5…−2% | cheapest 27B healing |
| 5 | **A1** oracle headroom | 1 GPU-h (4070) | decides everything | gate |
| 6 | **B1 + B2 + C4 + D1** core scale/norm KL healing | S27 build + 3 h/run | −3…−8% (median −5%) | the main lever |
| 7 | **C1/C2/C5/C7** depth-pathology fixes (after A2/A3) | P08 hours | could double #6 | field report 6–13% → 22% |
| 8 | **G3** asymmetric sequential refit | 2–4 h one card, no S27 | −1…−4% | first 27B number before S27 |
| 9 | **I1** level → mix transfer | 2 GPU-h (4070) | decides library vs per-mix | architecture call |
| 10 | **H1** heal(base) vs heal(bdiag) | 2 GPU-h (4070) | decides interplay | protects the H_out investment |
| 11 | **B4** conv1d taps | P08 arm | −0.5…−2% | free capacity in GDN |
| 12 | **B6** a/b full weights | P08 arm | −0.5…−3% | free capacity, bf16 already |
| 13 | **I4** heal the finalists | 1 night per size | 0…−3% | selection after healing |
| 14 | **E1** window sweep (+C8) | P08 + 8 h 27B | −1…−4% (long docs) | field report capped at 1k |
| 15 | **F1** scale-conditioned re-encode | 35 min + 1 h per mix | −1…−4% | cheapest code refresh |
| 16 | **B9** residual-writer biases | small kernel change, 0.0004 bpw | −0.5…−2% | closed-form init |
| 17 | **F3** CD Viterbi refresh per level | 2–3× quant time | −1…−4% at K2–3 | PTQ-only, library-friendly |
| 18 | **J2** tied suh + heal | P08 + time table | speed at ≈0 KL | V3 prefill |
| 19 | **G6** two-stage block → E2E | P08 | same ceiling, ≈2× cheaper | makes many healings affordable |
| 20 | **D4 + E4** tail-KL selection and dose | P08 1 h | 0…−2%, plus a data decision | decides E6 |

Honest aggregate: a healed 2.5 bpw mix at −5 to −12% KL compared with the same mix unhealed, if A1's ceiling is not small. The field report's 22% was probably on less-polished quants (unknown recipe) and measured on a proxy metric. Code refresh (F1/F2) might add a further −2 to −6%, with high uncertainty.

## The 5 cheapest diagnostics that would most change this ranking

1. **A1 oracle headroom** (≈1 GPU-h, 4070). If the ceiling is ≤5% at K3, gradient healing falls behind G1/G2/G3/K1 and S27 is deferred.
2. **A2 depth-pattern replication** (≈2 GPU-h). If "last quarter > all" reproduces, C1/C2/C7 move to the top; if not, the field report's pattern is model- or recipe-specific.
3. **A3 bf16 vs fp32 healing gradients** (20 min). Low early-layer cosine explains the field report directly (#33-03) and makes C7 the fix.
4. **A4 head-τ probe on the 27B** (4 evals, ≈20 min). This sizes K1/K2 and logit miscalibration at the real model with no new code.
5. **I1 level → mix transfer** (≈2 GPU-h). This decides whether healing lives in the library (knapsack prices healed options, I6) or is a per-mix final pass (I4/I5).

H1 (does healing absorb H_out's gain, ≈2 GPU-h) would be the sixth.
