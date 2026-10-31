# Weight-only PTQ: state of the art (late 2026) and what to try next for Qwen3.8-27B / EXL3

*Research brief. Scope: weight-only PTQ and light healing, measured by KL or perplexity. Date: 2026-10-29.*

**Labels used throughout**
- **[Lit]**: the source states it directly.
- **[Interp]**: my reading of a source.
- **[Own]**: my own speculation or reasoning, with no paper behind it.

**Comparability warning.** Almost every published number is on Llama-2/3 with WikiText-2 perplexity, at uniform bits and without embedding or head accounting. The only sources measured on *our* model and *our* metric are the two IQR whitepapers and the GDN NVFP4 paper. Treat cross-model "×% better" claims as directional only.

---

## 0. Executive summary

1. **The strongest competitor is IQR-V2 (Deadcode).** It is GPTQ with a trellis run down each column (exact error feedback), trained int8 trellis tables, E4M3 per-(row,128) group scales, measured band-KL allocation, and a rotated trellis embedding.
   - It uses **no fine-tuning**.
   - After size adjustment we are about **14% worse at 3.1 bpw**, **about 7% worse at 4.1**, **at parity at 5.1** and **about 8% better at 6.0**. At 2.1 bpw we are probably about 25% worse, but that rests on extrapolating their curve. Details in §5.
   - So the gap is a **low-bit** gap. That points at things that matter more at low rate: allocation non-additivity, head/embedding bit spending, gain adaptation (group scales), and error-feedback exactness. It does not point at the trellis code itself, because EXL3's L=16 code is closer to the rate-distortion bound than IQR's L=12 table.
2. **Literature SOTA for one-shot weight PTQ** is trellis (QTIP) + **YAQA** (KL-Hessian, two-sided LDLQ).
   - YAQA reports about **1/3 lower KL** than LDLQ without fine-tuning, and about **10–20% lower** after QuIP#-style recovery fine-tuning (Llama 3.x).
   - Nothing published since beats QTIP+YAQA on KL at equal bpw that I could find.
   - PV-tuning is the strongest *fine-tuning* method, but it was shown only at 1–2.3 bpw.
3. **Bit allocation.** The additive knapsack is still standard practice (HIGGS, Q-Palette, exllamav3's own optimizer, IQR). Additivity is known to break **below about 3 bpw** (HIGGS, EvoPress, IMPQ). The fixes are measured-KL local search (EvoPress) and pairwise interaction terms (IMPQ).
4. **Hybrid GDN layers.** A Sep-2026 paper on **this exact model** (Qwen3.8-27B) finds the following for weight quantization:
   - GDN state error does **not** accumulate with context.
   - The `a`/`b` gates are the *least* sensitive projections.
   - The quantization NLL gap **shrinks** with position.

   Nothing special is needed beyond keeping `conv1d`, `A_log`, `dt_bias` and the norms in bf16.
5. **Top recommendation (own, with Fisher math and indirect literature support).** The LM head's output-side Fisher is about `diag(E[p_v])·E[hhᵀ]`, so almost all of the 248k head rows are nearly free to quantize hard. Embedding rows matter only in proportion to token frequency. **Tier the head and the embedding by vocab probability or frequency.** That frees an estimated 0.05–0.15 bpw for the trunk at the ≤4 bpw tiers.
6. **Outdated in our current setup:**
   - The **Lloyd-Max embedding**. Rotated-trellis embeddings now exist (exllamav3 `-eb`, IQR-V2 at +0.00006 KL at 4 bits), and Lloyd-Max sits about 2.3 dB further from the rate-distortion bound.
   - **One-sided LDLQ**, superseded by YAQA. This is already planned.

---

## 1. Q1: strongest PTQ methods 2024–2026

| Method | What it is | Best evidence (model/metric) | vs trellis+LDLQ at equal bpw | Cost |
|---|---|---|---|---|
| **QTIP** ([arXiv 2406.11235](https://arxiv.org/abs/2406.11235)) | Bitshift trellis, L=16, 16×16 tiles, BlockLDLQ with feedback only *between* 16-column blocks | Llama-2-70B W2 ppl at 2/3/4 bit (fine-tuned): **3.70/3.26/3.16**, vs QuIP# 3.91/3.35/3.18 and AQLM 3.83/3.36/3.19 [Lit, Table 5] | This *is* the baseline (EXL3 is based on it) | Cheap |
| **YAQA** ([arXiv 2505.22988](https://arxiv.org/abs/2505.22988)) | Kronecker-factored Hessian of end-to-end KL (H_O ⊗ H_I) with two-sided LDLQ | "Reduces KL… by a factor of ≈1/3 across all models and bitrates" (Llama 3.1 8B/70B, 3.2 1B/3B, QTIP 2/3/4 bit, no FT). With recovery FT the gain is still "≈10–20%". Gemma-3-12B INT4: KL **0.056** vs Google QAT **0.089** [Lit] | **Yes**, about 33% (no FT) or 10–20% (with FT). Uniform bits, Llama | Sketch A ≈ **30 GPU-h** (~10B, 20M tok); B ≈ **50 GPU-h**. FP32 Hessians, about 2× LDLQ Hessian storage [Lit] |
| **GuidedQuant** ([arXiv 2505.07004](https://arxiv.org/abs/2505.07004), [code](https://github.com/snu-mllab/GuidedQuant)) | Layer objective weighted by end-loss output gradients (block-diagonal Fisher, output channels averaged in g groups); plugs into QTIP | "Consistently outperforms all VQ baselines" on Llama-2 with QTIP. I could not extract the numbers [Lit/partial] | Yes, but it is a cheaper, cruder version of YAQA's H_O [Interp] | One backward pass; ≤1.5× LDLQ time at g=2–4 [Lit] |
| **Q-Palette** ([arXiv 2509.20214](https://arxiv.org/abs/2509.20214), [code](https://github.com/snu-mllab/Q-Palette)) | Fractional-bit TCQ (1.5–5.0 in 0.5 steps; "half-TCQ" row-splits for .25/.75) plus linearity-theorem knapsack | Llama-2-7B 2 bit, data-aware: **6.47** vs QTIP 6.84. 3 bit: 5.34 vs 5.39 [Lit, Table 4] | Yes at 2 bit (+mixed allocation); about equal at 3 bit. Gain is mostly from allocation granularity [Interp] | Cheap; ILP in 60 s |
| **PV-Tuning** ([arXiv 2405.14852](https://arxiv.org/abs/2405.14852)) | Fine-tuning that also updates *discrete codes* (V-step), KL to teacher | Llama-2-7B AQLM 2.01b: 7.38 (calibration only) → 6.69 (continuous-only FT) → **5.90** (PV+STE). Llama-3-8B 2.0b: 6.99 [Lit] | Llama-2-7B 2b: PV 5.84 vs QTIP-FT 5.86, i.e. equal. YAQA authors say LDLQ+QTIP already beats PV-tuned AQLM on Llama 3.1 [Lit] | 2 days on 8 GPUs (up to 8 to converge), 1B-token RedPajama [Lit] |
| **AQLM / QuIP# / GPTVQ** ([2401.06118](https://arxiv.org/abs/2401.06118), [2402.04396](https://arxiv.org/abs/2402.04396), [2402.15319](https://arxiv.org/abs/2402.15319)) | Additive / E8 lattice / GPTQ-VQ | Dominated by QTIP in the table above [Lit] | No | — |
| **EfficientQAT** ([arXiv 2407.11062](https://arxiv.org/abs/2407.11062)) | Block-wise train all parameters (Block-AP), then end-to-end train quantization parameters only (E2E-QP) | Uniform scalar grids, not trellis [Lit] | Not directly comparable. The E2E-QP part ≈ our planned scale-only QAD [Interp] | Moderate |
| **GPTAQ** ([ICML'25](https://proceedings.mlr.press/v267/li25dn.html)) | GPTQ with "asymmetric calibration": match FP outputs given quantized inputs; about 20 extra lines of code | Improves GPTQ at low bit; numbers not extracted [Lit] | Unknown vs trellis | Cheap |
| **D²Quant** ([arXiv 2602.02546](https://arxiv.org/abs/2602.02546)) | Dual-scale quantizer for down_proj plus **mean-shift bias correction** at post-attention LN | Qwen3-8B 2 bit: W2 14.72 → 14.10. Bias correction alone: 14.72 → 14.63 [Lit] | Baselines are GPTQ variants, not trellis | Cheap |
| **CDQuant** ([arXiv 2406.17542](https://arxiv.org/abs/2406.17542)) | Greedy coordinate descent replacing or refining GPTQ | Not fetched in detail. IQR-V1 uses CD passes too [Lit/partial] | Small | Cheap |
| SpinQuant / QuaRot / OSTQuant | Learned or Hadamard rotations, aimed mainly at **W4A4** | — | Not relevant to weight-only with incoherence processing already in place [Interp] | — |

**Bottom line for Q1** [Interp]: for weight-only KL at 2–4 bpw, nothing published since 2025 clearly beats **QTIP + YAQA (+ light FT)**. The remaining headroom is in Hessian quality (YAQA), allocation, and per-group gain adaptation, not in a new codebook.

Above about 5 bpw every method converges, and remaining differences come from allocation and the head/embedding. IQR-V2's own data shows that above about 7 bits trellis has no advantage over int8 (its IQ8R-V2 trunk is int8).

---

## 2. Q2: healing and fine-tuning after quantization

| Evidence | Gain | What was trained | Data / compute |
|---|---|---|---|
| YAQA + QuIP#-style recovery FT ([2505.22988](https://arxiv.org/abs/2505.22988)) | FT shrinks the LDLQ↔YAQA gap but YAQA keeps 10–20% [Lit] | Unquantized weights blockwise, then end-to-end norms/scales | Small |
| QTIP ([2406.11235](https://arxiv.org/abs/2406.11235)) | Llama-2-70B at 3–4 bit: "fine-tuning has no effect"; "not a silver bullet" [Lit] | HYB codebook + not-yet-quantized weights | — |
| PV-Tuning Table 1 ([2405.14852](https://arxiv.org/abs/2405.14852)) | Llama-2-7B AQLM 2.01b: continuous-only 7.38 → 6.69 (−9% ppl); adding a code-update V-step → 5.90 [Lit] | P: codebooks, scales, norms, embeddings. V: codes | 1B tokens, 2–8 days on 8 GPUs |
| NVIDIA QAD ([arXiv 2601.20088](https://arxiv.org/abs/2601.20088)) | "Near-BF16" for NVFP4. KL loss beats CE (QAT) and MSE on logits. Teacher = the original model (a larger teacher was worse) [Lit] | Full weights (NVFP4 W4A4) | **0.3–6B tokens**, LR 1e-6..1e-5. Self-generated data works as well as SFT data, random tokens do not, and single-domain data transfers [Lit] |
| Google QAT vs YAQA (Gemma-3-12B) | QAT KL 0.089 at 4.5 b vs YAQA PTQ 0.056 at 4 b [Lit] | — | QAT about 100× more data |
| EoRA ([arXiv 2410.21271](https://arxiv.org/abs/2410.21271), [code](https://github.com/NVlabs/EoRA)) | Training-free low-rank error compensation in the activation eigenspace [Lit] | Side-path LoRA (bf16 or quantized) | Costs bpw |

**Realistic expectation for our planned fixed-code QAD** (training scales, norms, suh/svh only) [Interp/Own]:
- **5–15% KLD at ≤3.2 bpw, 2–5% at 4–5 bpw, ≈0 at ≥6 bpw.** Your own 5–10% estimate is consistent with PV-Tuning's continuous-only row and QTIP's "no effect at 3–4 bit".
- **Data:** the challenge calibration corpus, same distribution as the scored split, *including long agentic traces*. Use top-k teacher log-probs (k≈64–512, matching the metric's top-512 + tail), temperature 1, LR around 1e-5 for scales.
- QAD at NVIDIA scale (billions of tokens, full weights) is out of reach on 2×32 GB, but scale-only training needs about 1e-3 of the parameters and a few tens of millions of tokens.
- **Code updates (PV-style).** These are where the large 2-bit gains came from. On a trellis they need re-Viterbi of the tiles whose targets moved. This is feasible; see idea O7.

**LoRA-merge** [Interp]: you cannot merge into trellis codes without re-quantizing. A side-path adapter costs about r·0.004 bpw for a 5120×17408 matrix at bf16 (r=16 ≈ 0.065 bpw on that matrix). At our rates those bits are almost always better spent on K. **Low priority.**

---

## 3. Q3: bit allocation beyond Hessian proxy + knapsack

**Additive knapsack is the standard**, used by all of:
- HIGGS linearity theorem ([2411.17525](https://arxiv.org/abs/2411.17525)): ΔPPL ≈ Σ α_l t_l², with α_l fitted by Gaussian noise injection, solved with CP-SAT.
- Q-Palette (ILP on the linearity-theorem coefficients).
- exllamav3's own SC optimizer ([doc/optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md)): LDLQ-*shaped* noise injection, `kld = S_t·rfn_t(K)^α` with α≈2 and α_low≈2.35 for K2→K1 on Qwen3.8-27B, then exact greedy. Shaped noise reaches Spearman **0.84** against ground-truth swap attribution (unshaped: 0.61).
- IQR (band-measured KL shares, Pareto-front merge).

**Where additivity breaks** [Lit]:
- HIGGS: predicted ≈ actual only for **b > 3**; it "diverges at lower bitwidths".
- EvoPress ([2410.14649](https://arxiv.org/abs/2410.14649)): SPDY-style monotonicity and linearity assumptions "do not hold… especially… below 3 bits". The concrete demonstrations are for pruning, though.
- IMPQ ([2509.15455](https://arxiv.org/abs/2509.15455)): a Shapley-interaction quadratic MILP gives 20–80% lower perplexity than independent sensitivity baselines at 2–2.5 bit. However:
  - its baselines are weak first-order Taylor scores;
  - its own shrinkage ablation shows interaction terms change average perplexity by under 1%.
- IQR-V1 (our model): the band-additive model predicted final mixes **within 1–7%**, so additivity holds well at the band level for 2.5–7 bpw.

**Fixes people use:**
1. **EvoPress**: (1+λ) evolution with level-swap mutations under a fixed budget, KL fitness, multi-stage token budgets (64K tokens enough for 8B). Most of the gain arrives in about 2 GPU-h; converged in about 8 GPU-h on 8B.
2. **IMPQ**: a pairwise interaction matrix from Monte-Carlo permutations; about 18 h on an A40 for 8B.
3. **Re-measure in context**: sensitivity measured with the rest of the model already quantized (IQR's band shares are measured around a mixed base). [Interp]

**Our allocation is current, not outdated** [Interp]. The likely improvement is a measured-KL local search seeded by the knapsack solution, concentrated in the <3.5 bpw tiers (ideas O4, L9).

---

## 4. Q4: hybrid GDN layers, embedding/head, outliers

**Gated DeltaNet, on this exact model.** Source: "Why Gated DeltaNet Survives 4-bit Quantization", [arXiv 2609.04098](https://arxiv.org/abs/2609.04098), Qwen3.8-27B, NVFP4 W4A4. All points [Lit]:
- **Per-projection error on layer output y** (one projection quantized at a time):
  - out_proj 12.7%, in_proj_qkv 10.4%, z 9.9%;
  - gates `a` 2.1% and `b` 2.6%, i.e. the least sensitive.
- **Projection errors add in quadrature**: 19.4% predicted vs 19.2% measured. That is good for per-tensor additive allocation *within* a layer.
- **State error stays flat out to 32K** (≈12.6%). A 1% state impulse decays to 1/e within 80–1,382 steps.
- **The weight-quantization NLL gap shrinks over the context**: +0.081 nats in the first half, +0.011 in the second.
- **Kept in bf16:** embeddings, lm_head, conv1d, norms, A_log, dt_bias.
- **Fused-GEMM pitfall**: qkv+z and b+a fused with mismatched scales corrupted the gates (AIME 86.7 → 80.8), and perplexity *falsely improved*. This is relevant if your fused kernel shares scales across fused tensors.

**Other hybrids** [Lit]:
- NVIDIA QAD kept the attention layers and the first and last two layers in bf16 for Nemotron-H-style Mamba hybrids, "to maintain a better PTQ baseline". That is NVFP4 W4A4, a much coarser format, so the transfer to us is weak [Interp].

**Embedding / head** [Lit]:
- IQR-V2 measured a rotated trellis embedding at 4 bits: **+0.00006 KL**. An *unrotated* trellis embedding or head clips outliers: **40–120× worse** (+0.0024 embedding, +0.0070 head). IQR's head uses GPTQ-fitted scalar tables at 3–6 bit, or int8 at the top sizes.
- Each of embedding and head is 1.27B parameters, so 1 bit on either = **0.047 bpw**. IQR-V1 measured, *for the embedding*: 4-bit round-to-nearest costs +0.00058 KL and 6-bit costs +0.00003.
- exllamav3 now trellis-codes the embedding in rotated 256-wide groups (`-eb`, [convert.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md)).
- VBQ ([arXiv 2607.02893](https://arxiv.org/abs/2607.02893)) used a **frequency-tiered head**: the top 0.5% of tokens at 8 bit, then 4, 2 and 1 bit, for 1.18 bits mean. The tiering was "worth 0.6 PPL for +0.09 mean bits". This was a small model trained from scratch, so it is only weak evidence for PTQ.

**Outliers.** Incoherence processing handles weight outliers [Lit: QTIP, IQR]. The remaining outlier issue is the head and embedding if left unrotated. Ours are rotated, so this is fine [Interp].

---

## 5. Point-by-point: IQR-V2 (competitor) vs us

**Sources:** the local text versions of [IQR-V1](https://radiance.stilldeadco.de/whitepapers/iqr-v1.pdf) and [IQR-V2](https://radiance.stilldeadco.de/whitepapers/iqr-v2.pdf). Both were read in full. All IQR facts below are [Lit].

### Head-to-head KL

The competitor numbers quoted in the task are exactly IQR-V2's Table 2. "Adjusted" interpolates IQR's curve (log-KL linear in bpw between adjacent recipes) to our bpw [Interp].

| Our point | IQR-V2 nearest | IQR-V2 adjusted to our bpw | Us vs them |
|---|---|---|---|
| 2.12 bpw / 0.148 | 2.467 / 0.0730 | ~0.117 (extrapolated) | ~+25% worse (low confidence) |
| 3.12 / 0.033 | 3.145 / 0.0280 | ~0.029 | **~+14% worse** |
| 4.07 / 0.0095 | 4.130 / 0.00832 | ~0.0089 | **~+7% worse** |
| 5.07 / 0.0028 | 5.076 / 0.00273 | ~0.0027 | ≈ parity |
| 6.02 / 0.0011 | 6.062 / 0.00115 | ~0.0012 | **~8% better** |

### Component comparison

| Component | IQR-V2 | Us (EXL3) | Who likely wins / note |
|---|---|---|---|
| Code | Bitshift trellis **L=12**, k=2..7, trained 4096-entry int8 tables (Lloyd on Viterbi paths), tail-biting runs of 256 rows. 0.69 dB from D(R) at 2 bit | L=16 mul1 computed code on 16×16 tiles. QTIP L=16 TCQ: MSE 0.069 vs D(R) 0.0625 at 2 bit (≈0.43 dB) | **Us**, slightly. Not the source of the gap [Interp] |
| Error feedback | Trellis runs *down a column* inside GPTQ, so per-column feedback is **exact** | BlockLDLQ: feedback only between 16-column blocks; no feedback inside a tile | **Them**, size unknown; plausibly matters more at low K [Own] |
| Gain / scale | **E4M3 scale per (row, 128 inputs)** × f32 tensor scale (0.0625 bpw), chosen dynamically as GPTQ reaches each group | Per-row and per-column fp16 (suh/svh) only, i.e. a rank-1 gain model | Depends on how much group-RMS varies after rotation; testable in one hour (O2). IQR-V1 found 128→64/32 groups didn't help at 2 bit |
| Rotation | 128-point block WHT, inputs only | Hadamard both sides + sign flips | ≈ / us |
| Hessians | 10M-token calibration corpus (same challenge data); damping λ = **0.05 × mean diag** | Same published Hessians; EXL3's damping | Check damping (L3) |
| Allocation | 8 bands × 3 units (gate_up, down, attn+GDN). Each unit's KL share = band share × weight fraction. Bands measured on whole-model KL; Pareto-front merge; ~50 candidates scored on the public split. Projection types "about equally sensitive per weight"; **last 8 layers carry 28–31% of KL** | Per-tensor proxy calibrated by group/band swaps, knapsack | Ours is finer-grained. Theirs is directly measured on the metric. They chose only 2–3 adjacent widths per mix |
| Embedding | Rotated trellis; 4-bit costs +0.00006 KL; 6-bit asymmetric at top sizes | Lloyd-Max 3–6 bit under 128-WHT | **Them.** Lloyd-Max is about 2.3 dB (≈0.38 bit) worse in MSE than trellis [Lit numbers, Interp conversion] |
| Head | GPTQ-fitted scalar tables 3–6 bit (rotated); int8 at top sizes | EXL3 trellis at K chosen by the knapsack | Probably ≈. Neither tiers by vocab probability (O1) |
| Fine-tuning | **None** | None yet (QAD planned) | Open lever for both sides |
| Selection | Public split for selection, private split for reporting. IQ2R margin "within selection noise" | Same corpus structure | — |

**Where they likely gain at 2.5–4 bpw** [Interp/Own], in order:
1. Exact column feedback plus dynamic group scales, both of which matter more at low K.
2. Measured-KL allocation that captures interactions our proxy misses below about 3.5 bpw.
3. Better embedding bits.

Run diagnostic **O0** before guessing further.

**What transfers to an EXL3/trellis container:**

| IQR idea | Transfers? | How |
|---|---|---|
| Rotated trellis embedding | Yes, directly | exllamav3 `-eb`, or our own code |
| Measured-band KL allocation | Yes | Use our swap infrastructure as the *fitness* in a local search |
| Dynamic per-group gain scales | Needs a kernel change | One extra scale fetch per 128 (or per 16×16 tile); still a trellis kernel |
| CD refinement passes after GPTQ (V1) | Yes | Per-tile re-Viterbi coordinate descent under the exact objective (O3) |
| Exact per-column trellis | No | That would be a different format. Approximate it with smaller effective blocks or O3 |
| Damping 0.05 | Yes | A free sweep |
| Int8 at top sizes | N/A | Our 6 bpw is already better |

---

## 6. Original ideas [Own unless marked]

**O0. Diagnose the gap before building anything** (1–2 days, no risk).
- **(a) Per-tensor relative output error.** Compute `tr(EHEᵀ)/tr(WHWᵀ)` for our EXL3 layer-3 `ffn_gate_up` at K=2..6. Compare it with IQR-V2 Fig. 3 (and V1 Fig. 3, whose "falls about 4× per bit" and "23% less than V1" anchor the scale).
  - If ours is equal or lower, the gap is allocation or embedding/head.
  - If it is higher, the gap is quantizer, feedback or gains.
- **(b) Per-source KL breakdown.** Measure our KL on the long agentic traces vs everything else. Two 256K traces are about **512K of the ~1.45M scored positions (~35%)**, and IQR-V1 says its lead was widest on agentic traces and reasoning.
- **(c) Band KL shares.** Compare our band shares to IQR's Fig. 4 (last band 28–31%).

**O1. Probability-weighted head and frequency-tiered embedding** [Own; Fisher math is standard; VBQ gives weak literature support].
- **Reasoning.** For logits z = W_head·h, the per-position KL ≈ ½ δzᵀ(diag(p) − ppᵀ)δz. So row v's error costs about E[p_v·(δw_vᵀh)²]. The head's *output-side Hessian is analytically ≈ diag(E[p_v])* ⊗ E[hhᵀ] (dropping the rank-1 term). In a 248k multilingual vocab, E[p_v] is extremely skewed, so most rows can go to K=2–3 while the top few thousand rows get K=6–8.
- **The embedding** is symmetric in a different way: row v matters ∝ its input frequency f_v.
- **How to implement:**
  - Permute the vocab by E[p_v] or f_v and split into 2–3 tensors with different K. A gather fixes the order after the head GEMM; the embedding needs only a lookup.
  - Alternatively, keep one K and use EXL3 `--out_scales yaqa` with this analytic diag(H_O) on the head.
- **Cheap test:**
  - Compute E[p_v] from the bf16 teacher on about 1M calibration tokens.
  - Quantize the head at K=2 for the bottom 90% of rows by mass and K_current for the rest.
  - Measure KL and bits.
- **Expected outcome:**
  - Head + embedding currently cost roughly 0.19–0.28 bpw *each* at 4–6 bit. Tiering could free **0.05–0.15 bpw** at the ≤4 tiers.
  - At our 3-bpw slope (log-KL ≈ 1.4–1.5 per bpw) that is roughly **7–20% lower KL at fixed size**.
- **Effort:** low-medium. **Risk:**
  - Rare-token spikes: a large +δz on a rare token produces a false positive. Guard with a K floor.
  - Tokens unseen in calibration: give them a floor of K=3.

**O2. Per-group gain scales in the trellis kernel** [IQR-backed idea, Own adaptation].
- **Cheap test first** (one hour): after our rotation and suh/svh, compute the variance σ_g² of every (row, 128-col) group. High-rate theory says the bits lost without per-group gain are `Δb ≈ ½·log2(AM(σ_g²)/GM(σ_g²))`.
  - If Δb > ~0.08 bpw on the big tensors, an E4M3 group scale (0.0625 bpw) pays for itself.
  - If Δb < 0.03, drop the idea.
- **Implementation:** a per-16×16-tile 8-bit gain costs only 0.03 bpw and fits EXL3's tile structure.
- **Effort:** medium (kernel plus quantizer). **Risk:** medium. **Gain:** 0–10%.

**O3. Tile-level coordinate-descent re-Viterbi after LDLQ** [CDQuant and IQR-V1 CD are related literature; Own for trellis].
- After the LDLQ pass, revisit each 16-column block. Re-encode each tile with the *other* blocks' errors fixed, against the exact `tr(EHEᵀ)` restricted to that block. This approximates the in-tile metric with the conditional (Schur) diagonal, then runs 1–3 sweeps.
- Recovers part of IQR's "exact feedback" advantage without changing the format.
- **Gain:** 2–8% at K≤3. **Effort:** medium (offline only). **Risk:** low.

**O4. Measured-KL local search on top of the knapsack** [EvoPress/IMPQ-backed].
- Seed with the knapsack mix. Mutate with budget-preserving K swaps between tensors of equal size (or ±1 K across bands). Fitness is measured KL on a fixed ~200K-position public subset that **includes long-trace slices in the same proportion as the scored split**.
- Mixes are assembled by copying pre-quantized tensors, so each candidate costs only a forward pass.
- Accept a change only on a **paired, per-document bootstrap** improvement, to avoid the selection noise IQR admits to.
- **Gain:** 3–10% below 3.5 bpw, about 0 above 5. **Effort:** low. **Risk:** low (overfitting is controlled by the public/private split).

**O5. Re-linearize the proxy around the chosen mix** [Own].
- Re-measure each tensor's ΔKL for ±1 K *in the context of the current mix* rather than around bf16. Re-solve, and iterate 2–3 times.
- This captures the non-additivity that grows below 3 bpw. It is cheaper than full evolution and complements O4.

**O6. Long-context-weighted Hessians and allocation** [Own].
- If the published Hessians are dominated by short documents, the 35% of the score that comes from 256K agentic traces is under-represented. Re-weight H (or just the allocation's calibration KL) by the leaderboard's position mix.
- The GDN paper suggests the weight-quantization gap *shrinks* late in the context. So check O0b first: the effect may run in either direction.

**O7. PV-lite on trellis: periodic code refresh during QAD** [PV-Tuning-backed; Own trellis adaptation].
- During scale-only QAD, keep an implicit target `W_target = dequant + accumulated gradient step` per tensor. Every N steps, re-run Viterbi (with LDLQ) on the τ fraction of tiles whose targets moved most.
- PV's big gains came from exactly this discrete step (Llama-2-7B 2b: 6.69 → 5.90).
- **Gain:** maybe 10–20% at ≤2.5 bpw, small at ≥4. **Effort:** high. **Risk:** medium.

**O8. Shared "differentiable EXL3" infrastructure** [Own].
- One building block unlocks YAQA Hessians, QAD, PV-lite and GuidedQuant: a forward pass that dequantizes each layer to bf16 on the fly (178 MB for the largest tensor), plus a backward pass that is a plain bf16 GEMM with Wᵀ.
- Build it once, on a K≈8 (or competitor-like int8) surrogate of the 27B so it fits in 2×32 GB with activation checkpointing.
- Using an 8-bit surrogate (KL about 5e-4) for the Hessian and teacher forward passes is an approximation, but a small one.

**O9. Selection on a tier-aware target** [Own; check the rules].
- If tiers are ranked by best KL *within* a bpw band, put each entry just under the tier ceiling (2.99, 3.99, 5.49, 7.49, 9.49) rather than at round numbers.
- IQR has entries at 3.80 and 5.42. Our 2.12 entry sits in the <3 tier, where their 2.79 entry has 0.047.

---

## 7. Ranked shortlist

| # | Item | Type | Expected gain (KLD at fixed bpw) | Effort | Risk |
|---|---|---|---|---|---|
| 0 | **O0 diagnostics** (per-tensor error vs IQR Fig. 3, per-source KL, band shares, group-variance AM/GM) | Own | Decides what to build | 1–2 days | None |
| 1 | **O1 vocab-probability-tiered head + frequency-tiered rotated-trellis embedding** (replacing Lloyd-Max) | Own + Lit (Fisher, IQR, VBQ) | **7–20% at ≤4 bpw**, ~3% at 5–6 | Low-medium | Low-medium |
| 2 | **YAQA two-sided LDLQ** (planned). Stage it: (a) diag-H_O `out_scales yaqa`, which is cheap; (b) full Kronecker H_O for attention, GDN and down projections first (gate/up H_O is 17408², about 1.2 GB fp32 each, so process per tensor from disk) | Lit | 10–30% (paper: ~33% on Llama at uniform bits; expect less on top of our allocation and QAD) | High (needs O8; days of GPU time on RDNA4) | Medium |
| 3 | **O4 + O5 measured-KL allocation refinement**, with long-trace-weighted fitness and paired bootstrap acceptance | Lit (EvoPress) + Own | 3–10% below 3.5 bpw | Low | Low |
| 4 | **Fixed-code QAD** (planned) on scales, norms, suh/svh (+ group gains if O2 is positive), KL on top-k teacher, calibration corpus incl. long traces; later O7 code refresh | Lit (QAD, PV, EfficientQAT) | 5–15% at ≤3.2 bpw, 2–5% at 4–5 | Medium (needs O8) | Low-medium |
| 5 | **Fractional or intra-tensor K** (planned). Prefer Q-Palette-style *row-split* K (two K values per tensor, rows chosen by YAQA diag-H_O) over new half-bit trellis rates | Lit (Q-Palette) + Own | 5–15% on tensors sitting between integer K. Convexity: mixing K=3/4 gives 25% more distortion than a true 3.5-bit code at D∝2^{−2R} | Medium (kernel per-row-block K, or split tensors) | Low |

Alternate for slot 5, conditional on the O0/O2 diagnostic: **O2 per-tile gain scales**, if the AM/GM test shows more than 0.08 bit lost.

---

## 8. Long tail (all other ideas, including weak ones)

Format: idea — why it might help / why it might not — effort — source.

1. **Damping sweep** (IQR uses 0.05×mean diag) — free / probably already near-optimal — trivial — Lit (IQR).
2. **GPTAQ asymmetric calibration**: quantize sequentially with quantized-model inputs and FP targets — fixes accumulated drift / our per-tensor K mixing breaks "the" quantized-input model — medium — Lit.
3. **Sequential re-quantization of the final mix** (Hessians recomputed with upstream already quantized, as `convert.py` does) — matches the real input statistics / the challenge Hessians may already be better (10M tokens) — medium — Lit/Own.
4. **D²Quant mean-shift bias correction** after o_proj / GDN out_proj (a stored 5120-vector per layer) — cheap; 2-bit gains on Qwen3-8B / tiny at ≥4 bit; needs a kernel bias add — low — Lit.
5. **D²Quant dual-scale for down_proj** (a column scale folded into up's per-row svh) — smoother down_proj / EXL3's suh already does something similar — low — Lit.
6. **GuidedQuant grouped Fisher** as a cheaper fallback to YAQA for gate/up (H_O too large) — one backward pass / strictly weaker than YAQA — medium — Lit.
7. **Random-sign seed search per tensor** (try 4–8 Hadamard sign seeds, keep the lowest proxy error) — a cheap 1–3% per-tensor error reduction / tiny end-to-end effect — low — Own.
8. **Trained LUT codebook instead of mul1** (IQR trains tables; QTIP HYB) — fits the post-rotation non-Gaussianity / QTIP says L=16 codes are all about equal; also breaks EXL3 fast paths — medium — Lit.
9. **Pairwise interaction terms (IMPQ) between adjacent bands** — captures non-additivity / IMPQ's own ablation shows <1% effect — medium — Lit.
10. **Separate K for gate vs up** (exllamav3 ties them by default) — finer allocation / tiny; costs fusion — low — Lit (optimize.md).
11. **Residual K8+K2 head** (planned) — finer head rate / O1 likely dominates it — low — Own (yours).
12. **Head stored as 3–4 vocab tiers with different K** (generalizes O1) — more savings / more kernels or gathers — low — Own.
13. **Embedding rows unseen in calibration at K=2 with a norm-preserving scale** — saves bits / private set has rare tokens — low — Own.
14. **Head bias or logit temperature tweak learned by QAD** (scalar per head tier) — corrects systematic logit-scale shrinkage from quantization / small — trivial once O8 exists — Own.
15. **Learn suh/svh by KL gradient instead of statistics** — a subset of QAD — low — Lit (EfficientQAT E2E-QP).
16. **RMSNorm weight retuning only** (a few hundred thousand parameters, no kernel change) — the cheapest QAD variant; could be done blockwise without full backprop — low — Lit/Own.
17. **Blockwise (per decoder-layer) output-matching fine-tune of scales** (EfficientQAT Block-AP style, scales only) — fits in memory easily / ignores end-to-end KL — medium — Lit.
18. **Per-position KL weighting in QAD** to mirror the leaderboard mix (35% long traces) — aligns the objective / long-context backprop is memory-heavy, so truncate with state carry — medium — Own.
19. **Keep GDN `a`/`b` projections at very low K** (GDN paper: least sensitive) — saves bits / they are tiny anyway (≈0.1% of weights) — trivial — Lit.
20. **Fused-tensor scale harmonization check** (GDN paper's qkv+z, b+a bug) — avoids silent gate corruption if our kernel fuses with shared scales — trivial — Lit.
21. **bf16 for conv1d, A_log, dt_bias, norms** (verify this is already the case) — negligible bits — trivial — Lit.
22. **Higher K for full-attention o_proj / k_proj in the last band** (IQR: last band 28–31% of KL; exllamav3: mid-stack k/v valuable on self-sampled traces) — the allocator should already find this; use it as a sanity check — trivial — Lit.
23. **EoRA low-rank side path at r≈8–32 on the most sensitive tensors only** — cheap, training-free / bits usually better spent on K — medium — Lit.
24. **Larger LDLQ effective block** (feedback inside the tile via two half-tiles) — closer to exact feedback / needs a format change — high — Own.
25. **Self-generated calibration traces** (exllamav3 `sd_trace`) — the exllamav3 docs say sensitivities measured on the model's own outputs transfer better / our scored corpus is fixed and known, so in-distribution calibration is already optimal — low — Lit.
26. **Calibration Hessians from more tokens** (YAQA used 20M; the challenge offers 10M) — lower variance for big tensors / probably diminishing returns — medium — Lit.
27. **Mixed-codebook per tensor** (e.g. scalar int table for the head, trellis elsewhere, as IQR does) — the head may prefer GPTQ tables / our rotated trellis is already better than scalar in MSE — medium — Lit.
28. **Tail-biting check and start-state bit savings** — negligible (EXL3 already handles it) — trivial — Lit (IQR).
29. **Ensembling Viterbi with different trellis offsets per tile** (pick the best of 2) — 1–2% error / doubles quantization time; tiny KL gain — low — Own.
30. **Teacher top-k caching from bf16 on 2 cards** (offline, ~20 GB disk for 50M tokens × top-64) — makes QAD and O4 cheap — low — Own.
31. **Use an 8-bit EXL3 model as YAQA/QAD teacher-surrogate** — fits in VRAM / small bias (KL ≈ 5e-4) — low — Own.
32. **Hessian shrinkage toward the diagonal for rare-feature columns** (GDN inputs have kurtosis ≈1,560 per the GDN paper) — robustness / damping already does this — trivial — Own.
33. **Stochastic or dithered rounding** — PV Table 1 shows it inferior to STE and PV — skip — Lit.
34. **SpinQuant/OSTQuant learned rotations** — designed for activations; weight-only gains over Hadamard are minor — skip — Lit/Interp.
35. **Per-group precision learned by Gumbel (VBQ)** — only worked when training from scratch; post-hoc search was dominated by uniform 4-bit — skip — Lit.
36. **NVFP4/FP8 hybrids** — wrong hardware target and coarser than trellis — skip — Interp.
37. **Quantize the MTP head harder or drop it** if it counts toward bpw — free bits if counted / IQR counts only the 26.9B text parameters — trivial — Own.

---

## Contradictions
- **Does fine-tuning help at 3–4 bit?**
  - QTIP: "no effect" on Llama-2-70B at 3–4 bit.
  - YAQA: FT narrows the LDLQ gap, i.e. it helps LDLQ.
  - Likely both are true: FT helps weak one-shot results and helps little on strong ones.
- **Is additivity reliable?**
  - IQR measured 1–7% prediction error on our model.
  - EvoPress and IMPQ claim large non-additivity below 3 bits, but on weak baselines, and EvoPress's concrete evidence comes from pruning.
- **PV-tuning vs QTIP.**
  - The PV paper reports QuIP# 2-bit Llama-2-7B as 6.19 in one table and 8.22 in another (an internal inconsistency).
  - The YAQA authors claim LDLQ+QTIP beats PV-tuned AQLM on Llama 3.1, without a public head-to-head.
- **What drives the exllamav3 sensitivity profile?** exllamav3's docs report a flattened U-shape and hot mid-stack k/v on self-sampled traces. IQR reports the last band dominating on the challenge corpus. The two are data-dependent, so trust measurements on the challenge corpus.

## Missing evidence
- I could not extract numeric tables for YAQA Table 1, GuidedQuant+QTIP, HIGGS, EvoPress quantization, or the QAD absolute recoveries. Their gains are quoted only as the papers' prose summaries.
- No published result applies YAQA, QAD or PV to a GDN hybrid in a trellis format. All gain estimates for our model are inference.
- I could not read exllamav3's `science/yaqa.md` (the fetch failed). Two-sided LDLQ support is confirmed only via `convert.md` (`--hessians`, `--hessians_reg 0.025`, `--out_scales yaqa`).
- IQR's per-tensor error curve is a figure, not a table. Comparing against it (O0a) needs approximate reading of the plot, or reproducing their GPTQ-column trellis on one tensor.
- I have not verified the leaderboard's tier-ranking rule (O9).

## Sources
**Kept:**
- IQR-V1/V2 whitepapers (local, and [online](https://radiance.stilldeadco.de/whitepapers/iqr-v2.pdf)): same model, metric and hardware.
- [YAQA](https://arxiv.org/abs/2505.22988): SOTA one-shot.
- [QTIP](https://arxiv.org/abs/2406.11235): EXL3's basis.
- [exllamav3 convert.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md) and [optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md): what our format already supports.
- [GDN 4-bit paper](https://arxiv.org/abs/2609.04098): same model.
- [Q-Palette](https://arxiv.org/abs/2509.20214), [HIGGS](https://arxiv.org/abs/2411.17525), [EvoPress](https://arxiv.org/abs/2410.14649), [IMPQ](https://arxiv.org/abs/2509.15455): allocation.
- [PV-Tuning](https://arxiv.org/abs/2405.14852), [NVIDIA QAD](https://arxiv.org/abs/2601.20088), [EfficientQAT](https://arxiv.org/abs/2407.11062): healing.
- [GuidedQuant](https://arxiv.org/abs/2505.07004), [GPTAQ](https://proceedings.mlr.press/v267/li25dn.html), [D²Quant](https://arxiv.org/abs/2602.02546), [CDQuant](https://arxiv.org/abs/2406.17542), [EoRA](https://arxiv.org/abs/2410.21271), [VBQ](https://arxiv.org/abs/2607.02893).

**Deprioritized:**
- SpinQuant / QuaRot / OSTQuant: activation-quantization focus.
- ParetoQ: QAT from training scale.
- Blog and SEO comparisons of EXL3 vs GGUF: different models, non-reproducible.

## Next steps
1. Run O0 (diagnostics) and the O1 cheap test (head tiering at fixed trunk). Both take under 2 days and decide between O2 and O3 for the remaining work.
2. Build O8 (on-the-fly dequant forward/backward on an 8-bit surrogate). It is the shared prerequisite for YAQA, QAD and PV-lite.
3. Run O4 on the 3.12 and 2.12 mixes with long-trace-weighted fitness.
