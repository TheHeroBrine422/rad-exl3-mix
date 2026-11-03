# NVIDIA Model Optimizer: overlap with our work (read 2026-11-03)

Source: the ModelOpt docs site (<https://nvidia.github.io/Model-Optimizer/>), its eight 2026 announcement posts, and
the repo at `472ecb3` (2026-11-03; clone in `/workspace/references/model-optimizer`, outside the repo). ModelOpt targets NVFP4 / FP8 / GGML-IQ
formats on NVIDIA hardware, not trellis codes, but its quantizer-side and allocation methods are ours in another
format. They also publish **`nvidia/Qwen3.8-27B-NVFP4`, our exact model**.

## What they do, mapped to ours

| ModelOpt | what it is | ours | overlap / difference |
|---|---|---|---|
| **AutoQuantize, gradient score** (2026-08-24) | per-layer, per-format sensitivity S = Σ_k g_k² (Y_k − Y_k^Q)², g = ∂L/∂Y from one backward pass; knapsack ILP under an effective-bits budget | the Σdy² diagonal H_out (#33-18) and the knapsack (`kfit2` / knap4) | **The same estimator.** They use it only to *allocate* formats; we use it to *quantize* (output-weighted LDLQ), with sensitivities from measured swaps. Using the Σdy² diagonal also as the knapsack price is our w08-hprice (`ourq-hprice.patch` / `knap4-hratio.patch`, not merged yet). Their KL-scored variant (quantize one layer, measure KL) is our swap measurement; they quote 14 h vs 16 min on a 35B-A3B. |
| **Saturation-aware AutoQuantize** (2026-09-28, Baseten; arXiv 2607.12266) | damage f(S) = c(1 − Π(1 − a_i)): additive in b_i = −log(1 − a_i), not in KL; a_i from Aumann–Shapley path gradients (K = 2 nodes), c = KL of the all-quantized model; linear program on Σb | knap4 adds per-tensor KL deltas (measured on a 4.00 base; F4 in the mining reports) | **New to us.** Up to −37% KL at matched bits vs additive gradient scores (Qwen3-30B / 235B MoE, formats NVFP4 / FP8 / BF16). Matters most where total damage approaches the ceiling, i.e. our low sizes (1.75–2.5 bpw, KL 0.07–0.3). Cheap to test on data we have (below). |
| **Local-Hessian scales** (2026-09-09) | NVFP4 per-16-block FP8 scale chosen by the block's local H error over all 126 candidates; RTN; GPTQ optional | per-row scale-factor search by H-weighted error (`E3_FACTORS`), output scales, #85 fixed-code refits | Their point, that scale choice by output error rather than weight error buys a lot, is the same as ours. **On Qwen3.8-27B, Local-Hessian + GPTQ scored below Local-Hessian alone**, so the published 27B checkpoint skips GPTQ. Nothing new for trellis codes. |
| **Scale learning / LSQ** (2026-09-16) | NVFP4 block scales trainable (STE), scale-only QAD or with weights; tied or dual (pre / post) scales | #89 frozen-code healing (H1 / H4), S27 | **Same as our healing.** Their findings: scale-only is *more consistent* than full QAD (QAD regresses some benchmarks), lr 1e-4, ~50 steps can suffice, 1/16 the optimizer state. Dual pre/post scales change rounding, so with frozen trellis codes only the post-scale applies (what #89 does). Supports S27 as the safer QAD-class step. |
| **Layerwise calibration**, `get_qdq_activations_from_prev_layer` (2026-09-28) | depth-outer loop: one decoder layer live; activations of the whole calib set kept at the layer boundary; each layer's statistics from the *quantized* previous layers; per-layer export shards, resumable | #87 / #88: inputs from quantized siblings / upstream (sym arm = theirs; asym target is our addition) | Their sym form (GPTQ on quantized-upstream inputs) is our #87 **sym** arm, which was null on the 0.8B (K3 0.998); our gain came from the **asym target** (GPTAQ-style), which ModelOpt does not have. Their loop is the engineering template for #87 at 27B: 1M tokens × 5120 × bf16 = 10 GB per boundary (iceman: disk-backed). |
| **GGML IQ formats + GPTQ / imatrix** (2026-10-28, GLM-5.3-Flash) | IQ1/IQ2 lattice codebooks with GPTQ in PyTorch, imatrix-weighted search, INT8 conversion for vLLM | our trellis + LDLQ | Findings that match ours: **down projections carry most of the 2-bit error** (NVFP4 on down alone cut NLL error ~40%, +0.65 bpw on experts); **long calibration samples beat short** (256 × 32K beat 512 × 4K: GPQA +3.1, SciCode +6.1) as in #33-21; IQ2 failure mode = runaway generations, which a KLD score only sees indirectly. |
| **QAD** (2026-09-16 Qwen3.6 W4A4; GLM IQ2) | KL distillation on chat SFT, 300–500 iters | V5 (not started) | Recovers instruction-following and long-context runaways; **raises hallucination / lowers abstention** and lengthens outputs; knowledge recall does not recover with a generic blend. Costs ~4,500 GPU-h at 321B. |
| **KV AutoQuant** (2026-10-08) | per-layer K/V format pair (FP8 / NVFP4) by isolated-layer KL, ILP under a KV-bit budget | none: our evals use a bf16 KV cache | Only matters if the board ever counts KV VRAM (deadcode: VRAM is the target, currently weights). Note only. |
| GDN handling | `linear_attn.in_proj_a` / `in_proj_b` (β / α gates) kept BF16; GDN projections FP8 in the 27B recipe | our ladder quantizes the GDN projections at the trunk K | Their 27B NVFP4 recipe keeps all attention and GDN projections at FP8 W8A8 and puts only the MLP (+ head) at NVFP4: a strong prior that GDN / attention inputs are the sensitive part. Consistent with #82 (gdn_in carries 39% of the H_out gain). |

## `nvidia/Qwen3.8-27B-NVFP4`

Recipe `modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_local_hessian.yaml`: MLP NVFP4 W4A4,
attention and GDN projections FP8 W8A8, lm_head NVFP4, β / α gates BF16; the format assignment came from a
5.5-effective-bit AutoQuantize sweep; scales by Local-Hessian, layerwise with quantized upstream activations, no GPTQ;
calib 512 × 2048 of `nemotron-post-training-v3`. Roughly 5–5.5 bpw of weights (their effective bits exclude
embeddings). It is a direct reference point next to our V2 5.0 / 5.5 (board 0.00256 / 0.00153).

## Ideas for us (each a BACKLOG row)

1. **Saturation check of the knapsack (CPU, ~1 h, no GPU).** On existing measurements (ladder uniform levels, V2
   mixes with predicted vs measured KL, the kfit2 swaps), test whether predicted = c(1 − exp(−Σb)) with
   b = −log(1 − ΔKL_i / c) fits measured mixed-model KL better than ΣΔKL, at 1.75–2.5 bpw especially. If the additive
   prediction overshoots at low bpw by more than the seed noise, re-run the knapsack on b for the low V3 sizes.
2. **Reference KLD of `nvidia/Qwen3.8-27B-NVFP4`** on our verify slice (iceman: dequantize weights to bf16 for a
   weight-only number; W4A4 activations would need fake-quant), to place it on our frontier. Public data only.
3. **Σdy² as the knapsack price** (w08-hprice, ready): their AutoQuantize is exactly this; once the 27B diagonal
   exists, the price comes free.
4. **#87 at 27B:** follow their layerwise loop (activations at the layer boundary, one layer resident, disk-backed).
5. **Healing (#89 → S27):** their scale-only results argue for it before any full QAD; adopt their settings as the
   starting point (lr 1e-4, ≤ 100 steps, early stop) when the user approves.
