---
license: apache-2.0
pretty_name: Qwen3.8-27B diagonal output Hessians
tags:
  - quantization
  - hessian
  - fisher
  - exl3
---

# Qwen3.8-27B diagonal output Hessians (Σdy², Sketch B, xw)

Per-output-channel sensitivity statistics for every trunk linear of **Qwen3.8-27B** (64 layers, 400 tensors),
collected for output-aware weight quantization (the diagonal of YAQA's output factor, arXiv 2505.22988). We use them
in EXL3-V3's quantizer ([rad-exl3-mix](https://github.com/TheHeroBrine422/rad-exl3-mix)): LDLQ / trellis
quantization with the published input Hessians on one side and this diagonal on the output side.

## What is in here

| file | content |
|---|---|
| `block-normalized.safetensors` | **the recommended estimator**: Σdy² with each 128-sequence block normalized before averaging (tames a heavy tail of a few huge blocks in early layers), from 1,920 sequences; `block-normalized-mkbn.json` = per-tensor block statistics |
| `snapshots/nNNNN-dy.safetensors` | Σ_t dy_t² (plain output Fisher diagonal), mean over the first NNNN sequences |
| `snapshots/nNNNN-xw.safetensors` | Σ_t ‖x_t‖² dy_t² (YAQA Sketch A's first-step weighting with H_in ≈ I) |
| `snapshots/nNNNN-sketchb.safetensors` | diag(Σ_b G_b G_bᵀ), G_b = Σ_t dy_t x_tᵀ per sequence (YAQA Sketch B's output factor, diagonal); `nNNNN-sketchb-halves.json` = per-tensor cosine between the even- and odd-sequence means |

Snapshots every 128 sequences, 128 … 1,920. Each file holds 400 fp32 vectors, one per trunk linear, named by its
checkpoint key (`model.language_model.layers.L.<module>`), of length out_features (the mean over sequences). gate_proj
and up_proj are separate entries. Only relative values matter for quantization (our quantizer rescales each diagonal
by its mean).

## How it was collected

- **Code:** [`v3/yaqa/collect.py`](https://github.com/TheHeroBrine422/rad-exl3-mix/blob/main/v3/yaqa/collect.py)
  (`YAQA_DIAG=1 YAQA_DIAG_EXTRA=1 YAQA_FP32=1 YAQA_GENMASK=1 YAQA_SEED=0`), launched by
  [`v3/yaqa/hdiag27.sh`](https://github.com/TheHeroBrine422/rad-exl3-mix/blob/main/v3/yaqa/hdiag27.sh);
  block normalization by [`v3/yaqa/mkbn.py`](https://github.com/TheHeroBrine422/rad-exl3-mix/blob/main/v3/yaqa/mkbn.py).
- **Model:** Qwen3.8-27B bf16 weights; activations and gradients in **fp32** end to end (bf16 backward rows were wrong
  for a minority of positions, increasingly in early layers).
- **Data:** 2,048-token windows of the **qad pool** of the
  [challenge's calib corpus](https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge) (1,350
  distinct windows; sequences beyond that reuse windows with fresh labels). No other data.
- **Loss:** cross-entropy against labels **sampled from the model's own distribution** (true Fisher), only at the
  positions the challenge scores (gen mask); every position still runs.
- **Stop:** 1,923 sequences (the 1,920 snapshot is the last); the block-normalized estimate is flat from ~384.

## Result (provisional, 2026-11-05)

Swap test at K2.5 (rule written before the data: [`v3/yaqa/H27.md`](https://github.com/TheHeroBrine422/rad-exl3-mix/blob/main/v3/yaqa/H27.md)): the
whole trunk quantized at K2.5 with each estimator (our tuned EXL3 settings, int8-rounded codebook), swapped into our
EXL3-V2 3.00 bpw model, scored on 103 search-pool documents of the calib corpus (123k scored positions, bf16 KV),
paired per document with a bootstrap over documents:

| H_out | mean KL | vs none | z |
|---|---|---|---|
| none (input Hessian only) | 0.05903 | — | |
| **block-normalized Σdy²** | **0.05737** | **−2.8%** (95% upper bound −0.4%) | −2.2 |
| plain Σdy² mean | 0.05803 | −1.7% | −1.6 |

Block-normalized vs plain: −1.1% (z −1.4). **Provisional:** one level (K2.5) and one rotation seed; the quantization
seed-noise arm and a K4 level are still running and this section will be updated. The gain is about a third of what
the same method gave on Qwen3.5-0.8B (−8 to −9% at K2–K3), so treat 0.8B-sized effects as upper bounds.

## Use

**Reading the files** (no dependency on our code):

```python
from safetensors import safe_open
with safe_open("block-normalized.safetensors", "pt") as f:
    h = f.get_tensor("model.language_model.layers.10.mlp.down_proj")   # [5120] fp32
h = h / h.mean()   # only relative values matter
```

**How we use it.** The quantization objective becomes tr(Eᵀ H_in E H_out) with H_out = diag(h) (the Kronecker
approximation of the end-to-end KL Hessian, H ≈ H_out ⊗ H_in, YAQA arXiv 2505.22988), H_in the published input
Hessian. In exllamav3 / EXL3 the output channels are rotated (sign flips, channel scales and a 128-point Hadamard), so
in the quantizer's basis the diagonal becomes a dense 128-block coupling between rows, and LDLQ runs with error
feedback on both sides (`ldlq_2hess`, `prepare_H_out`: H_out normalized to mean 1, + 0.01 on the diagonal). With
our quantizer: split a file into its per-tensor layout with
[`v3/yaqa/hf/unpack.py`](https://github.com/TheHeroBrine422/rad-exl3-mix/blob/main/v3/yaqa/hf/unpack.py), then
`ourq.py ... --hout <dir> --sigma-reg-out 0.01`.

**In a row-independent quantizer** (plain GPTQ / LDLQ with no output-side rotation), scaling one output row's
objective by h_i does **not** change that row's rounding. The diagonal then acts only where rows interact: scale or
codebook search shared across rows, per-row / per-slab bit allocation (row importance), mixed precision between
tensors, or an output-side rotation added on purpose. Per-tensor sums of h (with the input-side error) are also a
usable sensitivity for knapsack-style allocation.
License: Apache-2.0 (statistics derived from the model's outputs on Apache-2.0 / permissively licensed text).
