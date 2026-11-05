---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
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

| path | content |
|---|---|
| `block-normalized/` | **the recommended estimator**: Σdy² with each 128-sequence block normalized before averaging (tames a heavy tail of a few huge blocks in early layers), from 1,920 sequences; `mkbn.json` = per-tensor block statistics |
| `snapshots/nNNNN/dy/` | Σ_t dy_t² (plain output Fisher diagonal), mean over the first NNNN sequences |
| `snapshots/nNNNN/xw/` | Σ_t ‖x_t‖² dy_t² (YAQA Sketch A's first-step weighting with H_in ≈ I) |
| `snapshots/nNNNN/sketchb/` | diag(Σ_b G_b G_bᵀ), G_b = Σ_t dy_t x_tᵀ per sequence (YAQA Sketch B's output factor, diagonal); `halves.json` = per-tensor cosine between the even- and odd-sequence means |

Snapshots every 128 sequences, 128 … 1,920. One file per linear, named by its checkpoint key
(`model.language_model.layers.L.<module>.safetensors`), one fp32 tensor `"hout_diag"` of length out_features (the mean
over sequences). gate_proj / up_proj are separate files. Only relative values matter for quantization (our quantizer
rescales each diagonal by its mean).

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

## Result

TODO: swap-test result (#33-28: K2.5 V2-3.00-based mix, base vs plain Σdy² vs block-normalized, paired KL on a
search-pool slice).

## Use

With our quantizer: `ourq.py ... --hout <dir> --sigma-reg-out 0.01` (any of the directories above).
License: Apache-2.0 (statistics derived from the model's outputs on Apache-2.0 / permissively licensed text).
