# [QC] entry notes — EXL3 with fixed-code QAD, and allocation mixtures

Status: **WIP**. Checkpoint, trained scales and the full public measurement are still being prepared.
Everything below is either measured or explicitly marked as a projection; nothing here is a claim of a
finished entry.

## 1. The base quant we start from

`turboderp/Qwen3.8-27B-exl3`, branch `SC_4.00bpw_H5` (revision `4acd9ad5`): EXL3 4-bit, `mul1` codebook,
`h128` rotation, 5-bit head. Text model only — 401 EXL3 linears + 450 plain tensors.

Canonical payload as imported into a `.rad` container: **15,566,829,568 B = 4.63023 bpw** over the
challenge denominator 26,895,998,464. The vision tower (`model.visual.*`) and MTP (`mtp.*`) are excluded
by design — this entry is a text model, no multimodal input, no speculative decoding.

## 2. What this entry actually changes: fixed-code QAD

The EXL3 trellis codes are **not** retrained. What is trained:

- the per-linear `svh` output scales (one f16 value per output row), and
- the RMSNorm weights,

both held as FP32 masters with a straight-through estimator to the checkpoint dtype, so a step never
moves the exported value more than the dtype allows. The loss is the competition metric itself
(the 513-cell lower-bound KL) computed in chunks through the full-vocabulary head.

Measured on `pilot_val`, window 512, seed 20261027, LR 5e-4, 20 updates:

| trained scope | val KL before → after | gain |
|---|---|---|
| layers 60–63 (`svh` + norms) | 0.101719 → 0.093154 | **−8.4 %** (bootstrap CI [5.96, 10.09] %) |
| layers 48–63 (`svh` + norms) | 0.101938 → 0.079873 | **−21.6 %** |
| all 64 layers | 0.101723 → 0.099036 | −2.6 % (worse — layers 0–35 sit in the frozen native prefix, their scales get no useful gradient) |

So the useful scope is the last 16 layers. The 16-layer gain has no bootstrap CI recorded yet; that is
one of the open items.

## 3. Allocation: which layers deserve more bits

Separate from QAD we measured, on the same metric, which layer groups profit most from swapping in a
higher-bit-rate donor (5 bpw, revision `dfe2f4fd`):

| upgrade | KL | delta vs base |
|---|---:|---:|
| GDN `linear_attn` L32–39 → 5 bpw | 0.294474 | −3.8 % (47/54 docs improved) |
| MLP L32–47 → 5 bpw | 0.248260 | −18.9 % (52/54 docs improved) |

These are two single-knob measurements, not a combined artifact; additivity is unproven.

## 4. The mixture this entry intends to ship

66 linears / 198 donor tensors: MLP L32–47 (48 linears) + GDN `linear_attn` L32–39 (18; L35/L39 are
full-attention layers in this model and stay). The donor is nominally 5 bpw but is an average, so the
selection is 54 × K5 + 12 × K6.

Byte accounting (canonical payload, TP-invariant):

- base + donors, before any embedding change: **16,330,323,968 B**
- with the token embedding re-encoded as E4M3 with a single F32 scale:
  **15,058,925,572 B ≈ 4.47916 bpw**

That second number is a **projection**, not a produced container: the embedding layout has not been
validated inside a real container yet.

The embedding experiment is measured as a simulation only: a single global E4M3 scale on the embedding
costs **+0.28 % KL** (0.306872 vs 0.306028 on the pilot split) for a theoretical 1.184 GiB saving.

## 5. The plugin in this PR

- `plugin/exl3dense` — the kernel library. The trellis is decoded **inside** the GEMM instead of
  materialising BF16 weights, reproducing exllamav3's rounding order: `x = f16(a·s)` before the
  rotation, `f16(had(x))` before the GEMM, fp32 accumulation over k ascending, `had(acc)·svh` in f32 and
  one RNE bf16 store. Measured against `exllamav3_ext.exl3_gemv` over 38 cases (real K3–K6 blocks, TP
  slices, all model shapes including the 248,320 × 5,120 head): worst **1.44e-4** against the native
  result rounded to bf16, i.e. the arithmetic matches; the remaining difference to the native f16
  output is the mandated bf16 store of the surrounding graph.
- `plugin/qwen35_exl3` — the architecture, derived from the stock `GdnBF16` / `AttnGatedBF16` /
  `MlpBF16` blocks so convolution, chunk scan, recurrence, RoPE, q/k-norm and gating stay engine code.
  Only the projection declarations are replaced.
- One engine limitation forced a small extra op: `gemm_nt` reads weight bytes in the dtype of the call,
  and this model's `in_proj_a`/`in_proj_b` are F16 while the graph is bf16. Declaring them F32 or F16
  either zeroes the output or makes the whole layer host-only. `exl3dense_gemm_f16` therefore does those
  two tiny projections itself (f16 weight, bf16 activation, fp32 accumulation).

## 6. Where the numbers stand

`models/rad/base-exl3.rad` loads and serves on gfx1201 with this plugin (TP1, bf16 KV).

| measurement | KL |
|---|---:|
| 3 calib documents (`golden/calib`) | **0.007405** [0.005323, 0.008359] |
| 5 public documents (`golden/public`) in this engine | **0.017907** [0.005838, 0.038030] |
| the same container in the exllamav3 reference harness, same 5 documents | 0.017443 |
| 60 public documents | **running** |

The engine agrees with the reference harness to 2.7 % on identical weights and documents, so the
plugin is not the limiting factor — the quantisation is.

Against the current frontier (UD-Q4_K_S, 4.461 bpw → 0.0106) this base container is roughly 60 % above
the bar. The entry becomes competitive only with the QAD scales trained on the mixture above, which is
exactly what the open items below are about.

## 7. Open items

1. Train the `svh` + norm scales on the **mixture** (not the base), then re-measure.
2. Produce the actual mixture container with the E4M3 embedding and validate its real byte count.
3. Upload the checkpoint to Hugging Face and link it here.
4. Record a bootstrap CI for the 16-layer training gain and measure the 60-document public set to a
   closed interval.
5. Report the GEMM speed on a real layer of the checkpoint, as the competition asks.
