# STEPQuant (2026) · arXiv:2609.38169 — GDN recurrent-state quantization, **Qwen3.8-27B results**

**Why in a KV corpus.** On the 27B, 48 of 64 layers keep a fixed-size Gated-DeltaNet state instead of a KV cache
(48 value heads × 128 × 128 per layer; ≈ 151 MB per sequence in f32, our arithmetic). For many concurrent short or
medium sessions it rivals the attention KV, and its errors compound over decode like KV errors.

**Idea.** Recurrent-state error matters along two axes: **temporal** (errors in long-lived memory — slow-decay
heads — persist over many steps) and **spatial** (key rows differ in output impact; magnitudes vary by row and
column). Allocate bits by error × lifetime; fit row (key) and column (value) scales jointly, weighted by row impact.

**Method.** Lifetime-aware mixed-precision allocation over state units (with "pivot" units kept high), key-row-aware
dual-axis scale fitting from calibration statistics; quantized write-back after each delta-rule update; SGLang
kernels.

**Numbers.** Qwen3.8-27B, BF16 weights, long-generation avg of 7 tasks (Table 1): FP32 state 80.60; **INT8 71.86;
INT6 45.04; INT4 12.73**; STEPQuant@6-bit 80.59; @4-bit 80.51. Short tasks (Table 2): FP32 87.78; INT8 86.25; INT6
82.49; STEPQuant@4-bit 87.63 — **uniform INT8 costs 1.5 points on short tasks but 8.7 on long generation**. With
4-bit AWQ weights (Table 3): FP32 79.32, STEPQuant@6 79.27, @4 78.98. Ablation @6-bit avg (AIME/GPQA/LCB): INT6 38.63,
spatial only 80.19, temporal only 75.63, full 84.51 (= FP32 84.61). 6-bit: 5.03× state compression, total serving
memory −68.7 % (Qwen).

**Runtime cost.** Calibration (statistics + allocation); per-step quantize/dequantize of the state in the GDN kernel.

**Failure modes.** Uniform low-bit states fail on long generations while short benchmarks hide it — the same
tail-vs-mean pattern as KV.

**New vs #416.** #416 added an f16 GDN state for Flash-Next (54 → 27 MiB per sequence and card; ~+0.0007 KL). Sub-8-bit
states with lifetime-aware allocation are new; for the 27B the delta-net state is the other per-sequence memory term.

**Radiance fit.** The state is per sequence (checkpointed for prefix cache / spec rollback); a packed state format
needs GDN-kernel changes on the read and write-back side, not the paged KV path.
