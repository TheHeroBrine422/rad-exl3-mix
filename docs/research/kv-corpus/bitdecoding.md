# BitDecoding (2025) · arXiv:2503.18773 — kernels, not a quantizer

**Idea.** Low-bit KV only pays off if the decode kernel uses the matrix units: pick a packed layout that dequantizes
straight into tensor-core fragments, overlap dequant (vector ALUs) with MMA, and pack the GQA query group into the
M dimension so a decode step fills a tile.

**Method.** (i) Layout induced from the MMA fragment layout (ldmatrix order) so a thread's packed int4/int2 lanes
dequantize to the registers it feeds; (ii) warp-level dequant parallelism + cooperative softmax across warps;
(iii) **query transformation**: [1, (g_q, h_kv)] → [g_q, h_kv] so the g_q query heads sharing a KV head form one tile;
(iv) residual fp16 kernel for the tail (KIVI-style), packing kernel; (v) Hopper wgmma and Blackwell native NVFP4/MXFP4
paths. Tensor-wise (per-token) and channel-wise scales.

**Numbers.** Llama-3.1-8B-Instruct, 32K (Table I): throughput FP16 49.25 → INT4 147.21 (2.98×) → INT2 209.48
(4.25×); LongBench 48.25 / 48.16 / 47.38. Kernel: avg 7.5× over FP16 FlashDecoding-v2, up to 8.6× on Blackwell with
NVFP4, up to 4.3× over prior low-bit systems; **single-batch decode at 128K: 3× lower latency (3.3× vs non-fused
attention)**. Quantize+pack latency at 128K (Table II): 0.0599 ms prefill / 0.008 ms decode vs Marlin 58.02 / 0.41.

**Runtime cost.** None beyond the kernels; channel-wise K needs a residual tail.

**Failure modes.** Quality is whatever the quantizer gives (INT2 −2.7 % LongBench). NVFP4 gains are Blackwell-only
(no FP4 on gfx1201).

**New vs #416.** Nothing new as a quantizer; the kernel lessons apply to the 27B, where (unlike QSA) decode reads the
whole cache: at 128K × 16 layers × 4 KV heads, fp8 is 4 GiB per sequence, i.e. ~2 GiB per card at tp 2 next to
~9.5 GiB of MXFP4 weights per card — **≈17 % of decode bytes at 128K, ≈46 % at 512K** (our arithmetic). The q-group packing
(g_q = 6 on the 27B; × (1 + n_spec) with MTP/DFlash2 verify rows) is what makes a 16×16 WMMA tile worth using.

**Radiance fit.** Yes; radiance's `attn_decode_h256_gqa6_*` kernels are the place. On gfx1201 int4 unpacks to int8 for
`V_WMMA_I32_16X16X16_IU8`-class tiles only if q is also int8 (dynamic per-head q quantization); otherwise dequant to
f16 (the current path). An int8-q decode path halves ALU time but decode is bandwidth-bound, so it matters only at
large batch.
