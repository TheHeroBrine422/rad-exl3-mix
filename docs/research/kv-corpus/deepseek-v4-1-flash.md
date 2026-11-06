# DeepSeek-V4.1-Flash (DeepSeek-AI, 2026) · arXiv:2609.19969 (architecture + serving)

**Idea.** Push KV bytes down at the architecture level: Compressed Sparse Attention 2 (CSA2) with **cross-layer reuse
of main KV and Top-K indices**, a hierarchical sparse indexer, **FP4 main KV via QAT**, and SWA Bounded Replay for
persistent (host/SSD) caches; global KV **890 bytes/token** (~¼ of V4-Flash, 437× below DeepSeek-V1), 552B MoE,
1M context.

**Method.** Each CSA2 layer is statically Full (own main KV + indexer + Top-K), Reindex (reuses a preceding layer's
main KV and indexer K, recomputes Top-K with its own query) or Reuse (reuses main KV *and* Top-K indices). Main KV
stored as **E2M1 with one E4M3 scale per 16 channels** (NVFP4 without the global scale; range ±2688 vs a bound of
≈22.6 after RMSNorm), learned by QAT in post-training; quantized *after* RoPE (pre-RoPE gave only a marginal gain);
**SWA (local window) KV kept in FP8 "due to its sensitivity to quantization"**. Dequantize before attention, so no FP4
matmul hardware is required.

**Numbers.** Global KV 890 B/token; persistent KV ~⅛ of V4-Flash with SWA Bounded Replay; decode FLOPs grow only ¼
from 4K to 1M context. Quality: GPQA Diamond 90.9, Codeforces 3471, Terminal-Bench v2.1 up to 90.6 (scaffold-
dependent) — model-level results, not KV ablations.

**Runtime cost.** All of it is trained in (QAT, attention-sharing training for CSA2). Inference: 15 / 11 kernels per
Reuse-mode layer (prefill / decode).

**Failure modes.** Not separable from training; the paper notes FP4 was chosen for portability "despite the higher
accuracy of alternative formats in our experiments", and that the recent window needed FP8.

**New vs #416.** Two transferable facts: (1) a **recent window kept at higher precision than the body** (production
evidence for SKVQ/KIVI-style windows); (2) 4-bit KV is shipped only with **QAT** at frontier scale. Cross-layer index
reuse: Code 000 measured selection reuse on Flash-Next (not trained for it) as ΔNLL +0.014 / +0.026 (long / recall) —
works only when trained.

**Radiance fit.** FP4-with-E4M3-scale storage decodes on gfx1201 (Code 000 emulated it: slightly worse than int4,
IKV Table 9). Cross-layer reuse needs a model trained for it.
