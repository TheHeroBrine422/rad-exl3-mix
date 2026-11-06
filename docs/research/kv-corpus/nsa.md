# NSA — Native Sparse Attention (DeepSeek, 2025) · arXiv:2502.11089

**Idea.** Train sparsity in from the start with a hardware-aligned hierarchy: **compressed** coarse tokens (block
summaries), **selected** fine blocks (top-n by compressed-attention scores), and a **sliding window**, combined by
learned gates; kernels designed around GQA groups sharing the same selected blocks.

**Method.** Three branches per query: compression (MLP-pooled blocks), selection (blocks scored from the compressed
branch's attention, shared across the GQA group), sliding window; gated sum. Pre-trained end to end (27B-parameter
MoE-style model, 260B tokens in the paper).

**Numbers.** 27B model general benchmarks avg (Table 1): NSA 0.456 vs full 0.443. LongBench avg: **NSA 0.469** vs full
0.437, Exact-Top 0.423, Quest 0.392, InfLLM 0.383, H2O 0.303. Distilled reasoning (AIME, Table 3): NSA-R 0.121 /
0.146 vs full 0.046 / 0.092 at 8K / 16K generation limit. Expected memory-access speedup in decode: 4× at 8K → 11.6×
at 64K (KV tokens loaded 2048 → 5632 vs 8192 → 65536).

**Runtime cost.** Trained architecture; inference reads compressed tokens + selected blocks + window.

**Failure modes.** Requires pre-training (or long continued training); per-head-group selection is a design
constraint, not a retrofit.

**New vs #416.** Context for QSA-class indexers (Flash-Next's QSA is a trained indexer + 4-token blocks). Its
evaluation table shows **training-free Quest at 0.392 vs full 0.437** on LongBench — the gap a retrofit (D-LSA,
indexer distillation) must close.

**Radiance fit.** n/a for the 27B (would need training); radiance's QSA family is the in-engine analogue.
