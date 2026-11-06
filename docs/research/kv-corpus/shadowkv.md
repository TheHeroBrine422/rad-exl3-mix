# ShadowKV (ByteDance Seed, 2024) · arXiv:2410.21465

**Idea.** For high-throughput long context: keep a **low-rank pre-RoPE key cache** on the GPU (keys are low-rank
before RoPE), **offload values to CPU**, and per decode step select a tiny sparse budget of chunks (1.56 %) via
**landmarks** (mean post-RoPE key of each 8-token chunk), reconstruct those keys and fetch those values.

**Method.** Prefill: SVD of pre-RoPE keys per sequence (pre-RoPE keys are the most low-rank, Fig. 1; >6× compressible without needle loss, Fig. 5), chunk
landmarks, and **outlier chunks** (0.2–0.3 % of chunks whose keys are far from their mean) kept exact on GPU. Decode:
q · landmarks → top chunks (budget 1.56 % of context) + outliers + recent → reconstruct K (low rank + RoPE), gather V
from CPU (overlapped), exact attention on the selection.

**Numbers.** Llama-3-8B-1M (Table 1), RULER avg: full 86.68, **ShadowKV 86.88**, Quest 82.03, InfiniGen 70.13, Loki
9.33; LongBench avg 39.86 vs 39.94. RULER 8K → 256K with MInference prefill (Table 2): full 89.92 → 74.57, ShadowKV
90.47 → 74.31. Throughput, A100 (Table 3): Llama-3-8B-1M at 122K: 80.77 tok/s (batch 4) → **239.51 (batch 24),
2.97×**; at 244K 40.37 → 119.01. Independent: xKV (arXiv:2503.18893 Table 2) RULER-64K ShadowKV 87.17 at 9.08×
effective, but **70.94 at 5.51× total** compression (N-MV 45.83, VT 47.29).

**Runtime cost.** Per-sequence SVD at prefill; CPU memory and PCIe traffic for values (budget-limited); landmark
scoring per step (q · n/8 landmarks).

**Failure modes.** Selection misses hurt variable tracking and multi-value retrieval when compression is pushed
(xKV's numbers above); relies on PCIe bandwidth for V.

**New vs #416.** A **dense-attention** analogue of #412's residency: on the 27B (no indexer), landmarks per 16-token
page could drive Quest/ShadowKV-style selection and let #412's pager move cold V pages to host memory. Low-rank keys
are optional; landmarks + paging are the transferable part.

**Radiance fit.** Landmark = one extra key vector (or min/max pair, as Quest) per page and KV head (1/16 of K bytes);
selection op + #412's pager. Needs a top-k over pages per decode step — exactly the QSA machinery (qsa_select,
qsa_page) without a trained indexer.
