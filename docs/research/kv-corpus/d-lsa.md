# D-LSA — Latent Sparse Attention, distilled (Prefrontal AI, 2026) · arXiv:2610.42147 — **evaluated on Qwen3.8-27B**

**Idea.** Sub-quadratic indexing for ultra-long context: project keys into a small learned **index space**, pool
4:1, cluster them hierarchically into buckets (Ward agglomerative), and let each query score **bucket centroids**
instead of every key; attend to the selected long-range keys plus sink and window. **D-LSA** distils the index
projections from a pre-trained model (frozen); N-LSA trains natively; a **Training-Free** variant uses the keys'
leading principal directions as projections.

**Method.** Per global-attention layer: index keys (projection + L2 norm), pooled, clustered into buckets of a
cluster cache; per query: sum over query heads → index query, top-p buckets (flat or tree search), expand to keys,
budget k long-range keys. Distillation: KL between the model's attention restricted to its top-2048 keys
(renormalised) and the indexer softmax (temperature calibrated per layer); **352M tokens** (16K–1M-token documents,
48K steps). Qwen3.8-27B settings: k = 4096, p = 1024 buckets, sink 4, window 512, d_idx = 64, n_pool 4, 512-key
clusters. Attention kernel unchanged (reads the index list as a page table). SGLang fork, H200 / B300.

**Numbers.** Qwen3.8-27B: **RULER within 5 points of full attention up to 1M tokens; at 10M D-LSA 63.9 vs full 59.9;
PaperTrail 29.1 vs 19.8**; knowledge / reasoning / LongBench v2 within the 95 % CI. Single-needle ≥ 95 to 5M for all
methods; MultiKey-2/3 and CWE ≤ 5 at 10M for every method incl. full. One layer at 10M: **72.4× prefill, 42.5×
decode vs full attention**. Whole model, H200 TP8 + DCP8 (table): full TTFT 21.7 / 65.3 / 4582 s and TPOT 7.8 / 9.1 /
32.1 ms at 512K / 1M / 10M; sparse variants TTFT 1.0–9.2× faster, **TPOT 0.87–1.00× at 512K–1M** (no decode gain
there: weights dominate) and up to 2.87× at 10M. Not-in-corpus abstention at 512K: 96.3–98.1 vs 100 (full).

**Runtime cost.** Indexer projections (small), clustering at prefill (amortised over decode), bucket cache memory
(much smaller than one index key per 4 keys); distillation run per model (352M tokens; Training-Free needs none).

**Failure modes.** Tier-3 (multi-hop) PaperTrail ≤ 19 at 10M for all methods; Tree search loses MultiKey/VT on
GLM; decode speedups only appear far beyond the 27B's native 262K.

**New vs #416.** A sparse-attention retrofit for the **dense** 27B (QSA exists only in Flash-Next), with a
training-free fallback — the prerequisite for using #412 window residency on the 27B. Distillation cost (352M
tokens) is within our means.

**Radiance fit.** Indexer → select → paged attention is radiance's QSA pipeline shape; bucket centroids + index
table are new KV groups; then #412 can page the unselected middle to host. Bytes/token on card would drop to
block/index keys + sink/tail, as on Flash-Next.
