# PyramidKV (2024) · arXiv:2406.02069

**Idea.** Attention is broad in lower layers and funnels onto a few tokens in upper layers ("pyramidal information
funneling"), so give **lower layers larger KV budgets and upper layers smaller ones**; select tokens SnapKV-style.

**Method.** Arithmetic budget schedule from bottom to top layers (hyper-parameters α, β), per-layer top-k by
observation-window attention.

**Numbers.** LongBench Llama-3-8B-Instruct full avg 41.46 (Table 1); keeps full-cache performance at 12 % of the KV;
at 0.7 % (KV size 64) beats H2O / SnapKV / StreamingLLM, up to +20.5 points on TREC. Memory table (Llama-3-8B):
512-token cache 428 MB (6.3 %): QMSum 22.80 vs 23.30, TREC 71.50 vs 73.00, PRe 69.50 vs 69.25. NIAH: LLaMA-3-70B
100.0 with 128 entries. Independent: KVarN (arXiv:2606.03458 Table 5) Llama-3.1-8B line retrieval at 7×: **61 % at 100
lines** (full 100 %); TurboQuant paper NIAH 0.895 vs 0.997 (arXiv:2504.19874).

**Runtime cost.** As SnapKV; per-layer budgets → ragged layer sizes.

**Failure modes.** Same query dependence as SnapKV; low-retention line retrieval failures at short contexts.

**New vs #416.** The **depth profile** matches Code 000's finding that early layers need more precision ("one early
layer quantised costs as much as all"; first QSA layer ~170× the KL cost of the last). PyramidKV allocates *tokens*
by depth; #416 allocates *bits* by depth — same lesson.

**Radiance fit.** Eviction part: poor. Depth-dependent budget idea: already in #416 recipes.
