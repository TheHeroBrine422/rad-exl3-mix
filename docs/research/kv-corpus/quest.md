# Quest (ICML 2024) · arXiv:2406.10774

**Idea.** Token criticality depends on the current query, so don't evict — **select pages per query**: keep, per
KV page, the element-wise **min and max of its keys**; upper-bound each page's q·k by Σᵢ max(qᵢ·minᵢ, qᵢ·maxᵢ); load
only the top-K pages for attention. All KV stays resident (O(n) memory), reads drop to the budget.

**Method.** Page metadata = 2 × head_dim values per page and KV head; score = channel-wise upper bound; top-K pages;
exact attention over them. First two layers kept dense (low sparsity there). FlashInfer kernels, page size 16.

**Numbers.** Passkey retrieval (Table, budget in tokens): at 10K context, budget 32 / 64 / 128 / 256 / 512 → **Quest
65 / 99 / 99 / 99 / 100 %**, H2O 0 / 1 / 1 / 1 / 3 %, TOVA ≤ 8 %, StreamingLLM ≤ 5 %; at 100K, budget 256 → 4096:
Quest 88 → 100 %, H2O 2–4 %. Up to 7.03× self-attention speedup, 2.23× end-to-end latency. Independent: RULER
(Llama-3-8B-1M) 82.03 vs full 86.68 (ShadowKV, arXiv:2410.21465); NIAH/GSM8K fine at 4× in the DMS comparison
(arXiv:2506.05345 Table 1: NIAH 95.8); **SCBench multi-turn 22.1 vs 48.7, string retrieval 8.2 vs 57.1** at 1/32
(arXiv:2412.10319) — fails when the budget is small and questions change.

**Runtime cost.** Metadata write per page (min/max reduce, 2/16 of K's bytes at page 16); per step q · metadata
over n/16 pages + top-K + gather. Prefill unchanged (dense).

**Failure modes.** Upper bounds loosen for diffuse pages (many mid-scored tokens); small budgets and multi-query
sessions; GQA: the selection must be shared by a KV head's query group (6 query heads on the 27B) or taken per query
head with more reads.

**New vs #416.** The closest **training-free** sparse decode for a dense-attention model like the 27B (QSA needs a
trained indexer). Page-granular, using exactly radiance's 16-token pages.

**Radiance fit.** Good: min/max (or mean, as ShadowKV) key metadata as an extra small KV group per page; selection
via the QSA-style select op; attention kernel reads a page list. Combine with #412 residency to page the unselected
middle to host memory (Quest keeps all pages resident).
