# SnapKV (2024) · arXiv:2404.14469

**Idea.** Each head's important prompt tokens can be read off an **observation window at the end of the prompt**:
pool the attention of the last w prompt queries over the prefix, keep the top-k clustered positions per head, drop
the rest once, before generation.

**Method.** At the end of prefill: attention of the last w (e.g. 32) queries → per-head scores → 1-D max/avg pooling
(clusters neighbours) → keep top-k + the window. One-shot, fine-tuning-free; decode then runs on the reduced cache.

**Numbers.** LWM-Text-Chat-1M LongBench (Table 1): budget 1024 / 2048 / 4096 vs full: NarrativeQA 18.02 / 17.92 /
17.92 vs 18.18; **GovReport 19.79 / 21.6 / 25.34 vs 27.97**; PassageRetrieval 3.0–4.0 vs 3.5. Command-R NIAH score
9.819 vs 9.866 (−0.5 %); RAG end-to-end F1 −2.1 %. 3.6× decode speed and 8.2× memory efficiency at 16K inputs; 380K
tokens on one A100-80GB (HF). Independent: KVarN (arXiv:2606.03458 Table 5) Llama-3.1-8B line retrieval at 7× (≈2.3
bits equivalent): 77 % at 100 lines vs 100 % full and 100 % for 2-bit KVarN; TurboQuant paper NIAH 0.858 vs 0.997.

**Runtime cost.** One extra attention-score pass for the observation window at prefill end; reduced decode reads.

**Failure modes.** **Query-dependent**: tokens are chosen for the question present at prefill. Multi-turn or follow-up
questions about other parts fail (SCBench, arXiv:2412.10319); divergence studies show SnapKV at 50 % retention
already changes long generations (arXiv:2609.16617). Summarisation (GovReport) degrades even at 4K budget.

**New vs #416.** Eviction — rejected for Flash-Next by Code 000. For the 27B, its value would be as a *precision*
selector (keep selected pages at higher precision) rather than deletion.

**Radiance fit.** Poor as eviction (prefix-cache pages shared across conversations must not be pruned per question).
Page-granular version (keep or demote whole 16-token pages chosen by pooled window attention) fits #412-style tiers.
