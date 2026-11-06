# Divergence Timing and Cumulative Disagreement under KV-Cache Eviction (2026) · arXiv:2609.16617 — evaluation

**Idea.** Measure *when* a compressed-cache generation first diverges from the full-cache one and how much it
disagrees afterwards: expected mismatch fraction = first-mismatch contribution + post-divergence exposure × its
mismatch rate (exact under a stepwise maximal coupling); residual-branch conditional Monte Carlo gives unbiased,
lower-variance estimates of occurrence, occupation and window/tail contributions.

**Numbers.** Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, full trajectories, 288 documents. SnapKV at 50 %
retention enters divergence later and less often than SnapKV-512 or recent-only retention at the same budget, but
post-divergence total variation stays high; **post-divergence exposure accounts for 85–90 % of four aggregate
mismatch gaps**. At 90 % retention, the paper's S_.99 TV endpoint (definition in the paper):
Llama SnapKV-90 28.44 %, recent-90 56.18 %; Qwen 43.63 % / 73.65 % (table). Branch-aligned TV higher in the late
than the early window for both models at 90 % retention.

**Runtime cost.** Paired rollouts (full vs compressed) — generation-time evaluation, more expensive than
teacher-forced KL.

**Failure modes exposed.** Teacher-forced metrics (per-position KL on a fixed text) measure the *first-mismatch*
term only; most of the cumulative disagreement comes after divergence, which only free-running generation shows —
and late windows are worse than early ones even at 90 % retention.

**New vs #416.** #416 and rad-autoquant score teacher-forced KL; IKV mentions "divergence timing of long greedy
generations against a bf16 noise baseline" only as future work. This paper is the method to do it with confidence
intervals.

**Radiance fit.** Judge-side: greedy (or fixed-seed) paired generations from the engine with and without a candidate
layout, first-divergence position and post-divergence token mismatch per document, against a bf16-vs-bf16 (other
chunking) noise baseline. Applies to quantization as much as to eviction.
