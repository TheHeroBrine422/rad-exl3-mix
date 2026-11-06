# SCBench — SharedContextBench (2024) · arXiv:2412.10319 — evaluation, not a method

**Idea.** Real serving **reuses** KV caches (multi-turn sessions, shared documents across requests); evaluating a
compression method on single requests hides its failures. 12 tasks (string retrieval, semantic retrieval, global
information, multi-task) in two shared-context modes (multi-turn, multi-request), contexts ~65K–1.5M tokens (avg
input e.g. Retr.KV 125K, En.QA 198K).

**Numbers.** Llama-3.1-8B (Table 4) multi-turn avg / string retrieval: **full 48.7 / 57.1**; MInference (sparse
prefill, τ = 1/32) 42.5 / 39.1; Quest (sparse decode, 1/32) 22.1 / 8.2; SnapKV (1/32) 20.9 / 6.1; StreamingLLM 15.5 /
0.4; **KIVI (2-bit, τ = 1/8) 32.0 / 12.0**; CacheBlend 49.3 / 56.7. Retrieval by turn (Table 8): SnapKV scores 100 on
LongBench but **0 on every SCBench turn**; StreamingLLM 84 on LongBench, 0 on SCBench; full model 62–70 per turn.
Finding: sub-O(n) memory methods fail from the second query on; O(n)-memory methods (sparse prefill, quantization)
degrade gracefully.

**Runtime cost.** n/a (benchmark; long contexts make it expensive: hundreds of 100K+ prefill per model).

**Failure modes exposed.** Query-aware eviction (SnapKV, PyramidKV) collapses on follow-up questions; 2-bit KV
quantization loses most of string retrieval at ~125K (12.0 vs 57.1) even though it keeps O(n) memory.

**New vs #416.** #416 measured on long / recall / multi-round corpora with 4 documents each — SCBench-style
multi-turn string retrieval at 128K is the hard gate the BACKLOG asks for (RULER-style probes at finalists).

**Radiance fit.** As a judge component: shared-context multi-turn mode maps onto radiance's prefix cache (and #412
residency) — prefill once, ask several questions; deterministic string-retrieval scoring (Retr.KV, Prefix-Suffix,
MultiHop) is cheap to grade.
