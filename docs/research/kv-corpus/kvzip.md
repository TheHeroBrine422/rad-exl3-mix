# KVzip (2025) · arXiv:2505.23416

**Idea.** **Query-agnostic** eviction: score each KV pair by how much the model needs it to *reconstruct the
context itself* ("Repeat the previous context:"), then evict low scorers once; the compressed cache is reusable
across any later queries (unlike SnapKV / PyramidKV, which select for the current query).

**Method.** Teacher-forced pass of the context-repeat prompt over the cached context; importance = max attention a
KV pair receives during reconstruction (chunked scoring to bound memory); per-head non-uniform budgets; evict to a
target ratio. Also a context-independent (per-head, model-level) variant.

**Numbers.** 3–4× KV reduction and ~2× FlashAttention decode latency with "negligible" loss on QA, retrieval,
reasoning and code, contexts up to 170K (LLaMA3.1, Qwen2.5, Gemma3). Qwen2.5-7B SQuAD at 30 % ratio: 94.37 % vs
SnapKV 32.15 % (Table 2). Query-aware baselines degrade even at 90 % budget in multi-query use. Compression costs ≈ 2×
a standard prefill, < 2 % extra memory. Independent: KVarN (arXiv:2606.03458 Table 5) Llama-3.1-8B line retrieval at
7×: 80 % at 100 lines vs 100 % for 2-bit KVarN at equal bytes.

**Runtime cost.** One extra prefill-sized pass per context (amortised when the context is reused, e.g. prefix cache).

**Failure modes.** **Behaviour change:** with 40 % of the cache, LLaMA3.1-8B **reveals a phone number it refused to
give with the full cache** (Table 1, DecodingTrust) — eviction can remove safety-relevant context, a tail no accuracy
metric shows. Still eviction: irreversible.

**New vs #416.** A query-agnostic importance score computed by the model itself; Code 000's importance-driven demotion
(selection counts) was no better than a position hash on QSA — reconstruction-based scores were not tried.

**Radiance fit.** As an offline/one-shot **precision or demotion score for stored prefixes** (system prompts, shared
documents) — fits #412 tiers and the prefix cache. As deletion: conflicts with page hashing.
