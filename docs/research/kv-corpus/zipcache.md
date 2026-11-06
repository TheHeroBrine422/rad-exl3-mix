# ZipCache (2024) · arXiv:2405.14256

**Idea.** Mixed precision **per token**: salient tokens 4-bit, the rest 2-bit, with saliency = attention *normalized*
by how many queries could attend to a token (removes the causal-mask bias toward early tokens), estimated cheaply
from a probe subset of queries so FlashAttention can still be used.

**Method.** Channel-separable token-wise quantization (per-channel K normalisation then per-token scales: 3hd + 2bl
parameters, Table 1) as the base; saliency from 10 % probe queries (5 % recent + 5 % random) through explicit
attention; top-x % tokens to the high bit width; re-done at prefill and periodically in decode.

**Numbers.** LLaMA3-8B GSM8k 4-bit granularity (Table 1): group-wise 54.51 (3.2×), token-wise 49.81, channel-sep.
token-wise **54.74 at 4.00×** vs FP16 55.88. Mistral-7B GSM8k CoT (Table 3): FP16 41.62; H2O (40 % kept) 1.67;
GEAR 4/4 39.42 (3.00×); KIVI 16/2 39.04 (3.46×); MiKV 4/2 36.32 (4.98×); **ZipCache 4/2 41.24 (4.98×)**. LLaMA3-8B
200-line retrieval (Table A): ZipCache 100 % at 4.43×, MiKV 90 %, H2O 0 %; prefill latency 2584 ms vs FP16 2340, GEAR
5958, KIVI 4010. LLaMA3-8B at 4096 input: −37.3 % prefill latency, −56.9 % decode latency, −19.8 % memory (vs MiKV-style
baselines).

**Runtime cost.** A probe attention pass that materialises scores for 10 % of queries during prefill; periodic
re-ranking in decode; two precisions in the cache (index or permutation per token).

**Failure modes.** Saliency is query-dependent: tokens judged unimportant at prefill can matter for a later
question (the multi-turn failure of SCBench, arXiv:2412.10319); tested at ≤ 4K context and 200-line retrieval only.

**New vs #416.** Precision chosen per token by importance (#416 is per layer, uniform over tokens); normalized
attention as the saliency signal. Code 000 found importance signals no better than a position hash for *demotion*
(IKV Table 9) — a warning for any saliency-driven per-token scheme on QSA; unmeasured on dense attention.

**Radiance fit.** Per-token mixed bit widths do not fit 16-token pages directly; a page-level version (a page's
precision chosen when it is written, from a probe) fits as two KV groups with a per-page layout tag — the same
machinery as #412's tiered precision. Requires a probe-attention op in prefill.
