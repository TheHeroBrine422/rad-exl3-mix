# MiKV — "No Token Left Behind" (2024) · arXiv:2402.18096

**Idea.** Eviction (H2O-style) destroys context (safety breaches, hallucinated details); instead keep "evicted"
tokens at **low precision** (INT2–4) and important tokens at higher precision.

**Method.** H2O importance (accumulated attention) splits tokens into an importance cache (INT4–8 or FP16) and a
retained cache (INT2–3). Low-bit retained cache uses a per-channel query-key outlier **balancer** (computed at
prefill, applied as element-wise scale to q and k), group size = head_dim / 2 to avoid RoPE artifacts. Decode uses
weight-only-quant GEMM kernels (KV treated like quantized weights) instead of batch-GEMV.

**Numbers.** Line retrieval, H2O importance ratio 50 % (Table 1): evicted → 43.2 % accuracy; retained in INT4 100 %,
INT3 99.8 %, INT2 84.6 %. At 25 %: INT4 100 %, INT2 68.0 %. Outlier balancer at INT2, 20 % importance (Table 2):
64.0 % → **92.6 %** (33 % cache). Importance cache INT4 + retained INT2: 92.0 % at 18 % cache size; both INT2: 65.0 %
(Table 3). MMLU Llama-2-7b at 20 % cache: 42.7 vs 44.0 (Table 5). AlpacaEval Llama-2-70b-chat at 20 %: 48.6 % win
rate vs full cache (Table 4).

**Runtime cost.** Importance tracking (accumulated attention scores need explicit attention probabilities — not
FlashAttention-compatible), per-token precision bookkeeping, balancer from prefill statistics.

**Failure modes.** INT2 everywhere collapses retrieval (65 %); importance estimates from H2O inherit its query
bias; ZipCache (arXiv:2405.14256, Table A) reports MiKV 90 % vs ZipCache 100 % on 200-line retrieval at 4.43×.

**New vs #416.** Key lesson for any eviction idea: **never drop, demote** — quantizing "unimportant" tokens instead
of evicting recovers 43 % → 100 % retrieval. The q/k balancer (SmoothQuant on the q·k product) is a zero-bit
transform #416 lacks, but on Flash-Next static key channel scaling after Hadamard gave no gain (IKV Table 9).

**Radiance fit.** Same as ZipCache: per-token precision → per-page precision (two groups + layout tag). The balancer
is a per-(layer, head, channel) scale that can be folded into q/k projections only if static; MiKV's is
per-prompt (computed at prefill).
