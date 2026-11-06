# DuoAttention (2024) · arXiv:2410.10819

**Idea.** Only some heads are **retrieval heads** that need the full context; the rest are **streaming heads** that
attend to sinks + recent tokens. Give retrieval heads a full KV cache and streaming heads a constant-size one.

**Method.** A trainable gate α ∈ [0, 1] per KV head (initialised at 1) blends full and sink+recent attention; optimised on
synthetic passkey-style data (model frozen) to keep outputs close to full attention while pushing gates toward
streaming; binarised by threshold to pick retrieval heads. Heads reordered so each class is contiguous; chunked prefill keeps streaming heads' cache constant.

**Numbers.** Memory −2.55× (MHA) / −1.67× (GQA); decode up to 2.18× / 1.50×; prefill 1.73× / 1.63×. Llama-3-8B-
Instruct-1048K LongBench avg (Table 3) at 50 % budget: **Duo 40.21 vs full 40.08**, H2O 35.76, StreamingLLM 32.26,
TOVA 35.55. Short benchmarks at 50 % (Fig. 8): MBPP 47.09 vs full 47.85 (H2O 32.12, StreamingLLM 5.57); MT-Bench 9.14
vs 8.93. With quantization: Llama-3-8B decoding with **3.3M context on one A100**.

**Runtime cost.** Per-model optimisation (synthetic data, hours on GPUs); two attention paths (dense paged for
retrieval heads, ring buffer for streaming heads).

**Failure modes.** Head roles calibrated on synthetic retrieval data; GQA limits the gain (one KV head serves 6 query
heads on the 27B — a KV head is "streaming" only if all its query heads are). Agentic/multi-turn generalisation not
tested. KVzip (arXiv:2505.23416) compares favourably in multi-query reuse.

**New vs #416.** **Per-head (not per-layer) treatment** — #416's recipe granularity is the layer. A precision analogue
("retrieval KV heads at 8 bits, streaming heads at 4 bits or windowed") is untested anywhere we found.

**Radiance fit.** The 27B has only 4 KV heads per layer × 16 layers = 64 KV heads total: a per-head classification
is cheap to measure. Window-only heads need a ring-buffer KV group (radiance has windowed layers for Gemma); per-head
precision needs per-head layouts inside one layer's page (two KV groups split by head).
