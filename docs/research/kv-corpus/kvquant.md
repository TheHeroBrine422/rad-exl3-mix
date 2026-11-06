# KVQuant (2024) · arXiv:2401.18079

**Idea.** Make 3-bit (and 2-bit) KV usable for 1M–10M context: per-channel **pre-RoPE** keys, per-layer
sensitivity-weighted non-uniform codebooks, per-vector dense-and-sparse outliers, fp16 attention sink.

**Method.** (i) K per-channel, V per-token; (ii) K quantized before RoPE, RoPE applied on the fly after dequant;
(iii) `nuqX`: a per-layer X-bit non-uniform datatype from Fisher-weighted k-means on calibration activations
(offline), rescaled per vector; (iv) top 1 % outliers per vector kept sparse in fp16; (v) first token kept fp16.
Effective rate nuq4-1% 4.32–4.35, nuq3-1% 3.33, nuq2-1% 2.33 bits/value.

**Numbers.** Wikitext-2 PPL, LLaMA-7B (Table 1): fp16 5.68; int4 5.98; nf4 5.87; KVQuant-4bit 5.72; under 0.1 PPL
loss at 3-bit for LLaMA/Llama-2/3/Mistral. LongBench (LLaMA-2-7B-32K, avg bits at 12.2K): fp16 31.96, KVQuant-3bit-1%
(3.33 b) 31.21, KIVI-2 (3.17 b) 30.04; PassageRetrieval 23 / 19.5 / 13.58.
**RULER 32K** (LLaMA-2-7B-32K): fp16 56.40, **KVQuant-3bit 53.65**, KVQuant-2bit 36.54, KIVI-2 39.78; MKey3
11 / 6.4 / 0 / 0. Passkey to 32K: nuq2-1% ≥ 0.98, KIVI-2 0.68–0.76. Llama-2-70B-32K nuq4 = fp16 on passkey.
Memory: LLaMA-7B 1M context on one A100-80GB, 10M on 8 GPUs (2-bit).

**Runtime cost.** Offline calibration (Fisher-weighted k-means per layer). Online: per-vector outlier threshold
(top-k per token or channel) at store time, sparse CSR/CSC fp16 outlier matmul next to the dense LUT matvec, RoPE on
the fly for keys. CUDA kernels: key matvec 2K/4K/16K = 25.6/39.9/126.3 µs vs fp16 33.3/59.1/219.4 (A6000) → ~1.7×.

**Failure modes.** RULER multi-key and multi-value tasks are where even 3-bit loses (MValue 47.5 vs 55.95,
MKey2 61.6 vs 68.2); 2-bit loses half of RULER. Sparse outlier paths are irregular (bad for WMMA tiles). Pre-RoPE K
requires re-applying RoPE per decode step over the whole context.

**New vs #416.** Non-uniform per-layer codebooks (Lloyd/k-means with Fisher weights; #416 tested a TurboQuant-style
Lloyd–Max only with per-vector norm), pre-RoPE K (radiance stores post-RoPE; only 64 of 256 dims rotate on the 27B,
so pre-RoPE costs little there), sink token in fp16 (radiance has no sink exception in #416). Outlier retention was
measured as a near-null by Code 000.

**Radiance fit.** Non-uniform codes: yes per-token (LUT dequant into f16/int8 before WMMA). Sparse outliers: poor fit
(variable-length side lists per page). Pre-RoPE: needs RoPE in the attention kernel over the 64 rotary dims.
