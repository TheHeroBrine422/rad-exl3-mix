# CommVQ (ICML 2025) · arXiv:2506.18879

**Idea.** Additive vector quantization of KV with a light learned encoder, and a key codebook built to **commute with
RoPE**, so attention can be computed against codebook entries (q·codebook once per step) instead of decoding every
cached key.

**Method.** Encoder maps a K/V vector to R residual code indices; decode = sum of codebook rows (a matmul). Key
codebook entries are 2×2-block rotation-commuting (RoPE applies to the codebook, not the code), trained with EM on
FineWeb-Edu; value codebook N_c = 1024 (1 b) / 2048 (2 b) rows, R = 11 / 21 residual stages for keys (Table 7),
codebooks 2.0–5.25 MB per model (Llama-3.1-8B).

**Numbers.** GSM8K Llama-3.1-8B-Instruct (Table 3): FP16 76.27; KIVI-2 (3.00 b) 73.69; VQLLM-2 (2 b) 52.69;
**CommVQ-2 (2 b) 76.04**; KIVI-1 (2 b) 2.20; **CommVQ-1 (1.03 b) 66.57**. Paper text: LongBench (8 KIVI tasks, max
128K) "almost the same average as FP16" at 2 bits, 2.63 % above KVQuant; InfiniteBench at 128K: retrieval (passkey,
number, KV) "retains some capacity" at 1 bit where others fail; NIAH: CommVQ-2 matches FP16. Domain shift (perplexity
FineWeb-Edu-trained codebook): +1.37 FineWeb, +0.47 GSM8K, +0.58 RepoBench, +0.79 KV_Retrieval. Attention latency,
optimized vs naive (8K / 32K / 128K): 0.4 / 1.1 / 3.8 ms vs 2.4 / 9.2 / 36.6. 128K context for Llama-3.1-8B on one
RTX 4090 at 1 bit.

**Runtime cost.** Trained encoder + codebooks (EM); per-step q·codebook products (cheap) plus gathers; encoder matmul
per stored token.

**Failure modes.** Learned codebooks are domain-dependent (+0.5–1.4 PPL off-domain); 1-bit loses ~10 GSM8K points.

**New vs #416.** RoPE-commuting codebooks; learned additive VQ for both K and V on the device tier.

**Radiance fit.** Codes per token fit pages; the 27B rotates only 64 of 256 K dims (partial RoPE), so the commuting
constraint touches a quarter of the key — the other 192 dims can use any codebook. Kernels are gather/LUT-bound, no
WMMA. Medium–low priority.
