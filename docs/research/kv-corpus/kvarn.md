# KVarN (2026) · arXiv:2606.03458

**Idea.** Under real decoding, KV quantization error **accumulates across steps**, driven mainly by wrong *token
scales*; fix it with Hadamard rotation plus Sinkhorn-style **dual-axis variance normalisation** (a scale per token
and per channel) before 2-bit RTN. Calibration-free.

**Method.** Every token rotated (WHT over head_dim). Per block of b = 128 tokens and KV head: 8 alternating
log-domain row/column std normalisations (keep best imbalance) → per-channel and per-token scales; asymmetric RTN
(K per channel, V per token) with the RTN scale/zero folded into one axis; reconstruction x = (q·s_a + z_a)·s_b.
Three-region layout: S sink tokens FP16, packed 2-bit body, R trailing FP16 tokens until a 128-token block completes.
Rate ≈ 2.3 bits/value at K2V2. **Pseudo-decode evaluation:** prefill in blocks of b; after each block the cache is
quantized and all later tokens' hidden states are computed *from the quantized cache* (Fig. 4) — static prefill
benchmarks (classic NIAH) miss the accumulation.

**Numbers.** Qwen3-4B (Table 1) AIME24 / MATH500 acc: FP16 61.1 / 82.6; **KVarN 2/2 (2.3 b) 60.0 / 79.2**; KIVI 55.5
/ 77.8; QuaRot 56.7 / 78.9; KVQuant-1% 40.0 / 67.5 (and 14 794 vs 12 477 generated tokens: quantization makes
reasoning *longer*); TurboQuant 3/3 (4.6 b) 48.9 / 77.0; Kitty 53.3 / 78.5. HumanEval Phi-4-14B (Table 2): FP16 88.9,
KVarN 88.2, KIVI 74.6. Line retrieval Qwen3-4B at 600 lines (Table 4): FP16 90 %, KVarN 85 %, KIVI 74 %, Hadamard
alone 83 %, KVQuant 57 %. Llama-3.1-8B at 2.3 equivalent bits (Table 5): KVarN 89 % at 600 lines vs SnapKV 85 %,
KVzip 86 % (7× eviction); at 100 lines KVarN 100 % vs SnapKV 77 %, PyramidKV 61 %.

**Runtime cost.** Normalisation per completed 128-token block: 1.9 ms per 128 generated tokens vs 1050 ms decode
(0.18 %, Qwen3-4B, vLLM); two-scale dequant ≈ +1 % vs RTN. FP16 sink + tail pool. vLLM fork (Triton).

**Failure modes.** Authors: unclear on MLA. The public KVarN repository's MLA status log (`KVARN_MLA_STATUS.md`) reports the
machinery inert on already-normalised caches (cos 0.99667 RTN → 0.99681 full) — radiance's K is RMS-normed (qk_norm)
before caching. vLLM integration needed four corruption fixes (prefix cache, chunked prefill, speculation).

**New vs #416.** Per-token × per-channel dual scales (Sinkhorn) and the 128-token tile; 2-bit K (#416's floor is
int4 K). **The pseudo-decode evaluation is the most important import for the judge** (see IDEAS.md).

**Radiance fit.** Tile-based: needs 128-token tiles (8 pages), an fp16 tail group, sink group, and a pack op on tile
completion (prefix hashing and spec rollback must only see packed tiles of accepted tokens). Per-token-only variant
(Hadamard + per-token scale, no channel scale) fits current rows.
