# kvtc — KV Cache Transform Coding (NVIDIA, 2025) · arXiv:2511.01815

**Idea.** For caches that are *stored* rather than read every step (prefix cache, offload, multi-turn reuse), use
classical media coding: **PCA decorrelation → dynamic-programming bit allocation → entropy coding (DEFLATE)**;
20× typical, 40×+ in some uses.

**Method.** Per layer/head a PCA basis fit once on calibration data (brief); per-component quantization with bits from
a DP allocation under a target ratio (low-variance components get 0 bits); DEFLATE on the codes (GPU nvCOMP). First 4
tokens (sinks) and a recent sliding window left uncompressed. Decompress before use.

**Numbers.** Llama 3.1 8B (Table 2), vanilla / KIVI-2 / GEAR-2 / H2O (8×) / FP8 / **kvtc 16× (CR 18–22)**: GSM8K
56.8 / 52.8 / 52.8 / 54.3 / 55.2 / 56.9; Lost-in-the-middle 99.4 / 88.8 / 96.9 / 20.2 / 99.4 / 99.3; RULER-VT 99.8
/ 98.9 / 99.8 / 50.4 / 99.9 / 99.1; kvtc 64× (CR 60–88): LITM 90.2, VT 95.9. Llama 3.3 70B kvtc 20×: NIAH 100,
LITM 100, MATH-500 72.6 vs 75.6. R1-Qwen 7B kvtc 8×: AIME24 52.5 vs 50.9. **Sink ablation (Table 6):** 64× without
the 4 sink tokens → GSM8K 1.6, LITM **0.0**, VT 61.8; with them 57.2 / 90.2 / 95.9. Latency (Mistral NeMo 12B, H100,
HF code, Table 5): compress 379 ms / decompress 267 ms for 8 × 8K tokens; TTFT from compressed cache 380 ms vs 3098 ms
recompute.

**Runtime cost.** Not a decode-time format: decompress on admission (PCA back-projection matmul + dequant + inflate).
Calibration: PCA per head on a short calibration set.

**Failure modes.** Sinks must be exempt (else collapse at high ratios); fixed PCA basis may not transfer to new
domains (authors list scalability/generalisation as limits); not composable with decode-time read formats.

**New vs #416.** A **host/store-tier codec** at 8–20× (#412's store "holds the pool's bytes, at the pool's precision";
Code 000's tiered precision transcodes to k4h layouts, ~3–4×). Code 000 found PCA useless as a *decode* format
(V needs 129–183 of 256 components), but entropy-coded PCA at 0-bit tails is a different regime.

**Radiance fit.** As a prefix-cache / idle-conversation codec (`--prefix-cache-mib` tiers, KV streaming host side):
yes, per 16-token page or per conversation chunk; decompress on promotion (GPU kernels for PCA matmul + Huffman /
ANS decode needed on gfx1201). Not for the resident pool.
