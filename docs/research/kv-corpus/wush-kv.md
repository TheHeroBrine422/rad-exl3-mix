# WUSH-KV (2026) · arXiv:2609.38121

**Idea.** Replace the fixed Hadamard with a **data-adaptive, non-orthogonal** per-KV-head transform built from
second-order statistics of *both* factors of the product it feeds (keys: query-side Hessian H and key covariance M;
values: the attention-output side), optimal for a clipped INT quantizer (QuEST) under mild assumptions.

**Method.** WUSH transform T minimising tr(T M Tᵀ)·tr(T⁻ᵀ H T⁻¹) (lower bound (Σσᵢ)², σ = singular values of
M^½H^½); block-diagonal with block = quantization group. Calibrated on 128 FineWeb-Edu sequences × 1024 tokens
(statistics only, no backprop). V transform folded into W_V / W_O; K transform applied after RoPE (q gets T⁻ᵀ).
Serving (SGLang): OSCAR-style percentile-clipped asymmetric INT2, BF16 sink 64 + recent 256 tokens, page-sized flush
of 8 tokens.

**Numbers.** Qwen3-8B WikiText-2 PPL (Table 1; FP 9.72), 4 / 3 / 2 bits: identity 11.36 / 15.29 / 38.08; random
rotation 9.77 / 10.18 / 14.22; **Hadamard 9.77 / 10.06 / 258.29**; OSCAR 9.77 / 10.19 / 13.74; **WUSH 9.73 / 9.82 /
10.51**. 2-bit reasoning (Table 2) Qwen3-4B-Thinking AIME25 / MATH500: BF16 77.8 / 89.5; WUSH-KV 71.1 / 90.1; OSCAR
65.6 / 89.3; TurboQuant-style H transform **0.0 / 18.7 with 91.8k-token outputs, 100 % hitting the length cap**.
Qwen3-8B AIME25: BF16 67.8, WUSH-KV 46.7, OSCAR 34.4. **RULER NIAH Qwen3-8B (Table 4), 4K → 128K:** BF16 99.71 →
79.17 (overall 91.48); WUSH-KV 97.44 → **57.26** (78.77); OSCAR 96.63 → 25.69 (71.74). **MRCR** overall: BF16 25.31,
WUSH-KV 9.55, OSCAR 6.79 (Table 5).

**Runtime cost.** One d × d (block-diagonal) K transform per (layer, KV head) applied to q and k per token; V free
(folded). Offline statistics pass. INT2 with sink/recent BF16 windows in SGLang.

**Failure modes.** At 2 bits, multi-needle retrieval (MRCR) loses >60 % even with the best transform, and NIAH falls
with context (−22 points at 128K vs BF16) — the clearest published evidence that 2-bit KV fails long-context tails
even when PPL and reasoning look fine. Degenerate never-ending generations (length-cap hits) are a tail failure mode
mean-KL would miss.

**New vs #416.** Code 000 tried a "WUSH-style" transform on Flash-Next at 4.5 / 3.5 bits and found it slightly
worse than Hadamard (0.0406 / 0.1086 vs 0.0397 / 0.1079 proxy, IKV Table 9). Here the gap appears only at **2-bit K**
(Hadamard 258 PPL vs WUSH 10.5), a regime #416 never uses (K ≥ 4 bits). New: sink + recent BF16 windows with
page-sized flushing — the exact shape radiance would need for any 2-bit K.

**Radiance fit.** Yes as a per-head matmul op replacing `had_rotate` for K (q·T⁻ᵀ, k·T), V folded into weights
(27B: output gate blocks W_O folding → online inverse). 16-token pages = their flush unit; sink/recent windows as
separate groups.
