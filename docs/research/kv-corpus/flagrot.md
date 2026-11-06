# FlagRot (HKUST, 2026) · arXiv:2611.03992 — **has Qwen3.8-27B results**

**Idea.** Rotation does not always help: it flattens a spiky 32-channel piece of a key but can put a new peak into a
flat one. Store each 32-channel piece **rotated or not, whichever has smaller calibrated (query-weighted) attention
error**, and rotate the query instead when reading. Per-channel caches (KIVI) decide once per head (free);
per-token caches decide per key, the flag hidden in existing sign bits. Plus calibrated key centering.

**Method.** Candidates per piece: identity vs H₃₂ (FlagRot-L adds local candidates; -C builds on WUSH-KV's
transform). Decision weight M and statistics from four calibration windows, frozen. Served as a vLLM plugin with
paged INT2 (g = 32, FP8 scale and minimum, 2.5 bits/value = 80 B/token/KV head), BF16 sink 4 + recent 128 (S4W128).

**Numbers.** 12-model geometric-mean KL ×10³ at 2.5 bits (Table 1): KIVI 18.7, RotateKV 18.1–25.6, FlagRot-auto 17.8,
**FlagRot-KIVI 12.4**; selection cuts RTN INT2/3/4 KL by 62/56/47 %. vLLM tasks (Table 2): **RULER 128k Llama-3.1-8B
BF16 73.7 → KIVI 62.3 → FlagRot 65.2**; LongBench v1 Qwen3-32B 44.97 → 42.80 → 44.73. **Qwen3.8-27B: LongBench
53.49 (BF16) → 53.82 (2-bit KIVI) → 53.25** — 2-bit KV costs nothing measurable on LongBench for the hybrid 27B.
**4-bit, no BF16 tokens, KL ×10³ on Qwen3.8-27B (Table 25):** INT4 RTN 6.7; TurboQuant 8.2; SmoothAttention 6.2;
whole-head Hadamard 6.2; calibrated base 5.2; FlagRot 5.8; FlagRot-L 4.4 (4.62 b); NVFP4 RTN 8.3. Other models gain
far more from Hadamard (Qwen3-8B 43.5 → 31.2; Gemma-3-27B 63.1 → 46.2). **Agentic, Qwen3.8-27B (Table 7):** vs
BF16, **FP8 KV: τ²-bench pass^1 −2.6 [−4.7, −0.5], pass^4 −4.1, SWE-bench Verified −4.8 [−9.0, −0.6]**; 4-bit
FlagRot −0.7 / −2.5 / −2.8 (inside the BF16 rerun spread).

**Runtime cost.** Qwen3.8-27B pure decode on H200 at 8K (Table 9): TPOT batch 1 / 64: BF16 14.9 / 29.1 ms, KIVI 2.5 b
15.5 / 33.2, FlagRot-KIVI 15.7 / 38.7, per-token FlagRot 16.5 / 30.5; prefill 10.9–11.8K vs 12.0K tok/s; capacity 0.93 M → 2.75 M
tokens (attention pages 6.3×, whole cache incl. GDN state 3.0×). Per-key mode costs 5 % prefill.

**Failure modes.** RULER 128k still −8.5 points vs BF16 at 2.5 bits; Qwen2.5-7B unusable without key centering
(KL > 600 ×10³ for every per-token 4-bit cache).

**New vs #416.** (1) Per-piece *selective* rotation and query-weighted calibration (#416 rotates all-or-nothing per
layer); (2) **evidence that whole-head Hadamard buys only ~7 % KL on the 27B at INT4** while calibration-based
transforms buy 22–34 %; (3) **fp8 KV measurably hurts the 27B on agentic tasks** — radiance's default.

**Radiance fit.** Yes for per-token caches: a flag per 32-value group fits into the group's scale word or sign bits;
the decode kernel rotates q per group (H₃₂ on 32 lanes is cheap in WMMA tiles). 16-token pages unaffected.
