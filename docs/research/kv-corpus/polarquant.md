# PolarQuant (2025) · arXiv:2502.02617

**Idea.** After a random rotation, the *angles* of a recursive polar decomposition of a key/value vector have a
tight, analytically known distribution, so they can be quantized with fixed codebooks and **no per-group scale or
zero point** (only the vector norm is stored).

**Method.** Random preconditioning (rotation) → recursive Cartesian-to-polar transform over blocks of 16
coordinates (log₂ 16 = 4 levels) → quantize angles: 4 bits at the first level ([0, 2π)), 2 bits at deeper levels
([0, π/2]); radius in fp16. Effective ≈ 3.9 bits/value in the LongBench comparison. Online variant (PolarQuant-R)
refits codebooks per prompt.

**Numbers.** LongBench-V1 avg, Llama-3.1-8B-Instruct (Table 1): exact 48.63; SnapKV 44.57; PyramidKV 44.03;
StreamingLLM 38.36; KIVI 46.70; **PolarQuant 48.11; PolarQuant-R online 48.37**. NIAH 4K–104K: 0.995 vs 0.997 full
(TurboQuant paper's table). Runtime at n = 16 384, 1024 generated (Table 2): prefill 11.623 s vs exact 2.934 s (the
polar transform is slow in their PyTorch code), generation 43.652 s vs exact 38.374, KIVI 49.564.

**Runtime cost.** Recursive trig (atan2 / sin / cos) per element on store and on load (or LUT); random rotation.
**4× slower prefill** in the reference implementation.

**Failure modes.** KVarN (arXiv:2606.03458) measures PolarQuant 4/2 bits (3.3 b) at AIME24 52.2 vs FP16 61.1 and
MATH500 71.1 vs 82.6 on Qwen3-4B (and +36 % longer generations) — weak under decode-time accumulation; line
retrieval at 200 lines 86 % vs 99 %.

**New vs #416.** Scale-free angle coding is new, but its gain is the 0.5 bits/value of scales, and the evidence on
reasoning models is negative.

**Radiance fit.** Per-token, pages fine, but trig-heavy decode with no integer/WMMA path. Low priority.
