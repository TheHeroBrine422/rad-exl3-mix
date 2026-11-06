# SAW-INT4 (Together AI, 2026) · arXiv:2604.19157

**Idea.** Under real serving constraints (paged layouts, regular access, fused attention, concurrency), the best
4-bit KV method is the simplest: **token-wise INT4 with block-diagonal Hadamard rotation (BDR)**, fused into the
store kernel; VQ and Hessian-aware schemes add little once serving cost is counted.

**Method.** Per token and head: block-diagonal Hadamard of order h ∈ {16, 32, 64, 128} (K only or K and V), then
token-wise INT4; fused rotate–quantize–save kernel writing straight into paged KV (SGLang). Compared against
k-means VQ (C = 1, 16, 256 centroids) and a Hessian-aware (damped eigendecomposition, calibrated) transform.

**Numbers.** Qwen3-4B-Thinking-2507, 5-task mean (GPQA, HumanEval, LCB v6, AIME, MATH; Table 2): BF16 75.64;
**plain INT4 0.00 on every task**; BDR-16 54.83; BDR-64 72.29; BDR-128 73.11; **BDR-128 on K only 73.78**;
Hessian+BDR-128 65.52; k-means C = 16 70.48. Qwen3-8B (Table 3): BF16 70.84, INT4 0.00, BDR-64 (K) 69.86, BDR-128
69.97. Throughput, 2×H100, 32 requests × 8192 in / 1024 out (Table 1): BF16 71.7 TPS/user, INT4 86.7, **BDR 88.3**,
k-means 24.2, Hessian+BDR 76.8. Decode step profile Qwen3-32B tp 2 bs 32 (Table 4): fused rotate 529.9 µs vs INT4
533.1 vs unfused 540.6.

**Runtime cost.** Zero measurable with fusion (rotation inside the KV-store kernel; q rotated once per step).
Long context: at high concurrency INT4 holds 4× more tokens, so the comparison flips between per-request TPS and
system TPS (BF16 queues requests) — a methodology warning for speed judges.

**Failure modes.** Small rotation blocks fail (BDR-16 −20.8 points on Qwen3-4B); naive INT4 KV is catastrophic on
Qwen3 reasoning (degenerate generations), so means over easy tasks can hide total failure. No RULER.

**New vs #416.** Confirms #416's design (full-head Hadamard + per-token groups) as the serving-optimal 4-bit point;
adds evidence that **K-only rotation** can match K+V rotation, and that rotation block size must be ≥ 64. Code 000
also found block-diagonal splits keeping RoPE dims apart equal to H256 (IKV E.5).

**Radiance fit.** Yes (radiance already does a full-head order-256 rotation; fusing `had_rotate` into the store
kernel is the radiance-side lesson: #416 runs it as its own op).
