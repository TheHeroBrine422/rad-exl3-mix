# Accelerating Long-Context Decoding in Large-Scale Hybrid LLMs Without Retraining (RIKEN, 2026) · arXiv:2611.06133 — **Qwen3.5-27B (same 64-layer / 16-attention-layer layout as the 27B)**

**Idea.** In hybrids, the few full-attention layers dominate long-context decode. Make each one **sliding window +
pinned prefix + Quest-style top-B token budget**, with window W and budget B chosen **per layer by a multi-objective
search** (NSGA-II over accuracy and speed); no training.

**Method.** Per full-attention layer: window W ∈ {256 … 4096}, Quest budget B ∈ {256 … 4096} (page min/max
selection), or "full"; selection re-done every step (R1) or every 32 steps (R32). NSGA-II on a calibration set,
then accuracy on 21 long-context datasets; vLLM with per-layer sliding windows and query-aware sparse attention.

**Numbers.** All six hybrids (4B–397B) keep a configuration within 0.5 pp of baseline averages on 21 datasets.
**Qwen3.5-27B: average 53.8 baseline vs 54.0 best-accuracy / 53.4 balanced / 53.9 best-speed / 53.8 R32 / 53.9 R1**;
best-speed = W 256, B 512 on all 16 layers; best-accuracy widens windows to 2048–4096 on layers 3, 31, 51, 55, 59.
Speed (batch 4): Qwen3.5-4B 1.26× / 2.22× at 64K / 256K; 122B-A10B 1.20× / 2.23×; **dense Qwen3.5-27B only 1.11× at
64K** (A100: 47.1 ms per step of which 19.4 ms weight reads; best possible 1.78×). Slower than baseline at short
context. Cost: 22.9 + 34.2 + 465.4 GPU-h (throughput, search, accuracy) for the 27B.

**Runtime cost.** Page metadata + selection per step (or per 32 steps); per-layer configs; prefill unchanged.

**Failure modes.** Gains limited where weights dominate decode (dense 27B at batch 4); accuracy measured on QA-style
long-context datasets (no multi-turn SCBench, no RULER multi-key tables in the main text).

**New vs #416.** **Per-layer window/budget recipes found by search** — exactly #416's rad-autoquant pattern, applied
to sparsity instead of bits, on our model family. Shows layer 3 (first attention layer) and a few late layers need
wide windows — consistent with Code 000's "early layers are sensitive".

**Radiance fit.** Good: Quest metadata per 16-token page + per-layer window groups; rad-autoquant could search (W,
B, layout) jointly per layer. Decode gains on radiance are larger than on A100 bf16 because radiance's 4-bit weights
shrink the weight-read share.
