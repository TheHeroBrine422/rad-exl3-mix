# KVTuner (ICML 2025) · arXiv:2502.04420

**Idea.** Layers differ in KV sensitivity, and **K usually matters more than V**; search offline for a per-layer
(K bits, V bits) pair from a small hardware-friendly menu with multi-objective optimisation, then serve the fixed
recipe with no online decisions.

**Method.** Menu {KV8, K8V4, K4V8, KV4, K4V2, K2V4, KV2}; intra-layer pruning to the Pareto pairs per layer, inter-layer
clustering by attention-output error to shrink the search; NSGA-style MOO over (memory, accuracy) on a calibration
set (GSM8K few-shot CoT prompts). Works on top of KIVI or per-token-asym quantizers.

**Numbers.** WikiText PPL (Table 2, KIVI-HQQ): Llama-3-8B KV8 9.95, KV4 9.99, K4V2 10.11, K2V8 31.92;
**Qwen2.5-7B-Instruct KV8 9.56, K8V4 9.39, K4V8 220.83, KV4 235.03** — Qwen2.5's keys break at 4 bits while V at 4
bits is free. Llama-3.1-8B relative attention-output error (Table 3): KV8 0.014, K8V4 0.100, K4V8 0.168, KV4 0.207,
KV2 0.962. Claims nearly lossless 3.25 bits (Llama-3.1-8B) and 4.0 bits (Qwen2.5-7B) on math reasoning. LongBench
avg, Qwen2.5-7B (Table 7): BF16 0.7956; per-token-asym KV4 **0.6343**; KVTuner-C4.0 0.7960; KVTuner-C5.0 0.8005.
Throughput (KIVI kernels, Llama-3.1-8B, Table 8): KVTuner-C3.25 +21.25 % over KV8 at bs 64 × 128 tokens.

**Runtime cost.** None online (offline recipe); offline search cost = many calibration evaluations (pruned).

**Failure modes.** Search objective is task accuracy on short CoT prompts; long-context and retrieval validated only
by LongBench averages. Assumes additivity is good enough after clustering (Code 000 found per-layer costs **do not
add** on Flash-Next and verifies every recipe on device).

**New vs #416.** Essentially #416's rad-autoquant idea published earlier, with two differences worth stealing:
**separate K and V bit widths per layer as independent knapsack items** (#416's menu couples them into fixed
layouts: k6_v4, k5_v4, …) and **inter-layer clustering** to cut the measured search space. The Qwen2.5 key
fragility is a reason to screen K-heavy pairs (K8V4-like) on the 27B; Qwen3.8's qk_norm likely removes most of it.

**Radiance fit.** Yes: per-layer layouts are exactly the #416 recipe mechanism; decoupled K/V widths need new packed
layouts (e.g. k8_v4, k4_v8) and kernels per combination (one per pair actually chosen).
