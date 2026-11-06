# Block-GTQ (2026) · arXiv:2606.24033

**Idea.** Allocate key bits **per RoPE frequency block** (each 2-D rotary pair or block of pairs) by a label-free
query–key energy score; low-frequency (long-range) and high-energy blocks get more bits. Greedy largest-marginal-gain
allocation is exactly optimal for the separable proxy.

**Method.** Score per block i: ½(E‖q⁽ⁱ⁾‖² + E‖k⁽ⁱ⁾‖²) from a short calibration (2048 tokens of WikiText-2), combined
with codec-specific rate–distortion penalties; equal-rate blocks grouped so packed attention runs without decoding
to fp16. Instantiated on TurboQuant-MSE.

**Numbers.** Layer-averaged logit MAE, K-only, 2-bit (Table): uniform TQ-MSE → Block-GTQ: Qwen3-8B 5.81 → 2.96 (−49 %),
Qwen2.5-14B 4.14 → 2.61, DS-R1-Qwen-7B 11.44 → 2.40 (−79 %), Llama-3.1-8B 3.80 → 2.55; improves all 367 layers.
Abstract: Llama-3.1-8B-Instruct K2V2 **six-task NIAH 70.6 → 97.4**, LongBench-EN 33.7 → 44.9; DS-R1-Distill-Qwen-7B
K3V2 without fp16 sink or recent buffers **AIME24 / 25 49.6 / 38.8 vs fp16 54.2 / 37.9, uniform TQ-MSE 0 / 0**;
Qwen2.5-3B packed K3V3 at 128K: 3.24× smaller resident KV, perplexity near fp16. Score ablation (NIAH avg): mean Q/K
energy 97.36; key only 95.29; weight-only 85.10. Calibration corpus sensitivity at K2V2: WikiText 97.36, code 93.66.

**Runtime cost.** Offline score (minutes); per-block bit widths → several packed sub-layouts per key; no decode-time
search.

**Failure modes.** Tied to TurboQuant-MSE (itself weak on Qwen-family models, see turboquant.md); calibration-corpus
sensitivity up to −3.7 NIAH points (code corpus); Appendix C.4 shows a 4K NIAH failure on Qwen where uniform K3 already
fails.

**New vs #416.** Frequency-aware K precision. Code 000 tried "RoPE dims of K at higher precision" (64 dims int8 + 192
int4: 0.0380 at 5.5 bpw, worse than int5 throughout; IKV Table 9) — the opposite allocation of what matters may differ
(Block-GTQ ranks by q/k energy, not by RoPE vs non-RoPE). On the 27B only 64 of 256 K dims rotate (partial RoPE
0.25); the 192 NoPE dims are where content matching lives.

**Radiance fit.** Static per-(layer, head) channel→bit-width map = a fixed sub-layout per KV group (e.g. 64 dims int6
+ 192 dims int4); fits per-token rows; needs a permutation folded into W_k/W_q.
