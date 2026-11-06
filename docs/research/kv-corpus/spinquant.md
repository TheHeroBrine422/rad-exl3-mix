# SpinQuant — KV part (2024) · arXiv:2405.16406

**Idea.** Random rotations vary a lot in quality; *learn* the mergeable rotations (Cayley SGD on the Stiefel
manifold) against the quantized network's loss. KV relevance: a learned per-head rotation R2 on the V path (merged
into W_v / W_o) and the online Hadamard R3 on q/k.

**Method.** R1 (residual stream) and R2 (per-head, V/O) learned with Cayley SGD on a small WikiText-2 calibration set (Table 11 ablates 128–800 samples) while
weights stay frozen (R1, R2 ≈ 0.26 % of weight size); R3 (q/k after RoPE) and R4 (down-proj input) are fixed
Hadamards. Then GPTQ or RTN on weights; KV asymmetric with clipping.

**Numbers.** LLaMA-2 7B zero-shot avg (8 tasks), W-A-KV bits (Table 13): 4-4-16 64.1 vs 4-4-4 64.0 (Wiki2 calib),
64.3 / 64.3 (C4 calib); Wiki2 PPL 5.9 → 5.9–6.0. Headline: SpinQuant_had W4A4KV4 LLaMA-2 7B avg 64.0, 2.9 points
below full precision. Random rotations spread zero-shot accuracy by up to 13 points (Fig. 4, W4A4).

**Runtime cost.** Offline optimisation (hours per model on GPUs); online cost only the fixed Hadamards (R3, R4).

**Failure modes.** No long-context evaluation; the learned R2 is fit to a short-context LM loss. KV at 4 bits is
nearly free on top of W4A4 here, so the paper says little about KV below 4 bits.

**New vs #416.** A *learned* V rotation. Code 000's "best of 16 random rotations" (SpinQuant stand-in) and a
WUSH-style data-adaptive transform were both slightly worse than Hadamard (0.0407 / 0.1104 vs 0.0397 / 0.1079
proxy, IKV Table 9) — but neither optimised the end loss as SpinQuant does. An end-to-end KL-trained per-head V
rotation (folded into W_v) is untested in radiance.

**Radiance fit.** Yes if folded into weights (needs a requant of v_proj / o_proj on our side, or a per-head 256×256
matmul op instead of `had_rotate` at the same place: ~2·256 MAC per value, negligible next to attention at long
context). On the 27B the output gate sits between attention and o_proj, so W_o folding needs the gate handled in
the rotated basis (not possible for an element-wise gate) → keep an online un-rotation.
