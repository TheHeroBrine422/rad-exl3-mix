# QuaRot — KV part (2024) · arXiv:2404.00456

**Idea.** Hadamard rotations remove outliers so weights, activations **and KV** quantize to 4 bits; for KV: rotate
V by a per-head Hadamard folded into W_v / W_out, rotate q and k by an online Hadamard after RoPE (scores unchanged).

**Method.** V: `W_v ← W_v H_head`, `W_out ← H_headᵀ W_out` (offline, zero runtime). K: online Hadamard on q and k
after RoPE (one fast WHT per head per token). KV quantized asymmetric RTN, group = head_dim (128), clip 0.9 not
used for KV. No calibration for the KV part.

**Numbers.** KV-only ablation, WikiText-2 PPL, Llama-2 7B / 13B / 70B, group 128 (Table 6): 16/16 bits 5.47 / 4.88
/ 3.32; **K4V4 5.51 / 4.91 / 3.33**; K4V3 5.54 / 4.93 / 3.35; K4V2 5.75 / 5.09 / 3.43; K3V3 5.68 / 5.02 / 3.39;
K2V4 8.06 / 6.42 / 3.89; K2V2 9.23 / 7.07 / 4.13. So on Llama-2, **K is the more sensitive tensor** (K2V4 ≫ K4V2).
End-to-end W4A4KV4 70B: 3.79 PPL (GPTQ weights) vs 3.32 FP16; ≥ 3.63× peak decode memory saving.

**Runtime cost.** One online WHT on q and k per head (FP32 Hadamard in their experiments; FP16 ablation Table 10
slightly worse), V rotation free. No calibration.

**Failure modes.** Only perplexity / zero-shot at short context; no long-context or retrieval test. Large drops at
70B with RTN weights (SpinQuant, arXiv:2405.16406, reports −28.1 / −33.2 points for W4A4 / W4A4KV4 RTN on 70B).

**New vs #416.** Nothing new as a transform (#416's `had_rotate` is this, order 256, on q/k/v after RoPE). Two
differences worth testing: (1) V's rotation **folded into W_v/W_out** removes #416's online V rotation and output
un-rotation op (on the 27B the gated attention's output gate is element-wise *after* attention, so H must be undone
before the gate; fold into W_o only works if the gate is applied in the rotated basis — it is not, so keep the op);
(2) the K-vs-V sensitivity ordering differs by model (Llama-2: K; Flash-Next: V), so the 27B needs its own screen.

**Radiance fit.** Yes, already in radiance (`had_rotate`, any per-token layout).
