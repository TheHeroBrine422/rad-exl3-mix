# KVLinC (2025) · arXiv:2510.05373

**Idea.** At 2 bits, rotate V (Hadamard) and K per-channel as in KIVI, then *correct the attention* for key
quantization error with a trained **linear-attention side term**: a fixed-size recurrent state per head that adds
residual weight mass for the error keys.

**Method.** Ŷ_n = [Σ exp(q·K̂ᵢ)V̂ᵢ + φ_q(q) S_n] / [Σ exp(q·K̂ᵢ) + φ_q(q) P_n] with
S_n = S_{n−1} + φ_k(K^e_n)ᵀ V_n, P_n = P_{n−1} + φ_k(K^e_n), K^e = K − K̂, and φ(X) = [softmax(XW₁), softmax(XW₂)]
(trainable feature maps, Eqs. 4–6). Training: adapters only, model frozen. Cost: O(dD) per decode step and a d × D
state per head. Average precision counted 2.71 bits (incl. scales and the correction state at their lengths).

**Numbers.** Base models (Table 1), 2-bit class: Llama-3.1-8B WikiText 6.2 (FP16) / KIVI 7.8 / QuaRot 7.3 / GEAR-L
7.3 / **KVLinC 7.1**; GSM8K 49.7 / 34.1 / 26.8 / 38.8 / **40.9**. Instruct (Table 2): **RULER 4K / 8K**, Llama-3.2-3B
FP16 92.5 / 88.1, KIVI 76.7 / 70.3, KVLinC 80.8 / 73.6; Qwen-2.5-3B FP16 90.3 / 85.0, KIVI 49.5 / 41.0, KVLinC
60.9 / 51.1. Ablation (Llama-3.1-8B): Hadamard-on-V alone GSM8K 36.9, linear correction alone 38.4, both 40.9. Kernel:
up to 2.55× over FlashAttention-2 (A40, prompt 256, 1024 generated tokens, large batch).

**Runtime cost.** Trained adapters (calibration training); a recurrent correction state per (sequence, layer, head);
an extra φ_q(q)·S product per decode step; error keys K^e computed at store time.

**Failure modes.** Even with correction, RULER at 2 bits loses 12–30 points on Qwen-2.5-3B at 4–8K — a reminder that
2-bit KV fails retrieval tails; evaluated only to 8K. The correction state is order-dependent (prefix-cache sharing
must checkpoint it like a recurrent state).

**New vs #416.** Correcting attention with a learned **linear-attention residual state** is a new zero-ish-byte
mechanism (state is O(d·D) per head, independent of context). Close to GEAR's low-rank correction, but O(1) in context.

**Radiance fit.** Plausible on the 27B: the correction state is a recurrent state like the GDN state (radiance
already checkpoints those for prefix cache / spec rollback). Needs a fused φ_q(q)S term in the decode kernel and a
store op that computes K^e and updates S, P. Trained adapters = a calibration run per model.
