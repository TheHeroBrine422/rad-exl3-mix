# QJL (2024) · arXiv:2406.03482

**Idea.** Remove the 1–2 bits/value of scale/zero overhead: apply a random Johnson–Lindenstrauss projection to each
key and keep only the **sign bits** (plus the key's norm); estimate q·k with an *asymmetric* estimator (unquantized
JL of q against sign bits of k), which is **unbiased**.

**Method.** k ↦ (sign(S k), ‖k‖) with S a Gaussian (orthogonalized) m × d matrix; ⟨q, k⟩ ≈ √(π/2)/m · ‖k‖ ·
⟨S q, sign(S k)⟩. Bits per key independent of d, logarithmic in context length for a fixed distortion. Outlier
channels (fixed channels in deeper layers) found at prompt time get a separate QJL copy. Values: standard per-token
quantization. 3 bits/value overall in experiments.

**Numbers.** LongBench (Longchat-7b-v1.5-32k, max length 31 500; Table): NarrativeQA / Qasper / MultiQA-en /
HotpotQA: FP16 20.79 / 29.42 / 42.83 / 33.05; KIVI-3 b 20.96 / 29.01 / 40.93 / 32.79; KVQuant 4.3 b 20.14 / 28.77 /
44.22 / 34.06; **QJL 3 b 21.83 / 29.44 / 41.52 / 35.62**. LM-eval Llama-3-8B at 3 b: within 0.2 of BF16 (Table 1).
">5× memory reduction" at 3 bits; faster runtime claimed at long context with its CUDA kernel.

**Runtime cost.** A random projection of q and k per token (m × d matvec), popcount-style or f16 dot
of S q with ±1 signs; no calibration.

**Failure modes.** Unbiased ≠ low variance: softmax amplifies score noise at the top of the distribution (where
retrieval lives). Code 000: "K is not the bottleneck below 6 bits (proxy k8_v2h = k4h_v2h = 0.18)" → rejected for
Flash-Next (IKV Table 9). WUSH-KV reports TurboQuant-style transforms collapsing 2-bit reasoning (arXiv:2609.38121).

**New vs #416.** Sign-bit keys with norm and an unbiased inner-product estimator; useful as a cheap **indexer key**
(selection) rather than the attention key itself.

**Radiance fit.** Per-token, scale-free: fits pages trivially (m bits + one f16 norm per key and head). As a 27B
sparse-attention indexer key (block summary) it is a natural candidate; as the main K it is a regression vs int4.
