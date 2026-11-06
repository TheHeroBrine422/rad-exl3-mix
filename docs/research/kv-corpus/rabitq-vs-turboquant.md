# Revisiting RaBitQ and TurboQuant (2026) · arXiv:2604.19528

**Idea.** A symmetric, reproducible comparison of the two random-rotation quantizers. RaBitQ: random rotation /
JL, a *uniform* codebook (shifted unsigned integers) with one rescaling scalar per vector, inner products in native
integer arithmetic **without decoding**; TurboQuant: k-means (Lloyd–Max) non-uniform codebook needing lookups.

**Numbers.** Theory (Table 1): RaBitQ has a sub-Gaussian tail bound (bits scale as log log(1/δ)); TurboQuant bounds
the MSE only (log(1/δ)). Quantization time, A100 GPU, d = 1536: RaBitQ 0.065 (FWHT variant 0.008) vs TurboQuant
0.114 (units of their Table 2). KV, LongBench-E, Llama-3.1-8B-Instruct (Table 3): full 50.39; **RaBitQ 2.5 b 48.64**
vs TurboQuant-mse 2.5 b 47.78; at 3.5 b 49.55 vs 49.57. NIAH re-run over 15 lengths 4K–104K × 10 depths with a
deterministic keyword scorer (the original GPT-3.5 judge was inconsistent; haystack order was filesystem-dependent).

**Runtime cost.** RaBitQ: rotation + integer dot products on codes (good fit for integer matrix units).

**Failure modes.** Both are mean-error guarantees per vector; neither addresses softmax-amplified errors at the top
scores. The note documents irreproducible runtime/recall claims in TurboQuant.

**New vs #416.** (1) A uniform-grid, decode-free, per-vector-scaled code is the integer-friendly form of "rotate then
quantize" — essentially #416's rotated int layouts with one scale per vector instead of per 32. (2) **Methodology for
our judge:** NIAH with an LLM judge and an unordered haystack is not reproducible — use deterministic scorers and
fixed haystacks.

**Radiance fit.** Yes (per-token rows); one f16 scale per 256 values instead of eight saves 0.44 bits/value, at a
quality cost Code 000 measured as negative (one norm per vector, IKV E.5).
