# GEAR (2024) · arXiv:2403.05527

**Idea.** Quantize KV to 2–4 bits, then correct the error with a **low-rank** matrix (coherent part) plus a **sparse**
outlier matrix (incoherent part); argues errors compound over autoregressive steps.

**Method.** X ≈ Q(X) + A_h B_hᵀ + S: base quantizer (KIVI or KCVT), rank r = 4 per head from a few power iterations on
the residual (r = 4 "sufficient" at n = 1024, d_H = 128), s = 2 % sparse outliers. Streaming buffer n_b = 20–64 fp16
tokens; low-rank factors recomputed per buffer flush.

**Numbers.** CoT tasks (Table 1, avg over LLaMA3-8B / LLaMA2-13B / Mistral-7B GSM8k, AQuA, BBH): FP16 40.52;
KIVI 4-bit 35.05; **GEAR(KCVT) 4-bit s = 2 %, r = 4: 40.80 at 31.0 % KV size**; KIVI 2-bit 25.25 (21.7 %);
**GEAR-L(KIVI) 2-bit r = 4: 38.34 at 23.6 %**. LLaMA3-8B GSM8k 2-bit: 52.62 vs FP16 54.21 vs KIVI 30.17.
V100 throughput, LLaMA-2-7B (Table 6): FP16 max batch 3 → 12.5 tok/s; KIVI-2 batch 18 → 56.6; GEAR-L-2 batch 18 → 63.38.

**Runtime cost.** Power-iteration SVD per head per flush, extra low-rank matmul in attention (n × r and d × r
factors), sparse fp16 outlier path; peak memory −2.39×, throughput 2.1–5.07× vs FP16 (V100/RTX Titan).

**Failure modes.** Evaluated at short context (prefill ≤ 3.6K, gen ≤ 256); no NIAH/RULER. Low-rank factors are per
(sequence, head) state that grows with n (A_h ∈ R^{n×r}); the sparse path is irregular.

**New vs #416.** Low-rank error correction of the residual and the explicit framing that **errors compound with
generation length** (CoT), i.e. exactly the tails the judge wants. Code 000 rejected sparse outliers (null on
k4h_v4) but never tested low-rank residual correction.

**Radiance fit.** Weak: per-sequence low-rank factors spanning all tokens break the page-local layout; a page-local
variant (rank-r residual per 16-token page) is possible but costs r·(16 + 256) values per page and head (r = 2 ≈
+0.85 bits/value at f16). Better fit as a host-tier codec.
