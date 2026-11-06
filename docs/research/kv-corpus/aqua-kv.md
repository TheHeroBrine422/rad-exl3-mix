# AQUA-KV — "Cache Me If You Must" (ICML 2025) · arXiv:2501.19392

**Idea.** Keys and values are largely *predictable* — K_l from layer l−1's keys, V_l from the same layer's K — so
store only the residual of a small linear predictor, quantized with a data-free vector quantizer (HIGGS).

**Method.** Linear probes on Llama-3.2-3B: one previous layer's keys explain about as much variance as a 2-bit
quantizer (1-bit ≈ 0.75, 2-bit ≈ 0.89 explained variance); previous *tokens* explain little. Algorithm: per layer, fit
ridge-regression predictors K_l ← K̂_{l−1}, V_l ← (K̂_l, V̂_{l−1}) on calibration data (one shot, layer by layer on
the *reconstructed* inputs), quantize residuals with HIGGS (Hadamard + MSE-optimal Gaussian grid VQ), first layer
kept at higher precision, 16-bit attention sink. Calibration 1–6 h on one GPU even for 70B.

**Numbers.** Llama 3.x / Qwen 2.5 (Table 2), WikiText-2 PPL (8K) and LongBench avg (14 tasks): uncompressed 3B 6.98
/ 44.61, 8B 5.61 / 48.13, 70B 2.54 / 52.92; **AQUA-KV 2.09 bits: 7.03 / 44.30, 5.72 / 47.77, 2.62 / 52.79**; HIGGS
2.02 b: 7.47 / 42.80; KIVI 2.25 b: 9.34 / 39.64; KVQuant 2.33 b: 9.43 / 20.56. Qwen2.5-7B: AQUA-KV 6.17 / 46.43 vs
6.13 / 46.82; HIGGS alone collapses (8.08 / 25.97), KIVI 7.02 / 32.78. Ablation (3B): without K predictor 7.50 /
42.92, without V predictor 7.06 / 43.91, without 16-bit sink 7.15 / 44.13. PassageRetrieval (3B instruct): 96.0
uncompressed, KIVI-2 59.5, KVQuant-2 4.5. With H2O at 20 % tokens: 38.3× memory saving, LongBench 38.43 vs 38.82.

**Runtime cost.** Predictor matmuls per stored token (cheap with GQA) and, at read time, reconstruction needs the
previous layer's reconstructed keys for every token; reported decode 23.31 vs 24.02 tok/s (8B, A100, 8K) and 5.76
vs 5.91 (70B) — but in an HF implementation that dequantizes whole caches.

**Failure modes.** Cross-layer K prediction makes layers' errors coupled (a chain); long-context retrieval only via
LongBench; Qwen-family keys need the predictor most (HIGGS alone fails).

**New vs #416.** #416's vpred is the "V from same-layer K" half. **Cross-layer K prediction is new** — but on the 27B
attention layers are 4 apart with 3 GDN layers between, and Code 000 measured only CKA 0.80 (K) / 0.52 (V) between
QSA layers on Flash-Next (IKV Table 9). Vector-quantized residuals (HIGGS grid) are new for radiance's device tier.

**Radiance fit.** V-from-K: yes (vpred exists; fused kernel missing). K from the previous attention layer: a decode
kernel would read layer l−1's packed K plus layer l's residual and apply the predictor per tile — doubles K reads and
couples layers; workable only via the vpred-style identity q·K_l = (qAᵀ)·K̂_{l−1} + q·R_l, i.e. a second score pass.
