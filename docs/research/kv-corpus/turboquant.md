# TurboQuant (2025) · arXiv:2504.19874

**Idea.** Data-oblivious online vector quantization near the Shannon bound: randomly rotate each vector so its
coordinates follow a concentrated Beta law and are nearly independent, then apply a precomputed **Lloyd–Max scalar
codebook per coordinate** (TurboQuant-mse); for unbiased inner products add 1-bit QJL on the residual
(TurboQuant-prod). One norm per vector, no group scales.

**Method.** x ↦ ‖x‖, R x/‖x‖ (random rotation), b-bit optimal scalar code per coordinate (codebook fixed by d and b);
prod variant: (b−1)-bit mse code + QJL sign bits of the residual. KV experiments split channels into outlier and
regular sets with two instances: **2.5 bits = 32 outlier channels at 3 b + 96 at 2 b** (head 128); 3.5 bits similar.
Distortion within ≈ 2.7× of the information-theoretic lower bound (proved).

**Numbers.** LongBench, Llama-3.1-8B-Instruct (Table): full 50.06; KIVI-3 b 48.50; KIVI-5 b 50.16; PolarQuant 3.9 b
49.78; **TurboQuant 2.5 b 49.44, 3.5 b 50.06**. NIAH score (Llama-3.1-8B, 4K–104K): full 0.997, **TurboQuant 0.997**,
PolarQuant 0.995, KIVI 0.981, PyramidKV 0.895, SnapKV 0.858. Near-neighbour indexing time ≈ 0 vs PQ / RaBitQ.

**Runtime cost.** Rotation per token (WHT), codebook lookup (non-uniform) on dequant, per-vector norm; outlier/regular
split needs a channel permutation; no calibration.

**Failure modes (independent evidence).** RaBitQ authors (arXiv:2604.19528): TurboQuant worse than RaBitQ in most
inner-product, ANN and KV settings, and several reported runtime / recall numbers not reproducible from the released
code. WUSH-KV (arXiv:2609.38121): a TurboQuant-style Hadamard transform at 2 bits gives Qwen3-4B-Thinking AIME25 0.0
with 100 % of generations hitting the length cap. FlagRot (arXiv:2611.03992, Table 25): TurboQuant INT4-class KL on
**Qwen3.8-27B 8.2 ×10³ vs 6.7 for plain INT4 RTN** at 4.9–7.1 bits. KVarN (arXiv:2606.03458): TurboQuant 3/3 (4.6 b)
48.9 AIME24 vs 60.0 for KVarN at 2.3 b. Code 000: per-vector norm + Lloyd–Max worse than int5 groups at equal bits
(k6_v4 0.0284 at 5.5 b → 0.0320 at 5.06 b; IKV E.5).

**New vs #416.** Already tested and rejected by Code 000 (per-vector norm + Lloyd–Max; QJL residual).

**Radiance fit.** Per-token and page-friendly, but non-uniform codes need LUT dequant (no integer WMMA path), and the
evidence on Qwen-family models (incl. the 27B) says it loses to plain group-scaled INT4.
