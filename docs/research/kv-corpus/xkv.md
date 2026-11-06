# xKV (2025) · arXiv:2503.18893

**Idea.** Per-token KV similarity across layers is weak, but the **dominant singular vectors of the KV cache align
across adjacent layers** (high CKA); factor a group of layers' KV into one shared low-rank subspace per prompt
(post-training, no fine-tuning), with Selective Reconstruction (SR) of only the tokens a decode step needs.

**Method.** Cross-Layer Factorization: concatenate a window of layers' K (and V) for the prompt, one SVD → shared
left singular vectors (token side) + per-layer right factors; done on the fly at prefill (custom SVD kernel). Decode:
reconstruct densely, or select chunks via landmarks and reconstruct only those (xK-SR / xKV-SR).

**Numbers.** RULER at 64K, Llama-3.1-8B-Instruct (Table 1): full 91.89; KIVI-2 (7.10×) 86.87; PyramidKV (8×) 89.94;
SnapKV (8×) 89.68; single-layer SVD (8.4×) **45.71**; MiniCache (1.30×) 45.04; **xKV (8.03×) 88.50** (VT 86.67 vs
97.29). Another model (Table 1 lower block): xKV-4 (6.21×) 90.19, MiniCache 13.78. With SR (Table 2): xK-SR 1.63×
total / 8.90× effective GPU memory 89.70 vs ShadowKV 87.17 and Quest 84.87. On-the-fly SVD latency at prefill
(Table 3, A100): 9.97 s prefill at 64K (overhead reported separately in App. C.1). Up to 4.23× end-to-end speedup with
SR.

**Runtime cost.** Per-prompt SVD over grouped layers at prefill (seconds at 64K–256K); reconstruction matmuls at
decode; decode-time tokens appended outside the factorization (authors: long generation is a limitation).

**Failure modes.** Variable tracking and multi-value retrieval degrade first (VT −10 points at 8×); long generation
not covered; per-layer single SVD collapses (45.71) — the cross-layer alignment is essential.

**New vs #416.** Cross-layer sharing of a low-rank basis. Code 000 measured linear CKA between QSA layers 0.52 (V) /
0.80 (K) on Flash-Next and rejected sharing; on the 27B attention layers are 4 apart (3 GDN layers between), likely
less aligned.

**Radiance fit.** Poor: per-prompt factors spanning pages and layers break the page-local, layer-local layout and
prefix-cache reuse (a new turn changes the factorization).
