# E8 Lattice Key Quantization (workshop, 2026) · arXiv:2610.15384

**Idea.** Use structured vector quantization from the **E8 lattice** (the QuIP# E8P-style 2¹⁶-word codebook, 8 dims per
16-bit index) for keys at 2 bits/dim; values stay scalar 2-bit. Score attention with **table-based asymmetric
scoring** (precompute q-block · codeword tables) instead of dequantizing keys.

**Method.** Keys rotated to unit norm (TurboQuant-style), split into 8-dim blocks, scaled by α and rounded to the
nearest E8-derived codeword (16-bit index → 2 bits/dim; 2.25 bits/dim with the norm). Values: scalar 2-bit groups.
Recent-token residual window in fp16. SQNR on 128-dim Gaussians: E8 10.47 dB vs scalar 3-bit 7.37 dB.

**Numbers.** Bytes per token (Table 2): Llama-3.1-8B fp16 131 072 → scalar k3v2 27 296 (4.8×) → **E8 k2v2 21 728
(6.0×)**. PPL deltas vs fp16 (Table 1), WikiText-2 at 4K: Qwen2.5-14B KIVI k2v2 +3.04, scalar k3v2 +0.68, **E8 +0.29**;
Llama-3.1-8B +15.17 / +0.60 / **+0.23**. E8 beats scalar k3v2 in 80 % of model–dataset–context settings. Decode tok/s
parity with scalar (Table 3; e.g. Llama-3.1-8B 4K: 87.0 both). Single-needle retrieval 4K–16K: 100 % for all methods.

**Runtime cost.** No calibration (fixed lattice codebook, one scale α). Encode = nearest-lattice-point search per
8-dim block (cheap for E8: two cosets). Scoring tables per query block; throughput parity only shown at 1K–4K
context where attention is a small share of decode.

**Failure modes.** Contexts ≤ 16K, no multi-needle or reasoning tests; values at 2 bits scalar (V is where Code 000
finds the bottleneck on Flash-Next); "parity" not demonstrated where KV reads dominate.

**New vs #416.** Lattice VQ for K. Directly reuses our own weight-quant experience (E8P codebooks, QuIP#/QTIP
lineage): the **trellis/lattice decoders we already run for weights could serve 2-bit keys**.

**Radiance fit.** Per-token 16-bit indices fit pages. On gfx1201, decode via E8P-style structured decode (256-entry
source table + sign/shift bits, as QuIP# does) into f16 before WMMA is cheaper than 2¹⁶-entry score tables (128 KB
f16 per q-block — exceeds LDS). Plausible for K at 2–2.25 bits if the 27B's K tolerates 2 bits at all.
