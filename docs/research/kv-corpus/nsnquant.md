# NSNQuant (2025) · arXiv:2505.18231

**Idea.** Calibration-free VQ: make every token look like a standard-normal vector with **Normalize → Shift → Normalize**
(token-wise L2 normalisation, channel-wise centering, second token-wise normalisation) plus a Hadamard transform, so
one fixed, reusable codebook fits all models and domains (fixes CQ's calibration/domain-shift dependence).

**Method.** Per token: N (scale to unit RMS), S (subtract the running per-channel mean, stored per head), N again,
Hadamard; 8-dim (2-bit) or 16-dim VQ against a single Gaussian codebook; scale adjustment, optional codebook tuning,
double quantization of the norms; residual window of recent tokens in FP16; fused kernels. KL to N(0,1) on LLaMA2-7B
keys (Table 1): N 0.2476 → NSN 0.0197 with Hadamard (oracle 0.0096).

**Numbers.** C4 PPL (Table 2), LLaMA3-8B / LLaMA3.1-8B: FP16 8.32 / 8.43; KIVI-2 (2.38 b) 16.43 / 15.80; CQ-4c9b
(2.26 b) 13.97 / 12.24; **NSNQuant-2b (2.23 b) 9.08 / 9.15**; 1-bit class: CQ-8c10b 43.78 / 49.60 vs NSNQuant-1b (1.23
b) 16.69 / 17.20. LongBench (KIVI subset) LLaMA3.1-8B-Instruct (Table 3): FP16 48.82; KIVI-2 48.17; KVQuant-2b-1 %
47.88; CQ 47.80; **NSNQuant-2b 48.56; NSNQuant-1b 47.38** (CQ-1b 40.78). Ablation LLaMA2-7B PPL: 5.285 → 6.293
without the first normalisation, 5.842 without centering. Up to 3× throughput vs FP16.

**Runtime cost.** Running channel mean per head (state), two norms per token, VQ encode (nearest codeword search,
8-dim) at store, LUT or gather decode; FP16 residual window.

**Failure modes.** No RULER/needle tests beyond LongBench; GSM8K and HumanEval at 2 bits still lose points (Table 4
KIVI-2 64.59 vs FP16 76.65; NSNQuant row in paper's Table 4).

**New vs #416.** The **channel-wise centering ("Shift")** is the K-side analogue of #416's static V mean, but
*dynamic* (running mean per sequence); token-energy normalisation exists in #416 only for k4h_v2h's V. Shift + double
normalisation is the cheap, calibration-free part of KVarN's dual scaling.

**Radiance fit.** Normalise/centre steps are per-token ops (fit pages; the running mean is a per-sequence state that
must be frozen per page for prefix caching). VQ part: same LUT-decode caveats as CQ.
