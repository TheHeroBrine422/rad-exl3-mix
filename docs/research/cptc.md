# CPTC — Computed Permutation Trellis Coding (Balın, Amazon Annapurna Labs, 2026-11-17)

**Status:** read (paper, 2026-11-19). Candidate codebook for the next 27B iteration and for
DS4.1 (P2+); no code released — reimplementation is fully specified in the paper.
arXiv 2611.22019 ("Near-Optimal Quantization from Two Multiplications and a 64-Byte Table").

## What it is

A trellis-code quantization **format** (same family as EXL3 / IQR / QTIP — all trellis codes over
RHT-processed weights), with a new decoder:

- Lmax = 16-bit tail-biting trellis state, Viterbi encoder (O(T·2^L)), **fixed coefficients
  (a0, a1) shared by all targets** (pairs listed per rate in Appendix A.2); the decoder multiplies
  the state by the two coefficients, adds the top Q=6 bits of the products, and indexes a
  **127-entry table (64 bytes, E2M5 or INT8-storable)**.
- Per-tensor (per-distribution) freedom = **only the 64-byte table** (Lloyd iteration on the
  trellis, offline) + scale. Retargeting is trivially cheap.
- **Proposition 1: every K-bit scalar format (INTK, FPK, Lloyd–Max) is a special case** (a0=0,
  L=K) — adopting the decoder never costs quality; a deeper trellis beat scalar in all their tests.
- RHT incoherence preprocessing adopted unchanged from QuIP#/QTIP → entries ≈ i.i.d. Gaussian;
  quality metric = effective bits (Shannon rate-distortion bound). K=4: 3.93 eff bits
  (0.07 from ideal 4.00); beats QTIP HYB/1MAD/3INST at every K by 0.014→0.29 bits; on
  generalized-Gaussian targets the same coefficients serve all β (Laplacian included).
- **Not Hessian-aware, no GPTQ-style error compensation, no fine-tuning**: it replaces the
  format (INT4/FP4 position), orthogonal to calibration methods. Their LLM runs = YAQA pipeline
  (Tseng et al. 2026 — the IQR lineage) with only the codebook swapped, "no fine-tuning".

## The number that matters for us (Table 22)

They ran a **preliminary single-seed pass on Qwen3.8-27B** (our model; "our own 65,536-sequence
Hessians" in the YAQA pipeline):

| K | CPTC DKL | HYB (QTIP, IQR default) DKL | ratio |
|---|---:|---:|---:|
| 5 | 0.0295 | 0.0719 | 2.4× |
| 4 | 0.0563 | 0.1175 | 2.1× |
| 3 | 0.1060 | 0.3063 | 2.9× |
| 2 | 0.2630 | — | — |

Same ordering on Llama 3.1 8B/70B (Table 4: CPTC < HYB at every rate; margin 9% @ K=2 → 27% @
K=4 at 8B). Zero-shot accuracy is rate-driven, not format-driven (within a point of BF16 at K≥3).
Caveats: single RHT seed (their 3-seed KL spread on Llama: 0.3–4.1%), their eval corpus (W2/C4
PPL + 0-shot; DKL column = their standard), no fine-tuning, and **not** compared against
EXL3-mul1/mad or our calibrated V3 pipeline.

## Decoder cost

- Trainium 4: decode in the systolic weight-load path, 8 values/lane/cycle, FP4-rate throughput
  at K≤4; negligible area; closed-timing, formally verified. (Not our platform.)
- **Stock GPU (Appendix D.1): fused decode + tensor-core GEMM**, low-batch (≤8) regime: each
  lane reads a 128-bit code window, decodes 32 values straight into mma operand registers; the
  stream→fragment map is a **fixed permutation pre-applied to the weights** (no runtime shuffles);
  63-entry FP16 table = one per shared-memory bank, conflict-free gather. INT4 baseline = same
  kernel minus trellis decode, so the CPTC-vs-INT4 gap isolates the decode cost (Table 23, L40S
  numbers; decode cost is small at K=4, growing toward K=6).
- RDNA4 translation: shfl ✓, per-lane table gather ✓, WMMA instead of mma.sync — the kernel
  structure ports; our BetterBench 27B finding (int8-ALU decode flips compute-bound at batch
  4–8) applies as the cost model to beat/measure.

## Encoder

- Viterbi O(T·2^16) tail-biting (two passes); their fused CUDA encoder: 8B linear layers in
  **<2 min on one GPU**, 2,400× faster than QTIP's released (batched-PyTorch) encoder at equal
  quality. A CPU reference is easy to write (L=16 × per-tensor is minutes on our CPU boxes;
  27B = 40 layers × ~10 linears = tens of minutes; DS41 = 46,080 experts is the big one — but
  experts are 2048×4096-class, so per-expert Viterbi on CPU is ~ms each → hours, fine).
- Table fitting: Lloyd iteration on the trellis over synthetic target samples, per (distribution,
  rate, depth), offline — trivial.

## Relation to our stack

- Our EXL3-V3 "mad" (single int codebook, decode fused into the GEMM as 2 GEMMs) and CPTC are
  the same category (trellis format, per-tensor table/scale, RHT) with different decoders:
  GEMM-factor fusion (mad) vs per-value scalar decode fused into the GEMM prologue (CPTC).
  The decisive question is KLD-per-decode-cycle on RDNA4 at our sizes — an experiment, not a
  paper argument.
- IQR's default codebook is the QTIP HYB family → bakon's DS41 IQR v5 (u3, 16-state trellis) is
  exactly the "HYB" arm of these tables: on 27B the paper says CPTC-K3 ≈ 2.9× lower DKL than
  HYB-K3. Relevant to their open question 3 (codebook) in the #266 thread; shareable if the
  user wants.
- 27B V3 card: frozen, unchanged (different pipeline and corpus; no comparability claim either
  way until we run it).

## Verification plan (proposed, cheap first)

1. **CPU reference implementation** from the paper (Algorithm 1, Appendix A.2 coefficient pairs,
   Lloyd-trellis table fit, tail-biting Viterbi encoder) — a day of CPU work (VM + Thor/colo).
2. **Proxy KLD** (0.8B/2B, 4070 box or CPU): CPTC K∈{2,3,4} vs our EXL3-V3 mad at the same K,
   same rotation/scale conventions, same slice — the 27B confirmation queue rule applies.
3. If the proxy gap reproduces: **RDNA4 decode kernel** (r4d unit, port of their D.1 structure)
   + full 27B KLD on our card's corpus (we have same-corpus IQR-V4 and V3 rows to hang it on)
   + prefill/decode timing vs exl3r4d mul1/mad.
4. Decision per the global-vs-knapsack rule: if CPTC-K beats mad-K at its measured decode cost
   for most tensors → adopt globally as a compute mode; if tensor-dependent → knapsack option
   with its KLD sensitivity + timings. DS41 P1 (mad baseline) proceeds as planned; CPTC enters
   after the mad baseline exists (it needs the same per-tensor scale/rotation plumbing and the
   per-expert encoder pass our G3 calibration already provides).
