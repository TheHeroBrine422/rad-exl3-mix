# New quant format (EXL3-V3): experiment tracker (the one place for this work)

Naming: EXL3-V1 = turboderp port, EXL3-V2 = our mixed-K EXL3 on the f16 kernel (finish first), **EXL3-V3 = this
(integer compute)**. The int8-codebook requant and other big V3 GPU work wait until V2 is done.

**Last updated:** 2026-10-30, ~17:00 card time.

This file holds every experiment for the new fast format: questions, status, results and decisions.
- Design rationale: `DESIGN-new-format.md`.
- Dropped ideas: also add a short entry to `EXPERIMENTS.md`.
- The EXL3-V2 search run is tracked in the working TODO and the pipeline's SUMMARY.md, not here.

**Status:** `queued` · `running` · `done` · `dropped` · `idea` (not yet planned).

**Where:**
- **card:** the GPU box, in gaps between evals.
- **cpu:** a CPU machine.
- Rule: run on CPU only when the GPU is less than ~20× faster, or when the job takes minutes.

**Shared setup unless noted:**
- E1b runs: our K6 EXL3 weights, scored on golden-calA (fit pool, 90.8k positions). Baseline KLD **0.00101**, top-1 98.75%.
- Paired SE on this slice is about ±1.5%, so smaller differences aren't real.

---

## Running / queued now

| id | what | where | status |
|---|---|---|---|
| E1b-batch1 | A4 per family, per depth band, A4 g128 | card | done |
| E1b-batch2 | A8: g128 / g32 scales, per family, all-but-down/head | card | done |
| E1b-batch3 | A4 fixes on gate_up and down: g128, clip 0.85/0.7/0.55, g128+clip | card | done |
| AQ-proxy | per-tensor activation-quant proxy on dumped real inputs (`e1b/aqproxy.py`), 337 tensors × 32 configs | CPU box A | done; calibrated (`aqcal.py`) |
| V0 | validation of 12 knap3 solutions (2/3/4/6 bpw × f16 control + faster points): EXL3 stand-in + E1b masks (`nf/nfval.sh`) | card | running |
| E3 data | all CPU sweeps collected from CPU box B + CPU box A into `v3/e3/data` (2026-10-31) | — | done |
| E3-sweep2 | integer codes (1024 sampled rows) + trellis codebooks (256 rows), every linear of layers 10/43 (`sweep2.sh`, resumable) | CPU box B | running, 44 jobs at once (`/data/work/e3`); 70/264 were done on the VM |
| DL | calib Hessians from HF onto the VM (47 GB, rev `2dcf0061`) | net | done → `/workspace/e3data/challenge` |
| E2 | int8 / int4 GEMM at real tiling (`igemm.hip`) | card | per-token, M sweep, group scales measured; next: decode pre-pass, scale tricks, split-K |
| YAQA | output-side Hessians (`v3/yaqa/`): gate 1 TP2 throughput, gate 2 two-sided LDLQ on blocks 22/23 at K2/K3 (`gate2.sh`) | both cards | gate 1 passed (5 s per 2048-token sequence); gate 2 (bf16 backward) failed: two-sided +23% (K2) / +28% (K3) worse; noise test: bf16 backward wrong in early/mid layers; gate 2b (fp32 backward, `gate2b.sh`) running |

---

## E0: hardware facts (done)

gfx1201, measured; ISA checked for every number.

| op | peak |
|---|---|
| iu8 16×16×16 | 323 TOPS |
| fp8 16×16×16 | 321 TOPS |
| iu4 16×16×32 | **685 TOPS** |
| iu4 16×16×16 | 328 TOPS |
| f16 (earlier) | 158 TF |
| fp4 / fp6 | none (gfx1250 only) |

Activation scales applied along K (`wmmagrp.hip`; a tight loop, so the worst case: a real GEMM overlaps some of
the scaling work with loads):

| scale group | iu4 TOPS | iu8 TOPS |
|---|---|---|
| 64 | 331 | 220 |
| 128 | 444 | 265 |
| 256 | 536 | 295 |
| 512 | 596 | 312 |
| 1024 | 631 | 323 |
| per token / per channel only (epilogue) | ~685 | ~323 |

**Decision:** compute is integer (iu8 + iu4), not fp8; W4A8 is dropped. Group scales along K are expensive, so the
scales are per token and per output channel.

## E1 / E1b: activation quantization cost (KLD; hook in `v3/e1b`, runner `e1brun.sh`)

The hook fake-quantizes the rotated GEMM input, one absmax scale per token row unless noted.
- No-op run: **byte-identical** to the normal K6 score.
- Map: 337 linears in issue order (`mkmask.py`).

### A8 (int8)

| run | KLD | added | top-1 |
|---|---|---|---|
| all linears | 0.00149 | **+0.00047** | 98.47% |
| all, **g128** activation scales | 0.00106 | **+0.00005** | 98.71% |
| all, g32 | 0.00104 | +0.00003 | 98.68% |
| all, g256 | 0.00118 | +0.00017 | |
| all, g512 | 0.00119 | +0.00018 | |
| all, g1024 | 0.00116 | +0.00015 | |
| all, per-row, clip 0.9 | 0.00221 | +0.00120 | (clipping hurts A8) |
| per family, per-row scale: attn_in / attn_out / gdn_in / ssm_out | 0.00103 / 0.00100 / 0.00101 / 0.00105 | ≤ +0.00004 | |
| per family, per-row scale: gate_up / down / head | 0.00112 / 0.00103 / 0.00111 | +0.00011 / +0.00002 / +0.00010 | |
| all but down and head, per-row | 0.00124 | +0.00023 | 98.57% |

- A8 everywhere costs more than our best 7.5 bpw EXL3 mixes measure in total (0.00026–0.00035). IQR's 8.5 bpw
  I8A8 entry (0.00046) is consistent with this.
- **Batch2 changes the picture:** the A8 floor is almost entirely the per-row scale. With one scale per 128
  inputs, A8 everywhere costs only +0.00005 (within noise of the ±1.5% SE at this level, ~5%).
- Per-row, the cost is spread (gate_up and head the largest) and compounds: the families sum to ~+0.00028 alone,
  +0.00047 together.
- **Open:** g128 scales along K are not free in an int GEMM (iu4 at g128: 484 of 685 TOPS). For iu8 a scale applies
  every 8 WMMAs instead of 4, so the cost should be about half that. Measure in E2 (iu8 with g128 float rescale).
- **Decision (revised):** A8 with g128 scales is near-free for quality. A16 stays as an option only for the very top
  of the Large tier, and only if E2 shows g128 iu8 is slow.

### A4 (int4)

| run | KLD | added | top-1 | ~share of linear FLOPs |
|---|---|---|---|---|
| all linears | 0.150 | +0.149 | 83.9% | 100% |
| attn_out | 0.0085 | +0.0075 | 96.5% | ~2% |
| ssm_out | 0.0141 | +0.013 | 95.5% | ~5% |
| attn q/k/v | 0.0237 | +0.023 | 94.3% | ~5% |
| gdn_in (qkv, z) | 0.0241 | +0.023 | 93.9% | ~13% |
| down | 0.0262 | +0.025 | 93.7% | ~14% |
| gate_up | 0.0264 | +0.025 | 93.0% | ~28% |
| head | 0.0333 | +0.032 | 91.0% | ~10% |
| layers 0–15, all linears | 0.0295 | +0.028 | 93.8% | 25% |
| layers 16–31, all linears | 0.0199 | +0.019 | 94.6% | 25% |
| layers 32–47, all linears | 0.0305 | +0.029 | 93.5% | 25% |
| layers 48–63, all linears | 0.0407 | +0.040 | 91.9% | 25% |
| all linears, **g128** scales | 0.0312 | +0.030 | 92.3% | 100% |
| gate_up, g128 | 0.0100 | +0.009 | 95.8% | ~28% |
| gate_up, per-row, clip 0.85 / 0.7 / 0.55 | 0.0183 / 0.0146 / 0.0195 | +0.017 / +0.014 / +0.018 | | ~28% |
| gate_up, g128 + clip 0.85 | 0.0097 | +0.009 | 96.0% | ~28% |
| gate_up, g256 | 0.0127 | +0.0117 | | ~28% |
| gate_up g64 / g64 + clip 0.85 | 0.0092 / 0.0115 | +0.008 / +0.010 | | ~28% |
| gate_up g256 + clip 0.85 / g512 / g512 + clip 0.85 | 0.0100 / 0.0144 / 0.0111 | +0.009 / +0.013 / +0.010 | | ~28% |
| down g64 / g64 + clip 0.85 | **0.0048** / 0.0052 | **+0.0038** / +0.0042 | | ~14% |
| down g256 / g256 + clip 0.85 | 0.0073 / 0.0064 | +0.006 / +0.005 | | ~14% |
| down g512 / g512 + clip 0.85 | 0.0088 / 0.0075 | +0.008 / +0.006 | | ~14% |
| down, g128 | 0.0070 | +0.006 | 96.8% | ~14% |
| down, per-row, clip 0.85 / 0.7 / 0.55 | 0.0205 / 0.0190 / 0.0365 | +0.019 / +0.018 / +0.035 | | ~14% |
| down, g128 + clip 0.85 | **0.0055** | **+0.0045** | 97.0% | ~14% |

- Crude A4 costs +0.0075 to +0.03 per family, about a full bit of weights for ~20% of GEMM time.
- **g128 scales cut A4 cost ~3–5×** (all linears 0.150 → 0.031; gate_up +0.025 → +0.009). Clipping helps less
  (gate_up 0.7: +0.014). The later layers are the most sensitive (48–63 is 2× layers 16–31).
- Still about a half-bit of weights for ~28% of GEMM time at gate_up, and g128 costs iu4 ~30% of its speed. Marginal
  at 3–4 bpw. Atom-style outliers, full Hadamard and QAD come after.
- **Batch3:** g128 is the big lever; clipping adds a little on top of g128 (down 0.0070 → 0.0055) and alone is
  worse than g128.
  - Best: down at A4 for +0.0045 (~14% of GEMM time), gate_up +0.009 (~28%).
  - In weight terms, +0.0045 is ~15% of the 3.0 bpw mixes' total KLD (0.028–0.032): affordable for Nano/Tiny if the
    speed is real.
  - Too much for 4 bpw (0.008 total).
- **Batches 4–7:**
  - **A8:** g256 / g512 / g1024 all land at about +0.00015–0.00018; only g128 and smaller go lower (+0.00005). So
    the A8 choice is effectively per-token / ≥256 (fast) vs ≤128 (most accurate).
  - **A4:** clipping helps at large groups (g256/g512) and hurts at g64. Best: down g64 +0.0038, gate_up g64 +0.008.
- **Status: A4 g128 stays a candidate option at ≤ 3.5 bpw**, pending E2 (iu4 at g128 measured 484 TOPS in the
  microbench = still 1.5× iu8).

## E1c: per-tensor activation costs (proxy, `v3/e1b/aqproxy.py`, `aqcal.py`, data in `v3/e1b/data`)

- **Dump:** the E1b hook's `EXL3_AQ_DUMP` saved 512 real input rows per linear (calib-A, K6 model). The dump run's
  KLD is byte-identical to the no-op.
- **Scoring:** CPU box A scored every tensor × {A8: per-row, g1024…g32; A4: per-row, g512…g32} × clip:
  rel_out = Σ‖W·e‖² / Σ‖W·x‖², with the hook's exact quantizer and rotation. 10,784 rows; ~10 s per tensor.
- **Calibration** (refit on 45 runs after batches 4–7: log-RMSE 0.25, factors unchanged within 3%), **first fit on 37 E1b KLD runs:** KLD increment ≈ Σ_families k_f · Σ rel_out.
  - Per-family factors give log-RMSE **0.26** (≈ ±30%) over four orders of magnitude; one global factor gives 1.19.
  - Factors: gate_up 0.020, attn_in 0.015, down 0.0097, attn_out 0.008, gdn_in 0.006, ssm_out 0.005, **head 1.07**.
    The head's logit error counts ~100× more per unit of relative output error.
  - Worst misses: tiny A8 runs near the noise floor.
  - Depth bands show late layers more sensitive than the proxy (L48–63 under-predicted 1.7×): add a depth term.
- **Group size matters very unevenly across tensors:** A4 rel_out per-row → g128 drops ×2 on L0 gate_up but ×28 on
  L3 q/k/v. This is exactly what a per-tensor choice exploits.
- **Use:** these are the knapsack's per-tensor activation costs.

 (`v3/bench/igemm.hip`)

- **Kernel:** T2's tile (128×256, LDS-staged activations, weight fragments from global), iu8 and iu4. Per-token ×
  per-channel scale in the epilogue; outputs exact vs a naive reference.
- **First run, no tuning, card 1:**

| shape (M = 4096) | iu8 | iu4 | f16 T2 core (~135 TF) |
|---|---|---|---|
| gate_up 34816×5120 | 5.60 ms (261 TOPS) | 2.96 ms (494) | ≈10.8 ms |
| down 5120×17408 | 2.91 ms (251) | 1.46 ms (500) | ≈5.4 ms |
| GDN in 10240×5120 | 1.71 ms (251) | 0.89 ms (485) | ≈3.2 ms |
| M=512, 5120×5120 | 195 TOPS | 379 TOPS | |

- **GEMM core:** iu8 ≈1.9× our f16, iu4 ≈3.7×; 78–81% / 71–73% of peak.
- **M sweep (TOPS):**

| M | gate_up iu8 / iu4 | down iu8 / iu4 |
|---|---|---|
| 512 | 237 / 385 | 189 / 357 |
| 1024 | 253 / 477 | 241 / 384 |
| 2048 | 259 / 493 | 248 / 482 |
| 4096 | 261 / 494 | 251 / 500 |
| 8192 | 264 / 495 | 255 / 510 |

  `down` at M = 512 is under-occupied (too few tiles): needs split-K.
- **Group activation scales in the real kernel** (float accumulators per group, gate_up M = 4096):
  - At the 128×256 tile, the extra float accumulators **spill** (119 VGPRs): iu8 g128 73 TOPS. Unusable.
  - At a 128×128 tile (MT4 NPW2), no spills:

| | per token | g1024 | g512 | g256 |
|---|---|---|---|---|
| iu8 | 254 | 225 (−12%) | 220 (−14%) | 210 (−17%) |
| iu4 | 488 | 390 (−20%) | 375 (−23%) | 346 (−29%) |

  - First version. To try: prefetching the group scales; power-of-two group scales applied as integer shifts (no
    float accumulators, ~1 bit of scale resolution lost).
- **To do:**
  - the trellis → int8/int4 fragment decode pre-pass;
  - the activation quantize step fused into the rotation;
  - an M sweep (incl. expert-sized shapes);
  - tuning.

## E3: weight codes, quality per bit (Hessian proxy; `v3/e3/e3.py`)

- **Metric:** rel = tr(dW H dWᵀ)/tr(W H Wᵀ), the same as hproxy.
- **Shown:** median ratio to our EXL3 ours-K at the same bits.
- **Data:** 3 tensors so far (layers 10/43).

| bits | GPTQ, per-channel int | GPTQ g128 → int8 | Lloyd LUT → int8 | RTN (rotated) |
|---|---|---|---|---|
| 2 | ×2.3 | ×1.7 | ×1.7 | ×12.7 |
| 3 | ×2.1 | ×2.0 | ×2.0 | ×11.3 |
| 4 | ×2.7 | ×2.3 | ×2.2 | ×13.0 |
| 5 | ×3.1 | ×2.5 | ×3.0 | ×13.8 |
| 6 | ×3.5 | ×3.4 | ×4.5 | ×15.3 |
| 8 | ×3.5 | ×19 | ×12 | ×16.3 |

- Plain integer codes cost about 0.6–0.9 bit against the trellis, and GPTQ is mandatory.
- Folding group scales into int8 has a floor around 1e-4 (≈ K6).
- **On paper:** the EXL3 mul1 codebook rounded to int8 adds ~6e-5 relative error. That's
  +4% on K4, +17% on K5, +60% on K6, and it dominates K7–8.
- **Decision (provisional):** a trellis decoding to int8 for ≤ ~K5. Above that, f16 compute (EXL3 as is).

## E3b: trellis decoding to int8 (`v3/e3/viterbi.c` + `trellis.py`)

- **Tool:** a CPU reimplementation of EXL3 quantization on sampled rows. It uses block LDLQ, a 16-bit bitshift trellis
  over 16×16 tiles, and a per-row RMS scale. Simplifications: free start state (not tail-biting) and no output-scale
  refit.
- **Validation:** layer 10 `ssm_z`, K4, 64 rows: rel 5.37e-4 vs our real ours-K4 4.16e-4 (×1.29). It's slightly worse
  than exllamav3 but good for relative comparisons between codebooks.

| tensor | K | float mul1 | mul1 rounded to int8 | int8 cost |
|---|---|---|---|---|
| L10 ssm_z (64 rows) | 4 | 5.37e-4 | 5.43e-4 | **+1%** |

- **First result:** at K4, rounding the trellis codebook to int8 costs about 1%, below the +4% paper bound. Trellis
  quality survives int8 compute at mid bits.
- tsweep1 covers K2–6 plus a Gaussian int8 table and an int4-grid codebook.

## E3b results: int8 / int4 trellis codebooks (tsweep2, all 11 tensors of layers 10 and 43, 256 rows)

Error relative to EXL3's float mul1 codebook:

| codebook | K2 | K3 | K4 | K5 | K6 |
|---|---|---|---|---|---|
| mul1 rounded to **int8** | ×1.000 | ×1.000 | ×1.001 | ×1.002 | ×1.004 |
| Gaussian int8 table | ×1.008 | ×1.008 | ×1.108 | ×0.975 | ×0.969 |
| mul1 on the **int4** grid | ×1.010 | ×1.065 | ×2.72 | ×10.5 | ×39.6 |

- **Int8 compute keeps all of EXL3's weight quality** through K6.
- **Full set (2026-10-31, layers 0/10/31/43/62/63, 33 tensors; data in `v3/e3/data`):**
  - mul1 rounded to int8: K2–K6 ×1.000–1.004 (range 0.99–1.01); **K7 ×1.25, K8 ×4.5** (9 tensors).
  - int4 grid: K2 ×1.01, K3 ×1.07, K4 ×2.7.
  - **So:** at K7–K8, 8-bit weight values can't hold the code. V3's top end needs f16 compute (or a 9-bit-ish
    decode) for K≥7 tensors; int8 is fine up to K6. The earlier "K≥5 requant" estimate was for *post-hoc* rounding of
    existing quants (k0 run); E3b's searched codes say K5–6 are fine at int8 when quantized with the rounded codebook.
- **W4A4 keeps trellis quality at K2–K3.** Only the activation side costs there.

## E3c: row-slab K (`v3/e3/rowslab.py`, `rowslab.jsonl`)

- **Method:** the ceiling of re-allocating bits across output-row slabs at the same total error,
  ½·log₂(AM/GM) of the slab Hessian norms (input side only). 49 tensors over 9 layers.

| slab | param-weighted gain, all tensors | gate_up + down only |
|---|---|---|
| per row | 0.29 bit | 0.18 bit |
| 128 rows | 0.12 bit | 0.05 bit |
| 256 rows | 0.11 bit | 0.045 bit |

- **Outliers:** a few small early tensors carry most of the gain: ssm_out L0 3.2 bit, attn_output L3 2.2,
  attn_qg L3 0.9, down L0 0.5 (at S128). They are residual writers with "massive activation" channels.
  - The whole-tensor knapsack already handles them cheaply (these tensors are small), and input-side norms likely
    overstate how much those channels matter downstream.
- **For the big tensors:** ~0.05 bit at 128-row slabs (a ceiling; integer K steps get less).
- **Decision:** not worth kernel complexity now; low priority. Revisit with output-side importance (YAQA H_out), which
  could widen the spread, or as "a few rows at bf16" for the outlier tensors.

## K3: the new-format solver (`v3/nf/knap3.py`, model from `nf/kfit2dump.py`)

- **Options per tensor:**
  - f16: any EXL3 option or bf16.
  - a8: ours-K1..8 trellis → int8 (×F8) × group {pt, 1024..64}.
  - a4: ours-K1..3 on the int4 grid (×F4) × group.
  - Shared-input linears (q/k/v, GDN qkv/z) share one mode + group.
- **KLD:**
  - weights: kfit2's fitted compounding model (551 measurements, dumped from round 9);
  - activations: additive, from E1c.
- **Time:** E2 TOPS at M = 4096 + a decode pre-pass (placeholder 0.006 ns/weight). Non-GEMM work is 8% of today's step.
- **Solve:** min KLD + λ·time, bytes strictly under the target. Layered DP with stage weights g_l = ∂KLD/∂X_l (two
  passes). ~2 s per point.
- **First frontier** (`nf/frontier-v0.jsonl`; prefill speedup vs today's all-f16 EXL3):

| target | all f16 | ×1.3 | ×1.5 | ×1.65 |
|---|---|---|---|---|
| 3.0 | 0.0256 | 0.0257 (+0.2%) | 0.0258 (+0.6%) | 0.0269 (+5%) |
| 4.0 | 0.00650 | 0.00657 (+1%) | 0.00665 (+2%) | 0.00707 (+9%) |
| 6.0 | 0.00063 | 0.00070 (+11%) | 0.00080 (+27%) | 0.00107 (+70%) |

- **Readout:**
  - At 3–4 bpw, a8 is nearly free: 90–95% of FLOPs at a8, a few sensitive tensors at f16.
  - At 6 bpw, the activation cost is a large share, so A16 matters.
  - A4 is almost never chosen.
- **Caveats:**
  - The decode pre-pass time is a placeholder; g128/g64 speeds are extrapolated.
  - The KLD model is unvalidated for new-format mixes. It needs the E4 prototype to measure real mixes, then a refit.

## E4: real int8 prefill kernels (`v3/nf/kern/exl3r4d`, build `/root/nfk`, runner `/root/e1b/nfkrun.sh`)

- **Kernels** (per-token activation scales only so far):
  - `nfq_act_i8` quantizes the prep kernel's rotated rows to int8 in fragment order.
  - `nfq_wdec_i8` is T2's trellis decode with int8 output, rint(v / 4 k_inv).
  - `nfq_gemm_i8` is the iu8 WMMA core with T2's epilogue.
  - Switched per tensor by `NFQ_MASK` ('8'), with the hook's indexing.
- **First end-to-end run** (k0, ours-K6, calib-A, all 337 linears int8): **KLD 0.00178, top-1 98.13%, eval wall
  74 s** (f16 100 s; fake-quant A8 per-token 0.00149).
  - Correct first try; the whole eval is 26% faster untuned (including load and scoring).
  - The +0.0003 over fake-quant is the **post-hoc int8 rounding of an existing EXL3 quant**, matching the paper bound
    (K4 +4%, K5 +17%, K6 +60% weight error). E3b's ×1.004 needs the trellis searched with the int8 codebook.
  - **So:** K ≤ 4 existing quants run as is; K ≥ 5 should be requantized with the int8-rounded codebook. That's a
    small change to exllamav3's codebook (same hash, rounded values), a ~6 h requant of K5–8 (or K1–8).
- **Next:** group scales in the real kernel, the requant, split-K for small N, and per-op timing (`--profile-ops`)
  for the real time table.

## E4b: end-to-end prototype at ~4 bpw (after the above)

Trellis → int8, W8A8 everywhere except the A16 picks. Measure KLD and prefill/decode speed against ours-K4 EXL3.

---

## Backlog (rough priority within each group)

### Activations
- [ ] A8 per family (batch2): find the tensors that need A16.
- [ ] A4 g128 / clip (batch3). Then Atom-style outlier channels at A8 within a tensor; full-dim / learned
      (SpinQuant) rotations.
- [x] Activation dump + CPU proxy (E1c): per-tensor costs, ±30% vs KLD with family factors. Next: a depth term.
- [ ] A4 by depth band at fine resolution, if bands differ a lot.
- [ ] Per-tensor clip ratios (searched on fit pool).

### Weight codes
- [ ] **Quarter, continuous and row-by-row K** (2026-10-31): a pattern-driven fractional trellis gives any
  K = a/b (quarter steps 2.25/2.75 in the focus range; dense / continuous menus per tensor). Row-by-row K = per row
  slab (rows ranked by sensitivity), cf. E3c's ~0.05-bit ceiling. Estimated gain beyond half steps: quarter ~1.5%,
  continuous < 1% more; row slabs ~0.05 bit on big tensors. Same machinery as half steps; cost is quant runs.
- [ ] **Fractional / sub-1 K for V3** (2026-10-31): K0, K0.5, K1.5, K2.5 (and other half-steps). K1.5 sits between the K1 and K2 tensors the < 2.5 bpw mixes use most. Half-steps help at every K (~5–6% KLD); once the fractional trellis exists, extra Ks are just quant time (~1 h each) -- order 1.5, 2.5, 3.5, 4.5, then 5.5+.
  - **K2.5 etc.:** a trellis shifting 5 bits per 2 weights. Estimate: half-steps cut the knapsack's discretisation loss
    from ~8% to ~2% of KLD (exponential D(R), rounding to ±0.5 vs ±0.25 bit) → ~5–6% lower KLD in 2.6–4.5 bpw,
    worth ~0.1 bpw.
  - **K0.5:** a 1-bit-per-2-weights trellis. Mainly for the < 2 bpw mixes; they put 51–154 tensors at K1.
  - **K0:** drop a tensor / replace it with a low-rank approximation. Candidates: early ffn_down, where K2 costs
    +0.00006 on layer 0. Testable now with probes (zero / low-rank donor containers).
  - **Cost:** quantizer (exllamav3 kernel), decode kernels, container format. Do with the int4-grid codes.
- [x] Small-sample Viterbi check (E3b): tool built, K4 int8 +1%; full K2–6 sweep queued.
- [ ] Trellis whose codes land on the int4 grid: the W4A4 path at 2–3 bpw.
- [x] Row-slab K (E3c): ceiling ~0.05 bit on the big tensors at 128-row slabs; low priority.
- [ ] Quantizer knobs: damping, column order, re-Viterbi. YAQA A/B on a few layers (low expectations).
- [ ] Full-tensor / full-model quants: **GPU only** (LDLQ + Viterbi).

### Kernels
- [ ] E2 timings (above). Then W8A8 with trellis decode in a pre-pass (T2-style) vs fused.
- [ ] iu4 kernel with g128 scales, if batch3 shows g128 makes A4 viable.
- [ ] Activation quantize kernel fused into the rotation (prep) pass.
- [ ] Grouped-GEMM speed at small M per expert (for Qwen Flash, later).

### Optimizer / frontier
- [ ] **Prime V3 from V2** (2026-10-31).
  - V3's fit starts from V2's final kfit2 model as a prior: per-tensor sensitivities, compounding, embedding /
    head costs.
  - Every V2 measurement enters as an all-f16 point (activation error 0). For a8 at K <= 6 the weights are the same
    (E3b ×1.000–1.004).
  - Fit only the new terms: activation factors, the activation × weight interaction, the int4-grid / int8-aware
    options.
  - Measure V3 mixes via the EXL3 stand-in + hook masks until the kernels land. Start from the best measured V2 mixes.
- [ ] **V3 pipeline: streaming instead of rounds** (2026-10-30).
  - **Loop:** fit runs continuously on a CPU box (CPU box A / CPU box B). Each fit re-solves all targets and replaces the
    not-yet-started mix candidates. The cards pull the freshest candidates plus uncertainty swaps (`uncswaps.py`).
  - **Gain:** a misfit is caught after ~10 evals, not a round, and the cards are never idle between rounds.
  - **Needs:**
    - explicit pairing for same-slice A/Bs (the slice offsets are already fitted);
    - fit→mix provenance for pred-vs-meas tracking;
    - Thompson exploration kept.
- [ ] **V3: ours-only option set** (2026-10-30). Drop turboderp's containers as options. V3's K≥5 requant
      with the int8-rounded codebook makes every option ours anyway, since turboderp's quants can't be requantized.
      - Keep their existing measurements in the fit, as data on the per-tensor sensitivities. Those don't depend on
        which quantizer made the tensor.
      - Drop their family factors from the option set.
      - Re-check CV and held-out ver1 after the switch.
      - Round 11's ours-only A/B (V2) gives the first evidence.
      - Later (a day or two in): drop turboderp measurements from the fit too, once our own measurements cover every
        family × depth band. Test: refit without them; CV and the ver1 ranking must not get worse.
- [ ] 3D knapsack: DP over bytes × time (or a Lagrangian on time), per-tensor options
      {W·A16, W·A8, W4A4} × K × activation group size {per-token, 1024, 512, 256, 128, 64} (decided 2026-10-30:
      32 and K/N splits dropped; per-token kept as the fastest option). Linears sharing an input
      (q/k/v, gate_up) are one item with one activation format. Per-tensor activation costs come from the
      activation dump + CPU proxy, scaled by the E1b family KLDs. Develop on CPU against estimated time costs until E2 lands.
- [ ] Frontier display: KLD vs prefill speed, one curve per bpw slice.
- [ ] Speed model: prefill from E2, decode from bytes / bandwidth + dequant.

### Embedding / head
- [ ] Vocab-tiered head (simulate first, multilingual guards).
- [ ] Trellis embedding vs Lloyd at tiered widths (E5).

### QAD (not started)
- [ ] Bit-exact torch reference of the kernel's dequant.
- [ ] Trainable scales/norms, STE fake-quant for A8/A4, teacher top-k logits cached from the qad pool.

### Other models
- [ ] Qwen Flash (MoE): W4A4 / A8 per expert by importance (routing frequency, H_out mass).

---

## Log (newest first)

- 2026-10-31 ~21:00: **radiance PR #362** (prefill softmax rescaled per row, not per wave) measured on our eval
  (`eval/pr362/run.sh`, golden-ver1, V2 3.00): KLD −0.12% (unpacked) / −0.13% (packed), z ≈ −2.2; 6.00 −0.5% ± 1.0%.
  Packed vs unpacked differs by +0.007% today, −0.001% with the fix. Real but ~10× below the ±1.5% decision noise:
  no V2 action; take it with the 1.15.x port (V3 re-baseline then).

- 2026-10-31 ~21:00: **YAQA: bf16 backward is too noisy** (`noise.sh`, `cmp.py`: same 8 sequences, positions and
  labels, rows kept in fp32). Per-row cosine bf16 vs fp32 for down_proj: layer 63 median 1.000 (p10 0.999), 40:
  0.998 (p10 0.66), 20: 0.988 (p10 0.02), 2: 0.92 (p10 < 0); the bad rows are 2.7–3.2× too large, so they dominate
  H_out. H_out diagonal cosine 0.88 at layers 48–63, 0.36 at 0–15. The bf16 forward itself diverges on 2/8
  sequences (seq 7 loss/token 1.03 vs 0.83). fp32 mode: 30 s per sequence (5.3 s bf16) after a ~15 min warm-up,
  28.4 GiB peak per card. Likely cause of gate 2's failure (blocks 22/23, diag cosine ~0.5).

- 2026-10-31 ~19:50: **YAQA gate 2 failed with the bf16 pilot** (200 sequences, blocks 22+23 swapped into the V2
  6.00 release, golden-ver1; base 0.000703). KLD increment over base: K2 one-sided +0.00290, two-sided +0.00357
  (+23%); K3 one-sided +0.00067, two-sided +0.00086 (+28%). Suspects: bf16 noise (confirmed above), 6,400 rows for
  17,408-wide H_out, output regularisation 0.025. Rerun in fp32 with sigma_reg_out 0.025 and 0.3 (`gate2b.sh`).

- 2026-10-31 ~20:00: **LM head sensitivity** (`v3/yaqa/headsens.py`, exact Fisher diagonal sum_t p(1−p) from the
  calib golden's top-512): ~400 of 248,320 rows hold 50% of the mass, 16–19k hold 90%, 68–90k hold 99%; the four
  pools agree (verify included). Supports a row-slab head; slab test next (needs `ourq.py --save-wq` head quants).

- 2026-10-31 ~19:00: **YAQA gate 1 passed.** `v3/yaqa/collect.py`: bf16 text model streamed tensor by tensor
  onto both cards (layers split 32/32; the transformers loader OOMs host RAM on the 30 GB box), fla 0.6.0 chunked
  gated-delta, gradient checkpointing, chunked sampled-label CE (labels drawn from the model's own distribution, as
  YAQA's Fisher). Per sequence it records r_t = ||x_t|| dy_t at 32 random positions for all 496 trunk linears (≈233 MB
  bf16). 5.0 s per 2048-token sequence including writes, peak 26.9 GiB per card; the qad pool (1,819 windows,
  3.7M tokens) is ~2.7 h and ~425 GB of rows. `fold.py` sums them into H_out (exllamav3 `--hessians` layout,
  "hout"). `ourq.py --hout DIR` passes them to exllamav3's two-sided LDLQ (`ldlq_2hess`). Gate 2 (`gate2.sh`):
  200-sequence pilot H_out (6,400 rows: rank-deficient for the 17,408-wide gate/up, which leans on sigma_reg_out),
  blocks 22+23 at K2/K3, one- vs two-sided, swapped into the V2 6.00 release, golden-ver1.

- 2026-10-31 ~04:30: **vocab-tiered embedding dropped**. Early data: 30k@3+rest@1 KLD 0.0140 vs uniform e4
  0.0081 (bf16 emb 0.0080); 30k@3+rest@2 (2.25 b/value) 0.0094 vs uniform e3 (3.125) 0.0086; on the multilingual
  guard slice 30k@3+rest@2 is +40% over bf16 (uniform e4 +4%). Rare-token rows matter more than frequency predicts.
  Est. gain 0–3% at medium-low confidence, not worth the risk to international users. Remaining plans finish for the
  record; results get added to EXPERIMENTS.md as a dead end.

- 2026-10-31 ~04:00: **IQR-V3 published** (whitepaper https://radiance.stilldeadco.de/whitepapers/iqr-v3.pdf, HF
  `StillDeadcode/qwen3.8-27b-iqr-v3`, kernels in radiance 1.15.0, ABI 32). Board numbers unchanged (they were
  already scored). What it means for us:
  - **Convergence on EXL3's design:** computed trellis code (3 ALU ops from a 16-bit state, like EXL3's mul1), L=16,
    one f32 scale a row (EXL3: svh). New for them, and the template for our V3: the code's values *are* int8
    (q(s) in [-126, 126], sum of four 6-bit bytes of a hash), so the Viterbi searches against the exact integers the
    int8 tile computes. Our E3 found mul1 rounded to int8 costs ×1.25 at K7 / ×4.5 at K8; an int8-native code
    removes that at the source (our planned int8-aware requant ≈ this).
  - **Prefill fold:** with one scale a row (applied at the store) and a per-128 block factor shared by all rows,
    their int8 tile folds like FP8's: 1.00–1.06× FP8 time vs 1.19–1.21× with per-(row,128) scales. Our E2 measured
    the same group-scale penalty (−12…−29%). EXL3 already has the row form, so our V3 int8 path can use it as is.
  - **Head-fed projections** (attn_output, ssm_out, 8% of trunk) need a larger scale gain from 5 bits (1.36–1.6 vs
    1.16–1.24) and still lose to block scales at 6–7 bits. Check our K5+ quants of these two families.
  - **GPTQ damping 0.05 → 0.01: −3.5% KL** on 8 layers at 3 bits. exllamav3 adds sigma_reg 0.025 × *mean*
    diag(H) to every channel, ≈ 22% of the median channel's curvature with massive-activation channels (mean ≈ 9×
    median): the same over-damping. `ourq.py --sigma_reg` exists. Cheap test: one layer's tensors at 0.025 / 0.005 /
    0.0025, hproxy error. If it holds, a requant of ours K2–6 (~6 h card time) is a V2.1/V3 quality step.
  - Their GEMM tables (decode GEMV vs FP8 per width, prefill per width) are V3's speed reference.
- **radiance 1.15.0 / ABI 32:** our plugins (exl3r4d, exl3dense, qwen35_exl3; PR #292 on 1.13.0) must be rebuilt
  and re-validated (byte-identical KLD) before V2 ships; V2 should target 1.15.0.

- 2026-10-31 ~03:10: **vocab-tiered head, step 1 (simulation)** started.
  - Ranking: token counts on search + qad pools (7.16M tokens; top 30k ids = 94.4% of tokens, 51% of ids unseen).
  - Head inputs: 32,768 rows of the head's input on calib-A (fit pool, disjoint from the ranking), all-K6 model,
    via the E1b hook's new `EXL3_AQ_DUMP_ONLY=336` (dump run KLD 0.00101 = the plain run).
  - Our head decoded at K1–8 (`mix/headdec`, the plugin's host decoder). `quantize/vtsim.py`: head-only KL vs the
    bf16 head for uniform and 2/3-tier plans, plus a rare-token guard (KL where the bf16 argmax is outside tier 1).
  - Validate first: uniform-K sim deltas vs the measured whole-model head swaps (sc4.00-e4 base: K8 0.00764, K7
    +0.00002, K6 +0.00009, K5 +0.0004, K4 +0.0016, K3 +0.0066; 0.047 bpw per K step).
  - **Head result (`data/vtsim-head-2026-10-31.json`): small.** The sim matches the measured swaps once logits are
    f32 (first run used f16 logits: a ~0.0002 KL floor). Uniform head-only KL: K3 0.0074, K4 0.0018, K5 0.00047,
    K6 0.00012, K8 0.000012. Tiering saves only 0.01–0.03 bpw at equal KL, or cuts head KL 30–45% at equal bytes
    (K4 budget: 0.0018 → 0.0010, top-1 97.7 → 99.3%). Against the head the mixes now pick (2.0 K3, 2.5–3 K4, 4 K5,
    4.5–5 K6, 6 K8) that is ~1.5–3% of total KLD. Why not more: noise in the ~150k tail rows lifts rare logits and
    exp() turns that into leaked mass, so the tail cannot go below K3 (tail K2 under 100k@K4: KL 0.0067 vs 0.0018).
    **Not worth a head kernel/format change now** (a 2-tier head stays a backlog item, ~2% gain).
  - Embedding next: rows only matter when their token is the input, so no leak; measured swaps on the same base
    e1 +0.015, e2 +0.0033, e3 +0.0006, e4 0, e5+ noise. `quantize/embtier.py` (per-row-tier embq) on a bf16-embedding
    copy of the base, 3 validation + 8 tier plans queued as `vt-emb-*` on calib-A.
  - Validation: uniform 2-bit sim 0.01096 vs the real e2 container 0.01131 on calib-A (fake-quant ≈ real).
  - **International-user guard (2026-10-31):** calib is ~94% English/code and only 51% of ids ever occur in
    it, so a calib-ranked low-bit tail could hurt other languages without the score showing it. Plan:
    1. language-neutral ranking `quantize/vtguard.py`: min(calib rank, token id) (Qwen BPE ids ≈ merge order ≈
       frequency in Qwen's multilingual data); keeps 99.1% of calib mass in its top 100k (calib ranking 99.6%);
    2. 2-bit tail floor as the default (`vt-emb-g-*` plans; 1-bit tail also measured);
    3. guard slice `golden-ml` = all 67 multilingual docs of the fit pool (77k tokens, out of sample for the
       ranking, which used search + qad); every variant measured on it as `vt-ml-*`, with base-bf16 and e4 refs;
    4. calib-ranked / 1-bit-tail plans ship only as an explicit "English/code-tuned" variant with an asterisk,
       unless the ml slice shows they cost international text about nothing.

- 2026-10-31 ~01:50:
  - V2 side-experiments for V3's benefit: per-layer edge offsets in the fit (no ranking gain) and a learned residual
    re-ranker (ridge / XGBoost no better than kfit2; GP too slow). Both logged in EXPERIMENTS.md. A re-ranker is worth
    retrying in V3 with more data. Edge offsets only if V3's new terms (activation × depth) need per-layer freedom.
  - 47 single-layer probes: the sensitivity spike is the last layer (L63 ffn_gate_up), not layer 0.
  - V3 backlog: K0 / K0.5 / K1.5 / K2.5 and half-steps 1–8 (fractional trellis).

- 2026-10-30 ~21:00:
  - **First real int8 prefill kernels run end to end:** K6 all-int8 0.00178 KLD, eval 74 s vs 100 s.
  - Post-hoc int8 rounding of EXL3 quants costs at K≥5; requant planned.
  - V0 validation: the weight model is off at 2 bpw (ours-K1/K2 overrated, fixed by the r9 refit); activation
    increments track.

- 2026-10-30 ~20:40:
  - Validation tooling: per-tensor group string in the hook (`EXL3_AQ_GROUPS`), knap3 `KN_WRITE` exports mix + masks,
    `nf/nfval.sh`. First 12 validation evals running.
  - Batches 4–7 in (A8 g256–g1024 plateau at +0.00015).

- 2026-10-30 ~20:15:
  - E3b full: the int8 trellis codebook is free through K6; the int4 grid is free at K2–3.
  - knap3 solver works; first frontier: ×1.5 prefill for +2% KLD at 4 bpw.
  - Found the bf16 head breaks mixes (EXL3 pipeline, r9 9.0/9.5); disabled.

- 2026-10-30 ~19:50:
  - E1c per-tensor activation costs done and calibrated (±30%).
  - E3c row-slab K: ~0.05 bit on the big tensors → low priority.

- 2026-10-30 ~19:30:
  - Activation dump: 512 rows per linear, 3.3 GB; the dump run's KLD is byte-identical to the baseline. Sent to
    CPU box A; the proxy scorer is running.
  - E2 M sweep, and group scales in the real kernel (spills at the big tile; −12…−29% at 128×128).

- 2026-10-30 ~19:00: E2 first run: iu8 ~255 TOPS, iu4 ~490 TOPS on real shapes (1.9× / 3.7× our f16 GEMM).

- 2026-10-30 ~18:50:
  - Group-scale speed for iu8 and iu4 measured (iu8 g512 −3.5%, g128 −18%).
  - A8 g256 +0.00017. Batches 4–7 (g64/g256/g512/g1024, A4 and A8) queued.
  - The group size becomes a per-tensor knapsack option.

- 2026-10-30 ~17:50:
  - **A8 with g128 activation scales costs only +0.00005** (per-row: +0.00047). The A8 floor was the scale
    granularity.
  - A4 g128: 5× better than per-row (all 0.031); gate_up g128 +0.009.
  - E3 sweep moved to CPU box B (44 jobs at once).

- 2026-10-30 ~17:45:
  - CPU work moves to CPU box B (48 threads) and the CPU box A (36 threads); the VM is downsized to 8 vCPU / 16 GB.
  - Sweep jobs rewritten: single-thread processes, row sampling, vectorized Viterbi. ~12× (trellis) and >100× (GPTQ on
    big tensors) faster.

- 2026-10-30 ~17:30:
  - E3b trellis tool built and validated (×1.29 vs exllamav3 K4); int8 codebook costs +1% at K4.
  - Calib Hessians on the VM (47 GB, 257 files).

- 2026-10-30 ~17:00:
  - Tracker created.
  - E1b A4 per-family results in.
  - Batch3 queued: A4 g128 and clipping.
  - Calib Hessians downloading to the VM from HF.
- 2026-10-30 ~16:40:
  - E1b A8-all +0.00047 (Large needs A16), A4-all 0.150.
  - E3 partial: int codes ×2–3.5 vs trellis.
- 2026-10-30 ~16:20: the E1b hook was built in a separate tree (`/root/e1b`); its no-op run is byte-identical to the
  normal K6 score.
- 2026-10-30 early: E0 iu4/iu8 peaks and the scale-group cost measured; fp4 absent.
