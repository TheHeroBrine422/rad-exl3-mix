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

## Plan to the V3 stream (2026-11-01, user)

Goal: start the eval stream soon after the level ladder finishes (late Nov 2 at the earliest). Phases:
1. **Quants:** the level ladder (running; ~3 h a level a card), head curve (running tonight), slab-head and
   trellis-embedding containers, a code-stamp (B13) deploy.
2. **Kernel / eval optimization: 2-4 passes before the stream** (each eval at the stream's pace pays back many
   times): the V3 int8 path (fold kernel, decode pre-pass vs fused, decided by the A4 timing), eval runner overhead,
   packing. Every pass ends with a byte-exact / KLD-identical check against the previous build.
3. **Quantizer optimization pass(es):** only output-identical speedups (profile one level end to end with a stage
   split first; the sigma sweep triples the input-group time). Pays off for any later requant (V3.5 sub-1 K, heads,
   fixes); changing the quantizer under a running ladder is out unless the shards are bit-identical (B13 stamps).
4. **Solver model, before the stream (CPU, needs no cards):**
   - [ ] **Uncertainty-aware solve (B11):** fit the model on bootstrap resamples; each option pays a penalty
     proportional to how much the fits disagree about it. Stays per-tensor additive (exact solver). Test on V2's data:
     does it avoid the V0-style extrapolation picks (K1 at 6 bpw, 31x misses)?
   - [ ] **Partial pooling** of the per-family / per-K factors (hierarchical kfit2): each factor shrunk toward its
     group unless the data supports it; new V3 terms (fractional levels, slab heads, trellis embedding, int8 mode)
     start from a prior, not a free fit.
5. **During the stream:**
   - [ ] **Measure where the model is unsure:** a fixed share of evals on the options the bootstrap fits disagree on
     most (sub-2 levels, new fractional levels, slab heads), not only near-optimal mixes.
   - [ ] **Residual re-ranker retry (XGBoost etc.) at the halfway point,** pass rule fixed now: beat kfit2 on grouped-CV
     pair ranking by more than the bootstrap noise AND on a held-out search slice; else dropped for good.
   - **Stop rule:** a size band stops when its held-out (ver3) best-so-far stops decreasing (no gain beyond its paired
     SE over N candidates, N fixed before the run). No fixed budget (~1,500-3,000 evals expected).
6. **D3** (re-run the 12 V0 mixes with the current fit on a search slice) as soon as the ladder has its first levels:
   the check of the solver's predicted -8...-21%.
7. Final: ver4 once; public only to report.

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
- [x] ~~**V3: ours-only option set** (2026-10-30).~~ SUPERSEDED (2026-10-31): DESIGN-v3 §1d keeps turboderp's tensors as options at every K. Drop turboderp's containers as options. V3's K≥5 requant
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

## Candidate experiments (2026-10-31; queue nothing until the YAQA decision)

Scope: sizes below 6.00 bpw only (6.00 is already 0.00076). Stop exploring when a full round finds nothing worth
more than ~1% KLD at the target sizes; then freeze menu + settings and build.

- [ ] **int8 codebook for K ≥ 5** (planned): write, validate one layer (DESIGN §6 A3), then requant K5–8 (~6–7 h).
- [ ] **Definitive A4 (W4A4) test at < 3 bpw** (user, 2026-11-01): the earlier "A4 almost never chosen" came from
      the activation side (E1c fake-quant); weights on the int4 grid were only CPU-measured (E3: x1.01 K2, x1.07 K3).
      1. full-model int4-codebook quants K2 / 2.5 / 3 (`EXL3_MUL1_I4`, exl3cb i4 build);
      2. a ~2.5–3 bpw mix with those tensors + the E1b hook at A4 g64/g128 on them, vs int8 codebook + A8 and vs f16:
         the real W4A4 KLD cost;
      3. time table with iu4 throughput (E2: ~3.7x f16, g128 −30%) + decode pre-pass -> knap3 at 2–3 bpw;
      4. only if knap3 picks A4: a real iu4 GEMM (nfq_gemm_i4) for an end-to-end timing.
- [ ] **Activation cost on a low-K base** (mining B1/B2): E1b on a K2/K3 base, int path gated at M ≥ 512 vs not (~1 h card).
- [ ] **Speed calibration** (mining A4/B7): key shapes at full/half N, stock/210 W, M = 16–511 (minutes, existing bench).
- [ ] **Best-of-N rotation seeds** (new): pick each tensor's sign/rotation seed from 2–4 by proxy; free at inference.
      CPU test of the per-tensor seed spread first (trellis.py with other seeds), expect 1–3% if anything.
- [x] ~~fp32 costs in the GPU trellis search~~ (2026-11-01): explained, nothing to gain. See log.
- [ ] Depends on tonight: YAQA count test / full collection (gate 2c/2e); gradient-weighted H_in if it wins;
      (dropped 2026-11-01: why turboderp wins at K6–8 -- damping and gain ruled out; the rest only matters ≥ 6 bpw,
      and his tensors stay options in the menu); quarter-step heads if the slab head fails; row-weighted trellis (PR #364) if gate 2d helps.
- [ ] **Head as a slab-plan curve in the solver** (planned): ~33 options, slab plans at 1/8-bit budgets 3–7 bits/row
      (headslab.py), each priced by headsim.py (exact head-only KL, validated vs whole-model head swaps); needs the
      int8-codebook head quants (one per K).
- [ ] **Later, during search/convergence: refine the head locally** (user idea, 2026-11-01): once the stream settles
      on a head budget, search finer around it (1/32-bit budgets, slab boundaries moved by single 128-row blocks,
      half-step K levels inside slabs), priced by headsim. Write after the search starts.
- [ ] Blocked / later: sub-1 K and K0 (needs the salted 0-bit read); fused-GEMM K ties (after the time table);
      radiance #328 check at the 1.15 port.

## Log (newest first)

- 2026-11-02 ~04:00 CT: **2B proxy: bf16 vs fp32 inference** (`proxy2b/halves.py`, model in two halves so fp32 fits
  the 4070; golden g2b-fit1 = 87 fit-pool docs, 2k cap, 86.9k gen rows, bf16, fla + causal-conv1d). Control: the
  split run in bf16 reproduces the golden (KL -4e-7, top1 1.0000). **fp32 vs the bf16 golden: KL 3.75e-4, p99 1.8e-3,
  top1 98.4%, flat by position** (3.9e-4 at 0-128, 3.6e-4 at 1k-2k: per-layer rounding, not state build-up). GDN
  kernel-path differences on the same rows ~1.5e-4. Decision: goldens stay bf16 (the challenge's reference is bf16;
  quants run on the same path, so the rounding is shared, not noise). Matters only at high bpw, where 2B KLDs approach
  ~1e-3: there, check a screening result with an fp32 halves run.

- 2026-11-02 ~02:00 CT: **exl3i serving path works on the GPU, up to the full model** (radiance-hero `qc-exl3-v3`
  4edad1f..340165b). Encoding `exl3i` (scheme exl3i, transform `<cb>/h128/k<rate>`, a u32 record TABLE plane of 16-row
  groups); the C oracle `common/exl3i_decode.c` (= the reference decoder, GPU quantizer vectors, exllamav3's oracle at
  integer K); the importer `exl3quant`'s `exl3i` quantiser (real K2.5 / K3 tensors: H-weighted error = ourq's logged
  proxy_err, ratio 1.000, stacked gate_up included); the kernel library `exl3i` (op `exl3i_dense`: exl3r4d's kernel
  with the decode a rate policy -- integer K exl3r4d's own plan, fractional / sub-1 a run-time plan). Selftest
  `exl3i_gpu_test`: decode bit-exact at all 64 rates x 2 codebooks; 256 GEMMs over every rate; 48 shape / dtype cases
  (M 1..952 across T2, y bf16/f16/f32, stacked seam, GDN 3-D); T2 = nt_pass bytes; integer K = exl3r4d bytes; 19
  refusals. Mutation check: 6 of 10 mutants run (all caught) before iceman went down. qwen35_exl3 routes exl3i weights
  to exl3i_dense (host test). Not yet run: d0.sh (a level's trunk in V2 3.00, real KLD).
- 2026-11-02 ~02:05 CT: **Embedding run6 (scale search 0.5..1.0): null.** K1 rel 0.25700 -> 0.25642 (-0.2%, edge rows
  0%), K1.25 0.18289 -> 0.18287. The scale search was never what limits the K1 / K1.25 embedding; menu unchanged.
- 2026-11-02 ~01:40 CT: **Embedding run5 was a no-op**: identical to run4 (K1 rel 0.25700, edge 0.2017). The edge rows
  sit at the LOW end: K1 826 / 4096 rows at factor 0.85 (the smallest), none above 0.95; K1.25 all at 0.95. run5 widened
  the top (to 1.75). run6 (colo) searches 0.5..1.0.

- 2026-11-01 ~00:10 CT: **Held-out provenance** (`eval/pools/provenance.py`, every golden's kld.json docs vs pools.json;
  `eval/pools/provenance-2026-11-01.json`). Verify pool: 845 docs; **317 never used by any slice (434k tokens), but
  none longer than 16k tokens and none agentic / long-agentic / long** -- those categories (43.8% of verify tokens)
  are entirely spent (calB/calBbig/ver1/ver2/long2). All 10 verify docs >= 32k are spent, both 250k traces 17 times.
  Overlaps: ver1 shares 8 docs with calB and 32 with calBbig; ver2 shares 30 / 73. calA/fit2/fit3/ml are fit pool
  (ml 8–12 docs shared with the fit slices). A ver3 drawn from clean docs only is clean but unrepresentative (no
  agentic text, code/multilingual 2x over-weighted). Decision needed before building ver3/ver4 (see TODO).

- 2026-11-01 ~23:20 CT: **Rounded-codebook quantizers pass** (exl3cb, `cbsmoke.py`, GPU): int8 on grid, x1.000 K1–6,
  x1.024 K7, x1.347 K8; int4 on its 16 levels, x1.001 K1, x1.007 K2, x1.020 K2.5, x1.059 K3, x2.5 K4.
  **Row-slab head works** (`headrun2.sh`, exact head-only KL on 32,768 held-out head inputs, paired vs uniform, 95% CI):
  K5 budget −40% (−48…−31), K6 −36% (−42…−29), K4 −9.6% (−23…+5, n.s.); 3 slabs = per-block; without the K3
  floor 3–12x worse; rare-token guard KL x1.2–1.5 at K5–6. Head ≈ 1.5–3% of total KLD: ~0.5–1% at 3.5–5 bpw.
  **knap4 (V3 menu, quality only) vs V2 menu:** 1.75 −25%, 2.0 −21%, 2.5 −15%, 3.0 −10%, 3.5 −13%, 4.0 −10%, 4.5 −11%,
  5.0 −6%; at 2.5: steps −9.6%, trellis embedding −5.4%, damping ~0, quarter heads 0. At 2.5 bpw nothing goes
  below K1.5 (sub-1 K would not be used there); at 2.0 the embedding hits its K1.5 floor (extend to K1).
  Speed: 3.0 at x1.53 prefill +0.6% KLD, 4.0 +2.5%. Gate 2e crashed twice more (profiler OOM, then expandable_segments
  at load); restarted 23:05 without either. Levels: settings = int8 codebook + per-tensor damping sweep on inputs.

- 2026-11-01 ~23:05 CT: **The GPU search's "4–6% gap" is the tail-biting constraint, not a flaw.** fp16 costs change
  nothing (CPU emulation, `f16gap.sh`: random tiles 1.000, real tensors with LDLQ median 1.000 at K2–6). The CPU
  reference Viterbi has a free start state, which EXL3's circular tiles cannot encode; a CPU tail-biting search with
  exllamav3's heuristic (`viterbi_tiles_tb`, NTRY 1) reproduces the GPU exactly (free/TB 0.948 K2, 0.962 K4 vs GPU
  0.949 / 0.960). Trying 16 wrap candidates recovers 12% / 8% / 4% of the gap at K1/2/3 (0.8% / 0.4% / 0.16% error):
  exllamav3's heuristic is near-optimal. Not worth a kernel change. Note: all earlier CPU trellis sweeps (E3b,
  fracsweep, gscale, damping) used the free start, so their absolute errors are 4–6% optimistic; ratios between
  arms are unaffected.

- 2026-11-01 ~5:45 PM CST: **Kernel pass, head curve, levels.**
  - **qtf2 v1** (`v3/e3/exl3frac2`, gate `qtf2check.py`): bit-identical to the fork's fractional kernel on all 15 old
    rates in both codebooks (0 index / value differences, tie-heavy and extreme tiles included); new rates (1/8 steps
    K0.125-2.875, sub-1 with the salted 0-bit read) pass the window-structure and pack round-trip checks. Speed: x1.27
    at KA 2-4, x2.1 at K5.5, x1.03 at KA 1 -- still ~2x slower than the integer kernel (K3 43k vs 18.6k tiles/s). v2
    caches every codebook value a thread needs in registers (as the integer K2-5 kernel does) and keeps KA 1's costs
    in one LDS buffer (register-staged): gate running.
  - **Head donors at half steps work alone** (K3.5 25.6 GB peak); K4.5 / K5.5 segfaulted in refit_scales: a kernel's
    code object could not load at full VRAM (HSA_STATUS_ERROR_OUT_OF_RESOURCES). Fix in ourq (empty cache + preload the
    refit kernels); K4.5-6.5 then fine. K7.5 hung (50 min): skipped.
  - **Slab-head curve is smooth except at the floor rule:** with half-step donors every 1/8-bit budget step gains 7-10%
    head KL, but the fixed tail floor round(budget - 1.5) jumps K3 -> K4 at budget 5 (-63%) and K5 -> K6 at 7 (-45%);
    between 4.25 and 4.875 the slab plans are worse than uniform half-step heads. Fix: one plan per floor, headsim
    chooses (`headslab.py HS_FLOOR_ALL`, `floorcurve.sh`). Quarter-step donors not needed for smoothness.
  - **Floor search fixes the slab head** (`fcurve-A/B.json`): with half-step donors and the tail floor chosen per
    budget by headsim, the curve is smooth (5-24% per 1/8 bit) and 43-54% below a uniform head at every budget 3.5-7.5
    (exact head-only KL, 32,768 held-out head inputs). The best floor sits ~0.5-1 bit below the budget. No quarter-step
    donors needed. (Whole-model check still to do, critique C3.5.)
  - **Speed pass 2 (2026-11-02, all output-identical, gated):** profile of K2.5 layers 3-4 (ladder settings): kernel 73%
    of wall, refit 11%, Cholesky / rotations / Hessians ~10%; the host thread waits on the GPU 86% of its samples (CPU is
    not a limit). Kernel time split (timing-only builds): traceback 3-5%, history writes 11-27% (K5.5: 45%, really the
    uncached path's per-candidate branches); tiles per call barely matters (64 tiles: 45k/s, 320: 50k/s).
    - qtf2 v5: packed history (one 16/32-bit store a thread a position) +5-11% at K1.x-2.x, +3-4% at K3.x; LDS-only step
      barrier ~+1%. v6: packed minimum in the uncached path: K5.5 14.1k -> 31.1k tiles/s (x2.2); K6.5 / 7.5 barely.
    - **Integer K1-4 on qtf2 (MASK 0): bit-identical to both integer kernels, x1.9 / 2.1 / 1.9 / 1.3** (K1 67k, K2 81k,
      K3 85k, K4 93k tiles/s vs 36k / 39k / 44k / 69k). K5-8 stay on the integer kernel (LDS barrier there: +1-6%).
    - quantize_exl3: the regularized-domain proxy (overwritten by the refit's) and refit's e_before (verbose only) are
      skipped; ourq caches the last Hessian (the sweep reloaded it 4x). e2e K2.5 L3-4: 52 tensors + 13 proxy rows
      byte-identical to the old code, 63 -> 55.5 s a layer (-12%).
    - **v7: step 0 peeled** (`first` a compile-time constant): with the 4-way width switch the compiler had kept an
      `if (first)` branch on every candidate (55 KB of code, ~1,100 branches vs 15 KB / 14 at MASK 0). K1.x +70-80%,
      K2.x +25%, K3-4.x +50-65%, K5.5 41k, K6.5 17k, sub-1 +50%.
    - **RACE found and fixed (v8):** the LDS-only step barrier (v5) made KA 1 output nondeterministic (qtf2race.py: first
      run after load differed in v6 / v7; random runs in a byte-history build). The cost arrays are reached through
      generic pointers (flat loads / stores), which an LDS-scoped fence does not wait for. The earlier e2e passes were
      luck. Every barrier is __syncthreads() again (the barrier was worth ~1%); the same patch on the integer kernel
      was reverted before use. v8 also picks packing per rate (KA 1 and integer KA <= 3; byte history for mixed KA 2-5).
      v8 gate: exact vs v4 arithmetic at all 34 instances, 0 vs the fork, integer identical, race 34 x 4 runs: 0.
    - Sub-1 (KA 0): 11.3k tiles/s at the ladder's batch before v7 (costs in global): ~3.9 card-h a level, ~27 for all seven.
    - Not done: reusing K-independent work across levels (~7%, needs a tensor-major ladder); refit via Cholesky instead
      of LU (faster, not bit-identical); KA 6-7 lane-cooperative search (head donors K6.5 / 7.5 only); sub-1 0-bit-step
      fusion. Checks: qtf2v5check (v6 vs v4 every instance), qtf2int6check, qtint5check, e2e5, speed2deploy.
  - **Speed pass 2, later results:** multi-level ourq (`-K a,b,c`) byte-identical to separate runs (K2.25 / 2.5 / 3,
    52 tensors + proxy rows + stamps), only -5% time (259 vs 272 s): the K-independent work is small. Integer K6 / K7 on
    qtf2: identical, x2.0 / x3.0 (i8); K5 slower on qtf2 (30k vs 45k tiles/s) -> stays on the integer kernel. Two ourq
    per card on v8: identical outputs, throughput x1.14.
  - **qtf2 v9 (QTF2_HIOPT: inlined forward lambdas, unroll capped at 16 for KOUT >= 6):** identical at every KA 5-7
    instance; levels it out at ~25k (i8) / ~35k (mul1) tiles/s: mixed KA 6 15-18k -> 25k, KA 7.5 4.3k -> 24k (i8) /
    5.6k -> 33k (mul1), but integer K6 / K7 and KA 5 slower with it. Final build: HIOPT only for mixed KA 6-7 (own
    compile units, qtf2_inst_a6f / a7f). K5 integer stays on the integer kernel.
  - **damp3 (running 2026-11-02):** damping held out across K0.5-7 (user: does the best spread change with K, and
    integer vs fractional?). Grid 0.025 / 0.01 / 0.005 / 0.0025 / 0.001 / 0.0003 / 0.0001, quantized on split-half
    H_A, scored on H_B; rule fixed before the run in `v3/e3/damp3/damp3report.py`. Old ladder data (K2-4.5): winners
    integer 51 / 42 / 7% (0.001 / 0.005 / 0.025), fractional 55 / 37 / 8%; neighbour-level agreement ~53% for int-int,
    frac-frac and int-frac alike -> no sign of a type effect, and the 0.005-vs-0.001 pick is mostly noise.
  - **Head mean-row subtraction: head-only KL -30%** (headremake.sh step 1, dry run 2026-11-02; exact headsim, uniform
    heads, 32,768 held-out head inputs): K3 0.00794 -> 0.00555, K5 0.000492 -> 0.000346 (geo ratio 0.70). Exact
    (softmax-invariant), free. Adopted for every head donor. Damping (with mean-row): K3 best sigma 0.005 (-2.6% vs
    default), K5 best default (0.01 / 0.005 ~+1%): they disagree, so per the rule fixed beforehand every remade donor
    runs all three and keeps the lowest headsim KL. `v3/yaqa/data/head-settings.json`. headsim's donor grouping
    (HS_MAXK) reproduced the earlier pricing exactly.
  - **radiance 1.15.1 port works** (fork branch qc-exl3-115: ABI 32 pins + layout-fingerprint export; all 4 plugins build,
    qwen35_exl3 synthetic tests pass): V2 3.00 on golden-ver1, stock 1.15.1 `--kld-ref` vs the frozen 1.13 recorder,
    gen-masked: 0.027122 vs 0.027122 (+0.002%, SE 0.001%; max per-row difference 0.026). Stock --kld-ref scores every
    row: mask with gen.u8 (`eval/cmp115.py`), as mergerep does.
  - **Output scales off on upstream-flagged gate/up: WORSE, +2.5% KLD** (z 2.1; K3 mul1, gate+up of the 36 layers with
    a flagged projection, 52 scales off, swapped into V2 3.00, golden-s2k: 583 search docs, 481k positions;
    `v3/e3/osr/result-K3.json`). Upstream's auto rule (ratio > 1.05) misjudges this model, like gate 2d's output-side
    model. Rule fixed beforehand: the requant keeps output scales always on; the K5 repeat was dropped.
  - **Low-K heads** (`lcurve.json`, donors K2 / K3 / K3.5 / K4, floors >= K2): uniform K2 head 0.0379 (x4.8 K3), but
    slab plans with a K2 floor are strong (budget 2.125: -48% vs uniform K2; budget 3.0: 0.0060 vs uniform K3 0.0079).
    Cliffs where the floor switches (2.0 -> 2.125 -48%, 3.0 -> 3.125 -41%) = donors too coarse: the remake adds K2.25 /
    2.5 / 2.75 (+ K1, K1.5), quarters at K2-3 if the cliffs remain.
  - **qtf2 v3/v4 gate PASS** (packed-min: x3.1-3.5 vs the old kernel at KA 2-4, faster than the integer kernel; sub-1
    table path = procedural, no speed gain: costs in global memory are the limit). **ourq --qtf2 end to end: 52 tensors
    byte-identical** to the old path (K2.5, i8, damping sweep, layers 10-11).
  - **exl3i oracle + decode plan** (`v3/infer`): plain-C reference decode (fractional, sub-1 salt, mul1 / mul1i8; codebook
    exact on all 65,536 states) and the generalized per-lane plan of exl3r4d (host emulation: 409,600 lane-position
    cases, every V3 rate, 0 failures).
  - **Extra trunk levels don't pay** (knap4 `KN_FRACS_EXTRA`, `k4levels.sh`): K5.25/5.75 <= 0.7% (6.0 bpw), 1/8 steps at
    K3-5 <= 0.8% (4.0). Base list stands.
- 2026-11-01 ~3:10 PM CST: **Full uniform requant planned (user):** K0.125-3 at 1/8 steps (sub-1 included as
  research), quarter steps to 4.75, K5/5.5/6; after a quantizer-kernel pass designed for 0-bit steps and 1/8 masks
  (~90 card-h at today's speed). CPU fracsweep below K1: rel error K0.5 0.227, K0.75 0.137, K1 0.089 (2.3-2.5x per bit),
  so sub-1 is a usable curve. Also today: search slices capped at 2k positions, held-out 8k (`eval/capstudy.py`:
  ~27% less noise per GPU-second, calls agree 0.993-0.997); damping sweep kept (per-tensor winners agree only
  65-70% across levels); upstream's output-scale test (`v3/e3/osr/osr.py`, all 64 layers, `v3/e3/data/osr.jsonl`) flags up_proj in
  35 of 64 layers (nearly all of L26-63) and gate_proj in 17 (L49-63, L0, L2): a paired swap test decides it before the
  requant.
- 2026-11-01 ~01:25 CST: **Night repairs.**
  - **A4 timing was invalid:** `/root/ntbench` (Oct 29) predates the T2 / NFQ struct fields, so both arms ran the
    fused NT kernel (identical hashes, no wdec kernels in the profile). Rebuilt as `/data/bench/ntbench2`; rerun in
    cardzero.sh with a hash sanity check. The A4 drop above rests on the matmul-core table, not on this slot.
  - **Fractional-K heads do not fit the GPU box's RAM** (K3.5 / K4.5 OOM-killed; integer heads ~22 GB fit alone).
    The slab-head curve uses integer donors K3-K8 (1/8-bit slab budgets still give a continuous curve). Head quants
    pause card 1's ladder (an OOM kill also took the card-1 K2.75 run once; it resumes per layer).
  - **Ladder profile (py-spy, 150 s of K2.75):** 84% in ldlq, 77% of wall time waiting on the GPU (trellis search
    kernel), 9% g_scale search, 4% refit. No host-side speedup is possible; only the frac Viterbi kernel or fewer
    passes (the input-group damping sweep runs that group 3x: ~+40% time per level for a gain D2 / damp2 put under 1%).
- 2026-11-01 ~01:55 CDT: **A4 (int4 activations) not worth building at any size** (knap4, V3 menu + measured speed
  table, sizes 1.75 / 2.0 / 2.5 / 3.0, lambda up to 1e-4): even under the strongest speed pressure the solver puts
  only 7-14% of FLOPs on A4, buying x1.88 -> x1.91-1.94 prefill for +2...+5% KLD. int8 already takes most of the
  time out; what is left is decode and the non-GEMM ops. So no int4-codebook quants (they would need their own ladder:
  post-hoc int4 rounding costs x1.06 at K3, x2.5 at K4) and no iu4 kernel. Revisit only if the A4 timing slot
  (int8 with decode) comes out far worse than the matmul-core table.
- 2026-11-01: **stop rule (user):** a size band's search stops when its held-out (ver3) best-so-far stops decreasing
  (no gain beyond its paired SE over N ver3 reads fixed beforehand); no fixed eval budget (~1,500-3,000 expected).
  ver3 is read for ~1 in 10 evals only (a candidate that becomes its band's best on the search slice), as in V2.
- 2026-11-01 ~01:50 CDT: **B19 closed: our K1 decode is bit-exact with exllamav3's** (`v3/e3/b19k1.py`: the plugin's
  C oracle `exl3_trellis_decode` vs exllamav3's `reconstruct` on turboderp's SC 1.40 tensors: 4 K1, 4 K2, 4 K3 incl.
  lm_head, 0 differing f16 values). The +12% of our 1.40 vs his chart is not a decode bug.
- 2026-11-01 ~01:47 CDT: **D2 done: our proxy edge over turboderp is ~9% in-sample, the rest is real; damping holds.**
  Split-half input Hessians (layers 3/10/43, 1,200 non-verify windows by document parity), ours quantized on H_A at
  K4 / K6, scored on H_A / H_B / published H (`v3/yaqa/d2report.py`, data `v3/yaqa/data/d2-score.jsonl`). Ours /
  turboderp proxy, median of 17 tensors: K4 0.775 in-sample, 0.851 out of sample, 0.863 published; K6 0.774 / 0.852 /
  0.862. So ~9% of the advantage is fitting to the Hessian sample at ~1.2M tokens per half; with the published 10M
  tokens it should be much smaller. Ours-vs-ours comparisons (fractional K, damping, codebook) are not affected.
  Damping: the H_A-best value is also H_B-best for 13/17 (K4) and 14/17 (K6) tensors; the median tensor keeps 0.025
  (gain small, as damp2). The per-tensor sweep does not overfit.
- 2026-11-01: **V3 stream budget raised to ~1,200-1,500 evals** (user), with a per-band stop on the ver3 validation
  curve. Solver-model plan: an uncertainty-aware solve first (bootstrap fits, penalty per option, B11), targeted
  evals where the fits disagree, partial pooling of factors; the XGBoost / residual re-ranker retried at the halfway
  point with a pass rule fixed beforehand (beat kfit2 on grouped-CV pair ranking by more than the bootstrap noise and
  on a held-out search slice).
- 2026-11-01 ~01:40 CDT: **knap4 wired to real options** (quality only, V3 menu: frac + damp, measured speed table):
  - `KN_EMBT=2`: trellis embedding priced from chain1's real KLD swaps (golden-ml increments scaled to the pooled
    corpus by the Lloyd points, 0.333; other K along the CPU curve from the nearest measured point, as damage vs a
    bf16 embedding). Result vs the CPU-priced menu (`KN_EMBT=1`): +0.0...+0.8% at 1.75-4.5 bpw, -0.8% at 5, -4.6%
    at 6. **The CPU-based embedding estimate holds on real KLD.** Embedding still sits at its t1.5 floor at <= 2 bpw.
  - `KN_HEADSLAB=1`: slab heads at K5 / 5.5 / 6 priced at the conservative end of headsim's interval (x0.69-0.71
    head KL; K4 not offered: n.s. and rare-token KL x2.5). Picked at 3.5-5 bpw: **-1.3...-1.5% total KLD at 4-5**,
    ~0 at <= 3 (the head stays at K <= 4 there) and at 6. Needs hdslab-K5/5.5/6 containers before validation.
  - `KN_LEVELS=/data/v3levels` (copy of the ladder's proxy.jsonl files): measured fractional levels replace the
    midpoint interpolation per tensor once the level and both neighbouring integers exist (tested on synthetic data).
- 2026-11-01: **YAQA scratch deleted** on the GPU box (qfull, rows32, qdiag, full32/64, hout*, n16/n32, q*, qe:
  192 GB; 649 GB free). Kept: headslab / headslab2 (head-slab quants), hs (head Fisher), pylib, scripts, logs.
- 2026-11-01 01:17 CDT: **Gate 2e FAILED** (gradient-weighted H_in from the 100-sequence snapshot, blocks 22/23,
  ver1, paired vs one-sided): K2 +44% (z 10.3), K3 +31% (z 11.6). By the rule fixed beforehand (either K positive ->
  drop), no blend test (`gate2e-blend.sh` / `hblend.py` written and tested, unused). All backward-based Hessian ideas
  are now closed (EXPERIMENTS.md). D2 started right after (split-half collection 1.05 s/seq).
- 2026-11-01 ~01:35 CT: **Document-reuse bias test: no measurable flattering from fit-slice reuse** (`eval/biastest.py`,
  existing scores only, `data/biastest-2026-11-01.out`). 76 V2 mixes scored on a fit-pool slice (calA / fit2 / fit3,
  where V2's models were fitted and the mixes picked) and on ver1; each as a ratio to turboderp's untuned SC reference
  at the nearest bpw. bias = ratio on the fit slice / ratio on ver1: median 1.024, weighted mean 1.014
  [1.005, 1.023]; by slice calA 1.027, fit2 1.009, fit3 1.033; by size < 2.5 bpw 0.97 (n = 8), above 1.02-1.03. If
  anything the mixes look ~2% WORSE on the slices they were fitted and selected on (slice-to-slice spread, not
  winner's curse). So fit / search reuse costs at most a couple of percent here and is not the problem; the
  ledger's strict "spent" rule can be relaxed for non-trained documents (QAD-trained ones stay excluded until the same
  test is run after V4). Caveat: ver1 itself was the V2 monitor slice (no selection on it, mixes were queued there
  because of their fit-slice scores).
- 2026-11-01 ~01:20 CT: **B10: the board's `bytes` is the tensor payload, not the file** (`store/sizedef.py`, 4 V1
  containers; the HF files are byte-identical to ours): board vs sum of plane bytes -0.13 / -0.04 / +0.18 / +0.01 MB,
  vs file size -9.6 to -9.9 MB (header, directory, alignment). So real bpw = payload bytes x 8 / 26,895,998,464;
  "container bytes" overstates by ~10 MB = 0.003 bpw. radiance 1.15.1's Qwen3.5 plugins do not declare or load the
  `mtp.` block; whether an MTP drafter shipped in the container would count toward size is not knowable from the
  board data (none of the entries has one): ask the challenge author before planning on it.
- 2026-11-01 (results from ~23:50 CT Oct 31, read now): **chain1.**
  - Real trellis-embedding swaps on V2 6.00, golden-ml (multilingual, 67 docs), KLD increment vs base: trellis t2
    +0.0047 vs Lloyd 2-bit +0.0092; t3 +0.0012 vs Lloyd 3 +0.0021; t4 +0.00017 vs Lloyd 4 +0.00030; t2.5 +0.0022 ~
    Lloyd 3. **Trellis costs ~0.5-0.6x Lloyd at equal bits, ~0.5 bit saved at K2-3**: confirms the CPU result on real
    KLD. The fit3 half of the comparison is empty (no base6 score on golden-fit3: the base eval did not run). On the
    fit pool, so not a held-out number.
  - Even-layer K6 swaps into V2 5.00 (fit3, 98 docs): ours K6 -24.2%, turboderp K6 -23.1%; turboderp vs ours +1.3%
    (z 1.26, n.s.). Ours and his K6 are equal on real KLD. The low-damping arm (ours-K6-d005) did not quantize.
- **tdverify complete:** all 16 turboderp downloads byte-exact against our packs (337 tensors each, 0 problems); each
  has 8 MTP trellis tensors that exist only in the downloads -- keep them (or pack them) before deleting.
- **damp2** (CPU, colo, layers 1/5/19/30/47/58, K3/5/7): best lower damping vs 0.025 is a much smaller win away from
  layers 11/43/63: attention inputs 0.989-0.996 (first sweep: 0.94-0.97), GDN inputs median 0.992-0.998 but one
  tensor 0.94, everything else >= 0.993. The gain is tensor-specific, which is what the per-tensor sweep in the level
  ladder (--sigma-in-sweep) is for; expect well under 1% total. Data `v3/e3/data/damp2.jsonl`.
- 2026-11-01 ~01:05 CT: **B4 closed: KLD rises with position for every quant alike** (`eval/posbins.py`, golden-long2,
  2 traces, ratio to turboderp SC 4.00 per bin 0-2k / 2-8k / 8-32k / 32-64k / 64-128k / 128k+). The reference's own
  KLD doubles (0.0040 -> ~0.008 past 32k), but the ratios are flat: V2 4.00 0.76-0.84, V2 4.50 0.42-0.51, td SC 3.00
  3.46-3.69, V2 3.00 2.5-3.3 (first bin highest). No quant compounds error with context faster than another, so
  short-slice rankings hold at long context; the rise is a property of the text / position. 2 docs only.
- 2026-11-01: **Sub-1 K moved to V3.5** (user): before QAD, since sub-2 quants are the natural QAD targets. **A5**
  selftest classes added to DESIGN §1c.4; **D6** GDN a/b merge closed (0.27-0.4% share); **B13** ourq.py code stamp
  (committed, deploy + smoke test in the morning); **B19** K1 bit-compare script queued after the gate 2e verdict.
- 2026-11-01 ~00:55 CT: **knap4 on the measured speed table** (`KN_SPEED=v3/bench/data/speedcal-stock.jsonl`, now
  k4grid's default; `KN_A4=a4time.jsonl` swaps in whole measured paths with decode once A4 has run). Also fixed: the
  LM head was timed at M 4096 (~77 ms, ~5% of the step) though prefill runs it on the last token only; flop shares
  now exclude it. Measured f16 core ~116 TF (not 135), iu8 ~2.2x. Same picks and KLDs as before at 2.5 / 3.0 / 4.0
  (the int8 choice was already saturated); all-int8 speedup estimate x1.65 -> x1.88, at +0.9% (3.0) / +5% (4.0)
  KLD. Decode is still the placeholder (0.006 ns/weight): A4 decides whether x1.88 holds.
- 2026-11-01: **Gate 2e MARGINAL rule** (decided before the verdict): both K negative (even at z ~ -0.5) -> a
  blend test alpha*H_grad + (1-alpha)*H_plain (quants + evals only, ~40 min) before any 200/400-sequence
  collection; either K positive -> drop.
- 2026-11-01 ~00:45 CT: **D1 closed by reading the code: no auto-vs-always confound in gate 2d.** In exllamav3's
  `regularize`, output scales on auto differ from "always" only for tensors given an `out_sensitivity` (the gated-MLP
  path of convert.py); ourq.py never passes one. With a diagonal, the Hessian-derived scales replace the plain ones
  and output scaling stays on; every gate-2d tensor took them (proxy.jsonl `hout_diag_scales` true for 13/13 in
  dsamp-K2, dfull-K2, dfull-K3; dsamp-K3 only got 5 tensors). Gate 2d's failure stands. No GPU run needed.
- 2026-11-01 ~00:45 CT: **A4 timing slot written** (`v3/bench/a4time.sh`, `a4report.py`): ntbench on the NFQ build,
  f16 T2 vs the int8 path *with* its decode (nfq_act_i8 + nfq_wdec_i8 + nfq_gemm_i8), 5 TP1 shapes, K2–6,
  M 512/2048/4096, rocprofv3 decode share at M 2048. Queued after D2 (`a4wait.sh`, card 0).
- 2026-10-31 ~22:30 CT: **Gate 2d failed: diagonal output scales hurt** (one-sided LDLQ, exllamav3's
  sensitivity_out_scales from diag(H_out), blocks 22/23, vs one-sided): sampled-H_out diagonal K2 +14.9% (z 7.1);
  full-token diagonal K2 +10.2% (z 5.9), K3 +4.3% (z 3.0). Output-side Hessian information hurts in every form we
  tried. Only gradient-weighted H_in (gate 2e) is left of the backward-based ideas.

- 2026-10-31 ~22:20 CT: **YAQA gate 2c failed: two-sided YAQA dropped.** Full-token fp32 H_out (200 seqs, every
  position) on blocks 22/23, KLD increment vs one-sided: K2 +30% (sigma_reg_out 0.025) / +7% (0.3), K3 +21% / +5%,
  all z ≥ 2.9; full-token is even WORSE than sampled at 0.025 (+15% K2, +10% K3), equal at 0.3. Truncated backward
  checked exact (0 difference). Sampling was not the problem. Remaining backward arms: gradient-weighted H_in
  (gate 2e: crashed on a profiler-path bug before collecting, fixed, not rerun) and diagonal output scales (gate 2d,
  evals running). Collector profile (seq 1): fwd 8.8 s, bwd 13.8 s, CPU-sum wait 11.3 s with H_in on (the CPU sums,
  not the GPU, are the bottleneck once H_in is added).
- 2026-10-31 ~22:20 CT: **Best-of-N rotation seeds: dead** (`seedtest.sh`, Thor, 9 tensors x 4 seeds x 2 row samples,
  K2–4): seed spread 0.5%, A-vs-B correlation +0.22, best-of-4 picked on A measured on B: −0.2% / +0.2% / +0.0%.
- 2026-10-31 ~22:20 CT: **Damping vs K** (`dampsweep.sh`, colo, 16 tensors, error vs undamped H; ratio to 0.025):
  attention inputs (q/k/v) gain with less damping, more at high K: K4 0.970 (0.0025) / 0.961 (0.001), K8 0.938–0.955;
  GDN inputs, out-projections, FFN: 0.994–1.000 at every K. 0.05 is worse everywhere. **Damping does not explain
  turboderp's K6–8 wins** (attention inputs are a small share). Settings: 0.005 (or 0.0025) for attention inputs,
  0.01 elsewhere is harmless. Depth extension running (colo).
- 2026-10-31 ~22:20 CT: **Gain (global scale) vs K** (`gainsweep.sh`, Thor, 16 tensors, RMS x f, f = 0.8–1.8): the best
  factor rises with K in every family: ~1.1 to K5, 1.2 at K6, 1.3 at K7–8; out-projections (attn_output, ssm_out)
  more: 1.2 (K5), 1.3 (K6), 1.4 (K7), 1.5 (K8; range to 1.8). A fixed 1.1 costs x1.05 (K6), x1.2 (K7), x2 (K8)
  error; for out-projections x1.2 (K6), x1.7 (K7), x4 (K8). This is the CPU tool's RMS scale, not exllamav3's own
  g_scale search: the GPU gaintest (running) shows whether exllamav3's choice already tracks this. If it doesn't,
  it is a strong candidate for the high-K gap.

- 2026-11-01 ~21:20 CT: **Quarter-step quantizer works on the GPU** (`fracsmoke.py`, 8192 Gaussian tiles, card 0):
  pack/unpack round trips and decode exact at every menu K; quarter/half steps on the curve (0.993–1.003 to K4.75).
  The first verdict said FAIL only because the CPU-vs-GPU check expected agreement within 3%: the GPU search is
  4–6% worse than the exact CPU Viterbi at every K, integer K (exllamav3's own kernel) included, and fractional K
  sits within 0.4% of its integer neighbours' offset. Check fixed; the levels (chain0) are unblocked. The 4–6% gap
  itself is a candidate (above). Overnight CPU chains queued: fine gain sweep on Thor after the seed test, damping at
  6 more depths (K3/5/7) on colo after the damping sweep.

- 2026-10-31 ~21:00 CT: **Embedding menu, token-frequency weighted** (`v3/e3/embfreq.py`, the sweep's saved per-row
  errors x search+qad token counts): the error curve is unchanged (x1.40 per quarter step; frequency-weighted error
  +1–3%). Per-row scale still wins to K4 (G1024 x1.011 / 1.006 / 1.003 at K2/3/4 frequency-weighted); from K5 G1024 is
  0.5–1% better frequency-weighted. Caveat: 48% of the 4096 random rows never occur and the rest carry 1.2% of the
  tokens, so the frequency weighting rests on few rows. Decision unchanged: per-row only (a 1% embedding-error
  difference at K ≥ 5 is far below anything measurable in KLD). Group-scale recheck final (16 tensors): same as before.

- 2026-10-31 ~20:40 CT: **Embedding menu sweep done** (`v3/e3/emb/run3.sh`, Thor, 4096 rows, `v3/e3/data/embmenu.jsonl`).
  One scale per row wins at equal bits at every K: G1024 is x1.015 worse at K1.5–2.5, x1.008 at K4, x1.003 at K5, even
  only at K5.75; G512 x1.01–1.02 worse throughout. The quarter-step curve is smooth (x1.40 error per quarter step at
  every K, i.e. ~3.9x per bit). **Embedding menu: quarter steps, one bf16 scale per row, no group options.** The
  best scale factor reaches the search edge (1.35) for 0.6–3% of rows at K5–6 only.
- 2026-10-31 ~20:40 CT: **Per-group weight scales stay dead at low K, Hessian-weighted** (`v3/e3/gscale.py`, colo, 12 of
  16 tensors so far): the raw gain at equal K is only 1.3–1.8% (ratio 0.982–0.987 at K1–2; at K0.5 groups are even
  slightly worse, 1.008–1.035), so after the 8/G-bit scale overhead groups lose: x1.08 (G128) / x1.17–1.20 (G64)
  worse at equal bits, K1–2. Confirms the Oct-30 amgm null result where it was untested (weighted, K ≤ 2).

- 2026-10-31 ~20:20 CT: **Overnight queue (TP2 to ~13:00 CT Nov 1), all on iceman, ordered by locks:**
  1. gate 2c (running) -> `gate2e.sh`: blocks 22/23 again with gradient-weighted H_in (`collect.py YAQA_HIN`); if 2c
     passed also 400 sequences with snapshots at 100/200/400 and split halves (`hhalf.py`) = the sample-count test.
  2. quarter-step smoke test, gate 2d, head rerun (card 0 / both).
  3. `v3/levels/chain0.sh` (card 0): gain sweep (layers 3/10/43, K2/4/6) and high-K damping sweep (K6/8), then waits
     for `/data/v3levels/settings.env` (default `--sigma-reg-in 0.005` at 2:30 AM CT) and builds the levels
     K2.25…K4.75 (exports, trunk only).
  4. `v3/levels/chain1.sh` (card 1): real trellis embedding K2–4 vs Lloyd e2–e5 on V2 6.00 (fit3 + ml); ours vs
     turboderp K6 (even layers on an ours-K5 base: ours-K6 / his plain6.00 / ours-K6 at sigma 0.005), fit3.
  Store index done: 53 packs; his plain6.00/5.00 packs are uniform K6/K5; 3–6 of his tensors per pack in some sizes
  are byte-identical to another of his packs (counted once).

- 2026-10-31 ~20:00 CT: **High-K half steps land on the curve** (`v3/e3/fracsweep-hi.sh`, colo, 18 tensors of layers
  0/10 (GDN), 11/63 (attention), 43 (MLP); scale search 0.85–1.75; `v3/e3/data/fracsweep-hi.jsonl`). Error / log-linear
  interpolation of the integer neighbours, median [range]: K5.5 0.997 [0.993, 1.042], K6.5 1.003 [0.980, 1.010], K7.5
  1.006 [0.994, 1.021]. Integer steps stay steep: ×3.77 / ×3.67 / ×3.81 error per bit at K5→6 / 6→7 / 7→8. So the
  K5–8 half steps of the menu are sound. Scale: the best factor rises with K (1.15 to K6, 1.3 from K6.5–7, some 1.5
  at K7.5–8): the old {1, 1.15, 1.3} grid capped high-K slightly; real quants search their own scale. K0.25: proxy
  rel err median 0.36 [0.05, 0.59] (1.0 = dropped): judge only by real swaps. 3 jobs failed on a wrong tensor list
  (layer 43 is full attention, has no GDN tensors); not rerun, every family is covered.

- 2026-10-31 ~19:45 CT: **Mining merge acted on** (`local/mine/merged-2026-10-31.md`, Opus 5.5 via `mergep.sh`).
  DESIGN-v3 corrected (YAQA status and costs: 3.7M qad tokens, ~0.2 TB full H_out; head K priced by KLD + bytes;
  his S not a prior) and the merge's checks added to §6. Queued on iceman behind gate 2c:
  - `gate2d.sh` (merge D1): **diagonal-only output weighting** -- one-sided LDLQ with exllamav3's sensitivity output
    scales from diag(H_out) (`ourq.py --hout-diag`), blocks 22/23 K2/K3, diagonal from the sampled fp32 H_out and from
    the full-token one, paired vs one-sided.
  - `headrun2.sh` (merge A2, replaces the invalid headrun): Fisher ranking from the **qad pool only** (vs fit+search+qad:
    top-set overlap 0.93–0.98, fit-pool Fisher captured in the top 1,280 rows 0.610 vs 0.613: the old leak was small),
    floor K3, budgets K4/K5/K6, judged by **`headsim.py`** (exact head-only KL on the 32,768 calib-A head inputs, fit
    pool = out of sample; paired vs uniform with a 512-position block bootstrap). Whole-model check for the finalist
    only, with the fixed plugin.

- 2026-10-31 ~19:15 CT: **Head slab test INVALID (plumbing bug), rerun pending.** Every spliced variant (u3/u4,
  slabs, Lagrange) scored KLD ~15 with top-1 0.0002 (V2 3.00 itself: 0.027). The donor heads are right (rows vs the
  bf16 head: K4 rel err 0.076, cos 0.997; K3 0.15), so quantization and the un-permute work. Cause: `qwen35_exl3`'s
  fallback for an unquantized head used `gemm_nt`, whose output is the activation dtype (bf16), into the f32 logits
  buffer -- never exercised before (V1/V2 heads are always EXL3). Fix: `logits_gemm` as in the bf16 arch
  (radiance-hero 9a156d0, local). Rerun: dev build of the plugin + the 14 evals (~20 min, both cards) after YAQA
  gate 2c. Plan numbers (proxy) are unaffected.
- 2026-10-31 ~19:10 CT: **YAQA gate 2c started** (`v3/yaqa/gate2c.sh`): full-token fp32 H_out for blocks 22/23 on
  the pilot's 200 windows and labels; backward truncated at layer 22 (gradients at/above unchanged; checked against
  an untruncated 2-sequence run) and CPU sums async. Then two-sided K2/K3 at sigma_reg_out 0.025 / 0.3 vs one-sided
  and vs gate 2b's sampled two-sided. First fracsmoke build failed (torch's ROCM_HOME = pip runtime without
  hipBLAS/thrust headers); now ROCM_HOME=/opt/rocm/core-10.1 as for the exllamav3 build.

- 2026-10-31 ~19:00 CT: **Quarter-step quantizer instances written** (exllamav3 branch `rxm-frac-quarter`,
  worktree `/workspace/forks/exllamav3-frac`, not yet built). The fractional kernel is already generic in the mask
  (step shape per bit, scratch sized by KA, ring end width from MASK & 1), so: 15 instances for the V3 menu (KA 1–4
  x 0x8888/0xAAAA/0xEEEE, KA 5–7 x 0xAAAA), one compile unit per KA; `frac_k` derives the canonical even mask for
  any K = KA + n/16 (n even); exllamav3's own decode refuses non-0xAAAA rates instead of misdecoding. The live
  exllamav3 is a prebuilt 159 MB .so (no build tree), so instead of a full rebuild `v3/e3/exl3frac` builds only
  quantize_tiles_frac + the 7 per-KA units as a side extension and swaps it in (`ourq.py --frac-ext SRC`; prebuilt
  pack/unpack already take any mask). `v3/e3/fracrun.sh` (iceman, queued behind the head test) builds it and runs
  `fracsmoke.py`: every menu K on 8192 Gaussian tiles -- on-curve vs integer neighbours, GPU vs CPU Viterbi error,
  pack/unpack round trip, decode check.
- 2026-10-31 ~18:55 CT: **Fractional-K at the menu's ends** (`v3/e3/fracsweep-hi.sh`, colo, 21 tensors = the 14 of
  fracsweep + layers 0 and 63): K5–8 half steps and K0.25, K4.5/5 overlapping. Running, ~1.5 h. Found while setting
  it up: `trellis.py`'s scale search {1, 1.15, 1.3} hit its top edge at K7.5 and its BOTTOM edge (1.0) for most
  tensors at K ≤ 2 in the earlier sweep, so those low-K proxy numbers may be slightly pessimistic (relative K steps
  less affected; real quants search their own scale). Now `E3_FACTORS`, here 0.85–1.75.
- 2026-10-31 ~18:45 CT: **Embedding menu sweep** (`v3/e3/emb/run3.sh`, Thor, 4096 sampled rows, 47 runs): quarter
  steps K1.5–6 x {one scale per row, G1024}, + G512 from K4; per-row error/energy/chosen-factor saved (tail rows,
  later token-frequency reweighting). Smoke test: at K5.75 9% of groups took the top factor 1.25, so the scale search
  is now 0.85–1.35. Running, ~1 h.
- 2026-10-31 ~18:35 CT: **Embedding scale group size** (Thor, 1024 rows, K2/3/4): error vs G128 = 1.016–1.018 at
  G256, 1.02–1.025 at G512, 1.022–1.031 at G1024, 1.023–1.040 at one scale per row (G5120). Scale overhead 0.125 →
  0.003 bit/value; on the trellis curve (~4x error per bit) 0.12 bit is worth ~18% error, so per-row wins by ~15%
  at equal bits. Per-row vs G1024: 0.013 bit (~1.8%) saved against +0.1/+0.2/+0.9% error at K2/3/4: G1024 likely
  wins only around K5–6 (menu sweep above settles it). Menu: per-row scale, G1024 where it is on the hull.
- 2026-10-31 ~18:05 CT: **Trellis embedding vs V2's Lloyd tables** (`v3/e3/embtrellis.py`, Thor, 1024 random rows,
  H = I, same fwht128 + per-(row,128) scale): relative error at equal bits, trellis/Lloyd 0.70 (1 bit), 0.56 (2),
  0.51 (3), 0.53 (4), 0.51 (5); trellis K matches Lloyd at K + 0.46–0.49 bits for K ≥ 2 (DESIGN's "trellis-3 ≈
  Lloyd-4" was too optimistic: trellis-3 ≈ Lloyd-3.5). ~0.023 bpw saved at equal error, ~1% KLD at ~3 bpw by V2's
  fitted embedding costs. Quarter steps adopted. `v3/e3/data/embtrellis.jsonl`.

- 2026-10-31 ~18:25 CT: **YAQA gate 2b failed too** (fp32 backward, 200-sequence sampled-row pilot, blocks 22+23 into
  V2 6.00, golden-ver1, paired vs the same one-sided quants). KLD increment over base, two-sided vs one-sided:
  K2 +17% (sigma_reg_out 0.025) / +6% (0.3); K3 +19% / +10% (all z ≥ 2). bf16 gate 2 was +23% / +28%, so fp32 helped and
  more output regularisation helps further -- but the trend runs toward one-sided (infinite regularisation), i.e. our
  sampled H_out still hurts. Remaining suspect: 32 sampled positions a sequence (1.6%) miss the few high-gradient
  tokens that dominate H_out (upstream's note). Last test before dropping YAQA: full-token sums for blocks 22/23
  (`collect.py YAQA_FULL=22,23`, fp32, CPU accumulators), ~1.5 h with quant + eval. Not run tonight (head test has
  the cards).

- 2026-10-31 ~22:45: **Upstream exllamav3 already has a YAQA collector** (`util/yaqa_hessians.py`, reference
  checkout v1.6.3), plus `convert.py -hess/-h1/-hreg/-sh` and `--out_scales yaqa`. Differences from ours: it sums
  d dᵀ over **every** token on the GPU (layer ranges per run for memory), we keep 32 sampled positions per sequence
  (1.6%) as rows on disk; it weights tokens by xᵀ H_in x (Sketch A), we by ‖x‖² (H_in = I); bf16 activations with
  fp32 sums. Its notes (Qwen3.5-2B): the benefit comes from the output side alone (online H_in + this H_out beat
  the iterated pair); H_out is "dominated by relatively few high-gradient tokens"; two seeds differ ~20% at 250 full
  2048-token rows. So our 6,400 sampled rows likely miss most of the mass: second suspect for gate 2 next to the
  bf16 noise. If gate 2b (fp32, sampled) fails too: full-token sums for blocks 22/23 (≈ 6 GB of fp32 accumulators;
  no room on the cards beside fp32 activations, so on the GPU box's CPU, ~10 s a sequence) before giving up.
  Also: turboderp's SC quants take their GPTQ Hessians from the self-sampled trace (`convert.py --cal_data`), i.e.
  model-written text only -- the kind of position the board scores; ours cover every calib token, prompts included.

- 2026-10-31 ~22:30: **Shared-input siblings do not share `suh`** (our K6 quant's dumped suh, every layer; CPU).
  Signs agree 100% (one seed per input-Hessian group), but the per-channel input scales differ: gate vs up median
  |ratio| 1.015 (p99/p1 spread 1.27), GDN qkv vs z 1.036 (1.19), attn q vs k / v 1.092 (1.92); no pair identical.
  The difference is exllamav3's per-tensor scale refit (`refit_scales`). Consequence: on V2's quants each sibling
  needs its own rotate + int8-quantize prep pass (cheap: ~20 MB of activations at M = 2048), not one shared tile.
  If shared tiles pay in the time table, the Phase-B requant can tie su across siblings (joint refit), at a quality
  cost still to be measured.

- 2026-10-31 ~22:10: **Fractional-K trellis works on real tensors** (`v3/e3/fracsweep`, CPU box A, 14 tensors of
  layers 10/11/42/43, every family, 128 sampled rows, same rows at every K; `fracreport.py`). Measured proxy error
  over the log-linear interpolation of the integer neighbours, median: K1.25 0.990, K1.5 0.988, K1.75 0.990, K2.25
  0.999, K2.5 1.000, K2.75 1.002, K3.5 0.998, K4.5 0.996 (all tensors within 0.95–1.02). Quarter and half steps
  land on the curve: the shift-pattern trellis costs nothing over the integer ones, so the −5…−7% menu estimate
  stands. Sub-1 (K0.5 / K0.75) can't be judged this way (K0 = dropped is not a curve point); they sit well below
  the line to "dropped" and need real swaps. Integer slopes match the hproxy table (K1→2 2.12, K2→3 1.99).

- 2026-10-31 ~21:50: **Sub-2-bit pricing, turboderp's tensors (CPU, V2 data).**
  - Proxy error per bit (ours, all 337 tensors, median log2 ratio): K1→K2 2.11, K2→K3 2.00, then 1.97 … 1.88.
  - V2's fitted model (q = 1.004, one factor for K1–K3) under-predicts the three K1-dominated band swaps by 20%
    in-sample / 36% CV (K2 5%, K3–K8 ±1–4%). A refit with its own K1 factor: K1 1.35 vs K2–K3 1.09 per unit proxy
    error; CV log-RMSE 0.870 → 0.845, pair ranking 3,115 → 3,119 / 3,786. Its solutions use far fewer K1 tensors
    (1.5 bpw 180 → 105, 1.75 106 → 52, 2.0 54 → 21; his low-bpw SC tensors and our K2 instead). The K1→K2 KLD slope
    is ~2.3–2.5 bits per bit, as turboderp notes for this model (~2.35). V3: K1 and each sub-2 level get their own
    factor and must be measured; K0.25/K0.75 only if K0.5 earns picks. (Thin evidence: 3 K1 swaps.)
  - What V2 takes from turboderp at high bpw: mostly the tensors his recipes **raised** to K7/K8 (6.0: 37 of 49;
    8.0: 36 of 42), i.e. his most sensitive ones by trace attribution (his 6.00 recipe: 22 at K8, 63 at K7). The fit
    prices our K7–K8 at 1.58× KLD per unit proxy error (K4–K6 1.25×, his ~1.0×): our quantizer lowers the input-side
    proxy harder, plausibly by moving error into output directions the proxy cannot see -- what YAQA's H_out targets.
    The ours-vs-his single-group swap test should use his 85 boosted tensors.
  - His per-tensor sensitivity S (`kld_table.json`) vs our fitted per-tensor K4 cost: Spearman 0.71 overall, family
    shares close (gate_up 45/41%, down 22/27%), within families weak (+0.6 gate_up/down, ~0 or negative attn k/v and
    GDN qkv/z), depth differs (layers 0–15: 14% ours / 23% his; 48–63: 45% / 39%). Not worth a per-tensor prior.
  - Final kfit2 model dump: `data/v3/kfit2-model-final.json`.

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
