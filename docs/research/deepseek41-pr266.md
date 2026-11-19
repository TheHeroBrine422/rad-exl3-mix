# DeepSeek-V4.1 on Radiance — bakon PR #266 + IQR-MoE result (2026-11-19)

**Status:** read (branch cloned 2026-11-19). Drives the DSV4.1-Flash-1017 EXL3 plan.
Links: codeberg.org/bakon/radiance `deepseek41-wip` (force-pushed to `9a963c2` 2026-11-18,
rebased onto radiance 1.17.0) + StillDeadcode/radiance!266 (open, 258 files, has conflicts with main).

## What exists in the branch (the engine blocker I flagged 11-19 morning is mostly solved)

- `arch/deepseek41`: full V4.1-Flash walk on one rank + across ranks (TP3/TP4 measured on 4x R9700),
  attention world, CSA2 sparse attention + two-level indexer, mHC mix/collapse, router, Engram
  (table read in place from planes; the release form uses F8_E8M0 scale planes), DSpark drafter
  (block drafter of three V4 stages, uneven expert split over ranks), fused decode chains
  (mHC post into collapse+norm in one launch; mixes beside MoE GEMMs; compressor/indexer
  interleaved with query/key), file-stager pass that reads only routed experts.
- Kernels: `libr4d/r4d_*` (DS41 units: sparse attention + indexer at decode and prefill heights on
  WMMA, GEMMs over GGUF native blocks, mHC/router, small-op chains in one launch each).
- GGUF import: lossless llama.cpp block import (IQ1_S/IQ2_XXS same path; GGUF expert family format 12,
  six-bit codebook format 14), split GGUF reading, `rad-convert --imatrix` fix, host-gather read-ahead.
- **IQR-MoE path on 1017**: "HF-direct Fp8Mx import; IQR-MoE converter and routing" (commit
  `9c7ce498`), "serve a dense IQR weight at i8a8h", "IQR shared expert in a mixed bf16 trunk",
  "packed stored expert plane, and the KV pool that survives it" (`79380811c0`). Kernels
  `r4d_moe_iqr.hip`, `r4d_gemm_iqr_p1/p2/p3.hip`, `r4d_iqr_layout.cpp`, `r4d_iqr_embed.hip`.
  NOTE: the IQR-MoE converter itself + the v5 recipe (`iqr-moe/ds41-moe-iq3r-full-v5.recipe`) are
  NOT in the public branch — bakon offers to share them ("happy to share both").
  The branch also carries an in-tree `exl3/` dir (quantizers + kernels + tools) — still the EXL3 side
  of the engine, pre-V3.
- Release-recipe form for the official checkpoint: `data/recipes/deepseek41-ud-iq1s.recipe`
  (Unsloth UD-IQ1_S import; their daily).
- PR structure plan (their draft comment): split into #242 (kvstream) first, imatrix fix,
  host-gather read-ahead, release reader + recipe, arch on existing kernels, per-family kernel PRs;
  #266 stays an umbrella draft. Conflicts with main on abi/rad_builder, rad_block_moe_fp8, issue.cpp.

## The IQR ladder result on DS41-1017 (bakon, last week; TP4 4x R9700, thinking off)

| container | experts | size | HE@164 | GSM@200 | decode |
|---|---|---|---|---|---|
| daily: Unsloth UD-IQ1_S import | IQ1_S/IQ2_XXS ~2.1 bpw | 251.1 GiB | 157 | 193–198 | ~85 tok/s |
| IQ3R v5 (pure IQR) | u3 all experts, 3.03 bpw | 307.7 GiB | 151 | 198 | ~52 tok/s |

At +22% bytes the IQR trellis is 6 HE tasks behind and ~40% slower; GSM fine. Width ladders
(mixing u2/u3), q6_K attention trunks, KV window residency all failed to close the gap.

**What they could not do (and why it matters to us):**
1. Per-expert Hessians never measured — `rad_block_moe_fp8.h` records one Gram per layer shared by
   all 384 experts; the per-expert tap exists but is ~8.5x slower (24 s/doc vs 2.8), a 15–25 h
   40-layer job; they stopped it.
2. Dense linears (attn q/kv/output, shared expert) have NO calibration tap — uncalibrated GPTQ
   fallback by construction.
Their open questions: (a) per-expert importance vectors (diagonal per expert, ~20 KB x 384) instead
of full Grams — cheap; (b) a Gram tap for dense linears; (c) trellis state count/rotation/codebook
(their build: u3, fwht128, 16-state trellis, tcq_gain 1.08, damp 0.01); (d) KL-guided per-tensor
allocation in the spirit of #364/#384. They will test anything that comes out of this on
HE@164 + GSM@200 + TP4 and post numbers. They are moving their own serving to Dynamic V3.1 IQ1_S.

## What this means for our DSV4.1-Flash-1017 EXL3 effort

1. **Our published calibration dataset is exactly the missing ingredient**
   (`Hero422/deepseek-v4.1-flash-1017-quant-calib`): G3 = per-expert Hessians for all 46,080 experts
   (D6 dump + Horvitz-Thompson), G1 = dense input Hessians for every trunk dense linear, G9 head,
   G12 indexer wk, G10 embed, routing counts, noise floors (G5/G6 on 1017), 100M-token golden.
   Open questions (a) and (b) are answered by our data with no new GPU work: (a) the per-expert
   importance vectors are the diagonals of our G3 Hessians; (b) the dense Grams are our G1 files.
2. **Engine base**: build on bakon's branch (or what lands from it) — no need to write the deepseek41
   arch from scratch. Two quantizer routes:
   - (i) IQR-layout route: calibrated per-expert trellis in HIS IQR-MoE format (feed per-expert
     allocation + fit into his converter; need the converter + recipe from him). Directly comparable
     to his v5 numbers on the same harness; least new kernel work.
   - (ii) mad-codebook route: port our exl3i/libr4d-mad decode to the DS41 op set (his libr4d DS41
     units already exist; the mad decoder is a sibling kernel family). Bigger work; on 27B our
     codebook beat IQR-V4 by 13–21% KLD, but his gap is 6 HE tasks at +22% bytes — calibration
     quality (our lever) vs codebook family (his) both need testing on the DS41 harness.
3. Decisions for the user: engage bakon (share data / get converter + recipe)? Route (i) vs (ii)
   first? Which engine base (his branch as-is vs wait for the split PRs to land on main)?
