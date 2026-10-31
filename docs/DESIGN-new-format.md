# New quant format — design outline (draft, 2026-10-30)

Goal: trellis-level quality per bit, but a prefill GEMM on the 8-bit matrix units (gfx1201 sustained: fp8/iu8
304–307 TFLOPS vs f16 158, our EXL3 core ~130–137), a fast decode (M=1, bandwidth-bound), and every continuous part
trainable for QAD. Built with our own quantizer on the challenge Hessians (+ YAQA), our own kernels, mixed per tensor
by the existing knapsack loop. Work starts after the overnight run; this file is the plan to argue about first.

## 1. The constraint that shapes everything

8-bit WMMA on RDNA4 multiplies 8-bit x 8-bit (iu8 x iu8 -> i32, or fp8/bf8 x fp8/bf8 -> f32). No f16 x 8-bit. So the
fast path is **W8A8 compute**: weights decode (in kernel) to 8-bit values, and **activations are quantized to 8 bit**
per step. Weight storage can stay at 1–8 bits (the code); only the compute operand is 8-bit.

- Activation quantization is a new error source our EXL3 path does not have. IQR does it (rotated int8 activations,
  one scale per (token, 128)); its 8.5-bpw I8A8 entry is 0.00046 KLD — that is roughly the floor it costs.
  -> Measure first (experiment E1): our bf16 model with activations fake-quantized to int8 / fp8 per (t,128) after the
  Hadamard rotation we already apply. If that alone costs >~0.0003 KLD, the Large tier keeps an f16 path.
- Option B (keep f16 activations, f16 WMMA): no new error, but the 2x is gone; only the decode gets cheaper.
  Likely answer: **both** — 8-bit compute where its KLD cost is small vs the weight error (Nano..Medium), f16 for
  the top tier / sensitive tensors (a per-tensor option in the knapsack, like bf16 is now).

## 1b. int4 compute (iu4 WMMA) — measured 2026-10-30, card 0

gfx1201 has **no fp4/fp6** (`wmma_f32_16x16x128_f8f6f4` needs gfx1250-insts), but it has an **iu4 WMMA at twice the
8-bit rate**. Peak, 8 independent accumulators per wave (`v3/bench/wmmabench.hip`; ISA checked, 8 WMMAs/loop):

| op | TOPS |
|---|---|
| iu8 16x16x16 / fp8 16x16x16 | 323 / 321 |
| iu4 16x16x16 | 328 (old shape, no gain) |
| **iu4 16x16x32** | **685** |

- iu4 is int4 x int4 (a sign flag per operand). The 2x needs **W4A4**: 4-bit weights *and* 4-bit activations.
- **Scales:** i32 partial sums must be scaled whenever a scale changes along K (`v3/bench/wmmagrp.hip`, 16x16
  tiles; the naive bench without scaling already runs at 460):

  | group along K | float (cvt + fma) | i32 multiply |
  |---|---|---|
  | 32 | 248 | 136 |
  | 64 | 366 | 236 |
  | 128 | 484 | 363 |

  -> i32 multiply is slow (quarter rate). Use a **per-output-channel weight scale and a per-token activation scale**
  (epilogue only, free), or groups of >=256 along K. Rotation (Hadamard) is what makes per-channel W4 tolerable
  (QuaRot/SpinQuant recipe). Weight codes must be uniform int4 (no codebook) to feed iu4 directly.
- **Benchmark pitfall:** identical accumulator chains or loop-invariant inputs get CSE'd (first runs showed
  1800 "TOPS"). Always count `v_wmma` in the ISA (`--save-temps`).
- **What iu4 buys:** prefill only. Decode is bandwidth-bound (4.0 bpw: 13.7 GB x 29.4 tok/s = 403 GB/s of ~640), so
  there only bytes and dequant cost matter; uniform int4 with a trivial unpack should get close to bandwidth
  (~1.3–1.4x decode headroom). Prefill now ~100 TF effective (1848 tok/s at 8k), mostly linears.
  - Ceilings: 8-bit ~3x, iu4 ~6.5x on the linears.
  - Realistic share of peak for a real kernel is ~50–65%; attention, GDN and norms don't speed up.
  - Guess: 8-bit ~1.7–2x end-to-end prefill, W4A4 ~2.5–3x.
- **Quality risk:** published W4A4 (QuaRot, SpinQuant) loses noticeably on 8B-class models. Expect KLD >= 0.05 for
  W4A4 everywhere, which is well above the 0.01 goal. So W4A4 has to be a **per-tensor option** chosen by
  sensitivity, not the whole model.
- **Precision menu per linear** (each is its own GEMM, so mixing per tensor has no kernel problem):

  | option | how | speed |
  |---|---|---|
  | W4A4 | one iu4 pass | 1x |
  | W4A8 | activation split into hi/lo nibbles, 2 iu4 passes on raw int4 weights (no weight dequant at all) | ~0.5x |
  | W8A4 | weight split into nibbles | ~0.5x |
  | W8A8 | iu8/fp8 | ~0.5x |
  | f16 | | ~0.25x |

  - **Within a tensor (Atom-style):** reorder K so outlier activation channels form the first K blocks. Those blocks
    get the 2-pass (A8) treatment and the rest a single pass, with cost (K + K_out)/K.
  - **Constraint:** linears reading the same activation (qg/k/v, gate_up) should share the A precision, or that input
    is quantized twice (cheap, but two outputs of the norm kernel).
- **Decision (2026-10-30):**
  - **Base format = W8A8 compute:** any weight code (1–8 bit) decoded to int8/fp8. W4A8 is dropped. It runs at the
    same speed as W8A8 (two iu4 passes) and would force uniform int4 storage.
  - **W4A4 = a per-tensor option**, only where its activation cost is small. It is plausible for <=0.05 KLD at
    3–3.5 bpw, tight for 0.01 at 4 bpw, since the weights already use ~0.008 there.
  - Amdahl: half the linear FLOPs at int4 -> GEMM time ~0.75x of all-8-bit.
  - Likely candidates: gate_up, attention/GDN input projections. ffn_down (post-SwiGLU outliers) likely not.
  - A full-model W4A4 quant is an optional side experiment, for speed curiosity only.
- **Knapsack fit:** add **time** as a second cost (per option, from a kernel microbench). Then solve
  min KLD s.t. bytes <= B and prefill time <= T, or a Lagrangian. The surrogate and DP machinery carries over;
  activation-quant cost per tensor comes from experiment E1b.



Candidates (all keep a trellis/vector code for the bits, differ in the decoded value):
- **A. Trellis on an int8 grid** (IQR-V2/V3 route): each state maps through a trained table to an int8 value; per
  (row-block x 128) scale folded once per k-block in the i32 accumulator. Decode = shift + mask + table lookup
  (IQR: one table lookup per weight; V2 found the *lookups* bound the speed, V1's scalar tables are faster).
- **B. EXL3-style computed codebook (hash) to fp8** instead of f16: keeps EXL3's quantizer almost unchanged, but the
  hash is ~50 VALU a 16x16 tile — the very thing that competes with WMMA issue. Only viable as a two-pass decode
  (decode once per step into fp8 fragments, like our T2 — half the bytes of f16).
- **C. Two-pass everywhere** (our T2 generalised): any code -> 8-bit fragments in a pre-pass, then a pure 8-bit GEMM.
  Pre-pass cost is fixed per weight per step (~10% now at f16; half at 8-bit) and is amortised by packing documents.
  Simplest kernel; decouples code design from GEMM design. Probably the first prototype.
Decision driver: decode VALU per weight (fused, A) vs pre-pass bytes (C). A fully optimised fused
trellis dequant with a table goes LDS-bound (IQR-V3 report §2: table gathers bound the GEMV); recomputing values
beats staging them.

## 3. Bits and options per tensor

- K = 1..8 bits per weight (whole and maybe half steps), per tensor; the knapsack mixes them (unchanged machinery).
- Scales: per (row-block, 128) group, fp8/E4M3 or f16, trainable. (Our AM/GM test says group scales add ~nothing to
  *quality* after rotation; here they exist for the int8 grid's range and for QAD.)
- Head: same format, **vocabulary-tiered** (rows permuted so frequency tiers are contiguous 128-row blocks, a K per
  tier, a gather after the logits). Design the tensor layout so a tensor can have row-block K now.
- **Embedding: trellis-coded rows** (IQR does it): a row-wise layout so a lookup decodes one row
  (EXL3's 128-row output rotation makes that impossible today). Same tiering by token frequency. Kernel: a gather that
  decodes rows on the fly.

## 4. Quantizer

- Our ourq.py loop on the challenge calib Hessians, with **YAQA** two-sided Hessians (output-side Kronecker factor
  collected once on calib; reusable for any format) and LDLQ/trellis rounding against the new grid.
- Trained per-K value tables (IQR trains its int8 tables) — fitted on calib weights, frozen before QAD or trained in it.
- Keep: per-layer resumable shards, low RAM (the shard-handle fix), proxy error out for hproxy.

## 5. QAD hooks (design-in, not bolt-on)

- Everything continuous is trainable with the codes frozen: scales, value tables, norms, (activation scales if static).
- Decode must have a differentiable reference (torch) implementation bit-equal to the kernel, for training.
- Data = the qad pool only (`eval/pools/`, `docs/METHOD.md` §6). Start from a fixed-code QAD.

## 6. Kernels

- Prefill: 8-bit WMMA GEMM, our T2 structure (128x256 tile, LDS-staged activations in fragment order, weight fragments
  from L2, grouped row blocks), activation quant fused into the producer (norm / rotation) like IQR.
- Decode (M=1): bandwidth-bound GEMV reading the codes (k bits/weight); IQR's window decode (one shift, mask, lookup,
  no sequential dependency) is the model.
- Embedding gather (trellis rows), tiered head + gather.
- Reuse: the standalone bench harness (v3/bench/tlgemm.hip, ntbench), byte-identity checks vs a reference.

## 7. Experiments, in order (each small, each decides something)

- E1: activation-quant floor — int8 / fp8 per (t,128) after rotation, bf16 weights, calib KLD. Decides 8-bit compute.
- E1b: per-tensor A4 / A8 sensitivity (rotated, per-token scale, bf16 weights): fake-quant one tensor family/band at a
  time, calib KLD -> the activation-cost term per option for the knapsack. Plus whole-model A4 to see the floor.
- E2: 8-bit WMMA core speed in tlgemm.hip (iu8 and fp8), same tiling — the real ceiling for our structure.
- E3: quality per bit of candidate codes on a few layers (Hessian error + small end-to-end band swaps vs our EXL3 K):
  trellis->int8 table (A) vs EXL3 hash->fp8 (B), with and without YAQA.
  Plus **codes on the int4 grid**: 2–3 bit trellis/codebook codes whose decoded values are int4 integers (per-channel
  scale), so a low-bpw tensor can feed the iu4 W4A4 path. Cost vs an unrestricted codebook at the same bits.
- E4: prototype C (two-pass, any code -> 8-bit fragments) end to end on one size (e.g. 4 bpw) vs ours-K4 EXL3:
  KLD and eval speed.
- E5: trellis embedding vs Lloyd at tiered widths (simulate first, as planned).
Then: full K sweep in the new format -> knapsack loop -> QAD on finalists.

## 8. Plan as of 2026-10-31 (brainstorm)

- **Integer everywhere (iu8/iu4), not fp8.**
  - Same speed (323 vs 321 TOPS).
  - An int4 grid is a subset of int8, so one stored int4-grid code serves W4A4 and W8A8 alike: the compute mode
    can be switched per tensor without requantizing.
  - Rotated activations are near-Gaussian, where int8 with a per-token scale has more SNR than e4m3 (3 mantissa
    bits).
  - fp8's only edge is outlier range, which the rotation removes.
  - Cost: scales must be per-token (A) and per-output-channel (W) to stay free. Group structure goes into the decoded
    int8 values (group scale x code rounded to the int8 grid), which adds an int8 weight rounding error. That error
    matters only at the top. **The Large tier keeps an f16/bf16 compute option per tensor.**
- **3D frontier (size : quality : speed):**
  - Each per-tensor option gets a byte cost and a time cost.
  - Time comes from a microbench per (shape, option) at prefill M = 4096. Decode time is ~bytes / bandwidth plus the
    dequant cost, also measured.
  - The DP solves min KLD s.t. bytes <= B and time <= T: a 2-D DP over byte buckets x time buckets, or a Lagrangian
    on time inside the existing byte DP. Sweep the (B, T) grid.
  - Display: one KLD-vs-prefill-speed curve per bpw slice.
- **Finer than whole-K steps:** rather than half-K codes, let K vary per **output-channel slab** (e.g. 128 or 256
  rows) inside a tensor.
  - The kernel reads the K per tile from metadata. The DP gets finer items, sensitivity varies by channel, and the
    effective bits can be any fraction.
  - The compute mode (A4/A8/f16) stays per tensor, since one activation format per GEMM.
- **QAD viability, designed in:**
  1. A torch reference dequant that is bit-exact with the kernel (test it like the T2 byte-identity checks).
  2. Trainable continuous parameters: per-channel scales, group scales, norms, embedding scales, and an optional
     learned codebook/LUT. Codes stay fixed during a QAD run, with optional re-quantize/re-Viterbi between runs.
  3. Activation fake-quant with STE and learned clip scales (LSQ-style) for A8/A4. Optionally learned rotations
     (SpinQuant), which help A4 the most.
  4. Memory on 2x32 GB: the student is quantized and dequantized on the fly, gradient checkpointing, optimizer state
     only for the small parameters. Teacher top-k logits are precomputed from bf16 over the qad pool (5.74M tok) and
     cached on disk, so no teacher in memory.
     - Order: blockwise distillation first (one layer at a time), then a short end-to-end run.
  5. QAD is per finished mix (per size). Budget compute for 2–3 finalists, not all 33 targets.
  6. Data: qad pool to train, verify pool to check, public only for the final pick.
- **Experiment order** (on cards once the overnight run ends):
  1. **E1/E1b:** activation fake-quant (A8, A4; rotation, per-token scale) on bf16 weights, per tensor family x depth
     band, calib KLD. Decides whether A8 is free and where A4 is affordable.
  2. **E2:** real-tiling int8 and int4 GEMM microkernels, which give the time-cost table. Include M sweeps for
     MoE-sized expert GEMMs.
  3. **E3:** codes on the int8/int4 grids vs trellis, quality per bit. Plus the slab-K check and the YAQA A/B.
  4. **E4:** end-to-end prototype at one size (~4 bpw): KLD and prefill/decode speed vs ours-K4 EXL3.
  5. Then the K sweep -> the 3D knapsack -> QAD on finalists.



- How much does 8-bit activation quant cost on THIS model (GDN layers included)? (E1)
- Fused decode vs two-pass at 8-bit: does the pre-pass at half the bytes make fused unnecessary?
- Half-step K worth it in the new format, or does tiering/mixing cover it?
- Compatibility: one new op row in radiance; containers via rad-convert (its own quant method) — check the plugin ABI
  for what a new dtype/encoding needs.
