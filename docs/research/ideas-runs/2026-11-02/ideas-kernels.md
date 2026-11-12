# Kernels lens: research ideas for EXL3‑V3 on gfx1201 (2026‑11‑02)

## 0. What I checked before writing, and four findings

The ideas below rely on these facts. Some I measured or read in the code today; the rest come from the attached documents.

- **F1. The V3 int8 codebook has an integer closed form (checked exhaustively on CPU).**
  - Let t = bytesum(state × 0x83DCD12D mod 2³²), so t ∈ [0, 1020].
  - The mul1i8 value is fp16(q · 4K_INV), with **q = (max(t + 3, 4) >> 2) − 128**.
  - This matches the reference (fp16 FMA → f32 rint → clamp ±127) on all 65,536 states. Only state 0 (t = 0) needs the `max`.
  - Consequences:
    - An int8 decode is: hash, `v_sad_u8` with accumulator 3 in each half, `v_pk_max_u16`, `v_pk_lshrrev_b16`, one `v_perm_b32` per 4 weights.
    - An fp16 decode adds `or 0x6400`, `pk_add_f16(−1152)` and `pk_mul_f16(4K_INV)`. Every step is exact, so the output is bit‑identical.
- **F2. The production path takes a slow route.** `radiance-hero/plugin/exl3i/exl3i_rate.hip.h` decodes mul1i8 by first computing the fp16 mul1 value. It then rounds each pair through f32 (`exl3i_i8_pair`: 2 cvt, 2 mul, 2 rint, 2 clamp, 2 mul, 2 cvt, pack).
  - By my reading of the source that is about 6–7 extra VALU ops per weight, on top of mul1's roughly 5. This is an estimate: no ROCm toolchain here, so no ISA count.
  - Every V3 tensor uses mul1i8, at every M. If decode at M = 1 is VALU‑bound (F4), **V3 containers could decode noticeably slower than V2's mul1 ones** unless F1 is used.
- **F3. Decode‑band and T2 configuration in the plugin** (`exl3i/nt.c`, `exl3r4d_dev.h`):
  - `sk = 1` (no split‑K) at every M. At M ≤ 16, down / ssm_out / attn_o (N = 5120) launch only 40 workgroups of 128 columns on 64 CUs.
  - The T2 decode chunk is 192 MiB, three times Navi 48's 64 MB Infinity Cache, so decoded weights round‑trip through DRAM.
  - T2 starts at M = 512; the NT form uses MT 4.
- **F4. Where decode time likely goes (estimate, to be tested).**
  - Measured: 32.5 tok/s at 1.40 bpw and 24 at 6.00, i.e. 30.8 ms vs 41.7 ms per token, while the bytes differ by 15.5 GB. The marginal cost is about 1.4 TB/s, which exceeds the card's bandwidth, so bytes do not set the time.
  - Two candidates for the fixed cost:
    - Per‑weight trellis decode: about 24.4e9 trunk weights × ~5 ops at ~12 T int‑ops/s ≈ 10 ms per token at any K.
    - Roughly 10–20 ms of per‑token overhead from small kernels, launches and sync.
  - At 4 bpw (13.7 GB, 403 GB/s = 63%) the bytes alone would take ~21 ms at full bandwidth.
- **F5. Prefill scale.**
  - iu8 matmul core at M = 2048 (`speedcal-stock.jsonl`): gate_up 2.81 ms, down 1.444 ms. f16 core: 6.10 / 3.17 ms. f16 T2 including decode: 6.74 / 3.65 ms.
  - My int8 per‑layer model (TP1, linears only): core ≈ 6.05 ms. On top of that: decode pre‑pass (≈ 382 MB int8 written, ~10%), prep + activation quant passes (~350 MB of traffic, ~9%), g128 fold (~12–17%, from E2), epilogue (~3%). Total ≈ 8.3 ms vs ~14 ms for f16 T2.
  - **The ~35% of overheads above the core is where most prefill ideas below act.**
- **F6. DFlash2 drafter on disk:** 5 dense layers at width 5120, 3.6 GB bf16, sharing the target's embedding and head. The bf16 checkpoint also has `mtp_num_hidden_layers: 1`, and radiance already ships a `spechub` MTP plugin.

Conventions below:
- "KL" means the change in mean KL at 2.5 bpw.
- "Speed" is relative to the stated baseline. Numbers marked *est.* are mine and untested.
- Test machine: iceman card 0 (R9700) unless noted; the 4070/proxy cannot run radiance kernels.

---

## A. Trellis decode cost

**1. Integer‑bytesum fp16 decode for mul1i8 (bit‑exact).**
- **Mechanism:** Replace `exl3i_i8_pair` with the F1 sequence. Per pair: sad_lo/sad_hi with accumulator 0x00030003, `pk_max_u16(·, 0x00040004)`, `pk_lshrrev_b16(2)`, `or 0x64006400`, `pk_add_f16(−1152)`, `pk_mul_f16(4K_INV)`. That is about 3.5 ops per weight after the hash, instead of fp16 FMA plus f32 rounding (~7.5).
- **Source:** new (found while reading the plugin today).
- **Why:** If decode is VALU‑bound (F4), the ops per weight set M = 1 speed. *Est.* about 12 → 7 ops per weight for V3 tensors. This also removes the f32 detour from T2 wdec and the narrow tile.
- **Fit:** kernel‑only, bit‑identical, transparent to the library and knapsack.
- **Test:** ntbench on gate_up and down at K2.5/K3 mul1i8, M = 1 and 16, old vs new, with an output hash equality check. About 1 h of card time; the code change is a few hours. Counts if µs per call drops ≥ 15% and hashes match.
- **Gain / risk:** KL 0. Decode tok/s +15–40% vs the current exl3i path (*est.*; depends on F4). Risk: the state‑0 corner; F1 covers it.
- **Conflicts:** none; #59 (exl3i) is where it lands.

**2. Integer int8 decode in the int8 wdec / fused tiles.**
- **Mechanism:** For the iu8 path, stop at the byte stage: two sad pairs, two pk_max, two pk_lshr, one `v_perm_b32` (bytes 0/2 of each dword), then `xor 0x80808080`. That is about 2.5 ops per weight after the hash and replaces `nfq_wdec_i8`'s fp16 decode → rint(v/4k_inv) → clamp.
- **Source:** new (F1).
- **Why:** About half the decode VALU for any int8 consumer. It matters most for the fused or narrow‑tile forms, where decode is repeated per M‑tile.
- **Fit:** kernel‑only, bit‑identical.
- **Test:** wdec standalone throughput (weights/s) at K2.5/K3/K4 before vs after; ~30 min.
- **Gain / risk:** KL 0. wdec itself is write‑bound at T2, so prefill gain is ≤ 2% there. The real value is enabling idea 15.
- **Conflicts:** #31 (E4 NFQ int8 crash) must be fixed first.

**3. Unsigned weight operand plus row‑sum correction (no XOR).**
- **Mechanism:** iu8 WMMA takes per‑operand signedness. Feed u = q + 128 as unsigned and subtract 128·Σₖx per (token, 128‑block). The activation prep computes that sum for free, and it folds into the same per‑block step as the scale.
- **Source:** new; a standard zero‑point trick.
- **Why:** Saves 0.25 op per weight; small.
- **Fit:** kernel‑only, exact in integer arithmetic.
- **Test:** igemm bench with an unsigned A and a correction term vs signed.
- **Gain / risk:** KL 0, speed < 1%. Only worth doing alongside 2.
- **Conflicts:** none.

**4. Pick the hash form by ISA count on gfx1201.**
- **Mechanism:** `EXL3R4D_V1CD` (2 `pk_mul_lo_u16` + 2 `mad_u32_u16` per pair) vs the default (1 pk_mul + shift + and + 2 mad). Count instructions and cycles per 256 weights with `--save-temps` for each rate.
- **Source:** existing flags in exl3r4d.
- **Why:** The hash is about 40% of decode VALU (*est.*).
- **Fit:** kernel‑only, bit‑identical.
- **Test:** compile both, count VALU per fragment, then ntbench M = 1. Cost is the compile (minutes) on a box with ROCm.
- **Gain / risk:** KL 0, 0–10% decode VALU.
- **Conflicts:** none.

**5. Dual‑issue (VOPD) friendly decode schedule.**
- **Mechanism:** RDNA4 can dual‑issue certain VALU pairs in wave32 (add, and, lshl, max, fma, mul, …). Restructure the decode so independent ops from two runs pair up — the two runs of a fragment are independent — and check the ISA for VOPD.
- **Source:** new for our kernel.
- **Why:** Up to about 1.3–1.6× VALU throughput on the dual‑issuable share. `sad` and `mad_u32_u16` are probably not dual‑issuable.
- **Fit:** kernel‑only, bit‑identical.
- **Test:** ISA inspection plus ntbench M = 1.
- **Gain / risk:** KL 0, decode +5–15% (*est.*). Risk: compiler control may be hard.
- **Conflicts:** none.

**6. Decode two fragments per step for ILP.**
- **Mechanism:** At M ≤ 16, each lane decodes one fragment per WMMA. Interleave two tiles' decode streams so the hash and sad latencies overlap. Registers allow it at MT 1.
- **Source:** new.
- **Why:** Latency hiding in a VALU‑bound loop.
- **Fit:** kernel‑only.
- **Test:** ntbench M = 1 with 1 vs 2 tiles per step.
- **Gain / risk:** KL 0, 0–10%.
- **Conflicts:** none.

**7. Fractional‑rate lane plan in scalar or uniform form.**
- **Mechanism:** `Exl3iLanePlan` holds woff/shift/off/xr per lane in VGPRs: about 20 registers. Many plan values are periodic in the lane index. Compute them from a small per‑wave table in SGPRs, or recompute cheaply, so fractional rates keep the same occupancy as integer K.
- **Source:** new; uses `exl3i_plan.h`.
- **Why:** Fractional K is the V3 menu (−9.6% KL at 2.5 bpw, knap4). If fractional tiles run slower, the speed table penalises the levels the solver wants.
- **Fit:** kernel‑only.
- **Test:** register count and ntbench M = 1/16 and wdec at K2, 2.25, 2.5, 2.75, 3.
- **Gain / risk:** KL 0. Removes a possible 0–20% penalty on fractional rates (*unknown until measured*).
- **Conflicts:** #36 / #37 / #59.

**8. Rejected, for the record: LDS LUT decode.**
- A 64K‑entry int8 LUT is exactly the LDS of a CU, and random `ds_read_u8` gathers bound IQR‑V3's GEMV (design A2).
- F1 makes arithmetic cheaper than any gather. Not proposed.

---

## B. Prefill int8 GEMM (M ≥ 512)

**9. Fuse the input rotation and int8 quantization into one prep kernel.**
- **Mechanism:** Today the prep writes rotated f16 (`[T][2K]` for stacked gate_up), then `nfq_act_i8` reads it and writes int8. Instead, read x once: rotate (DPP Hadamard), take amax per (row, 128), quantize, and write int8 in fragment order plus scales and the block sums for idea 3.
- **Source:** BACKLOG item; quantified here.
- **Why:** Down's input is 2048 × 17408. That is 71 MB f16 written plus 71 MB read plus 36 MB int8, ~0.25 ms against a 1.44 ms core (≈ 17%). Per layer ≈ 0.3–0.4 ms of ~8 ms (*est.*).
- **Fit:** kernel‑only. Numerically identical if the same rounding is used.
- **Test:** ntbench prep + act vs fused prep at the down and gate_up shapes, M = 2048; hours.
- **Gain / risk:** KL 0, int8 prefill −4–6%.
- **Conflicts:** BACKLOG "activation quantize fused into the rotation pass".

**10. Quantize down's input in the gate_up epilogue.**
- **Mechanism:** EPI 1 already holds a 16 × 128 block of h = silu(g)·u in LDS. Apply suh_down and the 128‑point Hadamard (the epilogue already has `rotate128`), then amax, int8 and block sum, and store int8 fragments directly. This removes down's prep entirely.
- **Source:** new (producer‑side fusion).
- **Why:** It removes the full h round trip (71 MB f16 write + read) and down's prep launch. Down + prep ≈ 22% of the MLP's int8 time.
- **Fit:** kernel‑only. Byte‑identical if rotation, rounding and amax order match the standalone prep.
- **Test:** compare gate_up + down time with fused vs separate at M = 2048; check the hash of down's output.
- **Gain / risk:** KL 0, −3–5% per layer (*est.*). Risk: epilogue register and LDS pressure (16 KB is already in use).
- **Conflicts:** none.

**11. Gated output norm, rotation and int8 fused for ssm_out (and the attention output gate for attn_o).**
- **Mechanism:** GDN's o ← RMSNorm(o) · silu(z) (and attention's sigmoid output gate) runs as a separate elementwise op before ssm_out's prep. Do it in one prep kernel: read o and z → gated norm → suh → rotate → int8.
- **Source:** new.
- **Why:** Saves one 2048 × 6144 round trip (25 MB ×2) per layer, on 48 + 16 layers.
- **Fit:** kernel‑only.
- **Test:** profile‑ops on the separate ops vs the fused kernel.
- **Gain / risk:** KL 0, −1–2% of prefill.
- **Conflicts:** none.

**12. RMSNorm kernel emits the reader's int8 tile.**
- **Mechanism:** The input norm writes bf16, then each reader's prep reads it. Make the norm kernel produce the rotated int8 tile directly for the reader group (needs idea 13's tied suh to serve q/k/v or gate/up from one tile).
- **Source:** new.
- **Why:** Saves 21 MB ×2 per reader group per layer.
- **Fit:** kernel‑only; needs a requant only if siblings must share.
- **Test:** profile‑ops.
- **Gain / risk:** KL 0, −1–2%. Caveat: if the board counts only the GEMM op, moving work into the norm flatters `gemm_us`. Report real prefill too.
- **Conflicts:** #42 (siblings don't share suh).

**13. Tie suh across siblings at the Phase‑B requant (gate/up first).**
- **Mechanism:** Today the stacked gate_up rotates the input twice (`[T][2K]` region, suh and suh.1). The median gate/up suh ratio is 1.015 (#42), so a joint refit should cost almost nothing. One rotated int8 tile then serves both halves. Later, apply the same to GDN qkv/z (1.036) and attention q/k/v (1.092).
- **Source:** design §1d / #42 follow‑up.
- **Why:** Halves gate_up's prep and activation quant, and enables 12.
- **Fit:** format/requant: tensors re‑quantized with a shared suh. It works inside the per‑level library if all levels use the tie.
- **Test:** CPU proxy error of a tied vs untied refit on 6 layers (Thor), then one 27B layer swap.
- **Gain / risk:** KL +0–0.3% (*est.*, gate/up), prefill −1–2%.
- **Conflicts:** #42 measured that they don't share; this proposes forcing it.

**14. Magic‑bias accumulator fold (one VALU op per accumulator per block).**
- **Mechanism:** Start each 128‑block's iu8 WMMA chain with C = 0x4B400000 instead of 0. The i32 result, read as a float, equals 12582912 + P_b exactly, because |P_b| < 2²² (128 · 127² = 2.06M). Then fold with a single `fma(asfloat(r), s_tb, acc)`, with no `cvt_f32_i32`, and subtract 12582912·Σ_b s_tb per row at the store (the prep provides Σs).
- **Source:** new; the classic magic‑number int→float conversion, done inside the WMMA accumulate.
- **Why:** E2 measured iu8 g256 at −17% and g1024 at −12% at a 128×128 tile. Halving fold ops could recover about half of that. Cost: an f32 cancellation error of about 1e‑3 relative per output (*est.*), roughly 10× below A8‑g128's own noise.
- **Fit:** kernel‑only. Not byte‑identical to the plain fold; deterministic, and the reference can emulate it.
- **Test:** wmmagrp.hip variant: g128 fold with cvt+fma vs magic+fma, TOPS and max rel error. Then one E1b‑style KL run at K3.
- **Gain / risk:** KL < +0.05% (*est.*), A8‑g128 GEMM +6–10%.
- **Conflicts:** #28 (E2 measured the fold penalty).

**15. Packed‑f32 / VOPD fold.**
- **Mechanism:** Use `v_pk_fma_f32` (2 FMAs per op) and dual‑issue the converts, so 16 fold ops per lane per block become about 8–10. Choose the WMMA accumulator layout so pairs of accumulators share a scale (same token row).
- **Source:** new.
- **Why:** Same target as 14, exact (no magic error).
- **Fit:** kernel‑only, bit‑identical to the cvt+fma fold.
- **Test:** wmmagrp.hip variant.
- **Gain / risk:** KL 0, +4–8% on A8‑g128 GEMMs (*est.*).
- **Conflicts:** #28.

**16. Static exponent groups along K (fold only at group boundaries).**
- **Mechanism:** For each tensor, calibrate a static power‑of‑two factor c_b per 128‑block (most blocks 1; outlier blocks larger). Permute the 128‑blocks of K at import so equal‑c_b blocks are contiguous (a free relayout of weight tiles and activation fragment order). Then accumulate each group in one i32 chain (overflow‑safe: 127² · 17408 < 2³¹) with a per‑token scale, and fold only 2–3 times per K loop.
- **Source:** new; combines E1b (g128 ≫ per‑token) with the static massive‑activation channels seen in #47.
- **Why:** Aims at g128 quality (+0.00005 at K6) at near per‑token speed (254 vs ~210–225 TOPS in E2).
- **Fit:** kernel plus import relayout; no requant.
- **Test:** CPU first, on the E1c activation dumps: A8 error of per‑token+static groups vs g128 vs per‑token (`aqproxy.py`). Then a time check in wmmagrp. Hours on CPU box A.
- **Gain / risk:** KL ≈ g128 if outliers are static (unknown for down's input). A8 GEMM +10–15% vs g128 (*est.*).
- **Conflicts:** #26‑01 / #27 (A8 group results), #47 (R1 spread outliers; this keeps them blocked).

**17. Per‑token scale plus a separately scaled static outlier block.**
- **Mechanism:** A simpler variant of 16. One per‑token scale computed with the outlier block(s) excluded, plus a dedicated scale for those 1–2 known blocks, folded only there.
- **Source:** new.
- **Why:** Per‑row A8 on gdn_in / attn / down costs ≤ +0.00004 each in E1b, but gate_up costs +0.00011. Fixing gate_up's outlier block might get most of g128's gain at about 1/40 of its fold cost.
- **Fit:** kernel‑only.
- **Test:** CPU aqproxy on gate_up dumps (layers 0–15 have the 65%‑energy channel).
- **Gain / risk:** KL −0.3–0.6% vs per‑token at 2.5 bpw (*est.*). Speed ≈ per‑token.
- **Conflicts:** #26‑01.

**18. Hi/lo residual activation pass on outlier blocks only (MSD‑style).**
- **Mechanism:** For the static outlier blocks, decompose x ≈ s(x_hi + x_lo/256) into two int8 operands and run an extra iu8 WMMA chain on those blocks only. That block gets A16‑grade precision at a cost of +n_outlier/n_blocks of the GEMM.
- **Source:** 2605.13915 (MSD: decompose activations instead of dequantizing weights). Here it is applied only to a few blocks.
- **Why:** It could let gate_up and head‑adjacent layers run A8 where the solver now picks f16.
- **Fit:** kernel‑only.
- **Test:** CPU aqproxy error, then wmmagrp.
- **Gain / risk:** KL −0.1–0.3% (*est.*), speed −2–5% on the affected tensors.
- **Conflicts:** none.

**19. Prefer cheaper folds at low bpw (solver input, not new code).**
- **Mechanism:** At K6, g256–1024 cost +0.00010–0.00013 over g128. At 2.5 bpw (KL 0.0666) that is ≤ 0.2% — below paired SE — while costing 10–15% less GEMM time. Make sure the time table prices g256/g512/g1024 and per‑token so knap4 can choose them per size.
- **Source:** design §1b.3; E1b numbers.
- **Why:** Speed comes free at the sizes that matter most.
- **Fit:** solver option, already in the design.
- **Test:** the planned time table plus one "activation cost on a low‑K base" E1b batch.
- **Gain / risk:** KL +0.1–0.2%, A8 GEMM +10%.
- **Conflicts:** BACKLOG "activation cost on a low‑K base".

**20. Activation‑aware suh at requant (makes fold‑free per‑token A8 viable).**
- **Mechanism:** suh already cuts per‑token A8 error by 35% vs signs only (#47 side result). Optimise suh in the Phase‑B requant for weight proxy error + λ·(per‑token A8 activation error from the E1c dumps). Then more tensors can use per‑token (no fold, ~254 TOPS) at g128‑like quality.
- **Source:** new; SmoothQuant‑like, but inside EXL3's existing column scale.
- **Why:** Removes the fold cost.
- **Fit:** requant (per‑level, all levels).
- **Test:** CPU on 6 layers: Pareto of weight proxy vs A8 per‑token error over λ.
- **Gain / risk:** KL ±0.3% (weight error may rise), A8 GEMM +10–15%.
- **Conflicts:** #42, #48 (output scales: different side).

**21. Overlap wdec of chunk i+1 with the GEMM of chunk i.**
- **Mechanism:** T2 runs wdec(chunk) then GEMM(chunk) serially. Use two queues, or a persistent kernel where a few waves decode ahead into a ring while the rest run WMMA.
- **Source:** new; producer/consumer in the spirit of BiLoCo's concurrent CUDA‑core work (2610.32243).
- **Why:** f16 T2 overhead is 0.64 ms on gate_up (8.08 → 6.74 vs core 6.10); about 10% at int8 (F5).
- **Fit:** kernel/engine‑only, bit‑identical.
- **Test:** ntbench gate_up M = 2048, chunk‑pipelined vs serial.
- **Gain / risk:** KL 0, int8 prefill −4–8% (*est.*). Risk: the board times one op; an overlapped op still counts its own time.
- **Conflicts:** none.

**22. Size T2 chunks to the 64 MB Infinity Cache.**
- **Mechanism:** The chunk is 192 MiB (F3), so decoded fragments spill to DRAM. Sweep 16/32/48 MB chunks so the GEMM reads them from MALL.
- **Source:** new (found in the code).
- **Why:** The int8 wdec writes ~178 MB for gate_up. A MALL‑resident chunk could halve the DRAM traffic of the pre‑pass.
- **Fit:** kernel‑only, bit‑identical.
- **Test:** ntbench gate_up / down at M = 2048 with `EXL3R4D_T2_CHUNK_BYTES` swept; 30 min.
- **Gain / risk:** KL 0, −2–5% (*est.*).
- **Conflicts:** none.

**23. Weight‑stationary GEMM that decodes each weight once.**
- **Mechanism:** Each workgroup owns a 128‑column block. It decodes its weights once into LDS (double‑buffered K chunks of 128 × 256 int8 = 32 KB), then streams all M = 2048 rows of int8 activations through them. Weights are never written to scratch; each is decoded once per call.
- **Source:** new (stream‑M instead of T2's tile‑and‑pre‑pass).
- **Why:** Removes wdec's write (~10%). Activation re‑reads (272 blocks × 10 MB for gate_up) are of the same order as T2's weight re‑reads and come from L2/MALL.
- **Fit:** kernel‑only, bit‑identical if the WMMA order is kept.
- **Test:** igemm.hip prototype of gate_up 2048 × 34816 × 5120, measuring TOPS.
- **Gain / risk:** KL 0, ±10% (*uncertain*; LDS bandwidth may bind).
- **Conflicts:** none.

**24. Re‑measure the fused (NT) vs T2 crossover with integer decode.**
- **Mechanism:** The f16 crossover is about 450–512 rows. With ideas 1–2 the in‑tile decode is about 2× cheaper, so the fused iu8 form may win up to M ≈ 1–2k for some shapes and skip the scratch entirely.
- **Source:** design §A2.C vs B.
- **Why:** Decides the int8 architecture.
- **Fit:** kernel‑only.
- **Test:** ntbench sweep M = 64–2048 for fused iu8 vs T2 iu8, after idea 2.
- **Gain / risk:** KL 0, 0–8%.
- **Conflicts:** #31.

**25. Re‑tile iu8 for fold registers.**
- **Mechanism:** At 128×256 the g128 fold spills (73 TOPS); at 128×128, 210–225. Try 256×128 (the M‑major split shares the per‑token scale across fewer accumulators), MT/NPW variants, and 64×256.
- **Source:** E2.
- **Why:** Recovers part of the fold penalty.
- **Fit:** kernel‑only.
- **Test:** wmmagrp / igemm grid sweep, a few hours.
- **Gain / risk:** KL 0, +3–8% for A8‑g128.
- **Conflicts:** #28.

**26. Split‑K / stream‑K at M = 384–1024 for N = 5120 shapes.**
- **Mechanism:** iu8 down at M = 512 reaches 189–210 TOPS vs 252 at 2048 (under‑occupied). Add split‑K 2–4 below 1024 rows.
- **Source:** E2 note "needs split‑K".
- **Why:** Real prefill has partial last chunks, and spec‑verify batches.
- **Fit:** kernel‑only. The reduction order changes numerics unless done deterministically in LDS.
- **Test:** igemm down M = 384/512/768.
- **Gain / risk:** KL ≈ 0, +15–25% on those calls; small overall.
- **Conflicts:** none.

**27. Measure the A8 M‑threshold with fused integer decode.**
- **Mechanism:** "Same speed below 512" is unmeasured. With idea 2 the fused iu8 narrow tile may beat f16 from M ≈ 64.
- **Source:** design checklist.
- **Why:** Batched decode and spec verify.
- **Fit:** dispatch policy.
- **Test:** the planned M = 16–511 sweep, repeated after ideas 1–2.
- **Gain / risk:** KL changes only for serving at M < 512 (the board is unaffected); serving +10–40% at M = 64–256 (*est.*).
- **Conflicts:** design §6 "before step 3".

**28. 4096‑row prefill chunks for long prompts.**
- **Mechanism:** The decode pre‑pass is paid per chunk. At 4096 rows it is amortised twice as well, and iu8 TOPS are the same (259 vs 260).
- **Source:** #6 found no gain on f16 (GEMM‑bound at 40% of peak); with T2/int8, decode is a separate fixed cost.
- **Why:** Real long prompts (agentic traces) cost decode work per 2k chunk.
- **Fit:** engine setting.
- **Test:** prefill of a 32k prompt at 2048 vs 4096 chunk (needs `--checkpoint-interval 4096`).
- **Gain / risk:** KL 0 (KLD identical in #6), prefill −3–5% (*est.*).
- **Conflicts:** #6 (null then; the reason differs now).

**29. 128‑point output rotation with an f16 WMMA (H₁₂₈ as a matrix).**
- **Mechanism:** The epilogue's Hadamard uses DPP/LDS butterflies. One 128×128 f16 WMMA per 16 rows could do it in a few instructions.
- **Source:** new.
- **Why:** The epilogue becomes a larger share as the core speeds up (~3–5% *est.*).
- **Fit:** kernel‑only; numerics differ (f16 rounding before rotation), within the store rounding.
- **Test:** epilogue microbench plus a hash/KL check.
- **Gain / risk:** KL ~0, −1–3%.
- **Conflicts:** none.

**30. Residual writers' epilogue: residual add plus partial sum of squares.**
- **Mechanism:** down / ssm_out / attn_o epilogues add the residual and write per‑(row, column‑tile) partial Σx². The next norm (or its fused prep, idea 12) only finishes the reduction.
- **Source:** new.
- **Why:** Saves one residual read per sublayer (21 MB).
- **Fit:** kernel‑only; deterministic if the partials are summed in fixed order.
- **Test:** profile‑ops.
- **Gain / risk:** KL 0, −0.5–1%.
- **Conflicts:** none.

**31. One launch per shared‑input sibling group (q/k/v, GDN qkv+z) with per‑segment rate.**
- **Mechanism:** Concatenate N across siblings in one dispatch. Each workgroup lies in one segment, branches uniformly to that segment's rate plan, and reads one int8 tile (with idea 13).
- **Source:** design §1d "fused GEMMs force K ties". Here no tie is needed.
- **Why:** Fewer launches, one activation read; at decode, fewer small kernels (see 34).
- **Fit:** kernel; mixes stay per‑tensor.
- **Test:** ntbench of separate vs fused qkv+z at M = 1 and 2048.
- **Gain / risk:** KL 0, prefill −1%, decode −2–4% (*est.*).
- **Conflicts:** none.

---

## C. Decode (M = 1–16) and the per‑token fixed cost

**32. Split‑K at decode for N = 5120 shapes.**
- **Mechanism:** With `sk = 1` (F3), down (K = 17408), ssm_out and attn_o launch 40 workgroups on 64 CUs. Use sk = 2–4 at M ≤ 16, with a deterministic LDS or second‑pass reduction (the split‑K LDS path exists in exl3r4d).
- **Source:** new finding in the plugin config.
- **Why:** Those tensors are about 30% of trunk weights; at 40/64 occupancy, *est.* 8–12% of decode GEMV time is lost.
- **Fit:** kernel‑only. Reduction order changes, but only for serving decode; the board is unaffected.
- **Test:** ntbench down/ssm_out at M = 1, sk = 1/2/4, measuring GB/s; 30 min.
- **Gain / risk:** KL 0 on the board, decode +5–10%.
- **Conflicts:** none.

**33. Profile, then cut the per‑token overhead (fewer launches, fused small ops).**
- **Mechanism:** F4 suggests about 10–20 ms per token is not the GEMV. Get it from `--profile-ops` at M = 1: Σ(op time) vs wall time per token, and the per‑op split. Then fuse the obvious chains: norm→prep, GDN conv+gates+recurrent+gated‑norm (35), q/k‑norm+RoPE (46), and sibling GEMVs (31). Push the decode step towards a few large kernels inside the recorded replay.
- **Source:** new; IQR‑V3 decode reference.
- **Why:** Potentially the largest decode lever after spec decode.
- **Fit:** engine and kernels.
- **Test:** diagnostic D1 (below). Then one fusion at a time with tok/s measured.
- **Gain / risk:** KL 0, decode +20–60% if the overhead hypothesis holds (*very uncertain*).
- **Conflicts:** #7 (rocprof unusable; use `--profile-ops`).

**34. int8 decode band: integer decode plus `v_dot4`/iu8 with A8 at decode (serving only).**
- **Mechanism:** At M ≤ 16, decode weights straight to int8 (idea 2) and quantize activations per token/128 to int8. This skips idea 1's fp16 conversion: about 6 instead of 7 ops per weight.
- **Source:** new; the design keeps f16 below 512 for quality.
- **Why:** Board KL is computed at prefill, so decode numerics never enter the score. They do change serving output slightly (A8‑g128 ≈ +5% KL at K6, ≈ +0.1% at 2.5 bpw).
- **Fit:** serving mode, optional flag.
- **Test:** ntbench M = 1/16 f16‑integer vs int8 band.
- **Gain / risk:** KL 0 on the board (serving +0.1–5%), decode +5–15% over idea 1.
- **Conflicts:** design §1b.2 (A8 only M ≥ 512): would need a user decision.

**35. One fused GDN decode step kernel.**
- **Mechanism:** At M = 1, a GDN layer runs conv1d‑state update, SiLU, q/k L2‑norm, a/b gates (softplus/sigmoid), the delta‑rule update of a 48 × 128 × 128 fp32 state, the output RMSNorm·silu(z) and residual pieces as several small kernels. Fuse them into one kernel per layer: a state read and write of 3.1 MB per layer.
- **Source:** LinQuant (2610.39418) / LeapQuant (2609.38166) kernel structure; new for us.
- **Why:** 48 layers × ~4–6 launches saved per token.
- **Fit:** kernel‑only, bit‑identical if the op order is kept.
- **Test:** profile‑ops M = 1 before and after; tok/s.
- **Gain / risk:** KL 0, decode +3–10% (*est.*).
- **Conflicts:** none.

**36. Halve GDN state traffic at decode and verify (ReplaySSM).**
- **Mechanism:** Keep a checkpoint plus a buffer of recent (k, v, β, g) inputs. Compute outputs from checkpoint + buffer, and write the state only every r steps.
- **Source:** 2610.13778 (roughly halves state traffic, buffer‑pointer rollback for speculation).
- **Why:** 48 × 3.1 MB fp32 read + write per token ≈ 0.47 ms (~1.5%). The main value is spec‑verify rollback (see 59).
- **Fit:** kernel/engine.
- **Test:** tok/s at M = 1; then with DFlash2.
- **Gain / risk:** KL 0 if replay is exact in fp32; decode +1–2% alone.
- **Conflicts:** none.

**37. Batch to M = 16 for free (serving policy).**
- **Mechanism:** If decode is VALU‑ or overhead‑bound, a 16‑row decode band costs about the same as 1 row: f16 core down is 0.569 ms at both M = 16 and M = 256 in speedcal. Batch concurrent requests (or spec rows) up to 16 per step.
- **Source:** speedcal table.
- **Why:** Up to 16× aggregate throughput at the same latency.
- **Fit:** serving.
- **Test:** tok/s at batch 1/4/16.
- **Gain / risk:** KL 0, multi‑user throughput 5–15×.
- **Conflicts:** §1c.1 (batching‑dependent outputs) applies only above 512.

**38. Head GEMV on the integer path, with slab rows WG‑uniform.**
- **Mechanism:** The head at decode is 1.27G weights (~5% of trunk) for ~1–2 ms per token. Use idea 1. For the slab head, make sure slabs are 128‑row aligned so each workgroup runs one rate plan (no divergence) and the row gather stays coalesced.
- **Source:** design §2 slab head.
- **Why:** The slab head adds plan switching; keep it free.
- **Fit:** kernel.
- **Test:** ntbench of the head shape, uniform vs slab.
- **Gain / risk:** KL 0, decode +2–4%.
- **Conflicts:** #34.

**39. Prefetch the next layer's weights into MALL during small ops.**
- **Mechanism:** While GDN/attention/norm kernels run (bandwidth mostly idle), issue a low‑priority prefetch of the next GEMV's first 32–48 MB into the Infinity Cache.
- **Source:** new.
- **Why:** Hides part of the fixed cost behind memory work.
- **Fit:** engine.
- **Test:** decode tok/s with and without a prefetch kernel on a side queue.
- **Gain / risk:** KL 0, 0–8% (*speculative*).
- **Conflicts:** none.

**40. TP2 decode for two‑card users.**
- **Mechanism:** Splitting N halves per‑card decode VALU and bytes. The 128 all‑reduces per token are latency‑bound (~10–20 µs each over the x4 link).
- **Source:** design ("TP2 only at times").
- **Why:** Up to about 1.5× decode on two cards (*est.*).
- **Fit:** serving.
- **Test:** tok/s TP1 vs TP2 at 2.5 and 4 bpw.
- **Gain / risk:** KL 0; risk: PCIe x4 latency.
- **Conflicts:** #9 (TP2 negative for evals: a different workload).

**41. Long‑context decode attention: GQA packing and split‑KV tuning.**
- **Mechanism:** Each KV head serves 6 query heads (24/4). Make sure one workgroup reads each K/V tile once for all 6 heads. Tune split‑KV per context length (the #328 fix exists).
- **Source:** design; 2609.17652 context.
- **Why:** At 128k context the KV read is 8 GB per token (64 KB per position), comparable to the 13.7 GB of weights at 4 bpw.
- **Fit:** kernel.
- **Test:** decode tok/s at 8k/64k/128k.
- **Gain / risk:** KL 0, +10–40% at long context.
- **Conflicts:** #49/#74 (attention port and fixes).

**42. Calibrated KV8 cache as a serving mode above 64k.**
- **Mechanism:** fp8 or int8 KV with calibrated static per‑channel scales. Static scales recovered 83% of FP8's long‑context loss on this exact model (2609.04098).
- **Source:** 2609.04098; process pitfall "fp8 KV moves KLD".
- **Why:** Halves the long‑context KV read and doubles cacheable context per card.
- **Fit:** serving option, never the board.
- **Test:** tok/s and verify‑long KL with KV8 vs bf16.
- **Gain / risk:** KL +0.5–3% at long context (serving only), decode +20–40% at 128k.
- **Conflicts:** process pitfall (fp8 KV): opt‑in only.

**43. Bandwidth tuning per rate for ≥ 5 bpw sizes.**
- **Mechanism:** At 6 bpw, decode reaches ~76% of bandwidth (*est.* from F4). Fractional tiles (e.g., K = 5.5 → 88 words per tile) may lose coalescing. Tune load width and nontemporal hints per rate.
- **Source:** design §1c.
- **Why:** The top sizes are bandwidth‑bound.
- **Fit:** kernel.
- **Test:** GB/s per rate at M = 1.
- **Gain / risk:** KL 0, +3–8% at ≥ 5 bpw.
- **Conflicts:** none.

---

## D. Fractional K, stacked tensors, format‑level kernel support

**44. Per‑side rate in the stacked gate_up kernel (untie gate and up K).**
- **Mechanism:** Every 128‑column workgroup is wholly on one side of the seam. Give each side its own (KA, MASK) plan, selected by a uniform branch: no divergence, two plans compiled in.
- **Source:** design §1d "fused GEMMs force K ties".
- **Why:** Lets knap4 price gate and up separately (their sensitivities differ, e.g. the L63 gate_up spike). This is a knapsack freedom gain, at no speed cost.
- **Fit:** kernel plus solver; per‑level library unchanged.
- **Test:** knap4 offline: tied vs untied gate/up at 2.5/3.0 (CPU, minutes). Build only if ≥ 0.5% predicted.
- **Gain / risk:** KL −0.3–1% (*est.*). Risk: code size.
- **Conflicts:** none.

**45. Census‑driven kernel instances (compile only the rates the mixes use).**
- **Mechanism:** 64 rates × 2 codebooks × modes × group sizes leads to an instance explosion. Generate the instance list from the shipped mixes' census; keep a generic fallback for the rest.
- **Source:** design §1c.4 census.
- **Why:** Build time, binary size and I‑cache; enables 44 and 31 cheaply.
- **Fit:** build system.
- **Test:** build time and binary size.
- **Gain / risk:** KL 0, speed 0–2% (I‑cache).
- **Conflicts:** none.

**46. Q/K‑norm + partial RoPE + output‑gate split fused into the attn_qg/k/v epilogue.**
- **Mechanism:** The epilogue holds output rows of 128–256 columns (head_dim 256). Apply per‑head RMSNorm, RoPE on 25% of dims (`partial_rotary_factor` 0.25) and the gate split before storing.
- **Source:** new.
- **Why:** Saves a round trip of 2048 × 14336 per attention layer, and decode launches.
- **Fit:** kernel‑only.
- **Test:** profile‑ops.
- **Gain / risk:** KL 0, −0.5% prefill, decode +1–2%.
- **Conflicts:** none.

---

## E. GDN and attention kernels

**47. Profile prefill shares after int8 (GDN chunk, attention, conv, norms).**
- **Mechanism:** With the linears about 1.7–2× faster, today's "8% non‑GEMM" becomes 15–20%. Measure at 2k and at 32k/128k context.
- **Source:** design §1.
- **Why:** Ranks ideas 48–53.
- **Fit:** diagnostic.
- **Test:** `--profile-ops` on one 2k and one 64k document; 1 h.
- **Gain / risk:** none directly.
- **Conflicts:** none.

**48. GDN short conv + SiLU + q/k L2‑norm in the qkv GEMM epilogue (3‑row halo).**
- **Mechanism:** The conv1d (kernel 4) runs along time. Tiles of 128 rows need 3 halo rows: recompute them (3/128 = 2.3% extra rows) or read the previous tile's raw rows from a small buffer. The chunk start reads the conv state.
- **Source:** new.
- **Why:** Saves a write and read of 2048 × 10240 (42 MB ×2, ≈ 0.13 ms per GDN layer, ~15% of an iu8 qkv call *est.*) on 48 layers.
- **Fit:** kernel‑only, bit‑identical if the arithmetic order is kept.
- **Test:** profile‑ops before and after on one layer; hash check.
- **Gain / risk:** KL 0, prefill −2–4%.
- **Conflicts:** none.

**49. INT8 GDN chunk matmuls (LinQuant recipe).**
- **Mechanism:** Run QKᵀ, the state read/write and the state‑correction contraction on iu8 with static per‑(head, value‑channel) scales. FP32 state carry, FP32 WY solve, Hadamard R_QK on q/k (valid because GDN's decay is scalar).
- **Source:** 2610.39418 (GDN 340M: INT8 per‑channel 15.019 vs 15.004 PPL).
- **Why:** Speeds the GDN core if 47 shows it matters.
- **Fit:** kernel; global switch with measured KL.
- **Test:** after 47, one GDN layer kernel microbench plus a whole‑model KL on a 2B‑style search slice.
- **Gain / risk:** KL +0.1–0.5% (*est.*), prefill −2–6% if GDN is ~15%.
- **Conflicts:** none.

**50. GDN chunk size (64 vs 128) and WMMA mapping for gfx1201.**
- **Mechanism:** Tune the chunk length and the intra‑chunk matmul shapes to the 16×16 WMMA. Check that the 128 × 128 head state maps to whole tiles.
- **Source:** new.
- **Why:** Kernel efficiency.
- **Fit:** kernel‑only; numerics change slightly with the chunk size.
- **Test:** GDN kernel bench.
- **Gain / risk:** KL ~0, prefill −1–3%.
- **Conflicts:** none.

**51. int8 QKᵀ for long‑context prefill attention (SageAttention‑style).**
- **Mechanism:** Smooth K by subtracting its per‑block mean (softmax‑invariant), quantize Q and K per block to int8, run QKᵀ on iu8 (2× f16), keep PV in f16.
- **Source:** 2609.33135 (MpFA: low‑bit QK, higher‑bit PV), 2609.39361.
- **Why:** At 128k context, attention FLOPs (~50 GFLOP per token over 16 layers) about equal the linears (~49), so real long‑prompt prefill is half attention.
- **Fit:** kernel; global switch, measured KL.
- **Test:** attention microbench at 32k–128k; KL on `golden-ver3-long`.
- **Gain / risk:** KL +0.1–0.5% (*est.*), long‑prompt prefill −15–25%.
- **Conflicts:** none.

**52. Prefix caching for multi‑turn agentic serving (exact replay from aligned GDN checkpoints).**
- **Mechanism:** Reuse KV and GDN checkpoints at 2048‑aligned boundaries of a shared prefix. Replay exactly from the last checkpoint to the divergence point (≤ 2047 tokens).
- **Source:** 2608.30310 (Tail‑Replay; we keep exact replay), 2608.30386.
- **Why:** Agentic traces up to 250k tokens resubmit the prefix every turn.
- **Fit:** serving.
- **Test:** turn‑2 time to first token on a long trace.
- **Gain / risk:** KL 0, turn TTFT −90%+.
- **Conflicts:** none.

**53. Decay‑aware compression of the prefix‑cache GDN checkpoints.**
- **Mechanism:** Store long‑retention heads in full, short‑retention heads compressed or dropped and refreshed from a bounded suffix.
- **Source:** 2608.30386 (2.63× smaller checkpoints, KDA).
- **Why:** Memory for many cached sessions.
- **Fit:** serving.
- **Test:** RULER‑style retrieval vs checkpoint memory.
- **Gain / risk:** KL serving only; mainly memory.
- **Conflicts:** none.

---

## F. Speculative decoding (DFlash2, MTP)

**54. Port DFlash2 first: our decode makes verification nearly free.**
- **Mechanism:**
  - The target's 16‑row decode band costs about the same as 1 row (VALU/overhead‑bound, F4; speedcal f16 core flat at M ≤ 256).
  - The block‑diffusion drafter proposes a block; verify up to 16 tokens in one step.
  - Taps at layers 5/19/33/47/61 per design §8.
- **Source:** design §8; IQR‑V3 ships it (published acceptance 4.1–5.5).
- **Why:** About (target step + draft) / accepted tokens ≈ (31 + 6) / 4.8 ≈ 7.7 ms per token → ~130 tok/s vs ~30 now (*est.*).
- **Fit:** serving; no board KL.
- **Test:** acceptance and tok/s on calib‑like prompts after the port.
- **Gain / risk:** KL 0, decode 3–4×. Risk: engineering size; GDN rollback (59).
- **Conflicts:** none.

**55. Quantize the drafter (3.6 GB bf16 → int8 or EXL3 ~4 bpw).**
- **Mechanism:** Drafts are verified, so drafter quality affects only acceptance. Its 3.6 GB read per draft pass is about 5.6 ms (≈ 18% of a verify step).
- **Source:** new (F6).
- **Why:** Cuts the draft cost to about 1.5–2 ms.
- **Fit:** serving; separate container, not counted in the board's bpw.
- **Test:** acceptance length at bf16 vs int8 vs EXL3 K4/K5 drafters; tok/s.
- **Gain / risk:** KL 0, decode +8–12% on top of 54. Risk: acceptance drop above ~5% erases the gain.
- **Conflicts:** none.

**56. Native MTP drafter through radiance's spechub.**
- **Mechanism:** The checkpoint has one MTP layer (`mtp_num_hidden_layers: 1`); turboderp's downloads include 8 MTP trellis tensors; radiance has an MTP coupler (`spechub`). Declare the MTP block in `qwen35_exl3` and use spechub.
- **Source:** config, #75 note, `plugin/spechub`.
- **Why:** A smaller step than the DFlash2 port, and a fallback.
- **Fit:** serving. Board byte counting for MTP is unknown (B10): ship it as a separate file or ask the author.
- **Test:** acceptance and tok/s.
- **Gain / risk:** KL 0, decode 1.5–2.2× (*est.*).
- **Conflicts:** none.

**57. Tree / multi‑candidate verification to fill the 16 rows.**
- **Mechanism:** If DFlash2 blocks are shorter than 16, use the spare rows for alternative tokens at the drafter's least confident positions.
- **Source:** generic tree speculation; new for our band.
- **Why:** The rows are free (37).
- **Fit:** serving.
- **Test:** acceptance with 1 vs 2–3 branches.
- **Gain / risk:** KL 0, +10–20% accepted length (*est.*).
- **Conflicts:** none.

**58. Drafter on card 1, verify on card 0 (pipelined).**
- **Mechanism:** While card 0 verifies block i, card 1 drafts block i+1 from the taps of block i−1, plus a correction once block i is known. Only taps (5 × 5120 per token) cross the x4 link.
- **Source:** new.
- **Why:** Takes the draft off the critical path.
- **Fit:** serving, two cards.
- **Test:** tok/s.
- **Gain / risk:** KL 0, +10–15% over 54+55.
- **Conflicts:** #9 (x4 link: small transfers only here).

**59. Rollback‑friendly GDN state for verification.**
- **Mechanism:** Verify computes outputs for 16 positions without materialising intermediate states (ReplaySSM), then advances the state by the accepted count only.
- **Source:** 2610.13778.
- **Why:** Without it, verify must store 16 × 151 MB of states or recompute.
- **Fit:** kernel/engine.
- **Test:** correctness against sequential decode, and tok/s.
- **Gain / risk:** KL 0, required for 54.
- **Conflicts:** none.

**60. Adaptive draft length.**
- **Mechanism:** Choose the block length per step from recent acceptance. Verify cost is flat up to 16, so err long.
- **Source:** standard.
- **Fit:** serving.
- **Test:** tok/s.
- **Gain / risk:** KL 0, +5%.
- **Conflicts:** none.

**61. Non‑idea, recorded: low‑bpw self‑drafting does not help while decode is VALU‑bound.**
- A 1.5 bpw copy of the model decodes the same 24.4e9 weights (F4: 32.5 vs 29.4 tok/s), so drafting costs about a full step.
- It only becomes useful if 1/33/34 make decode bandwidth‑bound. Re‑check then.

**62. Layer‑skip self‑draft as a no‑extra‑weights fallback.**
- **Mechanism:** Draft with every second trunk layer of the shipped model plus the head. Draft cost is ~½ step; acceptance is probably 50–70% for the first token.
- **Source:** LayerSkip/Draft&Verify‑style; new for us.
- **Why:** Works without DFlash2 or MTP files.
- **Fit:** serving.
- **Test:** acceptance on calib prompts with the existing engine.
- **Gain / risk:** KL 0, decode 1.2–1.5× (*est.*).
- **Conflicts:** none.

---

## G. Eval‑runner throughput (more evals per GPU‑hour)

Baseline: ~80 s fixed + ~0.4 s per 1k positions. Card 1 takes 152 s vs card 0's 119 because of the full‑vocabulary f32 logits copy over its PCIe3 x4 link.

**63. Fused head logsumexp plus top‑512 KL on the GPU.**
- **Mechanism:**
  - In the head GEMM epilogue, compute per‑(row, column‑tile) max and Σexp. Combine them in a second tiny kernel.
  - Gather the quant logits at the golden's 512 ids (golden resident in VRAM: 512 × 6 B per position, ~0.9 GB for 300k).
  - Return one KL per position. Logits (2 GB per 2k chunk) are never written or downloaded.
- **Source:** new (dead ends § Eval speed names the logits copy as the card 1 problem).
- **Why:** Card 1: 152 → ~119 s (−22%). Card 0: −2–5%. It also saves 2 GB per chunk of DRAM traffic.
- **Fit:** eval engine. The KL is identical if computed in f32 with the same formula (the check is equality to 1e‑7 on 20 candidates).
- **Test:** one slice on card 1, old vs new scores and wall time.
- **Gain / risk:** KL 0; about +12% evals per hour across both cards (*est.*).
- **Conflicts:** dead ends § card 1 (that fix was never done).

**64. Compute the head only for scored rows.**
- **Mechanism:** Search slices score only generated positions (gen mask). Skip head rows for prompt positions; the trunk still runs every row.
- **Source:** new.
- **Why:** The f16 head is about 10% of eval FLOPs now (T2 head 48 ms per 2k chunk) and about 20% after the int8 trunk. If ~40% of positions are unscored, that saves 4–8% of eval time.
- **Fit:** eval engine; scored values are identical (rows are independent).
- **Test:** gen fraction per slice (CPU, seconds), then wall time.
- **Gain / risk:** KL 0, +4–8% evals per hour.
- **Conflicts:** none.

**65. Prefix‑boundary residual cache (skip unchanged early layers).**
- **Mechanism:**
  - For the stream's current base mix, store the residual stream entering layers 16/32/48 for all slice positions (300k × 5120 × bf16 = 3.1 GB each; VRAM or host RAM).
  - A candidate whose first changed tensor is in layer ≥ L starts from the deepest stored boundary ≤ L. GDN and KV state below L are not needed. Results are bit‑identical, because layers below L have identical tensors and row‑independent kernels.
- **Source:** new.
- **Why:** Local‑search and targeted‑swap candidates change a few tensors. A single uniformly placed change skips about 37% of trunk compute on average.
- **Fit:** eval engine, together with the planned resident base.
- **Test:** emulate on V2's stream log: fraction of trunk skippable per candidate (CPU, minutes). Then implement.
- **Gain / risk:** KL 0, +20–40% evals per hour for local search (*est.*). Risk: invalidate whenever the base changes.
- **Conflicts:** design §5 resident‑base proposal: complementary.

**66. Layer‑major evaluation (decode each weight once per eval, very large M).**
- **Mechanism:** Run layer ℓ for every document of the slice before layer ℓ+1. Pack rows across documents up to M ≈ 8–16k, keep per‑document GDN state and KV within the layer, and carry one residual buffer (3.1 GB).
  - Weight decode drops from ~150× per eval (one per 2k chunk) to 1×.
  - GEMMs run at their best M; launches drop by about 150×.
  - Bit‑identical: per‑row GEMM results don't depend on M or MT (T2 and MT 1–8 measured byte‑identical).
- **Source:** new.
- **Why:** Removes the per‑chunk decode pre‑pass (~10%) and per‑chunk overhead. Together with 65 it gives the cheapest late‑layer candidates.
- **Fit:** eval engine (a new runner mode).
- **Test:** prototype on 8 layers; byte‑compare per‑row KL with the standard runner on 20 documents.
- **Gain / risk:** KL 0, +15–30% evals per hour (*est.*). Risk: engine work; attention and GDN need per‑document segmentation in a packed batch.
- **Conflicts:** #60‑02 (shape‑dependent rounding was cuBLAS, not our row‑independent kernels; still verify).

**67. Pipelined candidate loading (double‑buffer the fixed 80 s).**
- **Mechanism:** Load and relayout candidate n+1 into a second VRAM slot while candidate n is scored. Two 8–14 GB containers fit in 32 GB at ≤ 4 bpw.
- **Source:** new; a lighter cousin of the resident base.
- **Why:** The fixed cost is about half of a 120 s eval.
- **Fit:** eval engine.
- **Test:** step 1 of the resident‑base plan (split the 80 s), then measure back‑to‑back evals.
- **Gain / risk:** KL 0, up to +40–60% evals per hour if load (not graph build) dominates.
- **Conflicts:** design §5 resident base: try this first; it is simpler.

**68. Device‑ready tensor blobs in the library.**
- **Mechanism:** Store each (tensor, option) already in device layout (post‑relayout). A mix load then becomes mmap + DMA, with no relayout.
- **Source:** new.
- **Why:** Cuts the fixed cost's relayout share.
- **Fit:** eval engine and library.
- **Test:** time split from the resident‑base step 1.
- **Gain / risk:** KL 0, fixed cost −10–40 s (*unknown split*).
- **Conflicts:** design §5.

**69. Sequential racing: stop scoring clear losers early.**
- **Mechanism:**
  - Score documents in a fixed random order.
  - After each block of documents, run a paired test against the incumbent. Stop when the candidate is worse by z > 3, or better by z > 3 (then promote it to the ver3 check).
  - Fix the rule before the stream starts.
- **Source:** standard sequential testing; Loopy (2610.05265) uses successive halving.
- **Why:** Most local‑search candidates lose. Stopping losers at ~40% of documents saves about half their cost.
- **Fit:** search pipeline.
- **Test:** replay V2's per‑document scores (CPU): evaluations saved and wrong calls at z = 3.
- **Gain / risk:** KL 0 directly. +40–80% candidates per GPU‑hour (*est.*); worth more at ≤ 2.5 bpw, where V2's search had not flattened. Risk: weaker evidence for marginal accepts.
- **Conflicts:** #14 (single swaps too noisy); this is a stopping rule, not a model.

**70. Pack short documents into full steps.**
- **Mechanism:** Search documents are capped at 2k. Several documents shorter than ~512 tokens can share one prefill step, with GDN state reset and a block‑diagonal attention mask per document.
- **Source:** new.
- **Why:** GEMMs below ~384 rows lose efficiency (f16 down: 0.583 → 0.926 ms jump at 384).
- **Fit:** eval engine. Use it for every candidate or none (paired).
- **Test:** document length histogram (CPU) to size the gain.
- **Gain / risk:** KL 0 (consistent across candidates), +0–10%.
- **Conflicts:** #60‑02.

**71. Prefix‑sharing candidate batches.**
- **Mechanism:** Have the local search emit batches of candidates that share layers 0..L−1 (late‑layer mutations). Evaluate the shared prefix once and fork at L inside one run (two residual buffers).
- **Source:** new; pairs with 65 and 66.
- **Why:** Multiplies 65's gain.
- **Fit:** search and eval engine.
- **Test:** simulation on V2's local‑search logs.
- **Gain / risk:** KL 0, +20–50% for local search (*est.*).
- **Conflicts:** none.

**72. Two eval processes per card at low bpw.**
- **Mechanism:** If evals keep GEMMs at ~40% of peak (dead ends § Eval speed), a second process can fill the gaps (2.5 bpw ≈ 8.4 GB each). Quantizers gained +14% with two processes per card (#1).
- **Source:** #1 analogue.
- **Fit:** scheduler.
- **Test:** throughput with 1 vs 2 processes.
- **Gain / risk:** KL 0, +5–15%; risk: memory fragmentation.
- **Conflicts:** #1 (quantizer, +14% only).

**73. int8 eval mode equals serving mode, checked byte‑for‑byte.**
- **Mechanism:** Evals already run 25% faster on int8 (E4). Lock that the eval binary's kernel census equals the serving census per mix (design §1c.4 census), and record container sha256s.
- **Source:** design.
- **Why:** Prevents silent fallback to f16 (slower, unnoticed).
- **Fit:** process.
- **Test:** a census diff per eval.
- **Gain / risk:** KL 0; protects the gains of 1–31.
- **Conflicts:** none.

**74. Keep the golden resident in compact form.**
- **Mechanism:** Store golden top‑512 as fp16 log‑probabilities plus u32 ids on the GPU across evals of the same slice, uploaded once per slice rather than per eval.
- **Source:** new; prerequisite of 63.
- **Fit:** eval engine.
- **Test:** wall time.
- **Gain / risk:** KL ~0 (fp16 logprobs: check equality of KL to 1e‑6), +1–3%.
- **Conflicts:** none.

---

## H. Measurement and methodology

**75. Real‑prefill time table: core + wdec + prep + activation quant + fold + epilogue.**
- **Mechanism:** `speedcal-stock.jsonl` holds matmul cores with pre‑decoded weights (e.g., iu8 gate_up 2.81 ms at M = 2048). The solver needs whole‑op times per (shape, K, mode, group).
- **Source:** design §4.
- **Why:** F5 puts the non‑core overhead at ~35%. Without whole‑op times, knap4's speed choices are biased towards modes with expensive overheads.
- **Fit:** solver input.
- **Test:** ntbench of whole ops; a few card‑hours.
- **Gain / risk:** KL indirect (better speed/quality trade‑off per size).
- **Conflicts:** design §6 step 3.

**76. Static ISA op count per rate and codebook.**
- **Mechanism:** Compile every (rate, cb, band) instance with `--save-temps`. Count VALU ops per 256 weights in the decode loop, and dual‑issue pairs.
- **Source:** new; design A1's "count v_wmma in the ISA" rule.
- **Why:** A cheap, exact decode‑cost table: finds F2‑type regressions without a card.
- **Fit:** CI.
- **Test:** needs ROCm on a CPU box; under an hour.
- **Gain / risk:** none directly; diagnostic.
- **Conflicts:** none.

**77. Board‑faithful timing at TP2‑rank shapes, with clock and power state controlled.**
- **Mechanism:** The board's ~8,030 µs per layer for EXL3 is consistent with per‑rank (half‑N) timing. Time both splits, at stock and 210 W, with warmup and pinned clocks.
- **Source:** design §6 checks.
- **Fit:** measurement.
- **Test:** existing bench.
- **Gain / risk:** none directly; prevents misreading results.
- **Conflicts:** design §6.

**78. Three‑term decode cost model per K and pattern (for release notes and the final check).**
- **Mechanism:** t_token = a·(trellis weights) + b·bytes + c (+ KV term at long context). Fit it to tok/s at 4–6 sizes.
- **Source:** 2610.15629 (binding‑resource cost model).
- **Why:** Explains the flat‑in‑K decode and predicts the effect of 1, 32 and 33.
- **Fit:** analysis.
- **Test:** tok/s at 1.4/2/2.5/3/4/6 bpw (existing containers), 1 h.
- **Gain / risk:** diagnostic.
- **Conflicts:** none.

**79. Per‑build timing regression check.**
- **Mechanism:** Run a fixed set of ~20 ntbench cases (bands × rates × modes) after every kernel pass, next to the byte‑exact checks. Fail on a > 3% slowdown.
- **Source:** design §1c.4 lessons (a stale `.so` was benchmarked as new).
- **Fit:** CI.
- **Test:** 10 min per build.
- **Gain / risk:** protects the other ideas.
- **Conflicts:** none.

---

## I. Further ideas (smaller or speculative)

**80. Fold svh, 1/√128 and 4K_INV into one per‑row store constant.**
- One multiply per output at the store; check the current epilogue does this. KL 0, ≤ 1%.

**81. bf16 store of linear outputs where the consumer is a norm.**
- The design check (§1c.9) compares f16, bf16 and f32 store rounding. Measure the time side too. KL from the planned E1b hook test.

**82. Overlap the input‑rotation prep of the next GEMM with the epilogue of the previous one at decode.**
- Small kernels on two queues. Decode +1–3% (*speculative*).

**83. Persistent decode kernel per layer (megakernel‑lite).**
- One launch per layer at M = 1, with a work queue over GEMVs and small ops. The upper bound of idea 33.

**84. Precompute the block sums Σx per (token, 128) in the prep for any unsigned or zero‑point use (3, 14).**
- Free, enables 3 and 14.

**85. MALL‑aware ordering of decode‑time GEMVs.**
- Run gate_up then down of the same layer back to back (likely already), and keep the residual hot. Cheap check with profile‑ops.

**86. Head at M = 2048 in evals: T2 chunk and tile retune for N = 248,320.**
- Already 65.9 → 48.1 ms with T2; with 63/64 the head matters less, but a dedicated tile could give −10–20% of the head.

**87. Pin the per‑token activation‑scale memory layout to the fragment order.**
- Avoids a gather in the fold. Microbench.

**88. Unify the f16 path for K7/K8 with the int8 prep (A16 via two int8 activation passes).**
- Rejected: 2 × iu8 ≈ 128 TOPS ≈ f16 core (116). No gain; recorded so it is not retried.

**89. 2:4 activation sparsity for down's input via SWMMAC.**
- Rejected: h is not structurally sparse, and the KL cost would be large.

**90. FP8 activations instead of int8.**
- Rejected: the mul1i8 weights (−127..127 integers) are not exactly representable in E4M3, and gfx1201 has no fp8 × i8 WMMA.

**91. On‑GPU sampling and stop checks at decode (no host round trip per token).**
- Check whether radiance already does this; if not, it is part of 33. Decode +2–10% (*unknown*).

**92. Eval scheduler: longest documents first across the two cards.**
- Balances the tail when both cards score one slice. +1–3%.

**93. Quantize the drafter's taps (5 × 5120 per token) to bf16 or fp8 for the cross‑card pipeline (58).**
- Small; only with 58.

**94. Calibrate the activation‑cost model on a low‑K base with the real int8 kernels (not the fake‑quant hook).**
- E1b/E1c used a K6 base and a hook at T > 16. Re‑measure 3–4 modes with the real kernels at K2.5 so the solver's activation term is right where it matters (BACKLOG item, with real kernels).

**95. Decode at the narrow tile for M = 4–16 with integer decode and MT 1 (the verify shape).**
- Make sure the verify band uses ideas 1 and 32. It is the shape DFlash2 runs most.

---

## Ranked top 20 by expected gain per unit cost

| rank | # | idea | KL | expected speed effect | cost |
|---|---|---|---|---|---|
| 1 | 1 | Integer‑bytesum fp16 decode for mul1i8 | 0 (bit‑exact) | decode +15–40% vs current exl3i (avoids a likely V3 decode regression) | hours |
| 2 | 63 | GPU‑side fused logsumexp + top‑512 KL | 0 | card 1 evals −22%, card 0 −2–5% | ~1 day |
| 3 | 32 | Split‑K at decode for N = 5120 shapes | 0 (board) | decode +5–10% | hours |
| 4 | 64 | Head only on scored rows (evals) | 0 | evals +4–8% (more after int8) | hours |
| 5 | 2 | Integer int8 decode in wdec / fused tiles | 0 | enables 24; wdec VALU ×0.5 | hours |
| 6 | 67 | Pipelined candidate loading | 0 | evals up to +40–60% if load dominates | 1–2 days |
| 7 | 69 | Sequential racing of candidates | 0 | +40–80% candidates per GPU‑hour | 1 day + CPU replay |
| 8 | 22 | T2 chunk sized to the 64 MB Infinity Cache | 0 | prefill −2–5% | 30 min |
| 9 | 9 | Fused rotation + int8 activation quant prep | 0 | prefill −4–6% | 1–2 days |
| 10 | 65 | Prefix‑boundary residual cache | 0 | local‑search evals +20–40% | 2–3 days |
| 11 | 10 | Down's int8 input from the gate_up epilogue | 0 | prefill −3–5% | 2–3 days |
| 12 | 14/15 | Cheaper g128 fold (magic bias or packed f32) | ≤ +0.05% / 0 | A8‑g128 GEMM +5–10% | 1 day each |
| 13 | 21 | Overlap wdec with GEMM | 0 | prefill −4–8% | 2 days |
| 14 | 55 | Quantize the DFlash2 drafter | 0 | decode +8–12% on top of DFlash2 | 1 day after 54 |
| 15 | 54 | DFlash2 port (verify ≈ free on our band) | 0 | decode 3–4× | 1–2 weeks |
| 16 | 16/17 | Static exponent groups / outlier‑block scale | ≈ g128 | A8 GEMM +10–15% vs g128 | CPU test first, then 2 days |
| 17 | 44 | Per‑side rate in stacked gate_up | −0.3–1% | 0 | knap4 check, then 2 days |
| 18 | 48 | GDN conv + SiLU + norm in the qkv epilogue | 0 | prefill −2–4% | 2 days |
| 19 | 56 | MTP drafter via spechub | 0 | decode 1.5–2.2× | 3–5 days |
| 20 | 66 | Layer‑major eval runner | 0 | evals +15–30% | 1–2 weeks |

Notes on the ranking:
- Ideas 1, 32, 54 and 55 compound for serving decode.
- Ideas 63, 64, 67, 69 and 65 compound for search throughput. That is the only route by which this lens moves KL: more evaluations per GPU‑hour where V2's search had not yet flattened (≤ 2.5 bpw).

## Five cheapest diagnostics that would most change the ranking

1. **Decode profile (`--profile-ops`, M = 1) on V2 1.40 and 6.00 bpw.** Σ(op time) vs wall time per token, plus per‑op split (GEMV / GDN / attention / head / small ops). This decides whether the per‑token fixed cost is launches and overhead (→ 33, 35, 83 rise) or trellis VALU (→ 1, 5, 6 rise), and how much 32 buys. About 1 card‑hour.
2. **ntbench decode band, mul1 vs current mul1i8 (exl3i) vs idea 1**, at gate_up/down K2.5/K3, M = 1 and 16. This confirms or refutes the suspected V3 decode regression (F2) and sizes idea 1. About 1 card‑hour after a few hours of coding.
3. **wmmabench: interleave the hash + sad sequence between iu8 WMMAs.** Shows whether decode and fold VALU hide under WMMA on gfx1201. If they do, fused decode (23, 24) and the fold ideas (14–16) lose priority; if not, pre‑pass overlap and MALL sizing (21, 22) gain. About 30 min.
4. **Prefill profile at M = 2048 on the int8 path** (once #31 is fixed; otherwise f16 T2): per‑layer split of core / wdec / prep / activation quant / fold / epilogue / GDN / attention. This replaces my F5 overhead estimate with numbers and ranks ideas 9–31 and 47–51. About 1 card‑hour.
5. **CPU, on the E1c activation dumps:** A8 error for per‑token + static exponent groups (16) and for per‑token + outlier‑block scale (17), against g128 and plain per‑token, per family and depth band. If static groups match g128, the fold problem largely goes away, and the A8 GEMM gains 10–15% at no KL cost. A few CPU‑hours on CPU box A.
