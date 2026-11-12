The report is below. It is not kept anywhere except this reply, and I wrote nothing in the repo.

# Kernels lens: research ideas for EXL3-V3 speed (prefill A8, decode, evals, quantizer throughput)

**Scope note.** The shared brief says kernel work is tracked elsewhere, but this lens asked for it, so I kept it. I flag three kinds of item:
- **[pre-freeze]**: changes what the ladder quantizes. These are the format and quantizer choices that decide what the kernels can fuse.
- **[KL-indirect]**: more evals or cheaper levels for the same card-hours.
- **[board]**: moves `gemm_us`.

Everything else is runtime-only and can ship after the freeze.

## 0. Grounding: what the measured numbers say

**Measured** (bench_twopass on gfx1201 card 0 at 250 W, K4 mul1i8, M 2048; one layer, weighted 48 GDN : 16 attention; from `TWOPASS.md`, `A4.md` and `kernel-opts.md`):

| item | µs | share | note |
|---|---|---|---|
| f16 path (T2) | 14,474 | — | board V2 ≈ 8,050; the board is probably TP2-rank shapes at 210 W (TODO "pin protocol"), so roughly ×0.55 |
| A8 full path | 9,444 | 100% | ≈ 5,250 on the board scale vs IQR-V4's 3,829–4,043 |
| actq (rotate + quantize) | 768 | 8% | gate_up 172 µs for ~32 MB of traffic, about 185 GB/s, far from the roofline |
| wdec8 (separate decode pass) | 1,064 | 11% | gate_up 541 µs: 178 MB plane written + 89 MB read, bandwidth-bound at ~495 GB/s |
| A8 GEMM tile | 8,124 | 82–86% | gate_up 185 TOPS = 72% of E2's 255 TOPS for a plain iu8 kernel; the tile-only run is 7,005 µs with VOPD and 7,253 without |
| A4 no-fold tile, A4 g128, A4 g64 | 3,765 / 4,520 / 5,289 | — | A4.md: "the fold's VALU work does not overlap the WMMAs" |
| GEMV, M 1 / 4 / 16 | 613 / 658 / 867 | — | M 1 runs at ~48% of DRAM bandwidth; M 16 costs only 1.41× M 1 |
| A8 vs f16 time by M | 1.86 / 1.03 / 0.65 / 0.62 | — | at M 16 / 128 / 256 / 512; the crossover is about 128–256 rows |

What follows from the table:
1. **The gap to IQR is mostly outside the tile.** Our A8 tile runs about as fast as the FP8 tile on the board scale. IQR pays nothing for a separate actq pass (it is fused into the norm) or for a plane round trip (decode happens in the tile hook). Removing most of actq + wdec8 (1,832 µs) and switching VOPD back on would put us level with IQR.
2. **On RDNA4, VALU work inside the K loop adds 1:1 to GEMM time.** This cuts both ways:
   - Cheaper or sparser folds pay off fully.
   - Decode moved into the tile is not hidden. It only stops costing DRAM traffic. That still pays: decoding gate_up's 178M values is about 100 µs of VALU against the 541 µs the pass spends on bandwidth.
3. **Decode at M = 1 is VALU- and latency-bound, not byte-bound.** Evidence: 48% of DRAM bandwidth; tok/s is "nearly flat in K below 4 bpw" (32.5 tok/s at 1.40 bpw vs 24 at 6.00); IQR reaches 24.7 ms/token at a larger size. Smaller files do not decode faster until the per-weight decode cost falls. For the same reason, speculative verify tokens are nearly free (M 16 = 1.41× M 1).
4. **Card 0 reports a max sclk of 2,350 MHz; card 1 reports 3,139 (LOG 11-05).** The same tile also varies 7–12% from run to run (7,005 / 7,871 / 8,124 µs). Both hit every VALU-bound measurement and every 3–5% A/B call.

**New measurement (colo, this run)**, script `/data/work/ideas-kernels/hotblk.py`. A8 activation-quantization output error relative to g128, on real 27B E1c dumps; 54 tensors; hot blocks chosen on rows 0–255 and scored on rows 256–511; medians:

| family | per-token | 1 hot block | 4 hot blocks | 8 hot blocks | g256 |
|---|---|---|---|---|---|
| gate_up | 2.61 | 1.98 | 1.79 | 1.66 | 1.21 |
| gdn_in | 5.91 | 2.07 | 1.88 | 1.74 | 1.28 |
| attn_in | 5.20 | 2.37 | 1.99 | 1.78 | 1.31 |
| down | 5.72 | 5.67 | 5.54 | 5.35 | 1.29 |
| ssm_out | 6.01 | 5.87 | 5.23 | 4.51 | 1.44 |
| attn_out | 6.29 | 6.07 | 5.31 | 4.47 | 1.21 |

Applying the E1c family factors to all trunk tensors (K6 calibration) gives the predicted extra KLD from A8:
- g128: 7.2e-5
- g256: 9.1e-5
- g512: 1.17e-4
- g1024: 1.53e-4
- per-token: 3.49e-4
- per-token on gate_up only (gate_up holds 40% of the g128 total, ratio 2.69): about +4.9e-5

At 2.5 bpw (KLD ~0.05–0.066) that is **g256 ≈ +0.03% and per-token gate_up ≈ +0.1%**. Caveat: E1c is calibrated on a K6 base, and the low-K E1b check is still open in BACKLOG.

Fit labels below: **runtime** = kernel or engine only, no format change; **format** = a choice the ladder bakes in; **tooling** = eval or quantizer infrastructure.

---

## A. Measurement hygiene and protocol (diagnostics that set every other priority)

**A1. Pin the board's `gemm_us` accounting, especially where actq is charged.** Reproduce IQ4R-V3, FP8 and V1-4.00 on rad_kbench (TP2-rank shapes, 210 W, M 2048, cold weights) to within ~2% of the board (TODO item). Then check whether a fused norm+quant op (IQR's `rmsnorm_had_quant_i8`) counts inside `gemm_us` while our plugin's internal actq does.
- Source: new; radiance `rad_kbench` times each op with weights read cold.
- Why: decides whether ~8% of our board number (actq) is an accounting artifact that idea D10 removes. It also decides whether TP2-rank shapes change the tile's efficiency.
- Fit: tooling. Test: gpubox card 0, ~1–2 card-h. Gain: steers ~8% of `gemm_us`; KL 0. Risk: none.
- Conflicts: TODO "pin protocol"; #53 (bytes definition).

**A2. Fix or explain card 0's clock ceiling (2,350 vs 3,139 MHz).** Read `pp_dpm_sclk`, `pp_od_clk_voltage` and the power profile on both cards, then re-run one GEMV and one wdec8 case.
- Source: LOG 11-05 aside.
- Why: VALU-bound kernels scale with sclk: decode GEMV, wdec8, actq, the fold, and the quantizer's Viterbi (80% of a level's time). Up to +34% on card 0 if the ceiling is a misconfiguration.
- Fit: tooling. Test: 5 minutes plus one bench. Gain: 0–34% on card-0 VALU kernels, the ladder and evals. Risk: the "fix" may have been a deliberate thermal cap.
- Conflicts: #8 (card 1 is still slower because of its x4 link; the two effects are separate).

**A3. Deterministic-clock A/B protocol.** Fix sclk/mclk with determinism mode, interleave A and B builds, use 20+ repeats and report the median plus the IQR.
- Source: new.
- Why: the same tile measured 7,005 / 7,871 / 8,124 µs in different runs. Without this, the 3–5% items below (B1–B4, C1, D4) cannot be judged.
- Fit: tooling. Test: <1 card-h. Gain: enables decisions. Risk: a fixed clock under-reports peak; use it for A/B only.
- Conflicts: none.

**A4. Ablated-decode floor build at K2/3/4/5** (w29's `abl` build was never measured because of card contention). Gives the decode share of GEMV, narrow tile and wdec8 per rate.
- Source: w29.
- Why: H1–H5 are only worth doing if decode VALU is a large share at M 1 (my estimate is ~40–55%).
- Fit: tooling. Test: `/data/tp/w29/gbench.sh` on an idle card, ~3 minutes. Gain: ranks all decode ideas. Risk: none.
- Conflicts: #100 (same builds).

**A5. Instruction census of the A8 tile's K loop.** From disassembly: WMMA, fold FMAs, address arithmetic and LDS operations per 128-K block per wave, with and without RB 32 spill and VOPD.
- Source: new.
- Why: because VALU adds to tile time, this census is the ceiling for B1–B4 and B9. Today the fold share is only inferred from the A4 data (a model gives ~750 µs, ~11% of the tile).
- Fit: tooling. Test: CPU only, 1 hour. Gain: sizes B1–B4. Risk: none.
- Conflicts: none.

**A6. Achieved bandwidth and occupancy per pass.** rocprofv3 on `bench_twopass` standalone (#7 found rocprofv3 unusable inside radiance, not on standalone benches): actq, wdec8, epilogue.
- Why: actq reaches about 185 GB/s against ~600 available; this says whether D7/D8 or fusion is the cheaper route.
- Fit: tooling. Test: 30 minutes of card time. Risk: none. Conflicts: #7.

**A7. TP2-rank-shape sweep of the A8 path.** gate_up N 17408, down K 8704, attn_o/ssm_out K 3072, at 210 W and stock power.
- Why: the board likely times these shapes. Fold cost and tail effects change with K and N. VALU-heavy kernels throttle more under a power cap, so VALU-cutting ideas may gain more on the board than in our 250 W benches.
- Fit: tooling. Test: <1 card-h. Risk: none. Conflicts: design §6 "time gate_up/down at full and half N".

---

## B. The A8 prefill tile

**B1. Switch VOPD dual-issue back on and fix the reference instead.** The plugin builds with VOPD off because test_a4's g128 half-step terms differed. Model the fused rounding in the host emulation, or turn VOPD off per function only in the A4 fold.
- Source: A4.md (A8 tile 7,005 µs with VOPD vs 7,253 without).
- Why: −3.4% on the tile for free, plus whatever VOPD buys the integer decode kernels (H4).
- Fit: runtime. Test: rebuild and run test_twopass/test_a4 byte checks plus the bench, 2 hours. Gain: −3% layer; KL 0. Risk: emulation work if the fused op really changes bytes.
- Conflicts: none.

**B2. Switch to radiance's RB 8 fold form** (193 VGPR, no spill) from RB 32 (240 VGPR + 20 B/lane spill). It is equivalent with weight scale 1.
- Source: radiance tile notes (25–38% slowdowns measured when this tile spills inside the K loop); TWOPASS open question 3 (RB 8 never tried).
- Fit: runtime. Test: 30 minutes. Gain: 0–8% tile, 0 if the spill sits outside the loop; KL 0. Risk: lower ILP in the fold.
- Conflicts: P5 in kernel-opts.

**B3. A real g256 fold** (fold every second 128-K block). The biased-int window holds exactly 256 inputs (127·127·256 < 2^22), so today's g256 just folds a repeated scale every block at g128 speed.
- Source: TWOPASS open question 2; E1c; my run (g256 activation error is 1.21–1.44× g128 per family).
- Why: halves fold VALU. Predicted KL +1.9e-5 for the whole trunk (+0.03% at 2.5 bpw). Estimated tile −4–6% from the fold model.
- Fit: runtime (per-tensor mode option in knap4). Test: bench a8g256 vs a8g128 on the 48:16 layer plus byte tests, 1 day of dev. Gain: −3–5% layer; KL +0.03%. Risk: the gain is smaller if the fold is not 11% of the tile (A5 decides).
- Conflicts: #26-01 (A8 group-size prices are already measured).

**B4. Per-token A8 on gate_up only (a8t), offered as a knap4 option.** Pure int32 WMMA accumulation with the scale applied at the store; no in-loop fold.
- Source: E2 (per-token iu8 254 TOPS against fold variants 12–17% slower in that kernel); my run (gate_up has the lowest per-token penalty, 2.61×; the other families are 5–6×).
- Why: gate_up is ~45% of trunk FLOPs. Estimated −10–15% on gate_up's tile, ≈ −4–5% layer, for about +4.9e-5 KLD (+0.1% at 2.5 bpw, +0.6% at 4.0, ~+5% at 6.0). So it is a low-bpw-only option.
- Fit: runtime, per-tensor mode. Test: kernel variant plus bench; the KL side is already priced by E1c. Gain: −4% layer at ≤3 bpw. Risk: activation error may grow on low-K bases (E1b low-K check open).
- Conflicts: #26-01 (per-row +0.00047 was for all tensors at once; here it is gate_up only).

**B5. Hot-block K split (outlier 128-blocks keep their own scale, the rest share a per-token scale). Measured here: mostly negative.**
- Writers do not improve: down 5.67→5.35, attn_out 6.07→4.47 with 8 hot blocks.
- Norm-fed inputs drop only to ~2× g128.
- One exception: L3 attn_in (a massive-activation layer) goes from 28× to 1.25× with 4 hot blocks.
- Fit: runtime. Gain: none in general; at most a per-tensor option for L3/L11-type tensors. Record as tested.
- Conflicts: Appendix A's Atom-style idea.

**B6. Weight-stationary rasterization for the 64 MB Infinity Cache.** Schedule all 16 M-tiles of an N-slab back to back, so each weight fragment is fetched from DRAM once and re-read from L2/MALL. Today the plane is read 16× (2.85 GB for gate_up in 3.9 ms ≈ 730 GB/s, so LLC already absorbs some).
- Source: radiance raster grouping.
- Why: relieves memory pressure on the tile and is a prerequisite for C1.
- Fit: runtime. Test: counters (A6) plus a swizzle sweep, 0.5 day. Gain: 0–5% tile. Risk: the default grouping may already do this.
- Conflicts: none.

**B7. 256-row M tile** (two 128-row halves sharing weight fragments in registers): halves weight-plane traffic per FLOP.
- Fit: runtime. Test: needs fewer VGPRs (B2/B9). Gain: 0–5%. Risk: register pressure makes it worse; occupancy.
- Conflicts: none.

**B8. Direct global→LDS loads for the activation stage** (gfx12 buffer loads to LDS). Cuts VGPR staging, which may remove the spill (B2) and raise occupancy.
- Fit: runtime. Test: variant plus bench. Gain: 0–5%. Risk: ISA support and compiler maturity on clang 24.
- Conflicts: none.

**B9. Fewer address and branch VALU ops in the K loop** (SGPR-indexed buffer loads, hoisted offsets, unroll tuning). Under VALU-additivity every op counts.
- Fit: runtime. Test: guided by A5. Gain: 1–4%. Risk: low.
- Conflicts: none.

**B10. Epilogue output-Hadamard on WMMA** (a ±1 128×128 matrix, about 2.5% of the K loop's FLOPs) instead of LDS butterflies with barriers. Only if A6 shows the epilogue above ~5%.
- Fit: runtime (byte tests need a new reference: one f32 rounding instead of staged f16). Gain: 0–3%. Risk: changes output bytes.
- Conflicts: none.

**B11. Track radiance tile upgrades on every rebase.** 1.17 notes the i8a8 tile is "8–9% behind MXFP4's 4-bit tile"; any upstream fix is free for us because we carry a copy.
- Fit: runtime. Test: diff `r4d_gemm_8a8_prefill.h` per release. Gain: 0–9%. Risk: re-verification each time.
- Conflicts: #74 (ports).

**B12. Pin the release compiler and flags per kernel.** A4 measured ×1.25–1.38 vs ×1.40–1.50 depending on compiler and flags; LLVM 21 and ROCm clang 24 spill identically, but other choices differ.
- Fit: runtime. Test: build matrix plus bench, 2 hours. Gain: 0–5%. Risk: maintenance.
- Conflicts: none.

**B13. Accurate variant on the iu8 tile with hi/lo int8 activations** (MSD-style: x ≈ s·(q_hi + q_lo/128), two WMMA sets sharing each weight fragment). About 15-bit activations, so KL ≈ f16. It reads the same int8 plane as the fast variant (half the bytes of T2's f16 plane) and needs one decode path.
- Source: 2605.13915 (decomposes the activation, not the weight).
- Why: my model gives a tile ~7% faster than T2's f16 tile, and the plane write halves. Modest.
- Fit: runtime. Test: emulate first on CPU for KL (should be ≈ f16), then a kernel. Gain: −5–10% accurate-variant `gemm_us`; KL ≈ 0. Risk: the second fold may eat the gain.
- Conflicts: design §1 (f16 plane for the accurate variant).

**B14. Weight-side hi/lo for mul1 K6.5–8 on the A8 path (W16A8 as two iu8 passes). Estimated negative.** About 2× the A8 tile, i.e. ~1.3× the f16 path. The fast 7.0 should mix f16 and A8 per tensor instead (as the user asked).
- Recorded so nobody builds it. Conflicts: #94, #109.

---

## C. The decode pass and the int8 plane

**C1. LLC-chunked two-pass.** Decode an N-slab of ≤ 48 MB of the plane, run the GEMM on it, move to the next slab. Plane writes and re-reads then stay in the 64 MB Infinity Cache (if writes allocate there; C3 checks). DRAM traffic drops to the compressed trellis read only: 89 MB for gate_up at K4 instead of 267.
- Source: TODO "int8 T2 decode chunk sized to the 64 MiB LLC"; a coarse-grained cousin of IQR's tile-hook decode.
- Why: wdec8 is 11% of the layer and bandwidth-bound. Decode VALU alone is ~100 µs for gate_up. Estimated wdec8 1,064 → 250–400 µs, ≈ −7–8% layer at M 2048 and −20–30% at M 512.
- It also avoids multiplying GEMM instances by 64 rates × codebooks, which P1 would do.
- Fit: runtime. Test: slab loop in bench_twopass, ~1 day. Gain: −7% layer; KL 0. Risk: MALL may not keep the writes (then do C2).
- Conflicts: P1 (an alternative to it).

**C2. P1: decode inside the GEMM tile (IQR's claim counters).** On RDNA4 the decode VALU adds to the K loop rather than hiding, but the plane's DRAM write and re-read disappear.
- Source: `r4d_gemm_iqr.hip:670-768`.
- Fit: runtime. Test: as in kernel-opts. Gain: −6–8% at M 2048, −25–30% at M 512. Risk: one instance per rate × codebook; complexity.
- Conflicts: C1 (do C1 first).

**C3. Check that MALL allocates the plane's writes.** Probe whether stores use cache policy bits (GLC/SLC/NT) that bypass MALL; microbench write-then-read of a 32 MB buffer.
- Fit: tooling. Test: 1 hour. Gain: decides C1 vs C2. Risk: none.
- Conflicts: none.

**C4. Overlap slab decode with slab GEMM on disjoint CUs.** Run the decode of slab i+1 on a few workgroups (CU-masked queue, or a persistent kernel with producer/consumer roles) while slab i's GEMM runs. Decode is latency/DRAM-bound, so it overlaps better than in-tile VALU does.
- Fit: runtime. Test: after C1. Gain: a further −2–4%. Risk: graph replay and `rad_kernel_concurrent` rules.
- Conflicts: none.

**C5. Packed integer decode for wdec** (byte sum via `v_dot4_u32_u8` against 0x01010101, packed u16 shifts, 4 values per dword written with one store). Matters only after C1/C2 make the decode VALU-bound.
- Source: w29's closed form (5.8 ops/value int8).
- Fit: runtime. Test: disassembly count plus bench. Gain: −20–30% of the remaining decode time. Risk: low.
- Conflicts: #100 (closed form done; this goes further).

**C6. Let the knapsack's speed table see per-K decode cost.** Once C1 lands, decode cost scales with trellis bytes, so K2 vs K6 tensors differ ~3× in decode. That is ~2% at layer level, but it matters for M 256–1024 modes.
- Fit: tooling (knap4 table). Test: bench at 3 rates. Gain: correct prices. Risk: none.
- Conflicts: #45.

---

## D. Activation quantization and fusion (the actq 8% and the intermediates)

**D1. Multi-output norm kernel.** Read x once; RMSNorm; per-consumer suh; 128-point Hadamard; int8 quantization; write one plane per sibling (q/k/v, gate/up, GDN qkv/z) with each tensor's own suh. No requant needed.
- Source: radiance `rmsnorm_had_quant_i8`, generalized to several suh vectors.
- Why: removes the norm-output write and read plus repeated actq reads. Estimated actq 768 → ~350 µs (−4% layer), and the work moves to the norm side for the board (D10).
- Fit: runtime. Test: kernel plus byte tests vs today's actq outputs, 2 days. Gain: −4% layer; KL 0 (same bytes). Risk: the plugin must own the norm op (arch plugin).
- Conflicts: P3, P4.

**D2. [pre-freeze] Tie suh across siblings at requant.** Use one shared input scale per norm group (joint refit), or signs only.
- Source: #42 (siblings agree 100% in signs; scales differ 1.5–9% median); P4.
- Why: one plane per norm. That lets the stock radiance fused op feed us and lets siblings stack into one launch (E1). This is the only fusion that needs the ladder to change, so it must be decided now.
- Fit: format, quantizer. Test: (1) settle64 CPU proxy, tied vs free su on the 27B qkv/z, q/k/v, gate/up at K2.5/K4, ~3 CPU-hours on colo; (2) if within 0.5%, a 0.8B pair. Gain: enables −4–6% layer (D1 + E1 without multi-output plumbing); KL unknown, likely +0–0.5%. Risk: a real quality cost; D1 achieves most of it without a requant.
- Conflicts: #42; #108 (joint refits failed for quality, but that was a different objective).

**D3. [pre-freeze] su as signs only (drop the per-column input refit).** Signs fold exactly into the RMSNorm γ, so stock fused kernels apply everywhere.
- Fit: format. Test: same proxy as D2. Gain: same as D2, simpler. Risk: may cost more than D2.
- Conflicts: as D2.

**D4. SwiGLU epilogue fusion feeding down_proj.** The gate_up epilogue computes silu(g)·u·suh_down; suh_down folds exactly into up's svh because up's rows are down's columns. It then applies the 128-channel Hadamard and quantizes to int8, writing down's activation plane directly.
- How: permute gate and up rows at import so one tile holds 128 matching gate + 128 up channels. Alternatively use IQR's last-of-8-blocks counter scheme.
- Source: IQR `gemm_iqr_gated`; TWOPASS open question 6; TODO I13.
- Why: removes the bf16 gate_up output (2048×34,816×2 B ≈ 142 MB written and re-read), the silu kernel, and down's actq (294 µs, the largest actq). Estimated −4–6% layer.
- Fit: runtime (the row permutation is set at import). Test: kernel plus byte tests vs the unfused chain, 3–4 days. Gain: −5%; KL 0 (rounding order changes slightly). Risk: tile layout complexity.
- Conflicts: none.

**D5. Gated-norm fusion for ssm_out's input.** suh_ssm_out folds exactly into the gated RMSNorm weight; norm·silu(z)·γ′, then Hadamard and quantization in one kernel.
- Fit: runtime. Test: kernel plus bytes. Gain: −1% (actq 110 µs plus one pass). Risk: low.
- Conflicts: P4.

**D6. Attention output-gate fusion for o_proj's input.** sigmoid(g)·attn·suh_o, Hadamard and int8 in the attention epilogue or in one elementwise kernel. suh_o cannot fold into V under GQA (6 query heads share each KV head), but elementwise application is cheap.
- Fit: runtime. Gain: −1%. Risk: touches the attention kernel.
- Conflicts: #106 (head permutation must stay compatible, G3).

**D7. Hadamard on the WMMA unit inside actq.** x_block @ H128 as an f16 matrix product with f32 accumulation: about 2.7 GFLOP per input, ~20 µs. actq then becomes bandwidth-bound.
- Fit: runtime (new reference: one rounding instead of staged f16). Test: microbench, 0.5 day. Gain: actq ~3× faster (−4–5% layer) where not fused. Risk: changes bytes, so tests need updating.
- Conflicts: none.

**D8. Tune actq to its roofline** (DPP/permlane butterflies, 4 values per lane, one wave per 128-block row, wider stores). Today it reaches 185 GB/s on gate_up.
- Fit: runtime. Test: A6 counters plus a variant. Gain: −3–5% layer. Risk: low.
- Conflicts: D1/D7 supersede it if built.

**D9. One actq per sibling group (P3)**, exact only with tied suh (D2) or as D1's multi-output form.
- Gain: ~1%. Conflicts: P3.

**D10. [board] Put activation quantization in the norm-side op, as IQR does.** If A1 shows the board's `gemm_us` excludes norm ops, our op that does actq internally pays ~8% that IQR does not. D1/D5/D6 are real fusions in any case.
- Fit: runtime. Gain: up to −8% board `gemm_us`; real prefill −3–4%. Risk: depends on A1.
- Conflicts: A1.

**D11. Power-of-two activation scales so the fold becomes an exponent add or shift. Rejected.** `v_ldexp` or an int shift is still one VALU op per accumulator element, the same as the FMA. Quantization noise rises ~1.5 dB. Recorded.

---

## E. Launch structure and stacking

**E1. Stack sibling GEMMs into one launch**: q/k/v as N 14,336; GDN qkv+z as 16,384. This needs a shared activation plane (D2, or D1 with plane-sharing), but no K ties: the plane is int8 whatever each tensor's rate.
- Why: attn_k and attn_v cost 206 µs each as 1024-row GEMMs, mostly actq (76) plus tail effects. Stacked they cost ≈ their FLOP share (~120 µs). About −0.5–1% layer plus D9.
- Fit: runtime (+ D2). Test: bench a stacked qkv case. Gain: −1%. Risk: per-tensor mode must then be shared; the design already ties modes for shared inputs.
- Conflicts: §1d K-tie note (not needed here).

**E2. Mixed-K stacking needs no K ties.** Each tensor decodes into its own plane rows at its own rate; the tile never sees K. Keep knapsack K freedom while fusing launches (contrary to turboderp's fused K ties).
- Fit: runtime. Gain: knapsack freedom. Risk: none.
- Conflicts: §1d.

**E3. One launch per GEMM (P6 split finisher).** Removes ~2.8 µs dependent dispatches; ~0.4%.
- Conflicts: P6.

**E4. Split-K or stream-K for small GEMMs at TP2 shapes and small M** (attn_k/v, 256 ≤ M < 1024).
- Fit: runtime. Gain: small on the board, larger for real chunked prefill. Risk: determinism.

---

## F. A4 / i4 kernels (only if #46-03 passes)

**F1. Price A4-g128 next to g64 in #46-03.** Layer ×1.55 vs ×1.32 over A8; E1b gate_up A4 costs +0.009 at g128 vs +0.008 at g64.
- Fit: runtime + knap4 option. Test: existing costs plus timing. Gain: up to −15% at ≤3 bpw fast variants. Risk: KL at K2.
- Conflicts: #46-02, #46-03.

**F2. g64b4 flush variant** (×1.40–1.50; error 2.2–3.1e-2 for b8 vs exact b1), offered as a mode only if KL is unchanged.
- Conflicts: A4.md.

**F3. VOPD and compiler pin for the A4 tile** (×1.25–1.38 vs ×1.40–1.50 by flags). Same as B1/B12.

**F4. i4 weights at A8 quality via nibble-split activations** (two iu4 passes ≈ one iu8 pass, but on a half-size int4 plane, so wdec4 is cheaper). Low value: the int4 plane on the A8 path gave ≤ 2.8% (blockx.md).
- Recorded as low.

---

## G. Format and quantizer choices with speed consequences (pre-freeze checks)

**G1. [pre-freeze] Tie or simplify suh (D2/D3).** The one ladder-baked choice that decides how far prefill fusion can go. Decide before the ladder.

**G2. [pre-freeze] Check that #106's o_proj head-permutation act order folds at import.** Permuting o_proj input heads means permuting attention output heads, so q-gate/q, k and v head order must be permuted consistently, keeping GQA groups intact (6 query heads per KV head). If the permutation crosses KV groups it needs a runtime gather. Store "heaviest heads last" only within groups if necessary.
- Fit: format. Test: read the converter, CPU check, 1 hour. Gain: avoids a runtime gather (~1–2% of attention layers). Risk: constrains #106's gain.
- Conflicts: #106.

**G3. VALU per value for fractional rates.** Disassembly count for KA + ¼/½/¾ in GEMV, narrow tile and wdec8; w29 measured integer K only.
- Why: if fractional patterns cost more (variable shifts), decode-time pricing and a decode-fast variant need it.
- Fit: tooling. Test: CPU, 1 hour. Gain: correct pricing. Risk: none. Conflicts: #36/#37.

**G4. Sub-1 salted-read decode lives in its own instance.** Confirm the default rates pay nothing for the sub-1 flag (one compiled branch).
- Test: disassembly. Conflicts: #90.

**G5. [pre-freeze, done] Keep mul1i8 and the closed form.** iqrh would have bought −15–20% int8 and −32–35% f16 decode VALU, but cost +1.6–4% KL (#100); #113's 2^20 trellis would add +10–28% decode. Recorded so speed arguments do not reopen them.
- Conflicts: #100, #113.

**G6. GDN a/b projections to int8 weights.** Removes two bf16 `gemm_nt` launches per GDN layer (0.27–0.4% of the step) and saves 0.007 bpw. The quality side is the TODO item. Kernel: radiance's int8 weight op.
- Gain: ~0.3% speed plus a KL gain via bytes (TODO estimate −1.25% at 2–2.5 bpw). Conflicts: D6-design closed the merge; this is storage only.

---

## H. Decode GEMV (M = 1): VALU- and latency-bound

**H1. Split K across blocks (D1 in kernel-opts).** At N 5120 we launch 40 workgroups for 64 CUs. Use blocks = (128-column group × K slice); the last-arriving block sums partials in fixed order.
- Source: IQR 5120×6144 GEMV 42.3 → 30.0 µs.
- Fit: runtime. Test: GEMV bench. Gain: −20–30% decode GEMV, ~−15% decode tok/s cost. Risk: determinism handled by the fixed-order finisher.
- Conflicts: none.

**H2. Block per tile with KW waves splitting K**, plus a KW sweep (our decode costs more VALU per weight than IQR's, so the optimum KW differs).
- Source: radiance 1.17. Gain: part of H1's. Conflicts: none.

**H3. A8 decode for the fast variant.** int8 activations at M 1: `sudot4` gives 4 MACs per op, and the int8 closed-form decode costs 5.8 vs 7.4 ops/value on the f16 path. Estimated −15–25% GEMV.
- KL: the same A8-g128 noise the fast variant already pays in prefill (+0.00005 at K6). Not on the board (decode is unscored).
- Fit: runtime. Test: an int8 GEMV variant plus a decode tok/s and a short decode-time KL check. Gain: −15–25% decode GEMV. Risk: overturns design §1b.2's "A8 only for M ≥ 512". That rule assumed decode is byte-bound; the measurements say it is not.
- Conflicts: §1b decision 2.

**H4. VOPD for decode kernels.** The decode's integer ops (shifts, AND, add_nc, mad_u32_u16) are VOPD-pair candidates; the plugin currently disables VOPD globally.
- Fit: runtime. Test: rebuild plus GEMV bench. Gain: −5–20% decode VALU. Risk: as B1.

**H5. Drop the −0 fix in the f16 closed form** (`EXL3I_I8_CLOSED=2`: 6.4 vs 7.4 ops/value at K4). Only the sign of zero products changes; outputs are unchanged.
- Fit: runtime. Test: byte-compare outputs across builds plus bench. Gain: about −5% GEMV. Risk: tests compare plane bytes, so the test needs updating.
- Conflicts: #100 (w29's note).

**H6. Packed f16 decode tail** (`v_pk_*` for 2 values per op) and a `v_dot2_f32_f16` MAC on the f16 path.
- Gain: −5–10%. Risk: low.

**H7. Biased-int fold and SGPR scale loads in GEMV** (IQR). Gain: a few %.

**H8. Fused decode-time producers**: gated activation quantizer, GDN conv fold and attention-gate quantizer folded into the kernels that produce their inputs. IQR removed 128 launches per step.
- Fit: runtime. Gain: −1–3% decode. Risk: medium complexity.

**H9. Persistent per-layer decode kernel.** About 900 kernels per token × ~2.8 µs dependent-dispatch gaps ≈ 2.5 ms ≈ 7% of a ~34 ms token.
- Fit: runtime. Test: count the gaps with `--profile-ops` first. Gain: −5–7% decode. Risk: large engineering; graph replay may already hide part of it.

**H10. Software prefetch of the next tensor's trellis words** in GEMV to raise bandwidth utilization from 48%. Matters once H1–H5 cut the VALU.
- Gain: 0–10%. Risk: low.

**H11. Note for decode variants: lower bpw buys no decode speed until decode VALU falls** (flat-in-K data). A "decode-fast variant" means a cheaper codebook or H1–H5, not fewer bits.
- Fit: analysis for the 4-D frontier (ROADMAP "Later").

**H12. Decode-time head bytes.** An f16 head at K5–6 reads ~0.8 GB per token (~2 ms, ~6% of a token). IQR's vocabulary-weighted head takes 0.62 vs 1.02 ms. Our slab head needs a gather-plus-two-width kernel; bytes are priced in knap4.
- Gain: −2–4% decode. Conflicts: #34.

---

## I. Narrow tile, speculative decoding, MTP

**I1. IQR's barrier-free 16-token WMMA tile** (each wave decodes a 128-K unit, transposes it in its own LDS tile, activations straight from L2). 1.15–1.74× over V3's kernels from 4 tokens.
- Fit: runtime. Gain: verify and small-batch decode. Risk: medium.
- Conflicts: none.

**I2. DFlash2 port, sized by our M-curve.** Verify at M 16 costs 1.41× M 1 because the trellis decode is shared across tokens, so accepted drafts (4.1–5.5 tokens) give ~2.5–3.5× decode tok/s.
- Source: design §8. Fit: runtime. Test: acceptance vs bpw at 2.5/4/6. Gain: ×2.5–3.5 decode. Risk: drafter VRAM of 2.4–4.1 GB; the head-rerank detail (TODO).
- Conflicts: §8.

**I3. Native MTP drafter as the lighter option.** 8 MTP trellis tensors exist in turboderp's downloads; radiance has `spechub` MTP verify. Ask the challenge author whether an MTP block counts toward bpw (B10 open).
- Fit: format (container) plus runtime. Gain: ×1.5–2 decode at near-zero VRAM. Risk: the bpw-count question.
- Conflicts: #53.

**I4. Tree verify width from the cost curve, plus GDN state replay for rejected branches** (saber `p27_tree` / `gdn_replay`).
- Gain: +10–30% over linear drafts. Risk: GDN replay complexity.

**I5. int8-quantized DFlash2 drafter.** Draft-side decode cost halves; check acceptance.
- Gain: drafter step ~2× faster. Risk: lower acceptance.

**I6. An int8 narrow tile for M 16–255.** Today A8 is 1.86× slower than f16 at M 16 because the A8 path is a prefill tile. A dedicated narrow A8 tile would let the fast variant stay A8 through verify.
- Gain: verify −10–30% in the fast variant. Risk: medium. Conflicts: TWOPASS open question 4.

---

## J. GDN, attention, head and embedding kernels

**J1. INT8 GDN chunk kernels** (LinQuant recipe: per-(head, value-channel) static scales, FP32 state carry, int8 read copy, R_QK Hadamard on q/k; GDN's scalar decay commutes with it).
- Source: 2610.39418 (GDN INT8 per-channel 15.019 vs FP 15.004 perplexity).
- Fit: runtime. Gain: GDN core is a small share of 2k-token prefill; more at long context. Risk: low priority.
- Conflicts: none.

**J2. Keep GDN state in fp32 in every fast path, including int8 decode.** BF16 per-step state already cost −16 AIME points on Qwen3.5-9B (2609.38166); INT6 state collapses (2609.38169, 2610.39931).
- Fit: runtime rule. Gain: avoids a quality trap. Risk: none.

**J3. Head only on scored rows in KLD evals** (TODO: +4–8% eval throughput; scored KLD byte-identical). See K-section.

**J4. Embedding row-decode gather.** Check coalescing at prefill (2048 rows × 5120 values); negligible if done right.

**J5. Long-context attention (D-LSA, fp8 QK).** Out of scope per the brief; the V3 extra already exists.

---

## K. Eval-runner throughput [KL-indirect]
(~30 evals/h/card; ~80 s fixed + ~0.4 s per 1k positions; 22 streams × 100–300 evals ≈ 70–200 card-hours)

**K1. On-device KLD.** A fused per-row kernel: log-sum-exp over 248k logits, gather the golden's top-512 indices, compute KL. Only per-row scalars leave the card.
- Source: #8 (card 1 is 25% slower because of the full-vocab f32 logits copy over its x4 link).
- Fit: tooling. Test: byte-compare scored KLD vs the host path, ~1 day. Gain: card 1 +25%, card 0 +3–5%. Risk: low.
- Conflicts: #8.

**K2. Two concurrent evals per card for mixes ≤ 4 bpw.** If the 80 s fixed cost is mostly load, relayout, declare and scoring rather than GPU, overlapping two processes hides it.
- Fit: tooling. Test: two V2-3.00 evals in parallel vs serial, 1 hour. Gain: +30–60% evals/h at ≤ 4 bpw. Risk: 30 GB host RAM and page cache; VRAM above ~5 bpw.
- Conflicts: #1 (the +14% there was for the quantizer, already at 100% GPU).

**K3. Measure the 80 s split, then resident base / swap mode** (design §5 proposal, step 1 first).
- Gain: up to −40% per eval. Risk: engine work. Conflicts: §5 proposal.

**K4. Layer-major eval.** Run the whole slice through layer L before layer L+1 (residuals for 160k positions ≈ 1.6 GB per boundary).
- Each layer's weights are decoded once per eval, not once per 2048-chunk (decode is ~10% of the GEMM path).
- MLP GEMMs can run at very large M (no sequence dependence).
- It enables K5.
- Fit: tooling (engine mode or staged tool like mkgold27). Gain: −10–15% eval time. Risk: large engine change.
- Conflicts: TODO I12.

**K5. Exact prefix-layer cache for local-search mutations.** Store the base mix's residual stream at a few layer boundaries (on NVMe); evaluate a mutation from its first changed layer on. This is exact for prefill-only KLD because attention and GDN recompute from cached inputs.
- Fit: tooling. Test: byte-compare scored KLD vs a cold eval on 20 mutations. Gain: −30–50% on late-layer mutations. Risk: storage; ordering mutations late-first.
- Conflicts: none.

**K6. Sequential early stopping of losing candidates.** Paired per-document z vs the incumbent after 30–50% of documents, with a pre-registered alpha-spending rule. Most candidates are rejects.
- Fit: tooling (search rule, written before use). Gain: −30–40% eval time. Risk: false rejects; tune on V2 logs first.
- Conflicts: §5 stop rule (compatible).

**K7. Run fast-variant search evals on the A8 path** (×1.53 GEMMs). This is required anyway, since the fast variant's KLD must be measured on its own path.
- Gain: −30% per fast eval.

**K8. EXL3I_A8_MIN_ROWS = 256, and lower T2's threshold.** A8 is faster from about 256 rows; f16 nt_pass at 511 rows is 25% slower than T2 at 512.
- Gain: helps short documents and real-world chunks. Conflicts: TWOPASS open question 4.

**K9. Batch short documents so GEMM M ≥ 512.** Matters only for docs under ~512 tokens (per-token cost is flat above 512).
- Gain: small. Risk: batching changes numerics; it must be the same for every arm.

---

## L. Quantizer and ladder throughput [KL-indirect, pre-ladder]

**L1. FastKron recursive anti-diagonal two-sided LDLQ for the H_out levels.**
- Today: `ldlq_2hess` does one rank-16T addmm per anti-diagonal; 1,407 diagonals for gate_proj. The feedback is 72% of a gate_proj under H_out, and an H_out level takes 3.2 h vs 37 min.
- FastKron splits the diagonal range recursively with two masked matmuls per node: cubic instead of quartic work, same rounding decisions up to floating-point reassociation. Its benchmark at n = 8192: 23,084 → 250 ms.
- Source: 2608.06291.
- Why: about 24 levels ≥ K4 use H_out (adopted 11-07). Estimated 3.2 h → ~1.0–1.3 h per level, saving ~45–55 card-hours of ladder.
- Fit: tooling (quantizer). Test: port to ourq; compare `quant_w` and indices vs `ldlq_2hess` on two 27B tensors (expect identical, or within the rqah score band); time it, ~2 days. Gain: −45–55 card-h; KL 0. Risk: non-bit-identity with the research scripts (rqah verifies by score, so that is acceptable).
- Conflicts: #33-28 (cost), #87-06.

**L2. Remove host syncs from `ldlq_2hess`'s inner loop.** `int(a.max().item())` runs every diagonal; k_hi and n_hi are known on the CPU from s. Also batch several tensors' diagonals per Viterbi launch.
- Fit: tooling. Test: 1 hour. Gain: −5–15% of H_out level time. Risk: none.

**L3. Card-0 clock fix (A2)** speeds the Viterbi, ~80% of a no-H_out level. Gain: up to −25% ladder time on card 0.

**L4. GPU-batched tile coordinate descent for cd3 on attn_qg** (×3.7 that family's time, ~+5 min a level).
- Gain: −3–4 min/level × 71 levels ≈ 4 card-h. Conflicts: #77.

**L5. Pack stage (~1,600 CPU-seconds a level) in parallel on the CPU boxes.** Already off-card; just schedule it in parallel.

---

## M. Real-world prefill scheduling (not on the board, but it is what users see)

**M1. Recommend 4096–8192-token prefill chunks for A8.** Decode share is 11% at 2048 and 5.5% / 2.7% at 4096 / 8192; per-token GEMM cost is flat (A4 sweep 3.84–3.89 µs per token).
- Fit: runtime config. Gain: −5–8% real prefill. Risk: activation scratch memory.
- Conflicts: #6 (null under f16, where the GEMMs dominated).

**M2. Layer-major prefill for long prompts.** Decode each layer once per prompt and keep its plane slab in LLC across chunks.
- Gain: −10% for prompts over 8k. Risk: engine scheduling change.

**M3. Keep the int8 plane scratch at LLC size (C1)** instead of 178 MB per GEMM. That frees ~130 MB of VRAM for KV/context.

---

## Ranked top 20 (expected gain per unit cost)

| # | idea | gain | cost | KL |
|---|---|---|---|---|
| 1 | A2 card-0 clock check/fix | 0–34% on all card-0 VALU kernels, ladder and evals | 5 min | 0 |
| 2 | K1 on-device KLD | card 1 +25% evals, card 0 +3–5% | 1 day | 0 |
| 3 | L1 FastKron two-sided LDLQ | −45–55 card-h of ladder | 2 days | 0 |
| 4 | B1 VOPD back on | −3% layer, more on decode | 2 h | 0 |
| 5 | C1 LLC-chunked two-pass | −7% layer at 2048, −25% at 512 | 1 day | 0 |
| 6 | A1 pin board protocol (actq accounting) | steers ~8% `gemm_us` | 1–2 card-h | 0 |
| 7 | K2 two concurrent evals per card | +30–60% evals/h at ≤ 4 bpw | 1 h test | 0 |
| 8 | D1 multi-output norm-quant kernel | −4% layer | 2 days | 0 |
| 9 | D4 SwiGLU epilogue → down plane | −5% layer | 3–4 days | 0 |
| 10 | H1 split-K decode GEMV | −20–30% GEMV | 1–2 days | 0 |
| 11 | K6 sequential early stop in search | −30–40% eval time | 1 day + rule | 0 (pre-registered) |
| 12 | B3 real g256 fold | −3–5% layer | 1 day | +0.03% |
| 13 | B4 per-token A8 gate_up (knap4 option) | −4% layer at ≤ 3 bpw | 1–2 days | +0.1% at 2.5 |
| 14 | D2 [pre-freeze] tied-suh proxy test | unlocks D9/E1 (−1–2%) | 3 CPU-h | +0–0.5% |
| 15 | B2 RB 8 fold (no spill) | 0–8% tile | 30 min | 0 |
| 16 | L2 host syncs out of `ldlq_2hess` | −5–15% H_out level | 1 h | 0 |
| 17 | H3 A8 decode in the fast variant | −15–25% GEMV | 2 days | decode only |
| 18 | D7/D8 actq on WMMA / roofline | −4% (if D1 not built) | 0.5–1 day | 0 |
| 19 | K5 prefix-layer mutation cache | −30–50% mutation evals | 3 days | 0 |
| 20 | I2 DFlash2 (verify ≈ free on our M-curve) | ×2.5–3.5 decode | 1–2 weeks | 0 |

**Taken together:** C1 + D1 + D4 + B1 + B3/B4 would bring the A8 layer from 9,444 to about 7,200 µs on our bench, roughly 4,000 µs on the board scale. That is level with IQR-V4's 3,829–4,043, and none of it needs tile-core work.

## Five cheapest diagnostics that would most change this ranking

1. **A2: card 0's sclk (5 minutes).** A misconfiguration would re-rank every VALU-bound measurement, decode and ladder estimate, and give 25%+ for free.
2. **A1: board `gemm_us` accounting (1–2 card-h).** If norm-side ops are excluded, D10 becomes item 1 for the board. If TP2 shapes change tile efficiency, B-items move.
3. **A4 + A5: ablated-decode floor and K-loop instruction census (~3 card-min + 1 CPU-hour).** These fix the decode share at M 1 (ranks H1–H6) and the fold share in the tile (ranks B2/B3/B4/B9).
4. **K3 step 1: split the 80 s eval fixed cost (one instrumented eval).** If load and declare dominate, K2/K3 jump above K1; if GPU work dominates, K4/K5 matter more.
5. **D2 tied-suh proxy (settle64 on colo, ~3 CPU-hours).** If tying costs < 0.3% proxy error, adopt it before the ladder and drop D1's multi-output plumbing. If it costs more, D1 is the only route and G1 closes.
