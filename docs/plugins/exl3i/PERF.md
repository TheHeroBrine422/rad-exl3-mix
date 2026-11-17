# exl3i perf pass (w124, #164, 2026-11-12 / 13)

What the pass changed on `qc-exl3-v3-perf` (off `qc-exl3-v3` 42960d5), with the same stored bytes; how each change
is tested; what it measured on the RX 9070 XT (gfx1201, dad1: the R9700's chip and bus); what was tried and refuted;
what is left. Data and bench scripts: rad-exl3-mix `v3/bench/data/w124/` (README there has the per-shape tables).

## Commits

| commit | change | bit-exact? | test |
|---|---|---|---|
| 1406d75 | fractional / sub-1 lane plans from a compile-time table (`exl3i_plan_tab.h`) instead of `exl3i_plan()` per wave | yes | `exl3i_plan_tab_test` (host, every rate and lane) |
| 20740e4 | KS: the decode form (T ≤ 16) splits K across workgroups (~512), the last arriver of a 128-column block sums the slices in slice order and runs the epilogue | no (f32 order) | `test_ks` |
| ea3a9ea | DQ: the decode form on mul1i8 multiplies the grid index q, the step once after the sums | no (q·step exactly, not fp16(q·step)) | `exl3i_gpu_test` B |
| 3494a31 | DOT: T = 1 by `v_dot2_f32_f16` instead of a 16-row WMMA | no (f32 order) | `test_ks` DOT, `exl3i_gpu_test` B |
| 8d3f594 | R16 head row order: encoding version 3 (`/r16`), importer `rows=r16`, record dtype `EXL3I_DT_TILES_R16`, the epilogue stores output s at `exl3i_r16_row(s, N)` | yes (a store address) | `test_r16`, host H, `exl3quant_test` r16 |
| a7d1cf2 | A8 FUSE: the A8 GEMM's blocks decode their own weight tiles (claim / decode / wait counters) instead of a wdec8 pass (mul1i8); two passes for K > 8192 at M ≤ 512 (down at 512 rows: ×1.05 fused, from 768 ×0.92) | yes | `test_twopass` (planes and y byte for byte) |
| d517103 | FUSE opt-in (`EXL3I_A8_FUSE=1`): built with ROCm 7.14 / AMD clang 23 it hangs the R9700 on its first launch (below) | — | `test_twopass_fuse` (CMake, FUSE on) |
| 128d363 | AQ, interface version 3: `exl3i_actq` (the A8 quantiser for up to four consumers of one x, packed rows `[planes][K]` int8 + `[planes][K/128]` f32) and the dense op's optional `aq` operand (`aq_plane` / `aq_planes`): the A8 GEMM reads the producer's planes and runs no actq | yes (the planes are the op's own) | `test_aq` (planes vs the host reference; y with aq vs quantising itself), host L |
| 8c3ba37 | AQ producer stages: `exl3i_actq` `pre` = rmsnorm / silu_mul / mul (+ optional `h`), the value quantised being libr4d's unfused op's bf16 output (its arithmetic copied); a launch given aq below the A8 threshold is refused | yes | `test_aq` against libr4d's own `rmsnorm_bf16` / `silu_mul_bf16` / `mul_bf16` + actq: 0 bytes differ |
| e12c2bf | `exl3i_actq` layout hooks: its s0..s3 stored as exl3i_dense stores suh (`exl3i.f16.vec`) — a weight is stored once | — | host L |
| 1aa18de | qwen35_exl3 AQ: GDN norm → qkv / z (+ h for a / b), attention norm → qg / k / v, gate mul → o, MLP norm → gate_up, silu_mul → down, from 512 rows (`QWEN35_EXL3_AQ=0` off) | yes | `qwen35_exl3_test` test_aq (sites, planes, steps at 16 / 1024 rows, lint) |
| 5273e01 | inline-asm constants in the asm text (literal / inline constant), not `"s"` operands: LLVM's GFX12 SGPR hazard pass does not see an asm's SGPR read (radisa MISSING_ALU_WAIT 1,404 / 1,130 → 0, #175) | yes | w137 asmlit (each form, 2^26 inputs, both toolchains); whole-model KLD rows byte-identical |
| e776f32 | FUSE's claim loop with wave-uniform branches (the ROCm 7.14 hang, below); FUSE on by default again (`EXL3I_A8_FUSE=0` off) | yes | `test_twopass` (default = FUSE), `test_twopass_fuse`, `test_twopass_nofuse` on 7.14 and 10.1; whole-model KLD rows byte-identical |
| 3f41caf | bench_twopass: a `head` shape (248320 x 5120, when named) and an FNV-1a digest of y each row (two builds with the same output bits print the same digest) | — | — |
| a60dec3 | fractional-K decode (#175, below): RateFrac's states as packed pairs (one shift / alignbit a state, one v_perm a pair), the three-word runs (K7, K5.125-7.875) by buffer loads (`EXL3I_NW3_BUF`), the decode form's load stages 2 → 3 (`EXL3I_DEC_PF`) | yes | `exl3i_gpu_test` A (every rate x codebook vs the oracle), all gpu ctests on 10.1 + 7.14, y digests at M 1 (10 rates x 10 shapes) and the A8 path (M 2048), whole-model KLD rows byte-identical at K4.125 and K2.5 |
| 8afc1fc, a3799b5, ae909bf | qwen35_exl3: the vision tower and the DFlash2 drafter (stock radiance's blocks; below); media only when the container carries the tower's weights; the probe maps nothing | — | `qwen35_exl3_test` test_media_and_dflash2 |

Knobs (environment, benches only): `EXL3I_KSPLIT=<n>` (1: KS off), `EXL3I_KS_WG` (KS's workgroup target, 512),
`EXL3I_DEC_DOT=0` (WMMA at T 1), `EXL3I_A8_FUSE=0` (FUSE off; on by default since e776f32), `QWEN35_EXL3_AQ=0` (the
arch's AQ off); build: `EXL3I_DEC_Q=0` (no DQ), `EXL3I_DEC_PF` (the decode form's load stages, 3 since a60dec3),
`EXL3I_NW3_BUF=0` (the three-word runs by global loads, as before a60dec3), `EXL3I_PLAN_RUNTIME=1` (the old plan).

## What decode was bound by

At T = 1 the old decode was flat in K: gate_up took ~300 µs at every rate from K0.5 to K4 (30–290 GB/s of codes on a
640 GB/s card). With the decode replaced by the raw words (`EXL3I_ABLATE_DECODE`) the same kernel is bandwidth-bound
at K4 (gate_up 555 GB/s): the time was the trellis decode's ALU (54 VALU a 16×16 tile at K4: hash 16, byte sums 8,
the int8-grid finish 24, extraction 6) plus a grid of N/128 workgroups (40 on down, 8 on attn_k / v). A deeper load
pipeline (PF 4 / 8) and VOPD pairing did nothing. The decode's ALU per value does not shrink with K, so below K4 the
GEMV cannot get faster with bytes on this codebook: after KS + DQ + DOT, integer K1–K3 run at 0.84–1.02 of K4's time
per shape, fractional rates at 1.03–1.12 (their bit-offset state extraction: ~9 VALU a tile more than K4's
byte-aligned perm) and sub-1 at 0.94–1.17 (the salts). An extraction-free hash (states hashed from the shifted word)
gave nothing.

## Results (9070 XT, interleaved A/B, 48 GDN : 16 attention layer mean)

- Decode (nt_pass), T 1, base → HEAD: K0.5 681 → 433, K1 607 → 372, K2 632 → 382, K2.5 671 → 431, K3 658 → 390,
  K4 628 → 397, K6.5 1013 → 630, K7 812 → 635 µs; T 4 K4 641 → 409, T 8 704 → 431. Per shape: down ×0.46–0.50,
  attn_k / v ×0.30, attn_o / ssm_out ×0.51–0.64, gate_up ×0.73–0.86 (K7 ×1.05: KS's 2-way split on the widest weight).
- Prefill (the A8 path: actq + decode + GEMM), M 2048: K0.5 ×0.889, K1 ×0.969, K1.5 ×0.896, K2 ×0.968, K2.5 ×0.899,
  K3 ×0.964, K4 ×0.949, K6.5 ×0.892, K7 ×0.927; M 512 ×0.70–0.90. From the plan table (fractional wdec8 ×0.45) and
  FUSE (×0.94–0.97 at M 2048).

## Whole model and e2e (R9700, box2, same engine build)

- Prefill KLD (golden-search27c-c, 67,932 positions, bf16 KV) equal to today's build to 10 digits at K0.125, K2.625,
  K3, K4.125, K6.25: the prefill changes (plan table, FUSE) are bit-exact.
- R16 heads: whole-model KLD with the R16 K4 i8 / K7 mul1 head 0.65177 / 0.65232 vs 0.65231 with the bf16 head (K2.625
  trunk); the donors' logits vs their dequantized weight at every vocabulary id rel rms 5e-4, argmax equal.
- Decode numerics (radiance's KLD needs 64-token chunks on this model, so through the server: greedy, top-20 logprobs,
  today vs dev): per-step KL median ~5e-6, tails from state drift; KS + DOT alone (an f32 reorder) and DQ alone give the
  same signature.
- Decode step, batch 1: K4.125 + K7 mul1 R16 head 32.4 ms (today's kernels with a bf16 head: 49.0; dev with it 34.3),
  K2.625 + K7 31.6, K6.25 + K7 46.7; IQR-V4 S (4.14 bpw, its drafter off) 25.7 ms on the same card. Breakdown of the
  K4.125 step (--profile-ops): trunk GEMVs 26.8 ms (~485 GB/s of codes; the 640 GB/s floor is ~20), head 2.8, GDN 1.1,
  attention 1.4, norms 0.6, other 1.8, gaps 4.0. Prefill e2e at 8k tokens 1390 → 1566 tok/s (IQR 2461).

## AQ: the producer writes its consumers' int8 planes (#164 scope 2)

The A8 path quantised each consumer's input itself (`actq`: rotate with the weight's suh, absmax per 128, int8), reading
the producer's bf16 output once per consumer. `exl3i_actq` with `pre` folds the producer in: one launch computes the
norm / silu_mul / gate product exactly as libr4d does (byte-identical bf16 values), rotates them with each consumer's
suh and writes every consumer's plane; the dense ops read their planes (`aq`) and skip actq. Prefill only (≥ 512 rows);
decode issues as before. Planes byte for byte the unfused path's, so the outputs do not change at all.

- Kernel (9070 XT, K4 mul1i8, T 2048, producer + consumers, unfused → fused, round robin): attn norm→qg/k/v −6.7%,
  GDN norm→qkv/z −5.0%, MLP norm→gate_up −2.8%, silu_mul→down −11.5%, gate mul→o −12.9% (T 512: −1 to −13%; T 4096
  −2 to −14%). A shared quantiser alone (no producer folded in) saved 0–5%: the fold is what pays.
- Whole model (R9700, golden cut13, bf16 KV, act a8g128 on all 336 trunk weights): KLD rows byte-identical between today's
  kernels, dev with AQ off and dev with AQ on at K2.625 (0.291513), K4.125 (0.079819) and K6.25 (0.028256).
- e2e prefill (K4.125 A8, tok/s, two interleaved rounds): 8k today 2215 / 2181, dev AQ off 2282 / 2280, AQ on 2299 /
  2295; 32k today 2047 / 2037, off 2119 / 2116, on 2130 / 2128 — AQ +0.7% e2e, well under the kernel-level savings (the
  rest of a prefill step was not broken down). Decode C 1: today 47.5, dev 34.6 ms with or without AQ.
- The act f16 containers (every earlier e2e number here) never take the A8 path; with a8g128 on the trunk prefill is
  2290 tok/s at 8k against the f16 path's 1566 (IQR-V4 S 2461).

## Vision and DFlash2 in qwen35_exl3 (#164 scope 3)

Stock radiance's blocks, unchanged, over the exl3 trunk: the vision tower (`rad_block_vit.h`, bf16; declared through its
own weight gate, so a text-only container keeps serving text) and DFlash2 (`rad_block_dflash2.h`: qwen35_fp8's
declare_dflash2, the trunk's taps at layers 5 19 33 47 61, the drafter passes, `rad_arch_probe` for its fixed depth 7).
No core change. Packed on box2: K4.125 trunk (act a8g128) + the tower + incoai/Qwen3.8-27B-DFlash2 (rev 015e795) with
IQR-V4 S's drafter encodings (MXFP4 linears, u2 head); served by radiance 1.17.0 (the frozen source) built with
FFmpeg 7.1.1 (box2's install has no media decoder and refuses every image).

- **Trunk untouched:** text KLD of that container with today's kernels and with dev (AQ on, tower + drafter declared):
  rows byte-identical (0.077012).
- **Images (9 prompts: photos, a dashboard screenshot, synthetic colour fields; greedy, 64 tokens):** answers correct
  (portrait in U.S. Navy uniform, herring gull, stone cottage, the dashboard's panels, the 2 × 2 red / green / blue /
  white grid, a wide red strip) and the same prompt token counts as IQR-V4 S (stock path). Against IQR (top-20 logprobs,
  steps before the first different token): per-step KL median 1.0e-3 on images vs 5.8e-4 on text (means 0.020 / 0.016):
  the two trunks' difference, no extra from the image path. The stock bf16 reference (layer_offload on one card)
  segfaulted on its first image request, so the KLD against bf16 is not measured.
- **DFlash2 (spec on vs off, same server build):** temp-0 output identical to the end on 8 / 8 text prompts (512 / 512
  tokens); on images 6 / 9 (per-step KL median 6.6e-6: near-tie flips, verify steps run 8 rows through other kernels
  than T 1 — stock IQR's own drafter: 7 / 8 text, 7 / 9 images). Acceptance 34.8% of drafted tokens (IQR's drafter on
  its trunk 38.8%), ~3.4 tokens a step at C 1 (IQR ~3.9). Step time with speculation: pending (600-token requests ended
  inside dstep's window; the 2400-token rerun was stopped for the report).
- **VRAM (static weights, per card):** text container 14.11 GiB; + the tower 0.86 GiB (879 MiB); + the drafter 1.55 GiB
  (IQR's the same 1.55 GiB).

## FUSE hung the R9700 when built with ROCm 7.14 (solved, #175)

The first whole-model A8 run (box2, K4.125 a8g128) hung the card in `exl3i_a8_gemmd_kernel` on the first prefill launch;
reproduced with `test_twopass` and `bench_aq2` under box2's pip ROCm 7.14.1 (AMD clang 23); dad1's ROCm core-10.1
(clang 24) ran it. **Cause (w137, from the ISA):** the claim loop had two `if (tid == 0)` blocks around its barriers (the
release of the decoded strip at the end of an iteration, the next claim at the top). clang 23 threaded that condition
across the back edge and the structurizer made two nested loops: thread 0 alone in the outer one (release, claim, store
`s_v`) and every other thread of the block -- the 31 other lanes of wave 0 and waves 1-7 -- in an inner loop that starts
at the first barrier (`s_and_not1_b32 exec_lo, exec_lo, s56` drops lane 0 out of the inner loop; waves without thread 0
loop straight back to `s_barrier_signal`). The barriers are per wave, so they still completed, but thread 0's release
and claim only ran once the inner loop ended -- which needs `st >= 8`, which only thread 0 writes: the block decoded the
same strip forever. clang 24 kept one loop with the `tid == 0` blocks as masked regions inside it. **Fix (e776f32):** no
thread-dependent branch in the loop: wave 0 claims and releases with all its lanes (lane 0 adds 1, the others 0;
`readfirstlane` takes lane 0's returned value) and `st` is read with `readfirstlane`. Both compilers now emit uniform
scalar branches there. A `readfirstlane` on `st` alone (thread-0 branches kept) still hung, and so did the radisa-clean
build (5273e01 without e776f32): the lint's MISSING_ALU_WAIT errors are real hazards per LLVM's own model but not this
hang.

Checks: `test_twopass` with FUSE (808 checks) PASS on 7.14 (25 s; it hung before) and 10.1, every gpu ctest of the
plugins PASS on both; whole-model KLD (golden-search27c-c, 67,932 positions, the K4.125 A8 pack with a bf16 head) rows
**byte-identical** to today's kernels (rad10-perf home, 5c9956c, mean 0.285753) for 7.14 / 10.1 × FUSE off / on / the
new default. FUSE's kernel spills 3 VGPRs (7.14) / 7 VGPRs (10.1) per lane, unchanged by the fix.

## #175 (w137, 11-13): ISA lint, prefill and decode on box2 against IQR-V4 S

Scripts and summed data: rad-exl3-mix `v3/bench/k175/` (README there).

**ISA lint** (Code000's rdna4-isa-suite, `radisa check` per unit in source and object mode, every exl3i / exl3dense /
exl3r4d unit, with ROCm 10.1's VOPD clang and the pip ROCm 7.14.1): before, MISSING_ALU_WAIT 1,404 (10.1) / 1,130
(7.14) in 370-378 kernels, all from the inline asm's `"s"` constants (exl3r4d `hash_pair`, exl3i `i4n` / `fin16` /
`fin8` / `iqrh_hash`) and the SALU sites downstream whose SGPR pairs those asm statements read (exl3i_nt.hip.h
377-431, exl3r4d_common.hip.h 88-730). After 5273e01: 0 errors on both; left: REGISTER_SPILL (26 kernels: FUSE, the A4
GEMM, a few nt forms) and one SCRATCH_USE (exl3dense's dequant kernel, a 528 B private array), both review-only. `run.sh`
(collect + audit of every gfx1201 code object of the build: 6,439-6,510 kernels in 121-126 objects) on the staged perf
build and on our plugins with the fix, both toolchains: 0 errors, every byte decodes; INCOMPLETE only from stock
kernels' indirect calls (`sam_attn*`, `couple_verify`); LOCAL_MEMORY_HAZARD review findings everywhere (~90k in ours,
~670k in stock: the tool's unproven-wait heuristic). Cost of the literals: none measured (wdec8 / decode M 1 / 4 within
±0.4% at K2.5 / K4.125; the GEMM object byte-identical; e2e prefill base − fix −74 / +4 / +12 ms over three interleaved
rounds on card 1: 2,376 tok/s both, decode 32.0 vs 31.8 ms).

**8k prefill, e2e** (box2 R9700 at the donor's 210 W cap, rad10-perf binary, w138's #176 server settings, bench.py
chat requests, a nonce line each, 3 runs a round; client TTFT median / the web UI's tok/s from the server log):

| card 0 | round 1 | round 2 | tok/s | decode C 1 (ms / step) |
|---|---|---|---|---|
| V3 K4.125 A8 + R16 K4 head, today's kernels | 3,417 ms | 3,455 ms | 2,404 | 31.6 |
| the same, the fixes, FUSE off | 3,444 | 3,457 | 2,390 | 31.4 |
| the same, the fixes, FUSE on | 3,446 | 3,449 | 2,396 | 31.5 |
| IQR-V4 S (spec off) | 3,269 | 3,265 | 2,529 | 25.4 |

FUSE's kernel-level ×0.94-0.97 (9070 XT, M 2048) does not show end to end here: ±0.1%.

**Where the 8k gap is: in no op; both run at the power cap.** `--profile-ops` (one warm 8k prefill: the per-op difference between servers
that ran 1 and 5 prefills) has V3's ops at 3,338 ms against IQR's 3,378: GEMM 2,683 vs 2,775 (gate_up 1,274 / 1,301,
down 593 / 622, GDN projections qkv + z + a + b 448 vs IQR's fused inz + ab 480, ssm_out 181 / 181, attention projections
185 / 190), activation quant 209 (actq with the producer folded) vs 188 (IQR's fused norm / gate + Hadamard + int8)
+ 17, GDN 155 / 164, attention 203 / 213, norms 20 / 20, elementwise 67 (V3's separate residual adds 56 and the
sigmoid 10) vs 0.5 (IQR folds them). Profiled, V3 is not slower; unprofiled it is 4-6% slower. rocm-smi during six
back-to-back 8k prefills (card 1): both runs sit at the cap -- V3 212 W, sclk 1,865 MHz (3.36 s), IQR 208 W, 1,760 MHz
(3.26 s) -- so wall time follows energy: ~712 J against ~677 J a prefill (+5%). Profiling's per-op syncs leave idle gaps
that take the card off its cap, so the profiled times do not show the gap (card 1: IQR slows 3.26 → 3.43 s under
profiling, V3 does not, 3.44 → 3.38: unprofiled, IQR's kernels overlap more or draw less power; the two are not
separable without a trace). Junction temperature drifts the same way between back-to-back runs (64 → 71 °C; a run
straight after idle is ~2% faster), so A/B rows here are interleaved. Decode at C 1 (512 tokens) is
at the cap too: V3 208 W at 2,397 MHz, 33.5 ms / step (7.0 J), IQR 203 W at 2,431 MHz, 25.6 ms (5.2 J): the decode gap
is work (the trellis decode's ALU, above), not clocks. rocprofv3 cannot trace radiance (its queue interception breaks
the AQL backend's first upload: device error) and `RADIANCE_AQL_STAMP=1` doubles the prefill time, so there is no
unprofiled per-kernel split; reducing energy per prefill (fewer passes over the activations: IQR's folded residual add
/ gate, a GEMM that does less per byte) is what would move the 8k number on this card.

## Fractional-K decode (#175 w137, lead note 11-13 1:00 PM CST)

**Why fractional K decoded slower than its bytes.** Not the plan table (read once a wave; no table load in the K
loop) and not divergence (no branch in the loop). bench_twopass at T 1 on box2 (the decode form: KS + DQ + DOT, mul1i8;
a `head` shape added) against the ablation build (`EXL3I_ABLATE_DECODE`: the same loads, no decode ALU): at integer K4 /
K6 the head reads 626 / 628 GB/s with or without the decode (memory-bound, the ALU hidden), at K4.125 / K4.5 / K5.5 /
K6.5 531 / 555 / 506 / 537 GB/s against the ablation's 621 / 617 / 619 / 589, and integer K7 522 vs 620. The ISA of the
K loop (4 tiles, KS + DOT): K4 185 VALU (its byte-aligned perm extraction, 3 loads a tile); fractional KA <= 4 (MODE 0)
218 (four ubfe + two lshl_or a run, 4 loads a tile); KA 5..7 (MODE 2) 351 VALU and 140 VGPRs: both ubfe forms and a
select a state (the offset's side of 16 is a lane's run-time value) and the global-load path (`kBuf` was two-word only:
a 64-bit address add a word); K7 (NW 3) the same load path, 276. The extra work does not hide behind two load stages.
Below K4 every rate is ALU-bound (integer too: K2 249, K3 351 GB/s of codes over the layer; the ablation 369 / 453):
that is the codebook's per-value ALU (w124, above), not the fractional extraction; K2.5 runs at K2 / K3's time.

**Fix (a60dec3, same bits):** states as the packed pairs the hash takes (state 3 is the window's low half; states 0..2
one shift of the 32-bit window at KA <= 4 or one `v_alignbit` of the 64-bit window at KA 5..7; one `v_perm` a pair):
K loop VALU MODE 0 218 → 209, MODE 1 (sub-1) 251 → 226, MODE 2 351 → 218; the three-word runs by buffer loads: MODE 2
140 → 88 VGPRs, K7 276 → 224 VALU; the decode form's load stages 2 → 3 (VGPRs 88 → 91-134, no spills). Tried: KS
splits for the head (KSPLIT 2 / 4: the same gain as PF 3 on the head but other bits, ×1.2-1.3 slower on down / attn_o).
Left: the head at fractional K is still ~8% under integer K's read rate (576 vs 629 GB/s at K4.125); a fourth load a
tile and a tile that straddles cache lines (132 B at K4.125) are what differs from K4 now.

Decode GEMV, T 1 (us; box2 card 2, 3 interleaved rounds, before = qc-exl3-v3 3055e75, after = a60dec3; the layer is
the 48 GDN : 16 attention mean):

| K | gate_up | down | gdn_qkv | gdn_z | attn_qg | attn_k / v | attn_o / ssm_out | head | layer | head GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.5 | 167.0 → 156.4 | 95.0 → 90.2 | 60.4 → 57.7 | 43.5 → 40.7 | 68.1 → 65.0 | 19.9 → 19.5 | 42.9 → 42.1 | 1,023 → 1,016 | 409.8 → 388.5 (×0.948) | 388 → 391 |
| 4.125 | 177.0 → 172.6 | 96.2 → 94.3 | 61.0 → 61.5 | 43.5 → 42.7 | 71.4 → 67.9 | 19.7 → 19.7 | 43.3 → 43.1 | 1,234 → 1,139 | 422.6 → 415.1 (×0.982) | 531 → 576 |
| 4.5 | 188.3 → 183.5 | 101.9 → 102.2 | 65.6 → 63.5 | 45.4 → 45.5 | 74.6 → 72.3 | 19.9 → 19.9 | 45.1 → 45.8 | 1,287 → 1,201 | 447.2 → 441.3 (×0.987) | 555 → 595 |
| 6.5 | 266.2 → 254.5 | 143.5 → 131.6 | 88.0 → 83.6 | 60.2 → 55.5 | 101.0 → 96.4 | 22.9 → 21.6 | 60.2 → 55.4 | 1,923 → 1,865 | 617.8 → 580.8 (×0.940) | 537 → 554 |

Other rates, layer / head: K2 ×0.990 / ×1.046 (PF 3's one loss), K3 ×0.943 / ×0.943, K4 ×0.994 / ×0.995, K5.5 ×0.842 /
×0.893, K6 ×0.973 / ×1.014, K7 ×0.861 / ×0.911. Code read rate over the layer after: K4 482, K4.125 472, K4.5 485,
K5.5 524, K6 524, K6.5 532, K7 543 GB/s (fractional at integer K's rate); the head K4 629, K4.125 576, K4.5 595, K5.5
567, K6 619, K6.5 554, K7 573. T 4 / 8: ±1.5% (K6.5 ×0.93-0.97). A8 prefill path (M 2048): ±1%, same digests.

radisa (step 0's lint) on a60dec3, exl3i units, source + object mode: 0 errors except one MISSING_ALU_WAIT in
RateInt<3>'s KS kernels (ROCm 10.1: source mode only, the built object passes; 7.14: both) -- a false positive of the
suite's operand table: between `v_cmp_gt_i32_e64 s2, ...` and `v_cndmask_b32_e64 ..., s2` sits `s_mulk_i32 s3, 0x60`,
a SALU read of the same pair (SOPK's D = D * imm; LLVM's MIR has the tied use, so its pass counts the read and ends the
pending vector write), while radisa's isadb gives `s_mulk_i32` / `s_addk_co_i32` no read of D. A fix for the suite,
not for our code.

End to end (card 3, w138's #176 server, bench.py, ctx 0 / 256 tokens, server ms per decode step, medians of 3 runs a
round, rows interleaved): V3 K4.125 A8 + R16 K4 head **31.67 → 30.54 ms** (3 rounds), V3 K2.5 + R16 K2.5 head **30.83 →
29.64 ms** (2 rounds); 8k prefill unchanged (2,397 / 2,447 and 2,396 / 2,393 tok/s: order effects, the kernels ±1%).

## Codebook mad and its byte-sum decode GEMV (#177-02, w150, 11-14)

User decision 11-13 7:05 PM CST: mad (IQR's computed code, L16) replaces mul1i8. Commits on `qc-exl3-v3-mad` (off
`qc-exl3-v3` 179220f): 2bdd5ca (codebook + GEMV + tests), 92dcef4 (AQ form xb, pre-pass, PF 2, decode-band actq),
593473b (R16 heads, the GEMV's row rule, rxq / ourq mad stamps). Scripts, raw results and the box2 build recipe: rad-exl3-mix
`v3/bench/mad17702/` (README there).

**The codebook.** q = bytesum(((s · 0xD0474B + 0x2545F491) mod 2^32) & 0x3F3F3F3F) − 126 ∈ [−126, 126], value
fp16(q · 4 K_INV): mul1i8's int8 grid and step, another q a state; exactly what ourq / rxq `--codebook mad` (and w139's
`hash`) write (the oracle equals exl3hashm's numpy values for all 65,536 states). exl3i_decode.h API 3 (`EXL3I_CB_MAD`,
"mad"), encoding version 4 (`mad/h128/k<rate>[/r16]`), interface 4. Every exl3i path takes it: nt_pass (DQ: q = S − 126
as one packed fp16 subtract), T2, the A8 path (wdec8 / FUSE: the int8 index is S + 2 = q + 128, one SAD pair a pair of
values), and the new GEMV. The exl3i importer takes `codebook=mad` and checks the level's stamp (rxq `codebook mad` +
`mad {a, b}`, ourq `mad_a / mad_b`, w139's `hash` + tag); qwen35_exl3's AQ takes mad weights.

**The GEMV (exl3i_mad.hip.h; mad, act a8g128).** `sudot4(h, x · 0x01010101)` = x · bytesum(h): one instruction a weight a
row, nothing unpacked, no WMMA; the −126 x terms are summed once a (row, 128-block, lane half) by whoever quantises the
row, so the K loop never sees them. The tile scan is exl3i's own (nt_pass's lane plan; `madh()` gives a lane's eight
masked hashes in fragment order: one ubfe + v_mad_u32_u24 + AND a state, an alignbit a run; MODE 2 an alignbit + AND a
state), int32 a (row, block, lane half) folded into f32 with the row's activation scale, the lane pair added, × the step,
then nt_pass's epilogue (output rotation, 1/√128, svh, R16 rows) and KS split. The rows: from the producer
(`exl3i_actq` form **xb**, interface 4: x · 0x01010101 int32, scales, bias sums; qwen35_exl3 declares an xb op beside
each AQ site whose consumers are all mad and issues it at 1 row instead of the norm / silu_mul / mul) or, without one
(ssm_out, the head, AQ off), a one-launch pre-pass. With the producer's xb rows it takes any T ≤ 8 (qwen35_exl3 issues
them at 1 row); by itself T 1 (2 below K4: nt_pass's WMMA wins past that, gate_up T 2 K4.125 164 vs 159 µs, T 3 K2.5
232 vs 150) and only where the pre-pass pays: a trunk shape at K ≤ 2 or a head-sized N (layer K2.5 298 vs 280 µs on
nt_pass, the head K4.125 1079 vs 1273). Everything else takes nt_pass (DQ: mad's q is one packed fp16 subtract).

**What made it fast** (gate_up K2, T 1, box2 card 1, burst timing, cold weights):
- the hash as plain C, not inline asm: 138 → 101 µs. Same instructions; LLVM schedules (s_delay_alu) what it can see —
  the asm form issued at ~59% of the VALU peak, the C form ~83% (madgemv's rate). **The same applies to mul1i8's
  `hash_pair` / `exl3i_i4n` / the exl3r4d decode asm** (not changed here: a likely free gain for mul1 / mul1i8 / i4).
- no quantise prologue in the workgroup: compiled in, it held the K loop's registers at high K (K6.5 gate_up 314 → 238
  µs) on top of every workgroup redoing it (GSCR's redundancy); the producer or one pre-pass does it once.
- 2 load stages, not 4 (K2.625 140 → 122, K6.5 312 → 242; ±1% at K2).
Tried, no gain: row-major tile order (the container's) instead of k-major, NPW 1 / 4, activation loads once a block,
fewer weight words a run (the K2 loop is not load-bound once the hash schedules), KS splits other than nt_pass's.

**Decode GEMV µs, T 1** (box2 card 1, R9700 at the 210 W cap, burst of 20 launches, each on its own copy of the weight
(256 MB of copies), median of 12, 2 rounds; mad = the GEMV on the producer's rows as served (`a8q`); layer = 48 GDN :
16 attention mean; GB/s = the layer's trellis bytes / time; the card's practical read rate is ~585 GB/s, mul1i8 K8):

| K | codebook | gate_up | down | gdn_qkv | gdn_z | attn_qg | attn_k / v | attn_o / ssm_out | layer | GB/s | head (pre-pass) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | mad | 91.0 | 52.7 | 34.1 | 24.4 | 38.0 | 13.1 | 25.2 | 228.7 (×0.66) | 416 | 616 |
| 2 | mul1i8 | 144.3 | 80.0 | 50.0 | 33.9 | 56.1 | 15.3 | 35.8 | 344.2 | 276 | 970 |
| 2.5 | mad | 113.5 | 63.4 | 39.9 | 28.6 | 44.3 | 13.7 | 29.1 | 275.3 (×0.77) | 431 | 752 |
| 2.5 | mul1i8 | 150.3 | 83.0 | 51.8 | 35.0 | 58.7 | 15.5 | 36.9 | 357.6 | 332 | 1158 |
| 2.625 | mad | 117.9 | 65.3 | 40.5 | 29.1 | 45.8 | 13.7 | 29.7 | 283.4 (×0.79) | 440 | 793 |
| 2.625 | mul1i8 | 150.7 | 82.9 | 52.0 | 35.0 | 58.8 | 15.4 | 37.2 | 358.0 | 348 | 1167 |
| 3 | mad | 127.1 | 67.3 | 42.1 | 29.7 | 48.2 | 14.1 | 30.0 | 297.4 (×0.85) | 479 | 1006 |
| 3 | mul1i8 | 147.2 | 81.2 | 51.2 | 35.1 | 57.8 | 15.5 | 36.8 | 351.4 | 406 | 1153 |
| 4 | mad | 151.3 | 81.9 | 50.8 | 34.5 | 58.6 | 14.7 | 35.0 | 354.1 (×0.96) | 537 | 1026 |
| 4 | mul1i8 | 156.2 | 84.4 | 53.0 | 36.6 | 61.5 | 15.3 | 37.5 | 368.1 | 516 | 1018 |
| 4.125 | mad | 160.7 | 87.0 | 53.3 | 36.4 | 60.4 | 14.7 | 37.2 | 374.5 (×0.97) | 523 | 1074 |
| 4.125 | mul1i8 | 166.1 | 89.8 | 55.2 | 37.9 | 62.5 | 15.6 | 38.6 | 387.5 | 506 | 1258 |
| 4.5 | mad | 172.0 | 93.2 | 56.8 | 38.4 | 64.6 | 15.2 | 39.2 | 399.3 (×0.97) | 535 | 1156 |
| 4.5 | mul1i8 | 177.5 | 96.2 | 57.9 | 40.3 | 66.8 | 15.8 | 41.1 | 412.9 | 518 | 1272 |
| 5 | mad | 203.0 | 102.7 | 62.1 | 41.1 | 73.4 | 16.2 | 42.0 | 451.4 (×1.00) | 526 | 1615 |
| 5 | mul1i8 | 198.5 | 102.4 | 64.3 | 43.4 | 74.3 | 16.3 | 43.6 | 451.9 | 526 | 1381 |
| 6 | mad | 228.6 | 120.8 | 73.0 | 47.6 | 85.8 | 17.0 | 48.2 | 518.1 (×1.00) | 550 | 1597 |
| 6 | mul1i8 | 226.8 | 119.4 | 74.0 | 49.3 | 86.9 | 17.2 | 49.8 | 518.6 | 550 | 1538 |
| 6.5 | mad | 237.9 | 126.3 | 76.4 | 50.0 | 88.7 | 17.2 | 50.5 | 540.1 (×0.97) | 572 | 1658 |
| 6.5 | mul1i8 | 248.9 | 126.8 | 78.5 | 50.8 | 91.0 | 17.6 | 50.8 | 554.7 | 557 | 1855 |
| 8 | mad | 293.3 | 152.7 | 92.3 | 59.4 | 109.0 | 19.0 | 60.0 | 656.5 (×1.01) | 579 | 2094 |
| 8 | mul1i8 | 288.5 | 150.8 | 91.9 | 59.2 | 109.1 | 19.1 | 60.0 | 649.2 | 586 | 2021 |

From K4 up the GEMV reads at 523–579 GB/s, at or near bytes ÷ bandwidth (89–99% of the card's ~585); below K4 it is
VALU-bound (~4.3 VALU a weight: and 1.3, dot 1, mad 1, bfe 0.75, alignbit 0.25) at 416–479 GB/s — still 15–34% faster
than mul1i8 there. The small shapes (attn_k / v 13–19 µs, 1.6 MB) are fixed-cost bound (launch, KS reduce, epilogue).
Without the producer's rows (the pre-pass: ssm_out, the head) each launch costs ~+4 µs (the `a8` rows in
`data/dkf`). The head at K5 (1615 vs 1381) is the one row where mad is slower: not looked into (heads are K2.5–K4).

**e2e on w139's stage-3 packs** (box2 card 2, rad10-perf core, w138's #176 server settings, bench.py, ctx 0 gen 256 +
ctx 8k gen 32, 3 runs a round, 2 rounds, rows interleaved (b8 = 593473b; the two b7 rows 1 round); trunk mad vs its i8 twin, bf16 head, ssm_a / b bf16):

| pack | decode ms / step | 8k prefill tok/s |
|---|---|---|
| K2.625 i8 (base 179220f) | 32.43 | 2,407 |
| K2.625 mad on today's decode form (nt_pass DQ / DOT, f16 rows; `QWEN35_EXL3_AQ_XB=0 EXL3I_MAD_GEMV=0`) | 27.29 (×0.84) | 2,410 |
| K2.625 mad, the GEMV (pre-pass on every GEMV, no producer's rows; b7) | 27.11 | |
| K2.625 mad, the GEMV on the producers' xb rows (this branch) | **24.96 (×0.77)** | 2,447 (+1.7%) |
| K4.125 i8 | 33.58 | 2,369 |
| K4.125 mad on today's decode form | 31.40 (×0.94) | 2,383 |
| K4.125 mad, the GEMV, pre-pass everywhere (b7) | 32.54 | |
| K4.125 mad, the GEMV on the producers' xb rows | **30.36 (×0.90)** | 2,390 (+0.9%) |
| IQR-V4 iq2r, drafter off (10.89 GB, ~2.1-bit trunk) | 20.29 | 2,597 |

**Correction (w152, 11-14, #179; lead note 11-13 11:00 PM CST).** The mad rows above (b7, b8) ran on a race: KS's arrival
counters were one block a stream, and the engine starts z's GEMV beside qkv's and k / v beside qg (`rad_kernel_concurrent`;
on mad's xb rows nothing orders them), so the overlapped launches counted into each other's — mad decode emitted
non-finite logits after a few tokens (text "!!!!"; greedy decode with logprobs: 0 of 8 requests finish). Fixed by fdd480c
(counters keyed by (device, stream, output)). Re-measured on fdd480c (box2 card 1, same settings, 2 rounds interleaved,
server ms / decode step at ctx 0): K2.625 i8 32.53, mad on nt_pass 27.27 (×0.838), the GEMV with its pre-pass 27.17, the
GEMV on xb rows **25.39 (×0.781**, was 24.96 / ×0.770); K4.125 i8 33.58, nt_pass 31.44 (×0.936), pre-pass 31.44, xb rows
**30.61 (×0.912**, was 30.36 / ×0.904); with #179's step (qc-exl3-v3-step) 24.82 (×0.763) / 29.95 (×0.892). The xb rows'
gain over f16 rows is −6.9% / −2.6% (was −8% / −3%); 8k prefill as above. The decode-path KLD (EXL3I_MAD_SPLIT) **stands**:
a race-free build (every launch ordered) reproduces the rows above byte for byte at both K (the overlapped prefill launches
did not collide there). On fdd480c that knob's 8-row pieces each key their own counters (distinct y offsets) and use up the
64 slots, so later pieces run unsplit (deterministic, another f32 order: +0.065% ± 0.29% / −0.22% ± 0.55% vs A8): key the
pieces by the op's output base before relying on it again. Data: rad-exl3-mix `data/step179/remeasure177/`.

**Most of the gain is the codebook, not the byte-sum form.** mad on today's nt_pass (f16 rows) already takes −16% /
−6% of the step: its decode (one v_mad_u32_u24 + AND a state, q = S − 126 one packed subtract) is cheaper than
mul1i8's (packed hash + closed-form grid rounding) and is plain C. Per op, profiled (`--profile-ops`, K2.625), the GEMV
on xb rows and mad's nt_pass are equal (28.39 vs 28.30 ms / step; gate_up 7,788 vs 7,681 µs, down 4,248 vs 4,291, qkv
2,008 vs 2,065; ssm_out 1,721 on the pre-pass vs 1,567). The further −8% / −3% e2e shows only unprofiled, at the 210 W
cap (w137: decode runs at the cap): the GEMV does the same work with less energy (int8 rows, one instruction a weight),
so the card clocks higher. It costs the int8 activations at decode (below: +0.3% / +1.8% KL against f16 rows).

Decode per op (`--profile-ops`, 320 steps; profiled µs / step, K2.625): gate_up 7,785 vs 9,867, down 4,248 vs 5,600,
qkv 2,010 vs 2,627, z 1,478 vs 1,811, attn_qg 749 vs 989, attn_o 501 vs 627, ssm_out 1,722 vs 1,892 (pre-pass), the
decode AQ producers (actq xb, rmsnorm / silu_mul / mul folded in) 1,661 vs the ops they replace 1,685; total 28.4 vs
33.5 ms; IQR iq2r 23.0 ms. What IQR still has that we don't (iq2r profile, `v3/bench/mad17702/data/prof`): residual add
folded into the norm (1,025 µs vs 10), q / k norm + rope + gate + kv_store as one op (~730 vs 120), GDN qkv + z as one
GEMV (inz), gate_up with silu · mul in its epilogue, a vocabulary-tiered head (3,955 µs bf16 head here vs 1,280).

**Numerics.** Prefill (A8 path) is the reference dequant path byte for byte: whole-model KLD rows (golden-search27c-c,
67,932 positions) of the mad-labelled stage-3 packs on this branch = w139's `home-hq` (EXL3I_BENCH_IQRH 1) on the
mul1i8-labelled ones, K2.625 and K4.125; the i8 twins' rows = the base build's. Decode: the GEMV computes the A8 path's
integers (its int8 rows are the A8 quantiser's bytes) in another f32 order; whole-model KLD through it (EXL3I_MAD_SPLIT
=1, every prefill launch as 8-row GEMV pieces) against the A8 path, paired, gen-masked, per-document bootstrap SE:
K2.625 +0.008% (SE 0.13%), K4.125 −0.14% (SE 0.40%). Against f16 rows (nt_pass, today's decode numerics): +0.30% (SE
0.12%) / +1.8% (SE 0.7%) — the int8-activation trade the byte-sum form needs (IQR's too); decode now matches what the
prefill KLD measures.

**w151's ladder levels** (#177-03 stage B, `/data/q27/mad-packs`, codes stamp 635bac02): packed with this branch's
importer (`codebook=mad`; rad10-perf's rad-convert with this branch's plugins: the older home-pack's qwen35_exl3 refuses
encoding 4) and a8g128 act keys: whole-model KLD rows identical to their shim packs on w139's home-hq at K2.5
(0.681289) and K4 (0.307438).

**Tests** (box2, ROCm 10.1, build b8 = 593473b): host (`exl3i_twopass_host_test` M: codebook closed forms on all states, encodings, rows,
dispatch), `exl3quant_test` (mad stamp forms, import = oracle), `exl3i_gpu_test` (A: decode bit for bit at 64 rates ×
4 codebooks, madh's q at every position), `test_twopass` (A / B with mad: planes byte for byte, y = the device
emulation; M: the GEMV at every rate vs the exact GEMM of the quantised operands ≤ 2.8e-7 of max |y|, the trunk shapes
+ head × 11 rates × T, rows bit for bit independent, KS repeatable; M4: the row rule), `test_aq` (form xb rows, GEMV
with / without aq byte-identical, pre stages at 1 / 3 / 8 rows vs libr4d), `test_r16` (mad at a8g128: GEMV and f16
path at the R16 rows), `test_ks` (mad), `test_a4`: all PASS. mul1i8's device code is instruction for instruction the
base build's (only the argument block's size: +24 B).

## Refuted

- wdec8 / GEMM overlap on a second stream (the chunk cut in sub-chunks, events between): ×1.03–1.40 slower — the
  sub-chunk GEMMs lose efficiency and the waits cost more than the hidden decode while the GEMM fills the card.
- Decode prefetch depth 4 / 8 (no gain / slower: registers), VOPD on (no gain), the extraction-free hash (no gain).

## Left (proposals, not done)

- **Decode GEMV efficiency at K4** (the gap to IQR-V4: ~6 ms of trunk GEMV a step): the small weights (gdn_z, attn_o,
  ssm_out, attn_k / v) run at 140–385 GB/s against gate_up's 540: per-launch fixed costs (KS's last-arriver reduce, the
  epilogue's Hadamard, the GSCR rotation) and launch gaps; a layer-level fused decode (one launch for a layer's GEMVs)
  or a GEMV whose decode needs fewer VALU per value (int8 activations: a precision trade, judged relative).

- **AQ for ssm_out** (behind the GDN gated norm, a core op) and the head: still quantise for themselves.
- **FUSE under ROCm 7.14**: done (#175, e776f32): the hang was the claim loop's thread-0 branches; FUSE is on by default.

Left after #177-02 (mad, w150):
- **Decode at 3..16 rows on mad:** nt_pass (f16 WMMA, DQ) — already ×0.85 of mul1i8 there; an int8-WMMA tile that
  decodes each unit into fragment order (IQR's iqr_m16 form) on the xb / int8 rows would make T 2–16 flat in T at
  int8 cost (speculative verify, batches).
- **Producers for ssm_out (the GDN gated norm, a core op) and the head (final norm):** they take nt_pass / the GEMV with
  its pre-pass; an xb site there would give them the GEMV without the second launch.
- **The asm decode in mul1 / mul1i8 / mul1i4** (`hash_pair`, `exl3i_i4n`, exl3r4d's decode_run): the same scheduling
  loss mad's asm hash had (×1.37 at K2); plain C or builtins would likely speed the old codebooks' decode.
- **Decode numerics are now a knob:** a8g128 weights decode on int8 rows (the GEMV, +0.3% / +1.8% KL at K2.625 /
  K4.125 against f16 rows) — `EXL3I_MAD_GEMV=0` + `QWEN35_EXL3_AQ_XB=0` keep f16 rows at decode at −8% / −3% e2e. A
  per-weight choice would need an act mode that splits prefill A8 from decode (not added).
- **Non-GEMV step costs IQR folds** (residual add 1.0 ms / step, q / k norm + rope + gate + kv_store, GDN qkv + z as
  one GEMV, gate_up silu · mul epilogue at prefill, a tiered head): w152's scope.
- The mad head at K5 (1,615 vs 1,381 µs on the pre-pass) is the one rate where it loses: not examined.

