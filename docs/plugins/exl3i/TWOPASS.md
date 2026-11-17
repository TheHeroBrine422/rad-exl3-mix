# exl3i two-pass prefill (EXL3-V3, design option C)

Branch `qc-exl3-v3-twopass` (from `qc-exl3-v3` 008e8ff; merged with the A4 path in `qc-exl3-v3-merged`, A4.md's last
section). **Runtime:** changes **prefill** compute (the A8 path at M ≥ 512 for weights whose `exl3.act.<weight>` is an
A8 mode); decode (M < 512) and every f16 weight are unchanged.

## What runs

| act (metadata) | codebook | row | M < 512 | M ≥ 512 |
|---|---|---|---|---|
| f16 (or absent) | mul1, mul1i8 | `exl3i_gemm_nt` | nt_pass | T2: f16 fragment plane + 128×256 f16 WMMA GEMM (unchanged) |
| a8t, a8g128 … a8g1024 | mul1i8 | `exl3i_gemm_nt_a8` | nt_pass (byte-identical to the f16 row) | A8: actq → per chunk wdec8 → i8a8 GEMM |
| A8 mode | mul1 | none: refused (no row; launch `RAD_E_UNSUPPORTED`) | | |

A8, per weight and prefill step (`exl3i_a8.h` has the details):
1. `actq` (`exl3i_a8_prep.hip`): x·suh (suh.1 for gate_up's up half) rotated by `rotate128_math` (the f16 path's
   values bit for bit), int8 with one f32 scale per (row, G inputs).
2. `wdec8` (same file): the trellis decoded once into radiance's i8a8 fragment plane (`r4d_fp8_frag.h` order), each
   mul1i8 value as its int8 grid index (value / 4 K_INV, exact). Column chunks of ≤ 192 MB of plane; every trunk shape is
   one chunk.
3. GEMM (`exl3i_a8_gemm.hip` + `exl3i_8a8.hip.h`): radiance v1.17.0's `gemm_8a8_prefill_body` (RB 32 biased-int fold,
   weight scale 1, copied with attribution: plugins link only the ABI headers) with EXL3's store: the block's 128 rows
   are one output-Hadamard block, rotated back in the epilogue (in-lane, lane ^ 16, across the wave's tiles, across
   the four weight waves through the staging LDS), × 4 K_INV / √128 × svh, bf16 / f16 / f32.

Why no new f16 tile: radiance has no f16-weight prefill tile (its 16-bit tiled GEMM is bf16-in and would round our
f16 codebook values); T2 already is the two-pass f16 plane with a 16-bit WMMA tile (85% of sustained f16 peak).
Its plane now has a C reference and a byte check (`test_twopass` B).

**mul1i8's decode (w29, `exl3i_rate.hip.h`).** Both forms come from the byte sum S of mul1's hash (exl3r4d's state
extraction and `hash_pair` unchanged) and are bit-exact against the oracle for all 65,536 states (host test A, device
`exl3i_gpu_test` A and `test_twopass` A/B):
- int8 (wdec8, `decode8`): `n = max((S + 3) >> 2, 1)` per u16 half (`v_pk_lshrrev_b16`, `v_pk_max_u16`), then one
  `v_perm_b32` and `xor 0x80808080` per 4 values (q = n − 128 = max((S − 509) >> 2, −127)).
- f16 (nt_pass, T2's wdec, `decode`): `fp16(1024 + n) = fma(fp16(1024 + S), 1/4 + 2^-12, 768)` (one rounding to the
  integers of [1024, 2048)), `q = max(· − 1152, −127)`, `× 4 K_INV`, then the sign bit for S ≤ 510 (`v_pk_sub_u16`,
  `v_and_or_b32`): mul1's value is negative there, so the oracle's rint gives −0 at q = 0 (S 509, 510). Without that
  sign fix (bench flag `EXL3I_I8_CLOSED=2`) the values are equal and every output check passes, but T2's f16 plane
  differs in a zero's sign (test_twopass B).
- Flags (compile time, every TU alike): `EXL3I_I8_CLOSED` 1 (default) / 0 (the first form: mul1's f16 value rounded
  through f32, and wdec8's `exl3i_a8_wq` back to an int) / 2 (bench only, above). Bench only, never served:
  `EXL3I_BENCH_IQRH=1` (mul1i8's slot decodes IQR-V4's hash codebook, `(s·A + B) & 0x3F3F3F3F`, byte sum − 126, for
  the #100 speed comparison; not our values) and `EXL3I_ABLATE_DECODE=1` (words straight into the fragment: the floor).
- VALU a value (gfx1201 assembly of one tile's decode above the ablated floor, K 2 / 3 / 4 / 5): int8 15.3 / 13.6 /
  13.3 / 13.6 before → 7.8 / 6.1 / 5.8 / 6.1; f16 9.8 / 9.4 / 8.9 / 9.4 → 8.3 / 7.9 / 7.4 / 7.9. Timing: rad-exl3-mix
  `docs/design/kernel-opts.md` (#100 results).

Arena (`exl3i_a8_plan`): int8 activations [mats][M][K], f32 scales [mats][M][K/128], one chunk's int8 plane; shares
the scratch's start with the f16 region used below 512 rows. Largest answer over the trunk shapes at M ≤ 4096: 225 MB
(down at 4096, two activation planes, because the scratch hook sees no operands and cannot tell a stacked weight);
gate_up at 2048: 191 MB (f16 row at 2048 for down: 328 MB). Table: `exl3i_twopass_host_test` section F.

Knobs for benches/tests only (read by both the scratch hook and the launch): `EXL3I_A8_MIN_ROWS` (threshold),
`EXL3I_A8_CHUNK_MB` (chunk budget).

## Tests

- `tests/exl3i_twopass_host_test.c` (no device; `ctest -L host`): mul1i8 on the int8 grid (all 65536 states); the
  plugin oracle vs rad-exl3-mix `v3/infer/ref/exl3i_ref.c` (with `-DEXL3I_XREF_C=`); int8 plane vs the oracle's W_hat
  and vs radiance's own `r4d_fp8::frag_src`, all 64 rates; wdec8's and T2-wdec's lane arithmetic emulated byte for
  byte; `rotate128_math` transcription vs `exl3_had128`; quantiser invariants; A8 arithmetic vs x·Wᵀ (0.63% rel rms at
  g128); arena at M 1/511/512/513/2048/4096 for all 9 trunk shapes; `nt.c`/`rows.c` dispatch with stubbed launchers
  (54 trunk launches, chunking, threshold knob); refusals (act spellings, A8 on mul1, group ∤ K, rows past the scratch,
  stacked N % 256, encodings); the device-arithmetic emulation of the A8 GEMM near its exact fold; mul1i8's closed forms
  emulated for every state. PASS, 59 checks (58 without libr4d in the tree: no `frag_src` cross-check).
- `tests/test_twopass.hip` (device; **PASS, 808 checks**, gfx1201 card 0, 2026-11-05): the op's scratch read back and
  compared with `tests/exl3i_twopass_ref.c` byte for byte (int8 weight plane, int8 activations and scales at every rate;
  T2's rotated rows and f16 plane), every A8 output byte for byte against `exl3i_ref_a8_gemm_dev` (the device's fold,
  bias flushes and epilogue order emulated in f32) and within rel rms of the exact GEMM, then the v3.md §1c.4 matrix (real rows 1…2048 in a 2048 band, three store dtypes, bf16/f16 a with
  padded pitch, GDN 3-D input, stacked seam, a8t/a8g256/512/1024, K0.125…K8, three chunks, two scratch sizes and a
  replay byte-identical, A8 row = f16 row below 512, threshold knob), refusals. Fails on 0 checks.
- `tests/bench_twopass.hip`: f16 row vs A8 row vs A8-forced at the trunk shapes, M 2048 or the 16…4096 sweep, plus the
  A8 passes alone (actq / wdec8 / gemm); layer sums (GDN, attention, 48:16 mean).
- `tests/twopass_build.sh`: the same builds without the CMake tree (used for the gfx1201 compile check off-card).

## Card run (lead)

```
cmake -S plugin/exl3i -B build-tp -DCMAKE_PREFIX_PATH=<radiance prefix> -DEXL3I_XREF_C=<rad-exl3-mix>/v3/infer/ref/exl3i_ref.c
cmake --build build-tp -j8 && (cd build-tp && ctest -L host)
SO=build-tp/radiance_home/kernels/exl3i.so
build-tp/exl3i_gpu_test $SO                        # the f16 row still passes (sections A-D, F; E, the comparison with
                                                   # exl3_r4d.so, went with plugin/exl3r4d)
build-tp/test_twopass $SO                          # [--quick]
build-tp/bench_twopass $SO --m 2048 --csv tp-2048.csv
build-tp/bench_twopass $SO --sweep --modes f16,a8x,phases --csv tp-sweep.csv
```

## Card results (gfx1201 card 0, power cap 250 W, ROCm core-10.1 clang 24.0.0git; w28, 2026-11-05)

First run: test_twopass 366 of 711 checks failed. Causes and fixes (commits 886b46e, the next):
- **Kernel, compiler bug** (A, C): actq's `amax4` / `quant4` did `__builtin_bit_cast(v2h, v[1])` on an ext_vector `v2u`;
  this clang lowers a bit cast of an ext_vector element from the vector's base, so `v[1]` read `v[0]`. Each lane stored
  `[q0, q1, q0, q1]` (the rotation's features 2, 3 dead in the IR) and the absmax saw half the group: y ~1 rel rms off.
  Fixed by copying the dwords into scalars first. **Do not bit-cast an ext_vector element** (`v[i]`, `v.y`) in device
  code built by this compiler; a plain array element or a HIP struct vector (`uint4`) compiles right.
- **Test** (B, and A's byte check): the reference rounded rotate128's final `x 1/sqrt(128)` twice (f32, then fp16); the
  device's `(_Float16)(v * RS)` is one `v_fma_mixlo_f16` (one rounding) even under -ffp-contract=off: 1 ulp on ~4e-5 of
  values. T2 unchanged; the reference rounds once.
- **Test** (C, a8t): the worst element against the exact GEMM of the quantised operands passed 6e-3 of rms by 1.05x /
  1.18x at K 1024 / 2560 (the RB 32 fold's f32 rounding grows with K and the scale's width; rel rms 1.7e-3, limit 4e-3).
  The A8 output is now checked byte for byte against the emulated device arithmetic, and against the exact GEMM by rel
  rms only (the worst element is printed).
- bench_twopass's own copies of the passes (the phases rows) now build with the plugin's flags (rad_hipify).

After: test_twopass PASS (808 checks), exl3i_gpu_test PASS (E: byte-identical to an ABI 35 exl3r4d at K1-8), host test
PASS. GEMM register use under this compiler: 240 VGPRs, 20 B/lane scratch, occupancy 6 (as under Ubuntu LLVM 21).

Timing, `bench_twopass --m 2048` (K4 mul1i8, a / y bf16, median of 20; the passes timed alone):

| shape | f16 (T2) µs | A8 µs | f16 / A8 | actq | wdec8 | gemm |
|---|---|---|---|---|---|---|
| gate_up | 6749 | 4483 | 1.51 | 172 | 541 | 3948 |
| down | 3499 | 2190 | 1.60 | 294 | 242 | 1798 |
| gdn_qkv | 1941 | 1222 | 1.59 | 88 | 142 | 1097 |
| gdn_z | 1177 | 787 | 1.50 | 93 | 70 | 675 |
| attn_qg | 2319 | 1457 | 1.59 | 90 | 176 | 1317 |
| attn_k | 252 | 206 | 1.22 | 76 | 19 | 120 |
| attn_v | 250 | 206 | 1.21 | 76 | 18 | 121 |
| attn_o | 1185 | 795 | 1.49 | 96 | 62 | 645 |
| ssm_out | 1182 | 798 | 1.48 | 110 | 71 | 664 |
| **layer, 48 GDN : 16 attn** | **14474** | **9444** | **1.53** | 768 | 1064 | 8124 |

Run-to-run spread: the `a8x` rows (the same path at 2048) differ from `a8` by up to 7% per shape, 2% on the layer; the
sweep's f16 layer at 2048 came out 7% slower (15527 µs) than this run's. The passes alone sum 5% over the A8 launch.
The first (buggy) run timed the same within 1-3% except actq (now +25-30%: four values a lane, not two).

## Open questions

1. **Decode pass vs fused decode.** wdec8 is its own launch (the plane written once, then read by the GEMM). Its
   bytes (gate_up K4: 178 MB written + 89 MB read) are serial time; IQR's `gemm_iqr_tiled` decodes inside the GEMM's
   `begin` hook (claim counters, overlapped with other blocks' K loops). `bench_twopass --modes phases` gives the split;
   if wdec8 is a sizeable share at 2048 rows, port IQR's claim scheme (needs a zeroed counter block per launch in our
   scratch: one small memset). **Measured:** wdec8 is 11% of the A8 layer at 2048 rows (1064 of 9956 µs, passes
   alone), actq 8% (768 µs; attn q/k/v and gdn qkv/z quantise the same input once each: one shared actq would save ~0.1 ms of
   the layer mean, ~1%).
2. **Groups above 128** fold every 128-K block with a repeated scale (same speed as g128). Faster wide groups need a
   fold every G/128 blocks (the bias window holds 256 inputs; 512+ needs the int→float conversion). Recommend shipping
   a8g128 only unless the knapsack wants a8t/a8g256+.
3. **Registers.** The GEMM compiles at 240 VGPRs with a 20 B/lane spill under Ubuntu LLVM 21.1.8, the same as radiance's
   own RB 32 i8 tile under that compiler (its RB 8 form: 193, no spill). Check `-Rpass-analysis=kernel-resource-usage`
   under the card's ROCm compiler; RB 8's form is equivalent here (weight scale 1) if RB 32 spills there.
   **Measured** under the card's ROCm clang (24.0.0git): the same, 240 VGPRs, 20 B/lane scratch, occupancy 6; RB 8 not
   tried.
4. **Threshold 512**: the sweep's `a8x` vs `f16` decides. **Measured** (`--sweep --modes f16,a8x,phases`, the f16
   row's real path: nt_pass below 512, T2 from 512): A8 over f16 on the 48:16 layer 1.86 / 1.57 / 1.56 / 1.03 / 0.65 /
   0.53 / 0.49 at M 16 / 32 / 64 / 128 / 256 / 384 / 511, 0.62 at 512 and 0.60-0.64 up to 4096. Parity near 128 rows;
   A8 is ~35% faster from 256. Suggests EXL3I_A8_MIN_ROWS 256 (or ~192; not timed between 128 and 256); the A8 path at
   16 / 100 / 256 rows is in test_twopass C (byte-exact). attn_k / attn_v are faster on A8 at every M. Aside: the f16
   row at 511 rows (nt_pass) is 25% slower than at 512 (T2), so T2's own threshold (EXL3R4D_T2_MIN) may be high on this
   card at 250 W too.
5. **The scratch hook** cannot see suh.1 (no operands at declare): two activation planes wherever N % 256 (+11 MB at
   4096 rows). A schema parameter would fix it (interface bump).
6. ~~Not done: the i4 grid, A4, the gated activation quantiser.~~ **Done in A4.md (w34):** codebook mul1i4 (f16, A8
   and A4 rows), the A4 path (int4 plane, int4 activations g64 / g128, an iu4 tile). Still not done: the gated
   activation quantiser in gate_up's epilogue. Found there: T2 takes a stacked weight only with its seam on T2's
   256-column block (N/2 % 256; every trunk gate_up has it), else nt_pass; before, N/2 % 256 != 0 gave wrong up rows.
7. **`test_twopass --quick` fails C a8t K 2560** (3 checks: int8 activations and scales vs the reference, then y vs
   the emulation in 172 of 163,840 values) with the first and the closed-form decode alike (w29, 2026-11-05); the full
   run passes that case (808 of 808). The activation side does not depend on the decode, so this looks like a tie in
   actq against its reference at the quick run's random data. Not investigated.
