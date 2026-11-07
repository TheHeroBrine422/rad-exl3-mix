# V3 kernel optimizations to write (from IQR-V4's kernels, 2026-11-05)

Source: a read of radiance 1.17's IQR-V4 kernels (`libr4d/r4d_gemm_iqr.hip`, `r4d_iqr_dec.h`, `r4d_gemm_8a8_prefill.h`,
`r4d_gemm_fp8a8.hip`; Apache-2.0) against our two-pass exl3i path (fork branch `qc-exl3-v3-twopass`,
`plugin/exl3i/TWOPASS.md`). Gains are estimates unless marked measured. Status per item; results get added here.

**Branch status (2026-11-06, 6:10 PM CST):** the two-pass A8 path (`qc-exl3-v3-twopass`) and the A4 path
(`qc-exl3-v3-a4`: codebook `mul1i4`, act `a4g64` / `a4g128`, int4 plane, iu4 tile; `plugin/exl3i/A4.md`) were merged
on `qc-exl3-v3-merged` (fast-forwards, no conflicts; host and device tests pass, the f16 row byte-identical to
`qc-exl3-v3`'s). w53's i4 importer (`qc-exl3-v3-i4imp`, the quantizer's i4 output → `mul1i4` containers; byte-exact
vs the quantizer's i4 decode at every rate, real 27B i4 tensors, gfx1201 `test_a4` 1093 PASS) rebased cleanly on top,
and **`qc-exl3-v3` was fast-forwarded to 42960d5** (+12 commits, no history rewrite; the radiance PR tracks
`qc-exl3-v3`).

## Prefill (A8 path, M ≥ 512): in order

| # | what | IQR reference | est. gain | effort | status |
|---|---|---|---|---|---|
| P1 | **Decode inside the GEMM tile** instead of a separate decode launch: blocks claim 16-row weight tiles of their 128-row output tile through counters (claimed / decoded / finished), decode them, then run the K loop. Counters in our scratch, zeroed by actq (exl3i promises `rad_kernel_concurrent`: no state kept between launches) | `r4d_gemm_iqr.hip:670-768` | −6–8% of A8 GEMM time at M 2048, −25–30% at 512 | 1–2 days | to do after the first timing |
| P2 | **Integer int8 decode, bit-exact:** `q = max((S − 509) >> 2, −127)` from the hash's byte sum S (checked for every S; only state 0 needs the max), instead of value → f16 → f32 → int (~13 ops a value today) | — (w23's closed form) | halves the decode pass's ALU; what lets P1 hide the decode below M 2048 | hours | **done** (w29, fork `716b811`, bit-exact): wdec8 −5% at K4 (large shapes bandwidth-bound, unchanged), A8 path unchanged; see #100 results |
| P3 | **One activation-quant pass per sibling group** (gate_up reads its input twice today; q/k/v, GDN qkv/z) | `rmsnorm_had_quant_i8` pattern | 1–3% | small | to do |
| P4 | **Fold suh upstream where exact** (down's into up's svh; ssm_out's into the gated norm's weight), so each input needs one fused pass (activation, rotation, int8) like IQR's `gated_had_quant_i8`. Changes numerics (suh is a real per-channel scale): KL check needed. Tying suh across a norm's consumers at requant would let radiance's stock `rmsnorm_had_quant_i8` feed us directly (needs a requant: decide before the ladder) | `r4d_rmsnorm_had_quant_i8` | 2–4% | medium | to decide (requant part before the ladder) |
| P5 | **Register spill check:** the GEMM is 240 VGPRs + 20 B/lane spill under LLVM 21; radiance measured 25–38% slowdowns when this tile spills in the K loop. If the card's ROCm compiler still spills, use radiance's RB 8 fold (equivalent with weight scale 1; 193 VGPRs, no spill) | `r4d_gemm_8a8_prefill.h:24-84` | up to 25–38% if it spills | 30 min | first card run |
| P6 | Split finisher by arrival counter (last block sums partials in fixed order) instead of a reduce launch; one launch per GEMM | `r4d_gemm_fp8a8.hip:605-640` | ~2.8 µs a dependent dispatch, ~0.4% of a 2048 step | small | with P1 |

Already in our tile copy: the biased-int fold (`0x4B400000`), the fragment-plane layout, raster grouping, LDS staging
and barriers. Not applicable: IQR's LDS transposes, tables and scale-form tricks.

## Decode (M = 1–16, exl3r4d GEMV / narrow tile): later

| # | what | IQR reference | est. gain | status |
|---|---|---|---|---|
| D1 | **Split K across blocks:** at N = 5120 we launch 40 workgroups for 64 CUs. Blocks = (128-column group × K slice), the last-arriving block sums partials in fixed order, then the output Hadamard + store | IQR: 5120×6144 GEMV 42.3 → 30.0 µs (measured, theirs) | −20–30% on those shapes | later |
| D2 | **Integer byte-sum f16 decode** for mul1i8 (bit-exact; removes the f32 detour every V3 tensor pays at M = 1) | P2's form | — | **done** (w29, `716b811`): GEMV layer −7.0 / −6.4 / −4.8% at M 1 / 4 / 16 (K4) |
| D3 | **Fused producers:** gate_up's last-arriving block writes silu·up·suh_down already rotated for down | `r4d_gemm_iqr.hip:55-107` | — | later |
| D4 | Hash dotted with the activation byte broadcast by `v_perm` (sum and product in one `sudot4`), scales through SGPRs | `r4d_gemm_iqr.hip:170-233` | part of IQR's 42 → 30 µs | moot: the hash codebook (#100) was dropped (2026-11-06) |

## Format: the hash codebook (#100) — dropped (user, 2026-11-06)

**Decision (2026-11-06 6:20 PM CST): dropped; `mul1i8` (with the closed-form decode) stays the V3 codebook.** The real
run (#100c, w48, `v3/e3/sub1/r100c.md`; 0.8B, 2 seeds, fake-quant scoring): KL ×1.026 / ×1.040 / ×1.016 at K2 / 3 / 4
on search (verify ×1.025 / ×1.029 / ×1.000) vs the bar ≤ 1.01, ≈ +0.011…+0.025 bpw at equal quality; on gfx1201 decode
GEMV +15.6% / +17.6% / +12.6% (bar ≥ 15% at every K), prefill ±1%. It missed both bars; the per-tensor proxy (+0.8–0.9%)
underpriced its real KL cost 3–4×. The settle.py re-check (w46, `v3/e3/settle/reval.md` item 2) agreed with w27's proxy
table within ±0.3%. Kept below for the record.

**[superseded 2026-11-06] Status 2026-11-06 (morning): still open.** Two checks before the user decides: w27's proxy result is being re-run on the
exllamav3-faithful CPU proxy (settle.py; the old proxy distorted other results through its scale handling), and a
small real run (hash codebook behind a quantizer flag, 0.8B at K2 / K3 / K4, two seeds, scored on the calib goldens,
plus a gfx1201 decode microbench vs mul1i8 today and the closed form). The numbers below are from before both.
IQR's computed value is `sudot4((umul24(state, A) + B) & 0x3F3F3F3F, 0x01010101) − 126` (~3 ops a value incl. the
hash, int8-native, A / B stored per tensor) vs EXL3 mul1i8 (32-bit multiply, 4-byte sum, scale, int8 round). A
different codebook means a requant of every level (~200 GPU-hours if done after the ladder), so it is tested now on
the CPU proxy (rule: `v3/e3/sub1/README.md` "#100" + amendment: a quality cost is traded against the measured decode
/ prefill speedup — the user's call from a trade-off table — unless it is ≥ ×1.05). If adopted: exllamav3 codebook option, encoding tag
`cb=iqrh`, plugin decode paths, D4.

**Results (w27, CPU proxy, 20 tensors × 12 K):** iqrh / mul1i8 proxy error 1.011 (K0.25), ≈1.00 (K1–1.5), 1.012–1.014
(K2–3.5), 1.027 (K4), 1.035 (K5), 1.054 (K6), 1.065 (K7); equal-quality cost +0.008–0.010 bpw at K2–3.5, +0.02–0.06
above. A searched global A (0xd0474b) is ~1.01 at K2–4, 1.02 at K5–6. Proxy → KL has been ~1:1 to ~1.5–2× for codebook
changes. Decode ops a value (source count): mul1i8 today 14–17 (int8) / 9–11 (f16); **mul1i8 closed form 4.5 / 5.0
(no requant)**; iqrh 3.75 / 3.5. Card microbench of all three: w29. Decision: the user's, from both.

**#100 results (w29, card: gfx1201 card 0, 2026-11-05).** Three decodes on the same exl3i kernels (one build per
decode, `exl3i_rate.hip.h` flags; iqrh drops IQR's hash into mul1i8's slot, speed only): mul1i8 today (`EXL3I_I8_CLOSED=0`),
mul1i8 closed form (the default now, bit-exact: host test on every state, `exl3i_gpu_test` and `test_twopass` 808/808
PASS), iqrh. Runtime: changes **decode** (GEMV) and the prefill decode pass; no format change.

VALU ops a value, gfx1201 assembly (one tile's decode above a build with the decode ablated, so the state extraction
is included; `local/workers/w29/dis/`):

| K | int8 (wdec8): today / closed / iqrh | f16 (GEMV, T2): today / closed / iqrh | iqrh vs closed: int8 / f16 |
|---|---|---|---|
| 2 | 15.3 / 7.8 / 6.5 | 9.8 / 8.3 / 5.5 | −16% / −33% |
| 3 | 13.6 / 6.1 / 4.9 | 9.4 / 7.9 / 5.1 | −20% / −35% |
| 4 | 13.3 / 5.8 / 4.9 | 8.9 / 7.4 / 5.0 | −15% / −32% |
| 5 | 13.6 / 6.1 / 4.9 | 9.4 / 7.9 / 5.1 | −20% / −35% |

Against w27's source counts: today's form is cheaper than estimated (clang fuses the f16→f32 rounding into `v_fma_mix`,
`v_rndne`, `v_med3`: 4 ops a value). The f16 closed form spends 1 op a value on the sign of zero: the oracle's rint gives
−0 at S 509, 510, so bit-exactness needs the sign of S ≤ 510. Dropping it (`EXL3I_I8_CLOSED=2`, 6.4 at K4) gives equal
values. Every output check still passes on the card (`exl3i_gpu_test`; `test_twopass` fails only T2's plane bytes), and
a ±0 weight only flips the sign of a zero product, so outputs should not change; not compared byte for byte across builds.
The default keeps the fix. iqrh's hash is 2 ops a value like ours (`v_mad_u32_u24` + AND vs `v_pk_mul_lo_u16` +
`v_mad_u32_u16`); its gain is the rounding step and the f16 tail.

Card timing, K4, layer mean (48 GDN : 16 attention), median of 20 (the min-based change agrees within ~1 point):

| path | today µs | closed form µs | closed vs today | iqrh |
|---|---|---|---|---|
| (a) wdec8 pass alone, M 2048 | 1061 | 1009 | −5% (gate_up 544 → 540, down 241 → 240: bandwidth-bound at ~495 GB/s; attn_o 61 → 43, gdn_z 70 → 42) | not measured (see below) |
| (b) full A8 path, M 2048 | 8919 | 8972 | +0.6% (noise; GEMM 87%) | not measured |
| (c) GEMV (nt_pass) M 1 | 659 | 613 | −7.0% | not measured |
| (c) GEMV M 4 | 703 | 658 | −6.4% | not measured |
| (c) GEMV M 16 | 910 | 867 | −4.8% | not measured |

**Not measured:** iqrh, the ablated floor (so the decode share of each kernel), and K2/K3/K5. From 8:00 AM CST a
process outside our container (TP2, 23 GB on each card, presumably the user's) kept card 0 at 100% GFX, also while we
held `card0.lock`. Every run from then on was 1.5–3× slower and inconsistent (the ablated build came out slower than the
decoding ones), so all of those runs are discarded. A guarded retry (it runs only if card 0 samples idle for 5 s
while we hold the lock) found a lull at 8:29 AM CST, but the load came back within seconds (K4 attn_k at M 1: 110–170 µs
vs a clean 60). The K4 today / closed-form pair above ran just before 8:00, and its "today" numbers match w28's run of the same code (wdec8 gate_up 541, GEMV M16 attn_k 89 / gate_up 290 µs). To finish
on an idle card (~3 min, one hold): `/data/tp/w29/gbench.sh` (builds in `/data/tp/w29/b-{old,cf,cfz,iqrh,abl}`; it
waits for an idle card; check the K4 GEMV attn_k M 1 row ≈ 60 µs before trusting a pass), then
`local/workers/w29/summarize.py <res dir>`.

**Trade-off (for the user):** iqrh's decode is 15–20% fewer VALU than the closed form for int8 and 32–35% fewer for f16.
- **Prefill:** that buys nothing on today's separate wdec8 pass, which is bandwidth-bound at the large shapes; it would
  count only once the decode moves into the GEMM tile (P1).
- **Decode (GEMV):** the closed form's −1.5 ops a value bought −7% at M 1 (K4). Scaling that linearly, iqrh's
  −2.4 more would buy about −10–12% (**an estimate, not measured**). It can't go below the ablated floor, and at M 1,
  K4 the layer is at ~48% of DRAM bandwidth.
- **Cost:** +1.2–1.4% proxy error at K2–3.5, +2.7–5.4% at K4–6 (w27; ≈ +0.01 bpw equal-quality at K2–3.5, +0.02–0.05
  above), plus a requant of every level before the ladder.
- **Default without iqrh:** the closed form is in, bit-exact, no requant.

## A4 and the int4 plane (2026-11-05 / 11-06)
- **A4 path built** (w34, 11-05, the user's call before #46-03's prices): the int4 plane stores q of the half-integer
  i4 grid exactly and adds the ½·Σx term after the output Hadamard; byte-exact against the C emulation in every test.
  Full path, K2, M 2048: **×1.31 faster than A8** on the layer; the tile alone ×1.25–1.38 under the plugin's own build
  flags (VOPD off), ×1.40–1.50 with a cheaper fold flush; which compiler serves the card matters for A4. Whether A4
  ships is #46-03's rule (27B per-family costs need the i4 K2 / K3 levels and the importer).
- **Not worth building (2026-11-06, `v3/e3/a4/blockx.md`):** per-block A8 exceptions inside an A4 tensor (two A8
  blocks cut gate_up's A4 error only 6–11% vs the ≥ 30% rule: the error is spread, not concentrated); an i4 decode
  plane on the A8 path for speed alone (layer at M 2048: 2.8% faster if the int4→int8 conversion hides, 4.9% slower
  if it serializes; rule ≥ 5%), so i4 levels are not built for speed alone.

## Measurements (to fill in)
- First card run of the two-pass A8 path (w28, `plugin/exl3i/TWOPASS.md` "Card results"): K4 at M 2048, layer
  A8 9444 µs vs f16 14474 (1.53×); actq 8%, wdec8 11%, GEMM 82% of the passes alone. That makes P1 worth ≤ ~1 ms a layer.
- mul1i8 closed-form decode (w29, P2 + D2): wdec8 −5%, A8 path ±0, GEMV −7.0 / −6.4 / −4.8% at M 1 / 4 / 16 (K4); the
  table above.
