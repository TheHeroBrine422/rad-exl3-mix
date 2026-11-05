# V3 kernel optimizations to write (from IQR-V4's kernels, 2026-11-05)

Source: a read of radiance 1.17's IQR-V4 kernels (`libr4d/r4d_gemm_iqr.hip`, `r4d_iqr_dec.h`, `r4d_gemm_8a8_prefill.h`,
`r4d_gemm_fp8a8.hip`; Apache-2.0) against our two-pass exl3i path (fork branch `qc-exl3-v3-twopass`,
`plugin/exl3i/TWOPASS.md`). Gains are estimates unless marked measured. Status per item; results get added here.

## Prefill (A8 path, M ≥ 512): in order

| # | what | IQR reference | est. gain | effort | status |
|---|---|---|---|---|---|
| P1 | **Decode inside the GEMM tile** instead of a separate decode launch: blocks claim 16-row weight tiles of their 128-row output tile through counters (claimed / decoded / finished), decode them, then run the K loop. Counters in our scratch, zeroed by actq (exl3i promises `rad_kernel_concurrent`: no state kept between launches) | `r4d_gemm_iqr.hip:670-768` | −6–8% of A8 GEMM time at M 2048, −25–30% at 512 | 1–2 days | to do after the first timing |
| P2 | **Integer int8 decode, bit-exact:** `q = max((S − 509) >> 2, −127)` from the hash's byte sum S (checked for every S; only state 0 needs the max), instead of value → f16 → f32 → int (~13 ops a value today) | — (w23's closed form) | halves the decode pass's ALU; what lets P1 hide the decode below M 2048 | hours | to do |
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
| D2 | **Integer byte-sum f16 decode** for mul1i8 (bit-exact; removes the f32 detour every V3 tensor pays at M = 1) | P2's form | — | later (cheap) |
| D3 | **Fused producers:** gate_up's last-arriving block writes silu·up·suh_down already rotated for down | `r4d_gemm_iqr.hip:55-107` | — | later |
| D4 | Hash dotted with the activation byte broadcast by `v_perm` (sum and product in one `sudot4`), scales through SGPRs | `r4d_gemm_iqr.hip:170-233` | part of IQR's 42 → 30 µs | only with the hash codebook (#100) |

## Format: the hash codebook (#100) — decide before the ladder
IQR's computed value is `sudot4((umul24(state, A) + B) & 0x3F3F3F3F, 0x01010101) − 126` (~3 ops a value incl. the
hash, int8-native, A / B stored per tensor) vs EXL3 mul1i8 (32-bit multiply, 4-byte sum, scale, int8 round). A
different codebook means a requant of every level (~200 GPU-hours if done after the ladder), so it is tested now on
the CPU proxy (rule: `v3/e3/sub1/README.md` "#100"). If adopted: exllamav3 codebook option, encoding tag
`cb=iqrh`, plugin decode paths, D4.

**Results:** (pending: w27)

## Measurements (to fill in)
- First card run of the two-pass A8 path (`/data/tp/cardrun.sh`): device tests, per-shape timing at M 2048, per-pass
  split (actq / decode / GEMM) — decides P1's priority.
