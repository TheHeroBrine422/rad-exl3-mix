# qwen35_exl3 step pass (w152, #179, 2026-11-13 / 14)

The decode step outside the GEMVs: what the plugin's own small ops cost, what was fused, measured how, and what is left.
Branch `qc-exl3-v3-step` (on `qc-exl3-v3` b83698d; measured on fdd480c / 9f0a059 + the same counter fix). Box2 card 1 (R9700, gfx1201, 210 W cap), rad10-perf core (ROCm 10.1).
Data: rad-exl3-mix `data/step179/` (per-op tables, e2e, KLD / decode checks, device tests); scripts `v3/bench/step179/`.

## Commits

| commit | change | bits | test |
|---|---|---|---|
| f221bb0 | `exl3_fuse.h`: each block leaves its residual add to the next norm (`add_rmsnorm`, the final norm's too); `gdn_conv_recurrent_update` at decode (≤ 8 tokens a sequence); q / k norm + rope + kv_store as one `qk_norm_rope_gate_kv_store` at ≤ 64 rows (a `rope_table` fill a step); the attention gate's sigmoid · attn as one `act_mul`. libr4d ops, each declared only where a library resolves it; `QWEN35_EXL3_FUSE=0` (or a list `addnorm,gdn,qk,gate`) | identical | `test_fuse` (issue counts at decode / prefill / mixed / verify, off switches, a library resolving none), lint |
| 440b1f5 | `Exl3Aq::takes(T)`: the fusions ask whether AQ's site takes this step (A8 rows, or w150's 1-row xb band on mad) before folding a site's norm or o's gate product | identical | `test_aq` mad case; the old predicate fails it |
| def87e9 | exl3i interface 5: `exl3i_actq` pre rmsnorm with `res` = 1 (operands res, xo): x = bf16(x + res) written to xo, then the norm — `add_rmsnorm`'s bytes; the decode band's split rows leave xo / h to the row's last workgroup (counters keyed by (device, stream, xo)). `Exl3Aq::declare_res`: the int8 / xb forms of each rmsnorm site with the add; `issue_in_norm` folds the pending add into the site | identical | exl3i `test_aq` test_res (add_rmsnorm + actq vs fused at 1..2048 rows, int8 / xb, split / one workgroup, h over res), host checks |

## How it was measured

`--profile-ops` (the #177 / #124 profiles) synchronises around every op, so a small op shows its launch floor plus the
sync: the residual add read 1.03 ms a step there. The engine's own **`RADIANCE_AQL_STAMP=N`** times every dispatch of
every Nth recorded pass with device timestamps and nothing else changed (`stamp.sh`, `stsum.py`): the decode tables below
are those. Prefill chunks are not recorded passes (not stampable); the prefill table is `--profile-ops`. rocprofv3 sees
none of radiance's dispatches. Stamping costs ~0.9 ms a step (a signal a dispatch): read walls relative to each other.

**The 210 W cap moves the GEMVs.** Every exl3i GEMV of a fused step is 1–2% slower than in the unfused one, whatever
runs before it (`gemvpos.py`: by predecessor kernel and size class, all +1.5–2.3%, the head after an unchanged gather
too). ABBA with cardclock watching the SMU (1200-token decodes, K2.5 mul1i8, base / step2 / step2 / base): 209–210 W in
all four, gfx clock 2405 / 2374 / 2373 / 2403 MHz — fewer gaps, the same power, a lower clock. So part of each saving
goes back to the GEMVs: e2e gains are smaller than the removed kernel time.

## Results

**Decode e2e** (w138's bench.py, server ms / decode step, median of 3 runs a round; mad rows interleaved in rounds 6–9,
mul1i8 rounds 1–3 + ABBA). The mad homes carry the KS counter fix (fdd480c, below: without it mad decode is NaN).

| pack | base | fusions (f221bb0) | + actq's residual (def87e9) |
|---|---|---|---|
| mad K2.5 (w151 635bac02), ctx 0 | 24.94 | 24.74 | **24.51 (−1.7%)** |
| mad K2.5, ctx 8k | 25.96 | 25.75 | **25.53 (−1.7%)** |
| mad K4.125 (w150 144dd6e1), ctx 0 | 30.59 | 30.27 | **29.91 (−2.2%)** |
| mad K4.125, ctx 8k | 31.62 | 31.34 | **31.01 (−1.9%)** |
| mul1i8 K2.5 (w138), ctx 0–1.2k — ABBA, the engine's step counters (mean of 2 × 8 windows) | 29.70 | **29.05 (−2.2%)** | = (no decode site on mul1i8) |
| mul1i8 K4.125, ctx 0 | 30.53 | **30.00 (−1.7%)** | = |

8k prefill: unchanged within the rounds' drift (±0.5%: 2,383–2,409 tok/s mad, 2,376–2,498 mul1i8 across rounds; within
a round ±10 ms of 3.4 s). The prefill adds folded into actq save one pass over x a site (~10 ms an 8k prefill, ~0.3%);
the prefill is the A8 GEMMs (80% of it) at the power cap.

**Per op, decode, stamped** (µs a step / launches, ctx 0, K2.5; IQR-V4 iq2r drafter off; `data/step179/stamp-*.txt`):

| op | mul1i8 base | mul1i8 step | mad base | mad step | mad step + res | IQR iq2r |
|---|---|---|---|---|---|---|
| trunk GEMVs | 23,462 / 336 | 23,960 / 336 | 18,001 / 336 | 18,115 / 336 | 18,366 / 336 | 16,070 / 287 |
| head | 1,044 / 1 (exl3i R16) | 1,068 / 1 | 3,988 / 1 (bf16 in these packs) | 3,983 / 1 | 3,983 / 1 | 1,474 / 2 (vocab tiers) |
| GDN a / b (bf16 gemm_nt) | 333 / 96 | 338 / 96 | 1,669 / 96 (beside qkv: overlapped) | 1,692 / 96 | 1,705 / 96 | 2,396 / 48 |
| residual add (`binary`) | 470 / 144 (+ 16 gate muls) | 0 | 419 / 128 | 417 / 127 | **0** | 5 / 1 |
| block input norms + final norm | 573 / 161 (+ 32 q / k norms) | 559 + 4 / 129 | 498 / 128 (actq) + 97 / 33 (+ q / k) | 501 + 5 / 129 | 744 + 5 / 129 | 554 / 128 (+ fp8 quant) |
| GDN conv + recurrent | 504 + 146 / 96 | 579 / 48 | 510 + 151 / 96 | 582 / 48 | 587 / 48 | 647 / 48 |
| q / k norm + rope + kv_store | rope 93 / 32, kv 26 / 16 (+ norms above) | 33 + 7 / 17 | rope 92 / 32, kv 28 / 16 (+ norms above) | 33 + 7 / 17 | 33 + 7 / 17 | 41 + 4 / 17 |
| attention gate | sigmoid 66 / 16 (+ mul in `binary`) | act_mul 17 / 16 | sigmoid 67 / 16 (mul in o's actq) | 69 / 16 | 69 / 16 | 27 / 16 (gate quant) |
| silu · mul / actq producers | 85 / 64 | 85 / 64 | actq 167 / 80 | 170 / 80 | 169 / 80 | (in the gated GEMV) |
| attention core | 168 / 32 | 168 / 32 | 176 / 32 | 174 / 32 | 179 / 32 | 184 / 32 |
| small ops in all (`smallops.py`: not GEMV, head, a / b, attention core) | 1.97 ms | **1.29 ms** | 2.04 ms | 1.79 ms | **1.63 ms** | 1.29 ms |
| dispatches a step | 996 | 741 | 996 | 884 | **757** | 584 |
| idle (stamped wall − busy) | 3.85 ms | 2.98 ms | 3.24 ms | 2.97 ms | 2.42 ms | 2.18 ms |

The small ops are at IQR's level now on mul1i8 (1.29 ms both); on mad 1.63 ms, the difference the actq producers'
quantisation (0.17 ms: IQR's sits in its gated GEMV) and the split-row norm's tail. What is left is the GEMVs (mad K2.5
trunk 18.4 ms vs IQR's 16.1 at ~2.1 bits), the bf16 head of these mad packs (4.0 ms vs IQR's tiered 1.5), and ~170
launches IQR does not make (GDN qkv + z as one GEMV, gate_up with silu · mul in its epilogue, ssm_out's row pre-pass).

**The plugin's non-GEMV ops, all of them:** a diagnostic build that skips every op of the plugin's own but the GEMVs,
the attention core and kv (outputs garbage; GEMV time does not depend on values): e2e 28.78 → 27.17 ms (K2.5 mul1i8,
after f221bb0). ~1.6 ms a step was all that fusing the plugin's ops could still take; actq's residual then took 0.23 ms
(mad K2.5).

## Numerics

Whole-model KLD rows (golden-search27c-c, 67,932 positions), each against today's home: **byte-identical** for every
build here — mul1i8 K2.5 / K4.125 and mad K2.5 / K4.125, A8 prefill chunks (def) and 64-row steps (c64), on rad10-perf
(ROCm 10.1) and the frozen 7.14 install. Greedy decode with top-20 logprobs (8 prompts × 96 tokens, `gdec.py`):
bit-exact base vs every fused build on both mul1i8 packs; on mad, every build with `kctr-key.patch` bit-exact against
an all-ordered build (exl3i `rad_kernel_concurrent` 0).

## Tests

Host: `qwen35_exl3_test` (test_fuse, test_aq incl. the mad xb band with and without the add) and exl3i's host tests,
ROCm 10.1 and 7.14. Device (box2 card 1): exl3i_gpu_test, test_twopass (FUSE default / on / off), test_ks (mul1i8, mul1,
mad), test_r16, test_aq (+ libr4d: pre stages and test_res), test_a4 — all PASS on ROCm 10.1 and 7.14. After the rebase on b83698d
(w155's exl3i_rerank): host tests, test_aq and test_rerank PASS on both, KLD rows (mad K2.5) byte-identical, greedy decode
bit-exact. A standalone plugin build there needs the tree's `arch/common/rad_block_dflash2.h` over the installed one (w155's
op_rr_ext; `v3/bench/step179/plugb.sh` OVL=1, as w155's build155.sh).

## Found on the way: mad decode was NaN on qc-exl3-v3 (fixed: fdd480c)

With w150's xb band the engine starts z's GEMV beside qkv's and k / v beside qg (exl3i promises `rad_kernel_concurrent`;
80 unordered GEMVs a step on mad, none on mul1i8, whose GEMVs share a scratch range). KS's arrival counters are one
block a stream (`exl3i_kctr_for(stream)`), so the overlapped launches count into each other's: a column block summed
before its slices land, a reset that loses an arrival. Mad decode emits non-finite logits after 3–4 tokens ("!!!!"),
differently run to run (gdec: 0 of 8 requests finish on both mad packs, base and fused builds alike); prefill KLD is
unaffected (ordered). `v3/bench/step179/kctr-key.patch` (exl3i_nt.hip.h, exl3i_mad.hip.h; 13 lines) keys the counters
by (device, stream, output): two launches the engine may overlap write two outputs. With it decode is bit-exact to an
all-ordered build (768 / 768 steps, both packs). The lead applied it on qc-exl3-v3 as fdd480c (this branch's base); I
re-measured #177-02's mad numbers on it (exl3i PERF.md, the correction under its e2e table: decode ×0.781 / ×0.912 vs i8,
was ×0.770 / ×0.904; the decode-path KLD stands). The A8 FUSE counters (`exl3i_a8d_ctl_for(stream)`) have the same
keying; prefill rows are deterministic today, so its launches do not overlap — worth keying the same way.

## Left

- **GEMV-side folds** (exl3i, not this pass): the residual add into the out-projections' epilogue would also remove the
  actq res's split-row tail; GDN qkv + z as one GEMV; gate_up with silu · mul; an xb producer for ssm_out (behind the GDN
  gated norm in libr4d's recurrent op) and the head.
- **The packs' bf16 head** (mad packs here: 4.0 ms a step, 16% of the step) — a pack choice, not a kernel.
- Prefill: nothing in the plugin's own ops is visible at the cap; the q / k prologue past 64 rows and the attention gate
  at prefill stay unfused (0.8% of a profiled prefill).
- A stub test for an exl3i without interface 5 (the res forms unresolved: adds alone) needs the fakes to refuse one
  param form of an op (tests/arch_fakes.h refuses by op name); the path is the pre-def87e9 one.
