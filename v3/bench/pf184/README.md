# pf184 — prefill (gemm_us) pass on the mad exl3i A8 kernels (#184, w160, 2026-11-14)

Fork branch `qc-exl3-v3-pf184` (radiance-hero, off `qc-exl3-v3` 120132a; worktree `/workspace/wt/w160-radiance`).
Everything runs on box2 under `/data/q27/w160/` (HIP card index, `/data/q27/locks/card<N>.lock`, ram.lock shared for
model loads), ROCm 10.1 with the VOPD-repeatable clang, exl3i built standalone against `/data/q27/rad10-e157b/install`
(the reference build of this branch's base: same core). Results: `data/pf184/` (`pull.sh`).
**Runtime:** prefill (the A8 path at M ≥ 512; the board's `gemm_us`) and eval speed; decode code untouched.

## What gemm_us is and how it is measured here

The board's `gemm_us` (IQR-V4 technical report, Table 3): trunk GEMM time of one decoder layer at **2048 tokens, one
rank of two** (column-split weights half N, down / attn_o / ssm_out half K); FP8 = 3712 µs, IQR-V4 k2..k6 3801..3990.
`bench_gemmus` (fork `plugin/exl3i/tests/bench_gemmus.hip`) times, on one card, in interleaved rounds:

- **ours:** `exl3i_gemm_nt_a8` (act a8g128, codebook mad) reading the producer's int8 planes (AQ: what qwen35_exl3
  serves from 512 rows at every trunk site but ssm_out); `--self`: the same op quantising x itself;
- **iqr:** libr4d's `gemm_iqr_tiled` over IQR-V4's trunk form (2^16-state computed trellis, B-bit codes, one f32 scale
  a row × a factor a 128-K block; 256-row runs, 512 on gate_up / down; o / ssm_out from 5 bits in the head-fed bf16
  (row, 128) form), planes from r4d_selftest's generator through the row's own layout hooks;
- **fp8:** libr4d's `gemm_fp8a8_tiled` (block E4M3, the official FP8 entry's row), random operands;
- `--bare`: libr4d's int8 RB 32 tile alone on mad-valued int8 weights, and exl3i's A8 GEMM without the decode.

Ours and IQR read **the same activation bytes** (Gaussian rows quantised per (row, 128) as the A8 quantiser does), the
weights have the same value distribution (both are the mad computed code): **at box2's 210 W cap every GEMM is
power-bound** (constant-byte operands run ~1.7× faster than random ones: r4d_selftest --perfw8's FP8 1431 µs vs
2416 µs on random data at gate_up), so only same-data comparisons mean anything. Weights rotate over 256 MB of copies.
Default: sustained bursts (warm-up burst, then 20 launches timed, median of the rounds). `--cool 300 --burst 3`:
bursts after 300 ms idle (the clocks before the cap binds; big shapes only, small ones ramp from idle).

Board scale: ours / IQR in the same run × IQR-V4's Table 3 (same data distribution, so the calibration carries the
power regime); FP8 is not a good calibrator here (random E4M3 bytes cost more energy than real fp8 weights).

## Scripts

| script | what |
|---|---|
| `put.sh TAG` | the worktree (tracked files, uncommitted edits too) → box2 `/data/q27/w160/src/TAG` |
| `plugb.sh TAG [XFLAGS] [PLUGINS]` | exl3i (default) from src/TAG against rad10-e157b, host tests, `b/TAG/home` (e157b's home + this exl3i.so) |
| `gemmus.sh CARD BUILD TAG [args]` | `bench_gemmus` of a build → `res/gemmus-TAG.{txt,csv}` |
| `fp8cal.hip` | standalone FP8 / int8 tile timing at the trunk shapes with random operands (first calibration look) |
| `ct.sh CARD BUILD TEST\|all` | exl3i device tests of a build → `res/ct-*.txt`, `res/ct.log` |
| `kld.sh CARD TAG=HOME=PACK...` | whole-model KLD rows (golden-search27c-c, bf16 KV; an equivalence check of builds) |
| `e2.sh CARD ROUND TAG=HOME=PACK...` | e2e: w138's #176 server + bench.py, 8k prefill + ctx-0 decode |
| `isa.sh BUILD UNIT OUT.s` | device asm of an exl3i unit with the build's flags; `../k175/kloop.py` counts loops |
| `pull.sh` | box2 results → `data/pf184/` |
