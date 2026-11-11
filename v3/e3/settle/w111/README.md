# w111 · #146 P16 output-row interleave: A8 store-remap cost + second-slice 27B KL with v_proj

Rule: `../w111.md` (1fa29c9, committed alone before any build, timing, quant or score). Data: `../data/w111/` (`k/` kernel,
`kl/` the KL pair). Kernel code: radiance fork branch `qc-exl3-v3-p16` (worktree `/workspace/forks/radiance-p16`, commits
758528b, 4aee43b, fc1032a; never on qc-exl3-v3).

**Runtime:** prefill + decode if P16 is adopted (a store remap in every kernel that writes these tensors); measured
here for the A8 prefill GEMM only. The experiment itself changes nothing.

## The permutation, and a correction to w101's kernel note

P16 = `w101/perm101.py gperm(N, 16)`: stored row j holds natural row **16 (8 (k mod nb) + ⌊k / nb⌋) + (j & 15)**,
k = j >> 4, nb = N / 128. `w101/kernel-note130.md` gives "stored 128 b + 16 t + c → natural 16 (b + nb t) + c": that is
the **inverse** map (natural → stored). The two agree only when nb = 8 (k_proj / v_proj); for ssm_qkv (nb 80), ssm_z
(48) and attn_qg (96) a kernel built from the note would scatter the outputs wrongly. Checked in python against gperm
for N = 256, 1024, 6144, 10240, 12288, and on the card by C2 below. The forward map needs ⌊k / nb⌋ (nb not a power of
two): a multiply-shift, (k · mag) >> 20 with mag = 2²⁰ / nb + 1, exact for k < 8 nb at every nb < 378 (the launch
checks it per shape).

## Files

| file | where | what |
|---|---|---|
| `kl111.sh` | gpubox card 0 | #146-02 driver: v_proj units N / P × seeds 0 / 1 (w101's staged (c) code, stamp 87a7e4ab) → `mksplice111.py` → rad-convert `--reuse` + radmix of the 16 attn_v into w101's four mixes → gates (rxq.pairs, `bytes111.py`, stamp) → 16 scores in one card hold → `verdict111.py` |
| `mksplice111.py`, `bytes111.py` | gpubox | w101's mksplice101 / bytes101 for the 16 v_proj keys |
| `verdict111.py` | gpubox | D (128 tensors) and the 112-scope pair, docstat per pair, v_proj increment, joint z with w101 |
| `proj111.py` | anywhere | the kernel bar from bench_p16's CSV (projection over 48 GDN + 16 attention layers) |
| fork `plugin/exl3i/tests/bench_p16.hip`, `p16_build.sh` | dad1 / gpubox | the microbench (C1 / C2 byte checks, ABBA timing) and its build (also compares the off kernel's disassembly with the base's) |

Run:
```
# kernel (any gfx1201 box with /opt/rocm/core-10.1): BASE = qc-exl3-v3's exl3i_a8_gemm.hip + exl3i_8a8.hip.h
OUT=<dir> BASE=<dir> plugin/exl3i/tests/p16_build.sh
<dir>/bench_p16 <dir>/exl3i.so --reps 30 --m 512,2048,8192 --rates 2.5,4 --all-shapes-off --csv il1.csv   # registered
<dir>/bench_p16 <dir>/exl3i.so ... --il 0 | --il 2                                                       # A/A, raster
python3 proj111.py il1.csv [denominator.csv]
# KL pair (gpubox): code staged in the repo layout under /data/w111/code
setsid nohup bash /data/w111/code/v3/e3/settle/w111/kl111.sh prep 0 1 score verdict > /data/w111/run.out 2>&1 < /dev/null &
```

## #146-01 Kernel cost (A8 prefill GEMM, R9700 card 0)

Implementation: `exl3_8a8_body<IL>` (template), `exl3_store_at` (y column apart from the svh row), kernel
`exl3i_a8_gemm_kernel_il16` picked by `Exl3iA8Args.il16 = 1` (zero-initialised callers: today's kernel). ~6 integer ops
per 8-value store; every store stays one 16-byte vector. VGPRs 240, scratch 20 B, occupancy 6 as before (SGPRs 40 → 42).

- **Remap off = today:** the off kernel's disassembly is instruction for instruction the qc-exl3-v3 kernel's (2,975
  instructions; also the A4 kernel, which shares the store helper), and **C1** (off == base, byte for byte) holds at all
  9 trunk shapes × K2.5 / K4 × T 512 / 2048 / 8192.
- **C2** (remap on == remap off gathered through gperm on the host, byte for byte): pass at all 24 remapped cells
  (dad1 and card 0).
- **Timing** (`data/w111/k/card0-bench-il1.csv`, 30 ABBA rounds): GEMM on / off, geo over K × T (paired ratios):
  ssm_qkv **1.010**, ssm_z 1.004, attn_qg **1.016**, attn_v 1.004; worst cell at T ≥ 2048: 1.020 (qkv K2.5 T2048).
  A / A run (both arms the off kernel, `--il 0`): 0.998–1.001 by shape, cells within ±1%.
- **Projection** (A8 everywhere, 48 GDN + 16 attention layers; the four remapped families are ~21% of the trunk's A8
  time): **+0.05…+0.25%** of prefill gemm time (T 512: +0.05 / +0.08%, T 2048: +0.17 / +0.18%, T 8192: +0.16 / +0.25%,
  K2.5 / K4). **Kernel bar PASS** (≤ +0.5%, no shape > 1.03 at T ≥ 2048).
- **Where the cost comes from, and a fix (reported variant, `--il 2`, not the registered arm):** the remap's ALU is
  small; the cost is most likely write combining: each stored 128-row block writes 32-byte pieces of 8 different natural
  128-column lines (consistent with the variant below, which changes only the order the blocks run in). Running the stored blocks that together fill natural blocks 8c … 8c + 7 (stored c, c + nb/8, …,
  c + 7 nb/8) one after another (a raster change, one line in the body) brings it to the noise: qkv 1.003, z 1.001,
  attn_qg 1.005, attn_v 1.002; projection −0.06…+0.10% (`card0-proj-il2.txt`). dad1 (9070 XT, K4) agrees: remap alone
  +0.3…2.4%, with the raster +0…0.6%, A / A ±0.6%.
- **Not done:** no format bit, nt.c not wired, no e2e prefill (needs a packed P16 mix and the format bit), no decode.

**Decode (written up, not implemented):** the nt / rows / rate kernels (and T2's f16 prefill) store through
`exl3r4d_common.hip.h` `epilogue_block` (MATS 1 arm: `st_row(p.y + row · y_pitch + n0)`, n0 = cb · 128 + 16 t + 8 h, one
16-byte vector of 8 values; split-K reduces in LDS before it, so there is no second store site). The remap is the same
`il16_col(n0)` on the y address only (svh stays at n0), as a template flag or a wave-uniform kernel argument (a scalar
branch). A4's tile uses `exl3_store`, which now forwards to `exl3_store_at`, so A4 needs one line. At decode (T 1–8) a
call writes T · N · 2 B (≤ 200 KB) against N · K · bits / 8 of weights (attn_qg at K2.5: 19.7 MB), and adds ~6 integer
ops per 8 outputs to an epilogue of hundreds of FMAs: **expected ≈ 0 (< 0.1%)**, below decode's run-to-run noise; the
write-combining effect above needs many rows per block and does not arise at T ≤ 8. To be timed with the format bit.

## #146-02 Second-slice 27B KL (golden-search27d, gpubox card 0, 11-11 17:29–18:47 UTC)

Slice: w110's **golden-search27d** (fresh two-part: 130 xcorp pieces of xc8k-b2-c0, 222,107 positions + 63 calib
search-pool docs never in the ledger, 77,441 scored positions; built 17:28 UTC, before my gates; its first #130 use).
v_proj units 20 s each (stamp 87a7e4ab checked in all four), splices / packs seconds, radmix ~1 min, 16 scores 76 s
(calib part) / 187 s (xcorp part), all inside one card hold (`hold111.sh`) after the per-step lock waits behind w110's
level build proved slow (~7 min each). Gates: G1 PAIR OK (P-v vs N-v: 128 arm weights changed, 659 identical; X-v vs
w101's X: only the 16 attn_v), G2 BYTES111 PASS 16 / 16, G3 stamp — both seeds.

| pair (N → P16) | seed 0 | seed 1 | geo | xcorp | calib | SE comb | z |
|---|---|---|---|---|---|---|---|
| **D: 128 tensors (+ v_proj)** | 0.9911 | 0.9896 | **0.9903** | 0.9904 | 0.9900 | 0.0090 | −1.08 |
| 112 scope (w101's mixes) | 0.9993 | 0.9958 | 0.9976 | 0.9987 | 0.9843 | — | −0.16 |
| v_proj increment r(128) / r(112) | 0.9917 | 0.9937 | 0.9927 | | | | |

- **D PASSES** w101's bar on this slice (geo ≤ 0.995, both parts < 1). **attn_v joins** (increment ≤ 1).
- The **112 scope alone does not repeat** w101's pass here: geo 0.9976 (> 0.995), parts 0.9987 / 0.9843.
- **Joint with w101** (golden-search27c, 112 scope, 0.9933 ± 0.0070): same scope 0.9941, z −0.93; w101's 112 + D
  0.9922, **z −1.41**. Not significant on either combination.
- **Per document (D, docstat):** xcorp per-doc geo 0.987 / 0.990, 62% / 57% of docs better, the 28% swe-rebench doc
  0.977 / 0.989 (without it 0.998 / 0.989); calib per-doc geo **1.002 / 1.003** (median 0.991 / 1.000, 56% / 51% better):
  the calib gain is pooled only (seed 0's wildchat doc 0.854). Leave-one-out pooled ≤ 0.996 / 0.993.
- **v_proj's increment is large for 16 tensors** (−0.7%, more than the 112 tensors' −0.24% on this slice; f2 0.87,
  L63 0.66): seeds agree (0.992 / 0.994) and hygiene holds (only the 16 attn_v differ), but it is one slice: by the
  AGENTS.md confounder rule it is **unverified** until a second slice.
- Files: `../data/w111/kl/` (verdict111.txt / .json, docstat per pair, scores' json, census, patch stats, gate results
  in result.txt). The four D mixes are kept on gpubox `/data/w111/mix/` (66 GB) for a v_proj / third-slice check;
  w101's four 112-scope mixes are deleted (65 GB), as are the wq, patch and P16 checkpoints.

## Proposal (rule: `../w111.md` "Proposal rule")

D PASS and the kernel bar PASS → **adopt-proposal: P16 for ssm_qkv, ssm_z, attn_qg and attn_v** (k_proj: model 0.952
only, not tested). The evidence is a screen, not a significant result (joint z −1.41), and the 112-tensor part did not
repeat on its own (0.9976): most of this slice's gain comes from v_proj, which is unverified. Kernel cost: +0.05…0.25%
of prefill gemm time as implemented (≤ +0.1% with the raster variant), decode ≈ 0 (not timed). Needs a format bit and
the remap in nt / rows / rate / T2 / A4 before any adopted level ships. The user decides.

