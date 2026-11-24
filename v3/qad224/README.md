# v3/qad224 — w224: value-remap probe for table-QAD (lc1 + table-QAD question)

**Question:** would a lc1 (K <= 2) + mad (K > 2) mix with *table-QAD* (the lc1 decode tables trained under KL) beat scale-only QAD enough to justify the V4 work
(codebook header, per-tensor table plane, lc1 decode kernels)? Answered without lc1 codes, format or kernels: a trainable value map on the mad codebook integers of every K <= 2 tensor.
Training / eval tool only, **runtime: none** (nothing here reaches the format or the kernels). Report `rad-local/workers/out/w224.md`, raw data `../../data/qad224/`.

## What was added (all in `../qad220/qad220.py`; every flag off by default, bit-identical when off)
| flag | what |
|---|---|
| `--remap K_MAX[,KNOTS[,GROUPS]]` | on every trunk tensor whose container K (trellis bits / weight; `ql.K`) is <= K_MAX (2.50 student: 62 of 400 tensors, 14.96 % of the trunk's weights, K = 1.25 .. 2.0) a trainable **odd-symmetric piecewise-linear map f on the int8 codebook integers m**: the dequantized weight W' = f(m) = m + delta(m) enters the GEMM before the scales. KNOTS knots (default 64, at the quantiles of \|m\|, knot 0 at 0, last at the tensor's max \|m\|; integer positions, fewer when the tensor has fewer distinct values), knot value = x_j (1 + d_j), **d = 0 at the start = the identity map** (delta = 0 exactly); GROUPS (default 1) = contiguous row-blocks of W's input side, each with its own map (GROUPS = 1: one map a tensor). Same freedom as a 64-entry table per tensor / block. mad's \|m\| <= 127, so the map is a 255-entry table (delta table, gathered at m). **Forward: `y = [x, x] @ [m; delta]`, one bf16 GEMM with f32 accumulation (K doubled) and one rounding of y** (`_RemapMatmul`); backward dx = dy @ m^T, the table's gradient = a histogram of x^T dy over the entries (`torch.bincount`, f32 atomics: the remap gradient is not bitwise reproducible, the forward is) |
| `--remap-round` | the table's entries rounded to integers (int8 table) in the forward, straight-through gradient: the deployable-table constraint (off in the arms: they are the continuous upper bound) |
| `--eval-only GOLDEN [--eval-cap N]` | no training: the trainer's own forward (same `--a8` / `--compile-scales` / `--remap` flags; `--init CKPT/params.pt`; `--init` refuses parameters the student does not have, e.g. a remap checkpoint without `--remap`) over every document of the golden in `--Q` segments carrying the state (> 16,384 tokens: the w225 KV buffers), KL over the gen rows as the trainer's loss computes it; writes `OUT/eval.json` (per-document KL sums). Queue it as a `B` line |

Needs `--a8` and/or `--compile-scales` (the patched forward), not `--a8-int8` (a remapped weight is not an int8 integer: use `--a8`, the fake-quant activations + bf16 GEMM, whose numerics are within a bf16 ulp of the int8 path; w223).
Checkpoint: `ckpt-*/remap.safetensors` beside `scales/` (per tensor: `.knots`, `.d`, `.table`) and the d's inside `params.pt` (`<key>:m`). **The served read cannot apply a remap**: judge with `--eval-only`; the number is *in-trainer KL*, not served KL.

## First attempt failed (W' = bf16(f(m)), 11-24 ~2 AM CST)
The first version gathered f(m) into a bf16 weight. m is an integer, exactly on bf16's grid, so a map moved by less than half a bf16 step (2^-9 relative: any |d| up to ~0.2-0.4 %, i.e. the first ~10 steps at lr 3e-4) rounds back to m: the weights, and the KL, did not change (R25-m1: train KL identical to the control's to the 6th digit for steps 1-12). Scales do not have the problem (applied in f32 after the GEMM; the output rounding dithers). The gates (lr 0 = plain bit for bit) could not see it. Data: `../../data/qad224/probe1-bf16-W/`. Fix = the m + delta form above (unit test: a d = 1e-3 map moves y by the exact amount).

## Files
- `test224.py` — CPU unit checks (identity = plain bf16 weights bit for bit, forward vs a float64 interpolation, table gradient vs finite differences for 1 and 3 groups, rounded / straight-through, `apply_remap` selection by K and its off-switch). `python3 test224.py`.
- `cmp224.py CONTROL/eval.json ARM/eval.json ...` — paired per-document bootstrap of two `--eval-only` passes (cmp214's arithmetic).
- Queue lines (iceman card 0, `/data/queue`): `jobs.iceman.w224-gates.txt` (gates g0-g4 of the first attempt + e0), `jobs.iceman.w224-probe1-cancelled.txt` (first-attempt arms, cancelled), `jobs.iceman.w224-arms.txt` (gates g5 / g6, the arms r1 / r32 / r32hi, their evals); `jobs.box2.w224-jlong-pending.txt` (box2: gate + golden-jlong evals, waiting for box2's cards 2 / 3 to take `B` lines again).

## Results (11-24; iceman card 0; 2.50 student, 0.5 M rows, lr 3e-4, `--a8 --compile-scales --compile-ew`; in-trainer KL, **not served KL**)
Arms: `R25-ctl` scale-only (same flags, no remap); `R25-r1` `--remap 2.0,64,1`; `R25-r32` `--remap 2.0,64,32`; `R25-r32hi` = r32 with `--remap-lr 3e-3` (10x). Each run: budgets 2.5e5 / 5e5 (0.25 M = a WSD branch). 62 of 400 tensors, 14.96 % of the trunk's weights carry a map.

| judge | | r1 (G=1) | r32 (G=32) | r32hi (G=32, 10x lr) |
|---|---|---|---|---|
| j217, 163,534 gen rows, 166 docs (control 0.046515) | 0.25 M | 1.0055 [0.996, 1.021] | 1.0060 [0.997, 1.020] | **0.9843** [0.969, 0.998] |
| (control 0.042516) | 0.5 M | 1.0051 [0.997, 1.016] | 1.0025 [0.993, 1.015] | 1.0015 [0.984, 1.024] (0.989 [0.978, 0.996] without the 2 blow-up docs) |
| fixed eval (22 held-out calib qad docs, 21,090 rows, first 2,048 tokens) | 0.25 M / 0.5 M | 1.0038 / 0.9988 | 1.0013 / 0.9964 | 0.9946 / 0.9932 [0.988, 0.9994] |
| training KL (each batch before its update), mean ratio over the steps | 0.5 M | 0.9983 | 0.9989 | 0.9975 |

Ratios arm / control, paired per-document bootstrap (`cmp224.py`; cmp214's arithmetic). Noise floor: two scale-only controls (R25-ctl vs the a8-int8-trained I25-lr3e-4-4M 0.5 M, same recipe) differ by 0.985 [0.956, 1.003]. The maps move little: max |f(m) - m| 0.43 / 0.50 / 5.2 integer units (of +-127; a level spacing is ~1) at lr 3e-4 / 3e-4 / 3e-3. Cost: 129 / 119 / 119 s per step vs 91 s (K <= 2 tensors run a doubled-K GEMM, a weight-gradient GEMM and a histogram; peak memory +0.07 GiB).
Fidelity of the in-trainer judge: I25-lr3e-4-4M 0.5 M, in-trainer (a8 fake-quant forward) 0.043181 vs its served read 0.042757 on the same 163,534 rows: 1.0099 [1.004, 1.015] paired.
**Verdict: no gain beyond noise. Best arm r32hi: -1.6 / +0.2 % (j217 0.25 / 0.5 M), -0.5 / -0.7 % (fixed eval); r1 / r32: +0.3 .. +0.6 % on j217, -0.1 .. -0.4 % on the fixed eval; the pre-set gate was ~2 % (<= 1 % = drop): the table-QAD part of lc1 is not worth planning.** A deployable int8 table would be coarser than this continuous map (the maps moved by < 0.5 integer unit: they would round back to the identity), so this is an upper bound. Not run: the 2 M arm (step 4: gate > 1 % not met), golden-jlong (box2 queue, pending).
