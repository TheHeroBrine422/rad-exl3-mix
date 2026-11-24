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
- Queue lines: `jobs.w224.txt` (gates + e0) and `jobs.w224.arms.txt` (arms + evals) — iceman card 0, `/data/queue`.
