# v3/qad224 — w224: value-remap probe for table-QAD (lc1 + table-QAD question)

**Question:** would a lc1 (K <= 2) + mad (K > 2) mix with *table-QAD* (the lc1 decode tables trained under KL) beat scale-only QAD enough to justify the V4 work
(codebook header, per-tensor table plane, lc1 decode kernels)? Answered without lc1 codes, format or kernels: a trainable value map on the mad codebook integers of every K <= 2 tensor.
Training / eval tool only, **runtime: none** (nothing here reaches the format or the kernels). Report `rad-local/workers/out/w224.md`, raw data `../../data/qad224/`.

## What was added (all in `../qad220/qad220.py`; every flag off by default, bit-identical when off)
| flag | what |
|---|---|
| `--remap K_MAX[,KNOTS[,GROUPS]]` | on every trunk tensor whose container K (trellis bits / weight; `ql.K`) is <= K_MAX (2.50 student: 62 of 400 tensors, 14.96 % of the trunk's weights, K = 1.25 .. 2.0) a trainable **odd-symmetric piecewise-linear map f on the int8 codebook integers m**: W' = f(m) replaces the dequantized weight before the scales (`y = ... (x_rot @ W') ...`). KNOTS knots (default 64, at the quantiles of \|m\|, knot 0 at 0, last at the tensor's max \|m\|; integer positions, fewer when the tensor has fewer distinct values), knot value = x_j (1 + d_j), **d = 0 at the start = the identity map**; GROUPS (default 1) = contiguous row-blocks of W's input side, each with its own map (GROUPS = 1: one map a tensor). Same freedom as a 64-entry table per tensor / block. The forward is a 255-entry table gather (mad's \|m\| <= 127, so the PWL map is exactly a LUT) in bf16; the table's gradient = a histogram of dL/dW' over the entries (`torch.bincount`, f32 atomics: the remap gradient is not bitwise reproducible, the forward is) |
| `--remap-lr X` | the maps' peak lr (Adam, own group, same schedule; default = `--lr`; 0 = frozen at the identity: the equivalence gate). The maps' d are clamped to +-`--trust` (0.3) like the scales' log multipliers |
| `--remap-round` | the table's entries rounded to integers (int8 table) in the forward, straight-through gradient: the deployable-table constraint (off in the arms: they are the continuous upper bound) |
| `--eval-only GOLDEN [--eval-cap N]` | no training: the trainer's own forward (same `--a8` / `--compile-scales` / `--remap` flags; `--init CKPT/params.pt`; `--init` refuses parameters the student does not have, e.g. a remap checkpoint without `--remap`) over every document of the golden in `--Q` segments carrying the state (> 16,384 tokens: the w225 KV buffers), KL over the gen rows as the trainer's loss computes it; writes `OUT/eval.json` (per-document KL sums). Queue it as a `B` line |

Needs `--a8` and/or `--compile-scales` (the patched forward), not `--a8-int8` (a remapped weight is not an int8 integer: use `--a8`, the fake-quant activations + bf16 GEMM, whose numerics are within a bf16 ulp of the int8 path; w223).
Checkpoint: `ckpt-*/remap.safetensors` beside `scales/` (per tensor: `.knots`, `.d`, `.table`) and the d's inside `params.pt` (`<key>:m`). **The served read cannot apply a remap**: judge with `--eval-only`; the number is *in-trainer KL*, not served KL.

## Files
- `test224.py` — CPU unit checks (identity = plain bf16 weights bit for bit, forward vs a float64 interpolation, table gradient vs finite differences for 1 and 3 groups, rounded / straight-through, `apply_remap` selection by K and its off-switch). `python3 test224.py`.
- `cmp224.py CONTROL/eval.json ARM/eval.json ...` — paired per-document bootstrap of two `--eval-only` passes (cmp214's arithmetic).
- Queue lines: `jobs.w224.txt` (gates + e0) and `jobs.w224.arms.txt` (arms + evals) — iceman card 0, `/data/queue`.
