# #103 · down_proj block act order through a folded gate/up permutation (review 2026-11-05 idea 8), CPU proxy

Worker w38, colo. Source: `local/mine/review-2026-11-05-ideas.md` idea 8. Changes down's codes, so it must be decided
before the 27B ladder. **Runtime cost: none** (the permutation is folded into gate/up's stored row order and down's
column order; same format and kernels).

## Question

#95 found act order *inside* a 128-column Hadamard block useless (the rotation flattens sub-block H mass, CV
0.01–0.02), but the block traces still differ 10–150× *across* 128-blocks. For down_proj an across-block order is
exact and free: permute whole 128-row blocks of gate/up outputs (their tiles move, their output-Hadamard blocks stay
intact, so gate/up codes are unchanged) and the matching 128-column blocks of down. Does quantizing down's 128-blocks
in descending block trace(H) order lower its proxy error?

## Setup

- `trellis3b.py` = `v3/e3/trellis3.py` (#95's harness) plus `E3_ORDER=blk128`: down's 128-column blocks sorted by
  descending trace of their diagonal H block (rotated H; a block's trace is rotation-invariant), stable; sub-blocks
  inside a block in natural order. W and H permuted, LDLQ in that order, codes permuted back, error measured in the
  original order. Natural and blk128 run in one job on the same rows (paired).
- #95's settings: codebook cubic0.02_i8 (i8c), 128 rows, rotation seed 0, damp 0.01, factors {0.85, 1, 1.15, 1.3, 1.5},
  CD off, free start (as #95; the order is a quantizer choice, the ring constraint does not interact with it).
- Tensors: **#95's down tensors, ffn_down L10 and L30** (the rule), plus L0 / L43 / L63 (#57's down tensors, reported
  as a robustness check). K2 / 3 / 4. `run.sh` (colo, 4 single-threaded jobs, nice 10), `report.py data/ [r95.jsonl]`
  (the natural arm must reproduce #95's numbers from `v3/e3/sub1/r95.jsonl`).

## Rule (from the task, written before data; statistic fixed here 2026-11-05 ~11:05 PM CST)

**Adopt if blk128 / natural ≤ 0.98 at ≥ 2 of the 3 K, no K worse than 1.00**, the ratio per K being the geometric
mean over #95's two down tensors (L10, L30). The three extra layers do not enter the rule; if they disagree with it
(e.g. their mean > 1.00 where the rule passes) that is reported as a caveat.

## Results (2026-11-06 4:01 AM CST) — **fail: not adopted**

Full table: `report.md` (`python3 report.py data/ ../sub1/r95.jsonl`), raw lines `data/`. The natural arm reproduces
#95's numbers exactly (L10 / L30, K2 / 3 / 4: ratio 1.000000).

**Rule (#95's down tensors L10, L30; geometric mean of blk128 / natural):** **0.9942 (K2), 0.9970 (K3), 0.9976 (K4)**:
no K ≤ 0.98 (none > 1.00 either). **Not adopted.**

All down tensors run (the rule's two, #57's three, then three picked for a large block-trace spread):

| layer | K2 | K3 | K4 | block trace max / min | p90 / p10 |
|---|---|---|---|---|---|
| 10 (rule) | 0.9970 | 0.9973 | 0.9973 | 2.0 | 1.48 |
| 30 (rule) | 0.9915 | 0.9967 | 0.9978 | 2.4 | 1.18 |
| 43 | 0.9974 | 0.9977 | 0.9971 | 1.6 | 1.27 |
| 0 | 0.9757 | 0.9762 | 0.9763 | 4.7 | 2.80 |
| 63 | 0.9528 | 0.9588 | 0.9595 | 22.7 | 4.97 |
| 2 (extension) | 0.9887 | 0.9894 | 0.9910 | 82.5 | 2.08 |
| 58 (extension) | 0.9899 | 0.9905 | 0.9903 | 51.7 | 1.70 |
| 61 (extension) | 0.9936 | 0.9949 | 0.9937 | 20.7 | 1.83 |

Reading:
- The order never hurts (24 / 24 ratios < 1.00) and is free at runtime, but on a typical layer it gains 0.2–0.5%.
  The gain follows the *bulk* spread of block traces (p90 / p10), not the max / min (L2's 82× comes from one or two
  outlier blocks, −1%): ln ratio ≈ −0.023 · ln(p90 / p10) (residual sd 0.005).
- **Census** (`census.py`, `census.jsonl`: the 128-block traces of every layer's calib H_in, read from the diagonal
  only): down_proj p90 / p10 has median 1.35; only layers 0 and 63 reach ≥ 2.5. Predicted by the fit over all 64
  layers: mean ratio **0.992** (−0.8% of down's proxy error; median layer −0.7%; best L63 −4%). down is ~¼ of the
  trunk's weights (89M of ~380M per layer), so ~−0.2% trunk proxy error: far below anything a 27B KLD test can see. Not worth the converter
  work (permuting gate / up rows with their scales and down's columns, plus an exactness test) before the ladder.
- For idea 8's other targets (not tested, reported for whoever picks them up): the other inputs have a much larger
  bulk spread — p90 / p10 median 1.90 (residual readers, `input`), 1.45 (`post_attention.input`), 2.58 (GDN
  out_proj), 2.83 (attn o_proj). By the down fit that is ~−1.5…−2.5% proxy error on o_proj / out_proj (head-group
  permutations, exact) and ~−1.5% (`input` readers) / −0.9% (gate / up) on the residual readers (one global order shared by all readers;
  RMSNorm commutes).
  The fit is from down_proj only; any of these would need its own test with its own rule.

**Runtime cost:** none (a folded permutation; not adopted anyway).

## Files

`trellis3b.py` (trellis3.py + `E3_ORDER=blk128`), `run.sh` (`LAYERS` for the extension), `report.py`, `report.md`,
`census.py`, `census.jsonl` (all 256 input Hessians: down, input, post_attention, out_proj, o_proj), `data/`. Ran on
colo `/data/work/w13/w38/actblk`, 2026-11-05 11:00 PM – 2026-11-06 4:01 AM CST, ≤ 4 single-threaded jobs (~12 GB
each; total with #102 ≤ 10). The L2 / L58 / L61 extension was added at ~1:45 AM CST after the rule's tensors were in,
to check whether the order should be adopted anyway at zero runtime cost; it does not change the verdict.
