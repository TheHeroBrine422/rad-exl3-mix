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

---

# #106 · o_proj / GDN out_proj input-block act order through a head permutation (w46, CPU proxy settle.py)

From #103 (BACKLOG): the block-trace bulk spread p90 / p10 of these inputs (median 2.83 o_proj, 2.58 out_proj, census
above) is well above down_proj's (1.35), where #103 found −0.2…−0.5% on typical layers and −2…−5% at p90 / p10 ≥ 2.8
(fit: ln ratio ≈ −0.023 · ln(p90 / p10) → predicts ~−2% here). Written 2026-11-06 ~10:15 AM CST, **before any data**.
**Runtime cost: none** (a permutation folded into the stored tensors; same format and kernels).

## What is exact

- **o_proj** (24 q heads × 256, 4 KV heads, 6 q heads a group): its input columns are the attention heads' outputs.
  Heads inside a KV group can be reordered freely (q / gate rows of q_proj + o_proj columns); whole groups can be
  reordered with their K / V heads (k_proj / v_proj rows too). Moving whole heads moves whole 128-row output-Hadamard
  blocks of q / k / v, so their codes are unchanged (only their row order).
- **GDN out_proj** (48 v heads × 128, 16 k heads, 3 v heads a group, assumed contiguous as repeat_interleave; check
  in the converter): v heads inside a group freely (v rows of in_proj_qkv, z rows, the head's a / b rows, A_log,
  dt_bias, conv channels + out_proj columns); whole groups with their q / k heads. The gated RMSNorm weight is shared
  by all heads, so a head permutation leaves it alone.
- So the expressible orders are: groups in any order, heads inside a group in any order, a head's 128-blocks
  (2 for attention) in natural order. A free 128-block order is *not* expressible (it would split heads); it is run
  as an upper bound only.

## Setup

`v3/e3/settle/reval/reval.py order` (settle.py proxy v2: exllamav3's regularize transforms, its g_scale search,
`--scaled-hessian` LDLQ, the per-output-channel refit; codebook mul1_i8, damping 0.025, m = 1, 128 rows). The
permutation is applied in the transformed basis (whole 128-blocks of the transformed rows and of the LDLQ metric;
signs, rows, scale and tile sample identical across arms, so arms are paired), codes are permuted back and scored in
the original order. Block importance = trace of the block's diagonal block of the LDLQ metric (P diag(su) (H + δI)
diag(su) P). Arms: `natural`; **`head`** = groups by descending total trace, heads inside a group by descending trace
(the arm the rule judges); `free` = all blocks by descending trace (bound, reported); `rev` = ascending inside the
head constraint (sign check, reported). Samples: seed 0 (sign seeds 0, row block 0) and seed 1, as w43.
Check: the natural arm must reproduce w43's settle.py gain rel at m = 1.0 on the same tensor / K / seed.

**Rule tensors** (w43's three per family, picked over depth before any trace data): attn_output L19 / L43 / L63,
ssm_out L10 / L30 / L50; K2 / 3 / 4; seeds 0 and 1. Extension (reported, not in the rule): every other o_proj
(12 layers) and ssm_out L0 / 2 / 17 / 26 / 42 / 46 / 49 / 52 / 61 / 62, seed 0.

## Rule (#103's style; written before data)

Per family (o_proj and out_proj judged separately: each is its own converter change), at each K: ratio =
geometric mean of rel(head) / rel(natural) over the family's 3 rule tensors × 2 seeds (after the refit).
**Adopt for the family if the ratio is ≤ 0.98 at ≥ 2 of the 3 K and no rule tensor is worse** (no tensor's
seed-pooled geometric mean > 1.00 at any K). The extension layers do not enter the rule; disagreement (their mean >
1.00 where the rule passes) is reported as a caveat. Adoption path as every proxy pass: the 27B confirmation queue.

## Results (2026-11-06; rule tensors ~1:00 PM CST, extension and K1.5 / 5 / 6 pricing by 6:05 PM CST)

**o_proj: PASS (adopt, by the rule) — GDN out_proj: fail (not adopted).** Natural arm check: it reproduces
w43's settle.py rel for 19:attn_output K2 seed 0 exactly (0.02772837462809245).

Rule tensors × 2 samples (n = 6 per K), geometric mean of rel(arm) / rel(natural) after the refit:

| family | K | head (rule) | free (bound) | rev (sign check) | worst rule tensor (head) | tensors > 1.00 |
|---|---|---|---|---|---|---|
| o_proj | 2 | **0.928** | 0.926 | 1.055 | 0.956 (L19) | none |
| o_proj | 3 | **0.927** | 0.924 | 1.053 | 0.955 (L19) | none |
| o_proj | 4 | **0.927** | 0.925 | 1.054 | 0.954 (L19) | none |
| out_proj | 2 | 0.986 | 0.983 | 1.025 | 0.992 (L30) | none |
| out_proj | 3 | 0.986 | 0.983 | 1.027 | 0.991 (L30) | none |
| out_proj | 4 | 0.988 | 0.985 | 1.027 | 0.998 (L30) | none |

- **o_proj passes at all three K** (≤ 0.98 at ≥ 2 K, no tensor > 1.00): −7.3% proxy error. The head-constrained
  order gets nearly all of the free order's gain (0.927 vs 0.925), and the reversed order costs +5%: the effect has
  the expected sign and size ordering. Extension (sample 0, all 16 o_proj layers): 0.943 / 0.942 / 0.943 at
  K2 / 3 / 4, every layer < 1 (0.902 L43 … 0.979 L3), so no caveat. Priced outside the rule's K (block `i4k`, all
  16 layers): 0.944 K1.5, 0.942 K5, 0.942 K6; the rule tensors on both samples: 0.929 / 0.927 / 0.928. Flat in K.
- **out_proj fails** (−1.2 to −1.4%, none ≤ 0.98). Its extension (13 layers incl. the rule's three) gives
  0.977–0.978, best L61 / L62 (0.960–0.967, the largest trace spreads), but L30 is at 1.0017 at K4.
- Size vs #103's fit: it predicted ≈ −2% from the H-trace spread (p90 / p10 2.6–2.8). o_proj gains 3× that at a
  similar spread of the LDLQ metric's block traces (2.56), so the down_proj fit does not transfer to o_proj.
- What it is worth: o_proj is 31.5M weights × 16 attention layers ≈ 2% of the trunk's weights. −7% on its proxy
  error is roughly a −0.1% trunk proxy error, well below a 27B KLD test's resolution unless o_proj carries an
  outsized share of the KLD. The rule passes, so it goes to the lead and the 27B confirmation queue. It is free at
  runtime and needs only a converter / quantizer change: permute o_proj's input columns by head (groups by trace,
  heads in a group by trace) together with q_proj's q / gate rows and the group's k / v rows, plus an exactness test.
- Not measured: whether the real quantizer's tile sample (it would see the permuted tensor) moves g_scale. Here the
  arms share one scale (paired); the effect would be second order.

Raw lines: `data/e3/settle/reval/<box>/out/order/`. Tables: `data/e3/settle/reval/report.md` "## 4"
(`v3/e3/settle/reval/revalreport.py`); run details: `v3/e3/settle/reval.md`.
