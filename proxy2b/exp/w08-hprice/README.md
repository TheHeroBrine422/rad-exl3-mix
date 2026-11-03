# w08-hprice: give H_out quants their own price in knap4 (ranked.md F1)

**Question.** knap4 prices each measured ladder level from ourq's one-sided proxy error (`KN_LEVELS` reads
`proxy.jsonl` `proxy_err`, the error in the input Hessian alone). An H_out quant (two-sided LDLQ with the diagonal
output Hessian, #33-12/#33-16) trades one-sided error for error in the output rows that matter less. So the
one-sided proxy may call such a quant worse even when KL falls, and the solver would then never pick it. How large
is that effect, and what should replace the one-sided price?

**Mechanism.** For a linear W with output rows r, input Hessian H and output diagonal h:
- one-sided error e1 = Σ_r E_r H E_rᵀ / Σ_r W_r H W_rᵀ (what knap4 reads today);
- two-sided error e2 = Σ_r h_r E_r H E_rᵀ / Σ_r h_r W_r H W_rᵀ (what `ldlq_2hess` minimises; KL to second order,
  if h is the output Fisher diagonal).

The fix is an option `oursH-K<k>` next to `ours-K<k>`. It costs the same bytes. Its error is the one-sided
option's error × r × KN_HF:
- r = e2(H quant) / e2(one-sided quant), per tensor and level;
- KN_HF = the H family's own factor, fitted from uniform-trunk swaps (default 1).

## Result of the CPU check (done here; the rule is ranked.md F1's, written before this run)

ranked.md's F1 rule: *"one-sided proxy ratio bdiag/base per tensor. If > 1.05 where KL fell, the fit must switch."*

I recomputed e1 and e2 per tensor for the existing 0.8B arms, using their dequantized weights, the bf16
checkpoint and the calib Hessians (`rowerr2.py`). Arms: base / bdiag / edy over 3 rotation seeds, paired by seed;
bident at seed 0.

| arm (vs base) | K | whole-model KL | one-sided e1 (geomean) | tensors with e1 > 1.05 | two-sided e2 (geomean) |
|---|---|---|---|---|---|
| bdiag (8K Sketch-B diag, #33-16) | 3 | **0.919** | **1.065** | 49% | **0.862** |
| bdiag | 2 | 0.946 | 1.063 | 50% | 0.872 |
| edy (Σdy², 1K, #33-18) | 3 | 0.912 | 1.064 | 49% | 0.863 |
| edy | 2 | 0.937 | 1.063 | 49% | 0.872 |
| bident (H_out = I) | 3 | **1.049** | **0.919** | 0% | **1.037** |
| bident | 2 | 1.021 | 0.925 | 0% | 1.036 |

- **The rule fires.** The one-sided proxy calls the H_out quants 6% worse, and worse than 1.05 on half the
  tensors, while KL falls 5–9%. For the identity arm it gets the sign wrong the other way: it calls bident 8%
  better while KL rises 2–5%. e2 has the right sign in all six rows.
- **By type** (K3 bdiag, e1 → e2):
  - GDN in_proj_qkv 1.16 → 0.79, in_proj_z 1.22 → 0.77;
  - attention q 1.18 → 0.73, v 1.17 → 0.89;
  - gate 1.11 → 0.87, up 1.06 → 0.86;
  - residual writers (out_proj, o_proj, down) 0.94–0.95 under both.

  `f1check.out` has the full table.
- **Not in-sample flattery.** The bdiag rows above price bdiag with the same 8K diagonal it was quantized with.
  Pricing it with an independent diagonal (c18, calib qad windows, 1K sequences) gives e2 0.866 / 0.875
  (`f1check-c18.out`). The edy rows use edy's own diagonal (`f1check-edy.out`).
- **KN_HF on the 0.8B.** e2 overstates the KL gain somewhat. KL ratio ÷ e2 ratio = 1.06–1.07 at K3 and
  1.07–1.09 at K2. This is a whole-model fit with equal tensor weights. On the 27B it gets fitted from swaps instead
  (`hfit.py`).

**Found on the way: ourq's `--save-wq` `rowerr.npz` files are wrong.** On all 150 0.8B tensors, err / ref =
1.000–1.002.
- Cause: quantize_exl3 regularizes its weight argument in place, and ourq passes `w` itself.
- w04-blockgain found the same bug independently and has the fix (`w04-blockgain/rowerr.patch`). My patch builds
  on it.
- Consequence: headslab's slab-head plans (#34, E1 / w05-lowhead) were made from these numbers.

## Patches (for the lead to merge)

1. **`knap4-hratio.patch`** (v3/nf/knap4.py). Adds:
   - `KN_HRATIO=FILE`: options `oursH-K<k>`, int8 modes included;
   - `KN_HF`: the family factor;
   - `KN_HONLY=1`: drop the one-sided twin of every oursH option;
   - `KN_HRATIO_NEAR`: the nearest-level fallback.

   Containers go to `RXM_V3/oursH-K<k>.rad` (or `RXM_OURS` without `KN_V3`). The patch applies to the attached
   version and to HEAD. With w05-lowhead's `knap4-headcurve.patch`, apply w05's first.
2. **`ourq-hprice.patch`** (quantize/ourq.py), applied **after** `w04-blockgain/rowerr.patch`. Adds `--hprice DIR`,
   which is logging only. With it, proxy.jsonl gets two fields per linear: `proxy_err1` (one-sided, raw H) and
   `proxy_err2` (h-weighted). `rowerr.npz` now uses the same arrays.
   - It stacks with w01's sidecheck, w02's refit-ourq and w05's ourq-hout-bd (checked: apply and compile).
   - Like any ourq change, merging it changes the code stamp.
   - **Use:** add `--hprice <the 27B diagonal>` to ladder2.sh's settings in the Phase-B requant (both ladders) and
     to both arms of the 27B swap test. That makes the 27B ratios free, with no extra quant runs.

## Files and how to run

| file | what |
|---|---|
| `rowerr2.py` | e1 / e2 per linear from `wq` + checkpoint + calib H (CPU; 0.8B: 22 s an arm on the 4070's CPU, 4 threads) |
| `f1check.py`, `f1check*.out`, `rowerr2-0.8b*.jsonl` | the check above and its data |
| `hratio.py` | proxy.jsonl / rowerr2 pairs → KN_HRATIO lines (geomean over seed pairs) |
| `hfit.py` | KN_HF from a measured uniform-trunk swap, using knap4's weight costs |
| `mkratio.py` | test ratio files: identity, and the 0.8B per-type ratios mapped onto the 27B tensors |
| `knaptest.sh`, `knaptest/` | knap4 patch tests (colo CPU, ~2 min) |
| `f1fam.py` | per-family check against w10-hout-attrib's subset arms |
| `gatecmp.py` | ourq-hprice gate |
| `run.sh` | the 4070 run |
| `price27.sh` | the 27B pricing pipeline (CPU; inputs from iceman's ladders) |

- **4070:** `proxy2b/exp/w08-hprice/run.sh`. Results go to `/data/yb/w08-hprice/result.txt`.
  - Step 0, the patch gate: three two-layer K3 quants, ~3 min GPU.
  - Step 1, per family: CPU only, ~4 min. It waits for w10-hout-attrib's `ATTRIB DONE`. Rerun run.sh afterwards;
    it is idempotent.
- **27B (CPU, once the iceman ladders carry `--hprice`):**
  `price27.sh <one-sided ladder root> <H ladder root> 2.5,3,... [swaps.txt]`. `swaps.txt` has one line per swap:
  "K KL_one KL_H".

**GPU time:** ~3 min on the 4070. Nothing new on iceman: the 27B part piggybacks on the Phase-B requant and the
27B swap test.

## Draft pass rules (for the lead to commit before data)

1. **Gate** (run.sh step 0, `gatecmp.py`):
   - every shard tensor is bit-identical between REF (unpatched), OFF (patched, no `--hprice`) and ON (`--hprice`);
   - `proxy_err1` is within 2e-3 of `proxy_err` on every linear;
   - the new `rowerr.npz` sums equal `proxy_err1` (1e-9).

   Otherwise: do not merge.
2. **Per family** (run.sh step 1, `f1fam.py`, on w10's Σdy² subset arms, seed-paired, SE floored at 0.5%):
   - A (family, K) case *disagrees* if its tensors have e1 > 1.05 and e2 < 0.95. From the edy rows I expect GDN
     in, attention in and gate/up at K3 and K2 to disagree; the writers do not.
   - e2 is right in a case if the family's measured gain is ≥ 2 SE; e1 is right if it is ≤ −2 SE.
   - **PASS:** e2 is right in every disagreeing case → merge knap4-hratio and use KN_HRATIO for every H_out level.
   - **FAIL:** e1 is right in some case → that family gets no oursH option (drop its lines from the ratio file),
     and the per-type ratios go to the lead before anything ships.
   - Otherwise **INCONCLUSIVE:** the whole-model result above stands, and families are not split.
3. **27B KN_HF** (`price27.sh`):
   - KN_HF is the geomean of `hfit.py` over the 27B swap test's uniform-trunk swaps (K3 and K2.5, per ranked K1).
   - Use it if all swaps agree within ×1.1.
   - If they disagree by more, fit one KN_HF per level and interpolate in log K.
   - A KN_HF > 1.25 means two-sided pricing overstates the gain by more than a quarter. In that case, before
     knap4 picks oursH, also check a mixed-trunk swap (oursH on the types knap4 picks only).

## Self-tests (CPU)

- **`selftest_rowerr2.out`:** rowerr2's e1 matches ourq's own `proxy_err` (the refit error, damping removed) to
  ≤ 4.7e-5 relative on 4,200 tensor-arms.
- **`knaptest/knaptest.out`:**
  - patch off = attached knap4, byte for byte (V2-like menu and V3 table);
  - r = 1 everywhere with KN_HONLY gives the same KLD, bpw and time as without KN_HRATIO;
  - illustration: the 0.8B's per-type ratios on the 27B, KN_HF 1.07, both ladders. Predicted KLD falls 6.6–6.8%
    at 2.5 / 3.0. knap4 takes oursH for 65% of the trunk parameters and keeps residual writers one-sided
    (0.95 × 1.07 > 1). This is not a 27B prediction: the ratios are the 0.8B's.
- **`selftest_price27.out`:** price27.sh end to end on fake ladders built from the real V3 ladder's proxy files.
  The ratios, hfit and both knap4 solves run.
- **`selftest_hfit.out`:** round trip. Feeding hfit its own KN_HF = 1 prediction returns KN_HF 1.000.
- **`selftest_f1fam.out`:** planted family effects are called right (e2 / e1 / FAIL).
- **`selftest_gatecmp.out`:** PASS on matching dirs; FAIL on one changed trellis word; FAIL on an old-style rowerr.

## What changes what

- **F1 (ranked #5):** confirmed on the 0.8B. Without the patch, knap4 would rate the H_out ladder 6% worse per
  tensor and drop it, wasting the 27B collection. With it, the solver sees roughly the right sign and size.
- **INDEX:** a #33-xx row: "one-sided proxy misprices H_out quants".
- **B3 (w10):** the writers' e2 ≈ e1 ≈ 0.95 here fits w10's expectation that writers gain little from H_out.
- **E1 (w05-lowhead) / #34:** the slab-head plans need recomputing after the rowerr fix (w04 says the same).
- **F2** (kfit2 on a two-sided proxy for all options) gets cheaper: `--hprice` on the Phase-B ladder logs e2 for
  every new level.

## Runtime cost (user rule, 2026-11-03)
none (allocation pricing; picks among priced options).
