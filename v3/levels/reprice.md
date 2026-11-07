# V3 ladder re-priced in the knapsack with the 2026-11-06 facts (w58, 2026-11-06 evening CST)

**Status:** report only; nothing adopted. **Every KLD below is a kfit2-model prediction** (knapsack output), not a
measurement. **Runtime cost:** none (pricing only: same format, kernels and quantizer).
Solver: `reprice/knap5.py` = knap4 + switches (`reprice/README.md`); 834 solves (16 sizes 2.0–7.0, 18 arms, λ grid),
CPU on Thor + colo, 6:29–9:22 PM CST. Data `data/levels/reprice/out/`, all tables `data/levels/reprice/report.out`
(`python3 v3/levels/reprice/report.py data/levels/reprice/out`).

> **Corrected 2026-11-06 11:20 PM CST (w63): read the Correction section first.** The swj curve below (×0.78 flat from
> K4) used the withdrawn K4 number. Every "+ swj" / main / `gu` KLD at ≥ 3.0 in the sections after the Correction is
> too low by 7–33%. The menu-only (`noswj`), level-usage and fast-7.0 conclusions stand.

## Correction 2026-11-06 11:20 PM CST · swj curve re-based on the head-matched pairs (w63)

The swj factor used here (×0.9346 K2.5 → ×0.7786 K4, flat above) came from the 6:10 PM K4 pairs. Those pairs mixed
different heads and embeddings (LOG 2026-11-06 7:50 PM: withdrawn). Head-matched swj is ×0.935 at K2.5 (verify
×0.932) and ×0.989 at K4. The 0.8B gives the shape: −8.6% at K2, −3.1% at K3.

**Rerun:**
- **Arms rerun:** `main` (swj `all`, per level), `gu` (gate_up only) and `noi8k7`. Everything else is identical:
  knap5.py (same md5 as w58), env, sizes, λ grids and fast-variant rule.
- **The only change is `KN_SWJ_PTS=2:0.91,2.5:0.935,4:0.989,5:1.0`.**
  - Interpolation is knap5's `lin()`: linear in K between the points, flat outside. So r = 0.91 at K ≤ 2, 0.953
    at K3 and 1.0 at K ≥ 5.
  - The K3 value is above the 0.8B's 0.969 there: the line was not bent to fit it.
- **`gu` calibration.** The gate_up factors were re-solved to reproduce r(K) at uniform K: K2 ×0.830, K2.5 ×0.861,
  K4 ×0.973, K5 ×1.0 (w58: ×0.860 K2.5, ×0.462 K4).
  - K2 uses knap5's default V2 4.00 head and embedding for the calibration (K2 has no 27B pair).
- **Runs:** 214 solves on Thor + colo (nice 10), 10:23–11:12 PM CST, 0 failures.
  - Data: `data/levels/reprice/swj2/out/`. The new arms keep their names. w58's are copied in as `w58main` /
    `w58gu` / `w58noi8k7`, with `v2menu`, `old` and `noswj` as the ratio bases.
  - Tables: `swj2/report.out` (report.py, same layout as `report.out`) and `swj2/compare.out` (`reprice/compare2.py`:
    old → new per size, pick changes).

**Predicted KLD, old → new** (all predictions). Anchored = V2 board × (prediction / V2-menu prediction); "–" = no V2
board point at that size.

| size | accurate: w58 → new (change) | anchored acc. w58 → new | fast: w58 → new (change) | anchored fast w58 → new | vs V2-menu: w58 → new (menu only) | 11-06 "V3 at freeze" exp. | `gu` vs new main |
|---|---|---|---|---|---|---|---|
| 2.0 | 0.1233 → 0.1207 (−2.2%) | 0.1221 → 0.1194 | 0.1234 → 0.1207 (−2.2%) | 0.1221 → 0.1195 | −25.7% → −27.3% (−19.3%) | 0.120 | +0.5% |
| 2.25 | 0.07923 → 0.07871 (−0.7%) | – | 0.0793 → 0.07878 (−0.7%) | – | – | – | +0.0% |
| 2.5 | 0.05287 → 0.05340 (+1.0%) | 0.0528 → 0.0533 | 0.05294 → 0.05347 (+1.0%) | 0.0528 → 0.0534 | −20.8% → −20.0% (−13.7%) | 0.0487 | −0.2% |
| 2.75 | 0.03502 → 0.03639 (+3.9%) | – | 0.03509 → 0.03646 (+3.9%) | – | – | – | −0.5% |
| 3.0 | 0.02360 → 0.02533 (+7.3%) | 0.0246 → 0.0264 | 0.02367 → 0.02540 (+7.3%) | 0.0247 → 0.0265 | −19.6% → −13.7% (−9.4%) | 0.0243 | −0.5% |
| 3.25 | 0.01633 → 0.01808 (+10.7%) | – | 0.0164 → 0.01815 (+10.6%) | – | – | – | −0.5% |
| 3.5 | 0.01124 → 0.01286 (+14.4%) | 0.0129 → 0.0147 | 0.01131 → 0.01293 (+14.3%) | 0.0130 → 0.0148 | −24.7% → −13.8% (−11.1%) | 0.0136 | −0.5% |
| 3.75 | 0.00786 → 0.009246 (+17.6%) | – | 0.00793 → 0.009319 (+17.5%) | – | – | – | −0.5% |
| 4.0 | 0.005555 → 0.006670 (+20.1%) | 0.00618 → 0.00742 | 0.005625 → 0.006743 (+19.9%) | 0.00625 → 0.00750 | −25.4% → −10.4% (−9.1%) | 0.00656 | −0.4% |
| 4.25 | 0.003981 → 0.004830 (+21.3%) | – | 0.004051 → 0.004908 (+21.2%) | – | – | – | −0.3% |
| 4.5 | 0.002917 → 0.003532 (+21.1%) | 0.00352 → 0.00427 | 0.002987 → 0.003609 (+20.8%) | 0.00361 → 0.00436 | −26.6% → −11.1% (−10.6%) | 0.00380 | −0.2% |
| 5.0 | 0.001562 → 0.001904 (+21.9%) | 0.00192 → 0.00234 | 0.001633 → 0.001980 (+21.3%) | 0.00201 → 0.00243 | −25.0% → −8.6% (−8.4%) | 0.00226 | −0.1% |
| 5.5 | 0.0009065 → 0.001077 (+18.8%) | – | 0.0009766 → 0.001152 (+18.0%) | – | – | – | 0.0% |
| 6.0 | 0.0005706 → 0.0006634 (+16.3%) | 0.000716 → 0.000833 | 0.0006474 → 0.0007413 (+14.5%) | 0.000813 → 0.000931 | −21.6% → −8.9% (−8.9%) | 0.000806 | 0.0% |
| 6.5 | 0.0004003 → 0.0004517 (+12.9%) | – | 0.0004995 → 0.0005615 (+12.4%) | – | – | – | 0.0% |
| 7.0 | 0.0003106 → 0.0003382 (+8.9%) | 0.000361 → 0.000393 | 0.0004494 → 0.0004940 (+9.9%) | 0.000522 → 0.000574 | −19.8% → −12.7% (−12.7%) | 0.000397 | 0.0% |

**What this changes (predictions):**

- **swj's modelled gain** (main vs `noswj`): −10% at 2.0, −7% at 2.5, −5% at 3.0, −3% at 3.5, −1.5% at 4.0,
  −0.5% at 4.5, −0.2% at 5.0, 0 from 5.5. In w58 it was −8…−23%.
  - The 2.0 gain is now larger than before (r = 0.91 at K ≤ 2 instead of 0.935). That point rests on the 0.8B's K2
    shape, not a 27B pair.
  - From 5.5 up, main = `noswj` exactly.
- **Projections.** At ≤ 2.5 the 11-06 projection holds: −27% at 2.0 (projection −26%), −20% at 2.5. At 3.0 it is now
  −14% (projection −16%).
  - w58's "from 3.5 up, menu + swj alone already reaches the 11-06 'V3 at freeze' expectation" is **withdrawn**.
    Menu + swj alone is anchored 3–13% above that expectation at 2.5–6.0: +9% at 2.5 and 3.0, +8% at 3.5, +13% at
    4.0, +12% at 4.5, +3.5% at 5.0, +3% at 6.0.
  - The rest has to come from H_out, writers, healing and search, which the model does not price. At 2.0 and 7.0
    the menu alone meets the expectation (0.1194 vs 0.120; 0.000393 vs 0.000397).
- **Picks.**
  - **Tensor picks:** 28–132 of 337 tensor picks change at 2.0–4.5 (most at 3.5–4.5) and 66–73 at 5.0–7.0.
  - **Level usage barely moves.** Same never-picked set (the 17 i4 levels), and every pass-1 level is still used. i8
    stays to K5.75 and mul1 from K6 in the accurate variant. The fast variant stays all-A8, i8 at K ≥ 6.
  - **Embedding:** 0.25–0.5 bit lower at 3.75 / 4.5–6.0: t2.75 / t3.75 / t4.5 / t5 / t5.5 instead of t3.25 / t4.25 /
    t4.75 / t5.25 / t5.75. **Heads unchanged.**
  - **New: turboderp's tensors return at ≥ 4.25.** Accurate param share (`sc4.00-e4` / `sc5.00-e5` at 4.25–4.5, mostly
    `sc5.00-e5` at 5.0, sc5 + sc6 at 5.5, mostly `sc6.00-e6` at 6.0–7.0): 6% at 4.25, 8% at 4.5, 12% at 5.0, 13% at 5.5, 17% at 6.0, 18% at 6.5 and 10% at 7.0 (51–64 tensors at
    6.0–7.0). w58 had ≤ 0.3%.
    - Cause: the old ×0.78 discounted only our levels, so it priced turboderp's out. `noswj` already picked them in
      w58. The fast variant stays ≤ 0.6% turbo (no A8 path).
    - This lowers the accurate mul1 share at 6.0–7.0 (48 / 67 / 84% instead of 58 / 83 / 94%). It does not touch the
      i8-vs-mul1 split.
- **`gu` vs `all` is now moot.** Within −0.5…+0.5% at every size, identical from 5.5 up. w58 had −4.8% at 4.0 and
  −8.3% at 5.0. At K4 gate_up sits at 3.86 vs 4.12 for the rest (main 3.89 / 4.08).
- **Fast 7.0: conclusion unchanged, and a little worse.**
  - The fast 7.0 is all i8 K7 (87% of params), +46% vs the accurate 7.0. Anchored it is **0.00057** (w58: 0.00052),
    27% worse than V2 7.00's 0.00045.
  - Mixed-mode 7.0 points: ×1.24 GEMM 0.000379 (+12%, ~0.00044 anchored, ≈ V2 7.00) and ×1.45 0.000438 (+30%,
    ~0.00051).
  - The fast 6.0 now anchors at 0.00093, about V2 6.00's 0.000914. w58 had 0.00081.
  - **Dropping i8 K7 (`noi8k7`):** +0.0% at fast 7.0 under the 2% rule. The solver keeps ~1% GEMM
    (×1.516 instead of ×1.530) by putting a few tensors back on f16 mul1. At equal all-A8 (λ 1e-5) the cost is +4.2%
    (w58: +2.9% under the rule, +3.6% at λ 1e-5). The fast 6.5 cost is +0.2% (unchanged). So i8 K7 stays a
    fast-7.0-only level worth ≤ 4%.
- **Unchanged:** every menu-only (`noswj`), `old`, `v2menu`, pass-1, drop/add-level and CD result below. Their
  deltas were taken against the old main. They were not rerun, but they are relative and swj-insensitive at ≤ 2.5;
  at ≥ 3.0 treat them as approximate.

**For the lead:**

1. Fast 7.0: same options as below (skip it / mixed ×1.24 at ≈ V2 7.00 / point fast users at 6.5). The case for
   skipping it, or for a mixed fast 7.0, is now stronger.
2. Whether the V3 mixes should use turboderp's sc5 / sc6 tensors at ≥ 5.0 the way the model now prices them
   (10–18% of params). If not, the knapsack needs them removed from the accurate menu; that arm was not run.
3. Drop the per-family (`gu`) vs per-level swj question: it no longer moves any size by more than 0.5%.
4. Quote 3.0–6.0 predictions from this section, not from the tables below.

## Summary (predictions)

1. **Picks.** The accurate variant uses i8 up to K5.75, then **mul1 only at K ≥ 6**. mul1 is 0.2% of params at 4.5,
   4% at 5.0, 17% at 5.5, 58% at 6.0, 83% at 6.5 and 94% at 7.0. The accurate variant **never** picks an i8 level
   ≥ K6, because on f16 mul1 matches it byte for byte with less error. The fast variant is all A8 (a8g128, ≥ 99% of
   FLOPs) at every size, and all its K ≥ 6 bytes are i8 (mul1 ≤ 1%). So **i8 K6.25–7 are fast-only levels and mul1
   K6.25–8 accurate-only**.
   **i8 K7 is still picked:** 88% of params in the fast 7.0. At ×1.27 it buys little: dropping it costs +2.9% at
   fast 7.0 and +0.2% at 6.5. The fast 7.0 itself is weak: **+45% KLD vs the accurate 7.0**, and board-anchored
   0.00052, which is worse than V2 7.00's 0.00045.
2. **KLD vs the 11-04 projection.** The menu alone (no swj) is −19% vs the V2 menu at 2.0 (projection −21%), −9% at
   3.0 (−10%), −9% at 4.0 (−10%) and −8% at 5.0 (−6%): **the projection holds**. Applying swj per level (×0.935 K2.5
   → ×0.78 K4) adds −8% at 2.0–2.5 and −18% at 4–5. Combined: −26% at 2.0, −20% at 3.0, −25% at 4.0 and 5.0, −20% at
   7.0. The 11-06 projection said −26% at 2.0 and −16% at 3.0. **Above K4 the swj factor is extrapolated** (held at
   ×0.78) and drives the 4.5–7.0 numbers.
3. **Levels.**
   - **Never picked at 2.0–7.0:** all 17 i4 levels (K1 and the K1.125–3 twins). A4 is not picked even when offered
     (arm `a4`), and the sub-1 i4 levels are not picked at ≥ 2.0 (#90).
   - **Rarely picked:** i8 K1.125 and K1.25 (1–3 tensors, 2.0 only; dropping them costs +0.0%). They are still
     needed for the 1.5 / 1.75 research points.
   - **Duplicates:** mul1 K6 / K6.25 next to i8 K6 / K6.25 (dropping mul1 costs ≤ +0.14%).
   - **Pass 1 (25 levels) costs only +0.2…+0.8%** vs all 73 levels at every size, and every pass-1 level is used.
   - **Missing levels that would be picked but don't pay:** 1/8 steps at K5–7 (−0.1…−0.2%) and 1/16 head budgets
     (0.0% under conservative pricing; never picked).
   - Tile CD on attn_qg: −0.01…−0.06%, with 0–5 picks changed.

## What changed in the pricing (and how)

| fact (LOG 11-06) | old knap4 V3 config (11-01…05) | here |
|---|---|---|
| i8 / mul1 per K (`r94b.md`, `reval.md` items 2, 5) | f16 options = mul1 errors at every ladder K; a8 = ×F8 lerp {6: 1.004, 7: 1.25, 8: 4.5} → K6.5 ×1.12, K6.75 ×1.18 | separate options per codebook. i8-K: error ×R8 = 1.000 ≤ K3, 1.001 K4, 1.002 K5, 1.004 K6, **1.018 K6.5, 1.27 K7**, K6.75 ×1.070 (log-linear in the excess; sensitivity `r8ratio`: ×1.137). i8 takes f16 or a8; no i8 above K7. mul1 K6–8 in quarter steps, f16 only (the old menu lacked mul1 K7.25–7.75) |
| i4 vs i8 at K ≤ 3 (`reval.md` item 1, pooled) | int4 grid only as post-hoc F4 on a4 | i4-K1, K1.125–3: error ×R4 (1.001 K1 … 1.0088 K2, 1.018 K2.5, 1.062 K3); f16 / a8 / a4 |
| swj at every level | — | **KL factor on the weight-cost terms only** (kfit2's per-tensor X, which feed the depth compounding). Not on the A8 activation terms, the embedding, the head, c0 or turboderp's tensors. Main arm `all`: every trunk level × r(K), r linear in K between K2.5 ×0.9346 and K4 ×0.7786 (the pairs' geometric means), flat outside (≤ K2.5: 0.9346; ≥ K4: 0.7786, untested above K4). Arm `gu`: ffn_gate_up only (swj refits gate / up scales), its factor solved so that the model reproduces the measured whole-model ratio at uniform K2.5 / K4 with V2 3.00 / 4.00 head + embedding: ×0.860 at K2.5 and **×0.462 at K4** (gate_up carries 41–45% of the model's KL there) |
| tile CD attn_qg ×0.99 | — | arm `cd`: attn_qg error ×0.99 at every level |
| family factor φ (kfit2: ours_lo K1–3, ours_mid K4–6, ours_hi K7–8) | fractional K takes floor(K)'s φ: a ×1.26 step between K6.75 and K7 | φ log-linear between floor and ceil (arm `phifloor` = the old convention) |
| speed | speedcal cores + decode placeholder, groups 0…1024 | TWOPASS.md measured at M 2048, K4: f16 14,474 / A8 9,444 µs per layer (×1.53); a8g128 only (what the kernel ships) |

Unchanged: kfit2 model (V2 hproxy errors × DAMPF, measured fractional positions `data/v3/levels`), trellis
embedding (KN_EMBT=2), slab heads K5 / K6, E1c activation costs, 256 KiB buckets.

**Fast variant rule** (fixed before reading the results): the lowest-KLD solution on the λ grid
(1e-8 … 1e-5 KLD/ms) whose GEMM time is within 2% of the fastest one at that size. The accurate variant is the
f16-only solve.

## 1 · Picks per size (main arm: swj `all`, no CD)

| size | accurate KLD | emb / head | mul1 share (accurate) | fast KLD | fast vs acc | GEMM × | fast: K ≥ 6 params (i8) |
|---|---|---|---|---|---|---|---|
| 2.0 | 0.1233 | t1.5 / K3 | 0 | 0.1234 | +0.1% | 1.53 | – |
| 2.5 | 0.05287 | t2 / K4 | 0 | 0.05294 | +0.1% | 1.53 | – |
| 3.0 | 0.02360 | t2.5 / K4 | 0 | 0.02367 | +0.3% | 1.53 | – |
| 3.5 | 0.01124 | t2.75 / slab5 | 0 | 0.01131 | +0.6% | 1.53 | – |
| 4.0 | 0.005555 | t3.25 / slab5 | 0 | 0.005625 | +1.3% | 1.53 | – |
| 4.5 | 0.002917 | t4.25 / slab5 | 0.2% | 0.002987 | +2.4% | 1.53 | <0.1% |
| 5.0 | 0.001562 | t4.75 / slab6 | 4.1% | 0.001633 | +4.5% | 1.53 | 3.3% (K6, 6.25) |
| 5.5 | 0.0009065 | t5.25 / slab6 | 17.3% | 0.0009766 | +7.7% | 1.52 | 16.4% (mostly K6) |
| 6.0 | 0.0005706 | t5.75 / K7 | 57.8% | 0.0006474 | +13.5% | 1.52 | 61.6% (K6 40, 6.25–6.75 21) |
| 6.5 | 0.0004003 | t6 / K8 | 82.6% | 0.0004995 | +24.8% | 1.53 | 88.3% (K6.75 31, 6.5 26, K7 4) |
| 7.0 | 0.0003106 | t6 / K8 | 93.5% | 0.0004494 | +44.7% | 1.53 | 95.1% (**K7 88**, K6.75 7) |

(2.25 / 2.75 / 3.25 / 3.75 / 4.25 in `report.out` T1, same pattern.)

- **Fast-variant cost.** Up to 4.5 the fast variant costs ≤ 2.5% KLD for ×1.53 GEMM. That cost is mostly the A8
  activation term (~7e-5, constant), which at 6–7 is 12–23% of the whole KLD.
- **The 6.5 / 7.0 frontier** (`report.out` T1b): 7.0 at ×1.26 GEMM costs +14%, at ×1.47 +32%, at ×1.53 +45%;
  6.5 at ×1.39 costs +13%. The mixed points keep mul1 (mostly K7.25–8) on f16 for 13–50% of the tensors.
- **i8 K7.** In the fast 7.0 there is nothing else to spend bytes on with A8, since i8 stops at K7. So it is "i8 K7
  everywhere", which is about mul1 K6.82 (r94b): the fast 7.0 is predicted only 10% better than the fast 6.5.
  - Old pricing (`oldf`) also picked i8 K7 (78%). Holding swj off, new vs old pricing changes the fast 6.0–7.0 by
    +0.3…+2%: cheaper K6.5 / K6.75 and a dearer K7 cancel.
  - Under the φ-floor convention the fast 7.0 picks i8 K6.75 for 95% instead (0.000433). That is a pricing
    artifact: it strands ~0.2 bpw.
  - So whether K7 or K6.75 carries the fast 7.0 is within model noise. The level is worth ~3% at fast 7.0 only.
- **Shared-input units and slabs:** head = slab K5 at 3.5–4.5 and slab K6 at 5.0–5.5 (as before). Turboderp's
  sc5 / sc6 tensors appear at 6.0–7.0 (1–2 tensors).

## 2 · Predicted KLD vs V2 and the 11-04 projection

"V2-menu" = the same model's knapsack on V2's menu (k4grid `v2`), the base of the projection's percentages.
"Board-anchored" = V2 board KLD × (prediction / V2-menu prediction). The model under-predicts V2's real mixes by
10–20% at ≥ 3.5 bpw (V2-menu / board 0.80–0.90), so ratios are the safer read.

| size | V2 board | menu only (noswj) vs V2-menu | 11-04 proj. (menu) | + swj (main) vs V2-menu | 11-06 proj. (+swj) | main board-anchored: acc (fast) | noswj anchored | `gu` anchored | projections-11-06 "V3 at freeze" (exp.) |
|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 0.1644 | −19.3% | −21% | −25.7% | −26% | 0.1221 (0.1221) | 0.1326 | 0.1216 | 0.120 |
| 2.5 | 0.0666 | −13.7% | (−15%, 11-01) | −20.8% | – | 0.0528 (0.0528) | 0.0575 | 0.0527 | 0.0487 |
| 3.0 | 0.0306 | −9.4% | −10% | −19.6% | −16% | 0.0246 (0.0247) | 0.0277 | 0.0244 | 0.0243 |
| 3.5 | 0.0171 | −11.1% | (−13%) | −24.7% | – | 0.0129 (0.0130) | 0.0152 | 0.0126 | 0.0136 |
| 4.0 | 0.00828 | −9.1% | −10% | −25.4% | – | 0.00618 (0.00625) | 0.00753 | 0.00588 | 0.00656 |
| 4.5 | 0.00480 | −10.6% | (−11%) | −26.6% | – | 0.00352 (0.00361) | 0.00429 | 0.00325 | 0.00380 |
| 5.0 | 0.00256 | −8.4% | −6% | −25.0% | – | 0.00192 (0.00201) | 0.00234 | 0.00176 | 0.00226 |
| 6.0 | 0.000914 | −8.9% | – | −21.6% | – | 0.000716 (0.000813) | 0.000833 | 0.000676 | 0.000806 |
| 7.0 | 0.00045 | −12.7% | – | −19.8% | – | 0.000361 (0.000522) | 0.000393 | 0.000347 | 0.000397 |

- **Menu only.** The re-pricing barely moves the menu's predictions. Against the old knap4 V3 config (no swj), the
  new pricing is +0.0% at 2.0, +0.8% at 3.0, +2.9% at 4.0, +0.4% at 5.0, +3.6% at 6.5 and +2.6% at 7.0.
  - The rises are the φ interpolation (K3.x / K6.x levels now carry part of the next family's factor) and R8 at K4–6.
  - R8 at K4–6 is ×1.001–1.004.
- **swj.** At 2.0–2.5 the model's swj gain (−8%) exceeds the measured −6.5%. Per-level factors fall with K (×0.88
  at K3), and the knapsack re-allocates.
  - From 3.5 up, menu + swj alone already reaches the 11-06 "V3 at freeze" expectation, which also counted H_out,
    writers, healing and search. This rests on swj's K4 ×0.78 (LOG flags it "surprising, check on a verify slice")
    and on holding it flat above K4.
  - Without swj above K4 the 5.0–7.0 numbers sit between the `noswj` and main columns.
- **11-05 7.0 estimates.** Accurate ~0.00040–0.00043: now 0.00036 anchored (0.00039 without swj). Fast
  ~0.00045–0.00050 assumed ×1.02 for i8 K7: now **0.00052 anchored**. The all-A8 fast 7.0 is predicted worse than
  V2 7.00. The 7.0 fast frontier points (×1.26 / ×1.47) land at ~0.00041 / ~0.00048 anchored.

## 3 · Level usage, drops and additions (main arm unless named)

Per-level usage (sizes picked, max param share, tensor counts): `report.out` T4 / T5.

| candidate | evidence (prediction) | donor time it saves / costs |
|---|---|---|
| **drop i4 K1.125–3 (16 twins)** and **i4 K1** | never picked at 2.0–7.0, accurate or fast. With A4 offered (`a4`, E1c A4-g64 costs, A4 time = A8 / 1.31) the knapsack picks no A4 and no i4 at any λ up to 1e-5. The twins exist only for A4 (#46-03 still owes the 27B A4 costs). i4 K1 is only for < 2 bpw (research points) | 17 levels × ~37 min (×~5 with H_out) |
| replace i8 by i4 at K1.125–2 (`i4low`) | +0.5% at 2.0, +0.2% at 2.25, +0.1% at 2.5, 0 above | only relevant if the i4 twins are built anyway (A4) |
| drop i8 K1.125, K1.25 (`nolow`) | 1–3 tensors at 2.0 only; +0.00% at 2.0 / 2.25. The 1.5 / 1.75 research points need them (#90: K1.125–1.875 −12% there) | 2 levels if the research points move off pass 2 |
| drop mul1 K6, K6.25 (`dupm1`; accurate uses i8 K6 / K6.25 instead) | +0.02 / +0.06 / +0.13 / +0.14 / +0.03% at 5.0 / 5.5 / 6.0 / 6.5 / 7.0 | 2 levels |
| drop i8 K7 (`noi8k7`) | fast 7.0 +2.9%, fast 6.5 +0.2%; accurate never uses it | 1 level (it only serves a weak fast 7.0) |
| keep i8 K6.25–6.75 | dropping all of i8 K6.25–7 (`noi8hi`): fast +0.1% at 5.5, +4.6% at 6.0, +31% at 6.5, +44% at 7.0 | the fast 6.0–7.0 need them |
| pass 1 only (25 levels: quarter steps K1.25–5, i8 5.5 / 6 / 6.5 / 7, mul1 6 / 6.5 / 7 / 7.5 / 8; `pass1`) | +0.2…+0.8% accurate and fast at every size (max +0.8% at 5.5); every pass-1 level is picked somewhere (none to drop from pass 1) | pass 2 (48 levels) buys ≤ 0.8% at 2.0–7.0 (its 1/8 steps below K2 matter for the < 2 bpw research points) |
| add 1/8 steps K5–7 (i8 and mul1; `fine57`) | picked (i8 K6.125–6.875 take up to 17% at fast 6.5), gain −0.1…−0.2% | not worth 16 more levels |
| add 1/16 head budgets K3–8 (`h16`) | never picked; KLD identical at every size. Priced conservatively: a row mix of two existing heads on the lower convex hull of {K3, K4, slab5, slab6, K7, K8} (linear in error, +1 MB). A sensitivity-ordered slab at each budget (headsim) can only do better: this says nothing about headsim-priced budgets, only that blind row mixes don't pay | – (needs headsim's per-row curve before it can show a gain) |
| tile CD attn_qg ×0.99 (`cd`) | −0.01…−0.06% everywhere (attn_qg is small); 0–5 tensors change pick | ~+5 min a level (LOG 11-06) |

## Sensitivities (accurate / fast KLD vs main; `report.out` T3)

- **`gu`** (swj on gate_up only, calibrated): −0.2…−0.6% at ≤ 3.0, −4.8% at 4.0, −8.3% at 5.0, −3.8% at 7.0.
  - It re-allocates bits away from gate_up. At 4.0 gate_up averages K3.74 vs K4.25 for the rest; in main the gap
    is 3.91 / 4.06.
  - It changes 60–290 tensor picks. Its K4 gate_up factor ×0.46 is a strong claim from one size.
- **`noswj`:** +8…+23% (the swj effect in this model).
- **`phifloor`** (old φ convention): −0.0…−3.4%. Fast 7.0 moves from i8 K7 to i8 K6.75.
- **`r8ratio`** (i8 K6.75 ×1.137): fast +0.0…+0.5%; K6.75 picks move to K6.5 / K7.
- **`oldf`** (old pricing, measured speed table, no swj): T3. It also puts the fast 6.5–7.0 on int8 K6–7 (K7: 78% of
  params at 7.0).

## Caveats

- These are model predictions: kfit2 was fitted on V2 measurements, and V3 levels are priced through V2's hproxy
  errors × measured ratios. Paired real KLD (pass-1 mixes) is the test.
- The 27B A4 costs (#46-03) and the H_out / writers / healing gains are not in the model. H_out and writers would
  each change sensitivities, and the knapsack must be re-run after them.
- swj above K4, and the K4 value itself, are the largest single assumption in the 4.5–7.0 numbers.
- The speed table is one K (K4) and one M (2048). A8 activation costs are E1c (calib-A, K6 base, ±30%).

## For the lead (decisions, not taken here)

1. **Fast 7.0.** As priced it is all i8 K7, +45% vs accurate, and anchored ~0.00052, worse than V2 7.00's
   0.00045. Options:
   - skip the fast 7.0;
   - ship a mixed-mode "fast" 7.0 (×1.26–1.47 GEMM, +14…+32%);
   - point fast users at the 6.5 fast (0.00050 model; ~0.0006 anchored, using the 6.0 / 7.0 anchor ratios).
   Then decide whether the i8 K7 level is worth its donor time (+2.9% at fast 7.0 only).
2. **Ladder trims worth considering:**
   - the 16 i4 twins + i4 K1 unless #46-03 revives A4 (17 levels);
   - mul1 K6 / K6.25 (2 levels, ≤ 0.14%);
   - optionally i8 K7.
   Pass 1 as planned covers every size within +0.8%.
3. **Check swj at K ≥ 4** (verify slice, and one point above K4) before quoting 4.5–7.0 predictions. Also decide
   whether the knapsack should model swj per family (`gu`) or per level (`all`).
4. **1/16 head budgets** need headsim pricing to say anything. Under blind row mixes they are worth 0.
5. **Tile CD:** −0.01…−0.06% predicted at ~+5 min a level, so it is the user's call on cost alone.
