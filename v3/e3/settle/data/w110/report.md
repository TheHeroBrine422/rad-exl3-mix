# #145 · w110 · C9 and swj at sub-1 on the 27B (fixed codes): report

Rule `v3/e3/settle/w110.md`, committed alone before any data (c6e9cd1, 11-11 11:24 AM CST; its text says "~11:40");
amendments 1 (149396a) and 2 (70f8b72), both before any pack, mix or score (G1's bit-identity → rounding tolerance;
details below). Verdict text: `verdict.txt` (rerun: `python3 v3/e3/settle/w110/verdict110.py /data/w110` on gpubox).

## What ran (gpubox card 0, 11-11 CST)
- Levels `q27v3-159591b2-k{1.5,1,0.5}-i8` (stamp 159591b2, rad-exl3-mix c6e9cd1 incl. #142 fusion, fork b0052f7c),
  seed 0, via `rxrun.run` N = 3: 11:25 AM – 3:10 PM (768 jobs + 18 dup-seed jobs, interleaved with w111's scores);
  one 3-source swj pass 3:10 – 5:23 PM (8 stages × 993 s, peak 24.7 GiB: 0.74 h / level); finals + 12 packs (3 full ×
  ~23 min, 9 `--reuse` version packs 2–11 min: 723 / 787 weights copied) 5:24 – 7:24 PM; 18 mixes + hygiene to 7:33 PM;
  36 scores 7:46 – 9:37 PM.
- G1 (amended): CPU `rxq.post c9` vs in-job C9: max rel 6.4e-7 / 1.8e-6 / 1.4e-6 (K1.5 / K1 / K0.5), 0 bit-identical
  writers (GPU vs CPU), 0 f16 svh rows moved. e3 (e_clip 0.3) changes 40 / 1,394 / 270,193 of 327,680 writer rows at
  K1.5 / K1 / K0.5 (13 / 52 / 64 writers; after the f16 fold the shipped bytes differ in 6 / 52 / all writers, per hygiene).
- G2: all 16 arm mixes PAIR OK vs their T with `--planes svh` (787 weights, only the arm's svh planes differ);
  census PASS (336 K1, or 160 K0.5 + 176 K1.5 exl3i mul1i8 trunk weights; head / embedding / non-trunk = V2 3.00).
- Slice golden-search27d: xcorp 130 pieces of never-scored xc8k-b2-c0 (222,107 positions) + calib 63 search-pool docs
  never in the ledger (77,441 scored positions; calib_slice's 60k target overshoots to 84.8k tokens per category).
  T: K1 uniform KLD 1.0931 (top-1 0.668), splice 2.2824 (top-1 0.499). Top document ≤ 13% of a part's KL.
- One seed: every arm shares its codes with T (scale-only arms), as the rule allows.

## Results (r = KL(arm) / KL(T); pooled over both parts, 95% doc-bootstrap CI; per-doc geo over 193 docs)

| arm | what | pooled r | CI | x / c part | per-doc geo | pass |
|---|---|---|---|---|---|---|
| k1-os | C9 off at K1 | 1.0968 | [1.014, 1.199] | 1.097 / 1.095 | 1.117 | worse |
| k1-es | e3 at K1 | 1.0013 | [1.0004, 1.0020] | 1.001 / 1.001 | 1.0006 | worse (z +2.6) |
| k1-cn | swj off at K1 | **1.3558** | [1.294, 1.430] | 1.313 / 1.521 | 1.448 | worse |
| sp-oo-ss | C9 off, both K | 1.0429 | [0.956, 1.131] | 1.046 / 1.034 | 1.017 | worse |
| sp-ee-ss | e3, both K | **0.9590** | [0.944, 0.972] | 0.956 / 0.968 | 0.965 | **PASS** |
| sp-oc-ss | C9 off at K0.5 only | 1.0042 | [0.928, 1.076] | 1.004 / 1.005 | 0.988 | worse |
| sp-co-ss | C9 off at K1.5 only | 1.0040 | [0.994, 1.012] | 1.003 / 1.007 | 1.009 | worse |
| sp-ec-ss | e3 at K0.5 only | **0.9593** | [0.945, 0.972] | 0.957 / 0.968 | 0.965 | **PASS** |
| sp-ce-ss | e3 at K1.5 only | 0.9998 | [0.9996, 0.9999] | 1.000 / 1.000 | 0.9998 | – |
| sp-cc-nn | swj off, both K | **1.1867** | [1.132, 1.231] | 1.168 / 1.245 | 1.233 | worse |
| sp-cc-ns | swj off at K0.5 only | 1.1230 | [1.083, 1.158] | 1.108 / 1.169 | 1.160 | worse |
| sp-cc-sn | swj off at K1.5 only | 1.0606 | [1.042, 1.075] | 1.055 / 1.078 | 1.075 | worse |

C9 × swj (pooled r): M1 off/off 1.448 (interaction 0.974), e3/off 1.357 (0.9995); M2 off/off 1.285 (1.038),
e3/off 1.143 (1.005). Per-doc detail (medians, top docs, leave-one-out, drop-k): `docstat-M1.txt`, `docstat-M2.txt`.

## Verdict (the rule, mechanically)
- **C9 stays as today** (no global change; no band pattern). e3 passes at K0.5 (`sp-ec-ss`, and `sp-ee-ss` = the same
  effect) but the "below K1" pattern also needs worse(`ce`) at K1.5: `ce` = 0.9998 (e3 moves 6 writers there), so the
  pattern misses by 0.0002 → reported, not adopted. C9 off passes nowhere: #140's 0.8B "off at K0.5" does not carry
  (K0.5-only 1.004, per-doc 0.988, not significant), and off costs +9.7% at K1.
- **swj stays at every level ≤ K4.5**: off costs +35.6% at K1, +12.3% at K0.5, +6.1% at K1.5 (both parts, every
  statistic). #141-04's K0.5 seed disagreement on the 0.8B (0.972 / 1.048) does not carry to the 27B.

## Readings and proposals (lead / user)
1. **e_clip 0.3 below K1** (post hoc, outside the rule's pattern): −4.1% at K0.5 on every statistic (both parts,
   per-doc geo 0.965, 89% of xcorp docs better, z −4.5); +0.13% at K1 (worse, tiny); ~0 at K1.5. A K-dependent
   e_clip (0.3 for K < 1, 0.5 from K1) is the shape the data show; the boundary K0.625–0.875 is unmeasured (e3 changes
   82% of rows at K0.5 vs 0.4% at K1). Suggest a confirm on a second fresh slice with the two kept mixes
   (`/data/w110/mix/mix-sp-cc-ss.rad`, `mix-sp-ec-ss.rad`) + a K0.75 point if it is to be adopted. Settings-stamp
   change; runtime none.
2. **swj magnitude at K1 (−26%) is ~8× the 0.8B's (−3.3%)** → "unverified" under AGENTS.md until a second slice;
   hygiene (G2) already shows the two mixes identical outside the gate / up svh. It supports the status quo, so no
   decision hangs on it.
3. The three levels are today's ladder levels at stamp 159591b2 (assembled, swj'd, final + recipe):
   `/data/w110/lv/q27v3-159591b2-k{0.5,1,1.5}-i8` were today's when built; the lead has since moved to stamp 4f831984 (3e4bed5, swj bf16 numerics), so they seed the
   ladder only if the lead re-stamps them (codes unaffected by post.swj); otherwise a fresh ROOT. The ledger lines for
   golden-search27d-c went into the lead's commit 6e0a7c0.
4. Dup-seed SD (18 pairs, `seedsd.txt`): pooled 0.028% of the proxy (K < 1 0.024%, K1–2 0.030%).

**Runtime cost:** none (scales only). Card time: ~3.75 h quant (incl. lock waits), 2.2 h swj, ~1.3 h scores.
