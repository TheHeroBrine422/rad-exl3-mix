# search/v3/a8172 — A8 in the search (#172, w132, 11-13)

Every search and calibration container so far was mixed without `act` lines, so it runs the f16-activation prefill
path. V3 ships A8 (int8 activations, a8g128 on every trunk weight; decision 11-09). This folder measures how much that
matters for the search (1), prepares the switch (2) and checks whether the search's decisions bought anything (3).

**Status (11-13 12:40 PM CST).** The user paused the search at 10:11 AM CST (lead note 10:15) before any rotation, so
the live switch was never made: no `act` event in state, and `driver.json` is back to its pre-w132 config (the A8
version is kept as `driver.json.a8-w132`). The patch is applied in the repo as the A8 mix writer for everything from
now on (lead note item 3). It is opt-in: an eval request's `"act"`, or driver CONF `"act"`. hostd is staged on ds8
(`hostd.py.bak-w132` = the old one; requests without act are unchanged).

## 1. The gap (ds8, one card at a time through the search's card locks, 11-13 8:34–9:48 AM CST, ~1.2 card-h)

The four search incumbents at 2.5 / 3.5 / 5.0 / 6.0 (drv/state.jsonl, 8:30 AM CST), and the incumbent each one replaced,
were remixed from the search's own `.mix` files plus `act <w> a8g128` on the 336 trunk weights. f16 rows are the
search's own (same rad10 build). As a control, the f16 5.0 mix was rebuilt and rescored on ver3-8k: its rows are
byte-identical to the search's monitor rows (sha256 0a7ff91f…). KLD = the search's statistic over the scored
(generated) positions. Ratios on s171b are the equal-weight mean of the x / c parts' log ratios (seqrule pmean).
95% CIs: 2,000 per-document bootstrap resamples.

| size | ver3-8k f16 → A8 | ratio [95% CI] | s171b f16 → A8 | ratio [95% CI] (c / x) |
|---|---|---|---|---|
| 2.5 | 0.076663 → 0.076180 | 0.994 [0.987, 1.006] | 0.198981 → 0.199520 | 1.008 [0.999, 1.018] (1.013 / 1.002) |
| 3.5 | 0.017956 → 0.017863 | 0.995 [0.987, 1.009] | 0.090088 → 0.089571 | 0.995 [0.984, 1.005] (0.996 / 0.994) |
| 5.0 | 0.004340 → 0.004633 | **1.068 [1.025, 1.090]** | 0.036211 → 0.037343 | **1.045 [1.015, 1.082]** (1.060 / 1.031) |
| 6.0 | 0.002010 → 0.002107 | **1.048 [1.023, 1.072]** | 0.029652 → 0.030923 | **1.051 [1.032, 1.076]** (1.059 / 1.043) |

- **A8 is free up to 3.5 and costs ~5–7% KLD at 5.0 / 6.0.** This fits an activation-quant error floor that shows once
  the weight error is small. Between 3.5 and 5.0 is not measured.
- **w130's "K4 0.157 → K4A 0.146" is not an A8 effect.** val10.sh's K4A is k4.125 + A8 and its K4 is k4, so the
  comparison is confounded by the extra 0.125 bit. Its numbers are also all-position KLD (0.10–0.46 here vs the
  search's 0.002–0.2).
- **Bytes:** identical with and without the act lines at all four sizes (10,509,639,680 / 13,712,711,680 /
  18,556,968,960 / 21,760,040,960 B). radmix's act metadata fits in the alignment padding.
- **Speed:** the A8 path is active (radiance: `exl3i_actq` ops, 1,394 declared ops vs 1,186). Score wall time ("N
  documents in S s") fell 9–13%: ver3-8k 235–242 s vs 258–271 s, s171b 339–352 s vs 375–382 s. The only same-card pair,
  5.0 on ver3, went 268 → 238 s. That is ~10% more evals per card-hour; the pack / mix steps are unchanged.

**Ranking.**
- **Across sizes, ratios change** when one incumbent is ≤ 3.5 and the other ≥ 5.0. The log KLD ratio shrinks by
  +0.04 to +0.07 under A8, CI excluding 0 (e.g. 3.5 → 5.0: ver3 +0.071 [+0.020, +0.099], s171b +0.049 [+0.013, +0.093]).
  2.5 ↔ 3.5 and 5.0 ↔ 6.0 are unchanged within noise. So the monitor's "vs V2" reads at ≥ 5 are ~5% optimistic for
  what ships.
- **Within a size, the same decisions** (each incumbent vs the one it replaced, s171b, the driver's rule with the
  recorded size offset):

| size | f16 log r (z) | A8 log r (z) | A8 − f16 [95% CI] | f16 / A8 decision |
|---|---|---|---|---|
| 2.5 | −0.0301 (−1.93) | −0.0151 (−1.79) | +0.0150 [−0.0016, +0.0368] | accept / accept |
| 3.5 | −0.0141 (−1.64) | −0.0202 (−2.00) | −0.0061 [−0.0188, +0.0042] | accept / accept |
| 5.0 | −0.0390 (−1.84) | −0.0244 (−1.87) | +0.0146 [−0.0178, +0.0473] | accept / accept |
| 6.0 | −0.0081 (−0.77) | +0.0012 (+0.09) | +0.0093 [−0.0156, +0.0370] | accept / **reject** |

  Three of four accepts hold. The marginal 6.0 accept flips. No single per-pair shift is outside noise, but the
  pairwise shifts are of the same order as the accepted gains (0.01–0.04).

**w129's calibration rows:** 83 of the 90 prior rows (fit129h2.obs) have mean K < 4.5, where A8 ≈ f16: no rescore
needed. The 7 uniform rows at K ≥ 4.5 (4 at 4.5–5.5, 3 above) anchor the fit's high-K curve and read ~5% low for A8.
Recommended: rescore those 7 in A8 on fit129 (~0.6 card-h) and swap them into prior_obs before or at the switch.
The search's per-slice offsets cannot absorb a size-dependent shift.

## 2. The switch (prepared): `hostd-driver-a8.patch` + `RESTART-a8.md`

hostd writes radmix act lines from the eval request. The driver's CONF `"act": "a8g128"` takes effect at the next
slice rotation, where it rescores every incumbent on the new slice plus an A8 monitor read. Tests (patched tree;
colo / VM, CPU only):
- `test_hostd_act.py`: 4/4 PASS
- `../test_hostd_es.py`: ALL PASS
- `test_driver_act.py` (stand-in with an A8 executor): 7/7 PASS
- `../standin.py` (stock, f16 path): 7/7 PASS

`git apply --check` passes against 98c231d.

## 3. Search-value check (`RULES-value.md`, committed 62d6d4a before any run; ds8 8 cards 10:40–11:35 AM CST, ~4.3 card-h)

For each size: the initial pick (the starting fit's knapsack solve) vs the current incumbent, both as A8 containers, on
golden-s171c + golden-s171d. These two slices were built for the search and never activated: 0 evals, 0 decisions,
0 fit rows. Pooled: 112 c docs + 260 x pieces. Primary statistic: the size-adjusted pmean D (log ratio, current over
initial). 95% CIs from 2,000 document bootstrap resamples stratified by part.

| size | D [95% CI] | reading | raw d | s171c / s171d | c / x raw | KLD s171c init → cur |
|---|---|---|---|---|---|---|
| 1.5 | −0.019 [−0.031, −0.009] | gained | −0.033 | −0.019 / −0.020 | −0.042 / −0.025 | 0.4853 → 0.4700 |
| 1.75 | −0.059 [−0.075, −0.045] | gained | −0.069 | −0.058 / −0.060 | −0.080 / −0.057 | 0.3471 → 0.3297 |
| 2.0 | −0.085 [−0.110, −0.060] | gained | −0.094 | −0.094 / −0.077 | −0.118 / −0.069 | 0.2617 → 0.2412 |
| 2.5 | −0.060 [−0.083, −0.036] | gained | −0.067 | −0.044 / −0.073 | −0.074 / −0.060 | 0.1496 → 0.1428 |
| 3.0 | −0.094 [−0.122, −0.069] | gained | −0.101 | −0.103 / −0.086 | −0.132 / −0.071 | 0.0955 → 0.0882 |
| 3.5 | −0.075 [−0.095, −0.056] | gained | −0.081 | −0.083 / −0.069 | −0.065 / −0.097 | 0.0639 → 0.0575 |
| 4.0 | −0.084 [−0.110, −0.065] | gained | −0.089 | −0.083 / −0.089 | −0.071 / −0.107 | 0.0418 → 0.0369 |
| 4.5 | −0.115 [−0.147, −0.081] | gained | −0.120 | −0.084 / −0.132 | −0.094 / −0.145 | 0.0294 → 0.0270 |
| 5.0 | −0.085 [−0.128, −0.027] | gained | −0.089 | −0.064 / −0.103 | −0.069 / −0.110 | 0.0231 → 0.0211 |
| 5.5 | +0.028 [−0.016, +0.083] | null | +0.025 | +0.068 / −0.002 | +0.053 / −0.003 | 0.0179 → 0.0185 |
| 6.0 | −0.038 [−0.077, +0.001] | null | −0.040 | −0.077 / −0.006 | −0.061 / −0.018 | 0.0159 → 0.0154 |
| 6.5 | +0.016 [−0.030, +0.080] | null | +0.015 | +0.073 / −0.021 | +0.047 / −0.016 | 0.0137 → 0.0145 |
| **pooled** | **−0.056 [−0.070, −0.040]** (×0.946) | **gained** | | | | |

- **The ~520 decisions bought 2–11% KLD at 1.5–5.0, and nothing measurable at 5.5–6.5.** 7.0 never moved (excluded).
- **Why the ver3 monitor said otherwise:** it never scored the initial picks. Its first read per size is the first
  *accepted* candidate, so "flat on ver3" compares later accepts with early ones, not with the knapsack solve. Its
  first-to-last ratios (f16) are noisy (×0.54 to ×1.08 on ~100 docs) and still do not cover the initial pick.
- **Open:** how much of the gain came from the first accept per size and how much from later decisions. Scoring the
  first accepted incumbents (12 more A8 mixes, ~2 card-h, ~25 min on 8 cards) on the same docs would split it.

### 3b. Third arm: w133's M1 re-solve picks (amendment 1, committed 1e0f7cf before any M1 mix; 11:33 AM–12:38 PM CST, ~2.6 card-h)

M1 picks are w133's `picksT` on menu133e: the R16 heads plus w135's embedding options. Built as A8 containers from
ds8's packs and donors and scored on the same documents. Negative = M1 better; the same size adjustment applies.

| size | M1 emb / head | M1 vs current D [95% CI] | reading | M1 vs initial D [95% CI] | KLD s171c current → M1 |
|---|---|---|---|---|---|
| 1.5 | e2 / u2 | −0.046 [−0.086, +0.000] | null | −0.065 [−0.110, −0.014] | 0.4700 → 0.4541 |
| 1.75 | e2 / u2 | −0.094 [−0.109, −0.079] | gained | −0.154 [−0.174, −0.132] | 0.3297 → 0.3021 |
| 2.0 | e2 / u2.25 | −0.059 [−0.075, −0.042] | gained | −0.144 [−0.169, −0.120] | 0.2412 → 0.2311 |
| 2.5 | e2 / u2.5 | −0.055 [−0.082, −0.030] | gained | −0.114 [−0.141, −0.090] | 0.1428 → 0.1340 |
| 3.0 | base / u3 | −0.070 [−0.097, −0.045] | gained | −0.164 [−0.208, −0.127] | 0.0882 → 0.0817 |
| 3.5 | base / u3.5 | −0.070 [−0.101, −0.033] | gained | −0.144 [−0.179, −0.104] | 0.0575 → 0.0537 |
| 4.0 | base / u3.75 | −0.076 [−0.106, −0.037] | gained | −0.160 [−0.190, −0.128] | 0.0369 → 0.0349 |
| 4.5 | e5 / u4 | −0.100 [−0.141, −0.047] | gained | −0.215 [−0.265, −0.149] | 0.0270 → 0.0241 |
| 5.0 | e5 / u4.5 | −0.275 [−0.344, −0.214] | gained | −0.360 [−0.421, −0.303] | 0.0211 → 0.0163 |
| 5.5 | e5 / u5 | −0.481 [−0.540, −0.410] | gained | −0.453 [−0.508, −0.381] | 0.0185 → 0.0115 |
| 6.0 | e5 / u6 | −0.643 [−0.723, −0.554] | gained | −0.680 [−0.768, −0.582] | 0.0154 → 0.0090 |
| 6.5 | e5 / u6.5 | −0.799 [−0.912, −0.679] | gained | −0.783 [−0.876, −0.683] | 0.0145 → 0.0080 |
| 7.0 (not pooled) | e5 / u7 | −0.720 [−0.831, −0.599] (vs initial = current, 6.84 bpw) | gained | | 0.0080 (M1) |
| **pooled (12)** | | **−0.231 [−0.252, −0.206] (×0.794)** | **gained** | **−0.286 [−0.308, −0.260] (×0.751)** | |

- **M1 beats the search's current incumbents at every size from 1.75 to 7.0;** 1.5's CI touches 0. Both slices and
  both parts agree in sign at every size except 1.5's x part.
- **The large high-size gains come with the embedding.** The search's menu (menu129h) had only V2 3.00's l3 embedding
  ("base") at every size. Every M1 pick from 4.5 up uses l5 (e5) and gains −10% at 4.5, then −27% to −55% at 5.0–7.0.
  At 3.0–4.0, where M1 also keeps "base", the gain is −7%: that is the M1 fit and re-solve's own trunk / head
  allocation. At ≤ 2.5 the smaller l2 embedding frees bytes (−5% to −9%). The split between the embedding and the M1
  model is not measured: that needs M1 on the old menu, or the current incumbents with e5.
- **Reading for the lead:** the search beat its own starting knapsack (−5.6% pooled), but a one-shot M1 re-solve on the
  extended menu beats the search's incumbents by −23% pooled. Where the menus are comparable (3.0–4.0, same "base"
  embedding), the margin is −7%.

## Files

| file | what |
|---|---|
| `run132.sh` | ds8 runner (remix f16 / A8, score under the search's locks; restartable; `PREV=1` = phase 2, the prev incumbents on s171b) |
| `docsums132.py` | ds8: per-document KL sums of every rows file (hostd.doc_sums) → `data/searchv3/w132/docsums132.json` |
| `a8gap.py` | the tables above → `data/searchv3/w132/gap132.{json,out}` |
| `hostd-driver-a8.patch` | hostd act lines + driver CONF act at rotation (against 98c231d) |
| `test_hostd_act.py`, `test_driver_act.py` | the patch's tests |
| `RESTART-a8.md` | the restart plan (not executed: search paused) |
| `RULES-value.md`, `run-value.sh`, `docsums-value.py`, `value.py`, `mkm1mix.py` | the search-value check (+ amendment 1's M1 arm: `M1=1 run-value.sh`, `T7=1` for 7.0's initial pick) → `data/searchv3/w132/value.{json,out}`, `run-value.log`, `m1mix.json` |

Data use: golden-ver3-8k (verify, the monitor golden: one more routine read of 4 mixes), s171b (the active search
slice, already scored for these mixes) and s171c + s171d (search-pool slices, first use: the value check; they are
now spent as fresh slices for this run's incumbents). No new documents, no ledger change. Rows files stay on ds8
(`/data/q27/w132/scores/`).
