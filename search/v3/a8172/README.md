# search/v3/a8172 — A8 in the search (#172, w132, 11-13)

Every search and calibration container so far was mixed without `act` lines, so it runs the f16-activation prefill
path. V3 ships A8 (int8 activations, a8g128 on every trunk weight; decision 11-09). This folder measures how much that
matters for the search (1), prepares the switch (2) and checks whether the search's decisions bought anything (3).

**Status (11-13 11:30 AM CST).** The user paused the search at 10:11 AM CST (lead note 10:15) before any rotation, so
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

## Files

| file | what |
|---|---|
| `run132.sh` | ds8 runner (remix f16 / A8, score under the search's locks; restartable; `PREV=1` = phase 2, the prev incumbents on s171b) |
| `docsums132.py` | ds8: per-document KL sums of every rows file (hostd.doc_sums) → `data/searchv3/w132/docsums132.json` |
| `a8gap.py` | the tables above → `data/searchv3/w132/gap132.{json,out}` |
| `hostd-driver-a8.patch` | hostd act lines + driver CONF act at rotation (against 98c231d) |
| `test_hostd_act.py`, `test_driver_act.py` | the patch's tests |
| `RESTART-a8.md` | the restart plan (not executed: search paused) |
| `RULES-value.md`, `run-value.sh`, `docsums-value.py`, `value.py` | the search-value check → `data/searchv3/w132/value.{json,out}`, `run-value.log` |

Data use: golden-ver3-8k (verify, the monitor golden: one more routine read of 4 mixes), s171b (the active search
slice, already scored for these mixes) and s171c + s171d (search-pool slices, first use: the value check; they are
now spent as fresh slices for this run's incumbents). No new documents, no ledger change. Rows files stay on ds8
(`/data/q27/w132/scores/`).
