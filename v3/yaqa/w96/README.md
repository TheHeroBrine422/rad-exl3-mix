# w96 — head Fisher: rare-row smoothing and more data, scored on held-out rows (#127, user 11-09)

Rule: `RULES.md` (72e2191, committed alone before any data). Results: LOG 2026-11-08 9:05 PM CST (#127); tables in `data/`.

## Outcome

- **No arm is useful** (rule: ≥ 3% better held-out head KL at equal bytes on ≥ 2 of s4-3 / s5-3 / s6-3, CI excluding 0).
  Keep today's qad-gen Fisher for the head plans.

  | arm (tuned on the qad split) | s4-3 vs base | s5-3 vs base | s6-3 vs base | rare-token KL (s4/s5/s6) |
  |---|---|---|---|---|
  | smooth (prior U, κ 1) | +0.0% [0.0, 0.0] | −0.6% [−1.0, −0.3] (5.000 vs 4.996 b) | −0.7% [−1.1, −0.2] | −0.0 / −0.7 / −1.6% |
  | more-q (qad prompt rows, w 1) | +0.1% [−0.0, +0.4] | −2.7% [−4.1, −1.3] | −1.0% [−1.8, −0.4] | −0.1 / −3.7 / −2.8% |
  | more-x (xc8k-b1, w 0.5) = best | +0.0% [−0.3, +0.4] | −3.8% [−5.1, −2.3] (5.000 vs 4.996 b) | −1.0% [−1.7, −0.4] | −0.5 / −5.0 / −2.8% |
  | both (more-x + U κ 1) | = more-x (κ 1 changes nothing) | | | |

  Base (today): s4-3 / s5-3 / s6-3 = 5.09e-4 / 1.21e-4 / 4.65e-5 on search-gen (−60 / −64 / −50% vs u4 / u5 / u6).
- **knap5 re-solve** (w94's procedure) with the more-x grid: −0.00 / −0.02 / −0.01 / −0.02% predicted KLD at
  2.0 / 2.5 / 3.0 / 4.0 vs the base grid (`data/knap-report.md`). The base grid on the CPU sums reproduces w94's
  g-search-gen to 6 digits.
- **Census:** the Fisher uses 3,836,849 qad gen positions (2,799 qad documents, pools-v3). 673 rows have Fisher 0,
  49,606 rows < 10 positions with p > 1e-4, 129,017 rows < 100. Those row sets hold 0.0002% / 0.22% / 2.6% of the held-out exact
  Fisher (search-gen dump; 0.12% / 1.4% on the search pool's golden). Too little mass for smoothing to matter.
- **Post hoc, report only: w94's "qad rows are in-sample" explanation is wrong.** On the qad-dump rows, plans from half
  A's Fisher (never saw those documents) score the same as today's full Fisher (+1.1% [−0.1, +2.5], 0.0%, 0.0%), and
  the slab / uniform ratios are the same either way (qad 0.14 / 0.18 / 0.41 vs search 0.40 / 0.36 / 0.50). The gap is in
  the row sets: the five worst held-out documents for s4-3 are 4 Japanese (fineweb2 jpn) and 1 Thai Wikipedia
  (slab worse than uniform on them, s/u 1.0–2.3), 43% of the s4-3 KL; without the 6 jpn / th documents s4-3 / u4 is 0.24.
  The search dump holds 5 of the search pool's 6 jpn documents (5% of documents); calib is 0.24% jpn tokens, 0.20% Thai
  (search pool: 0.45% / 1.05%; qad 0.31% / 0.12%). Token-length weighting does not remove it (0.43).

## Files

| file | what | where it ran |
|---|---|---|
| `mksplit96.py` → `split96.json` | qad split (A 1,400 / B 1,399 documents; B ⊇ w94's 85 qad-dump documents) | VM |
| `mkgroups96.py`, `fisher96.py`, `gold96.sh` | Fisher Σp(1−p), Σp, n(p > 1e-4), N per row group of a golden (calib: A / B gen, A / B non-gen) | gpubox CPU (the 42 GB golden stays there) |
| `xc96.sh` | the same over night1105's `xc8k-b1` minus golden-search27b-x's 65 pieces | Thor |
| `unigram96.py` | token counts (calib fit + qad documents; xcorp v1) for prior U | gpubox / colo |
| `score96.py` | headsim94's exact head-only KL decomposed per 128-row block and donor (CPU, f32): any plan scored without matmuls | Thor |
| `plans96.py` | g1 / census / tune / final / curve on the block sums, plans from `headslab.py` (unchanged) | Thor |
| `run96.sh` | the Thor pipeline (restartable) | Thor |
| `copy96.sh` | donors + dumps gpubox → Thor (through the VM) | VM |
| `knaprep96.py` | knap5 table | colo |
| `posthoc96.py`, `doc96.py`, `tw96.py` | report-only checks (in-sample, per-document shares, token weighting) | Thor |

Machine data: Thor `/data/work/w96` (donor copies `in/K*`, 12.7 GB; block sums `sums/`, ~8 GB; delete when done),
gpubox `/data/w96` (groups, Fisher), colo `/data/work/w13/w96/knap`.

**Runtime:** none (plan statistics only).

**Proposal (item 3, GPU, not run):** re-quant K3 / K4 head donors with two-sided LDLQ and `hout_diag` = the smoothed
Fisher (prior U, κ 10–30) vs w94's T0 donors, headsim94 on the same dumps (~10 card-min). Low priority: σo 0.1 already
floors zero / rare rows, and rows with n < 100 hold 2.6% of the held-out Fisher.
