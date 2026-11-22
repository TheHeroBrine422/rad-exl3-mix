# RULES217 — does a larger Adam eps fix the scale-only QAD noise? (w217)

Written 2026-11-21 9:28 PM CST, after step 1 (the held-out judge, the gradient-noise medians) and before any step-2 run (the only healed checkpoints read so far are w214's no-eps ones, the controls; their j217 reads are in the report); committed alone. Amendments are appended below with a
time and a reason, never edits of the text above. Brief: `rad-local/workers/instr3/w217.md` (lead). Runtime: none (scales fold into the container's f16 suh / svh planes).

## Question
w214 (scale-only QAD, trunk suh / svh log multipliers on frozen codes, KL to the stored bf16 golden, batch 24 x 1,024 = ~21k gen rows an optimizer step, WSD) found every gain in
the first 0.25 M rows, then a stall or a drift back at constant lr, and a usable lr that falls with the size (2.50 works at 1e-3; 3.00 drifts at 1e-3 and wants 3e-4; 4.00 diverges at
1e-3). Hypothesis: Adam divides each scale's step by its own gradient size, so scales whose true gradient is ~0 still move ~lr per step in a random direction; a larger eps in the
denominator damps those moves and leaves strong-gradient scales alone. Test: lr x eps at all three sizes, 0.25 M and 0.5 M rows, nothing else.

## Judge, controls, data (fixed before any eps arm)
- **Held-out judge j217** (`slices/golden-j217`, box2; `v3/qad217/mkjudge217.py`, seed 217; ledger role `check`, use `qad217-judge`, committed `84d15d13` before any read): 166 documents of
  the calib **qad pool**, first 8,192 tokens, 163,534 gen rows, in no w214 / w217 training set (not the `train214` pool, no `qad214` / `qad217` use: so not w214's fixed-eval documents either),
  not read by the release's own fits (ledger uses `swj27-qad`, `w94-headx-qad*`; roles select / final / fit / fit-hessian), nor trained on by other QAD tasks (`qad215-*`, `qad218`, `qad218-tune`,
  `qad181-fixed`); drawn per category in proportion to the candidates' gen rows (1,157 candidates). Served scoring as w214's j214 reads: the checkpoint's scales folded into the released
  container (`swap214.py`, `--verify`), plain radiance `--kld-ref` (`score217.sh` = `score214.sh` with j217). j214 (calib verify) is not used in this task. Never public.
- **Controls:** the unhealed releases 2.50 / 3.00 / 4.00 (host-emb files, `ctl-<size>`), scored once each. w214's no-eps checkpoints (25A, 25A-lr3e-4, 30A, 30A-lr3e-4, 40A, 40A-lr3e-4 at
  0.25 M = `ckpt-2e+05` and 0.5 M = `ckpt-5e+05`) are **re-read on j217, not retrained**.
- **Training:** exactly w214's: same `pools214.json` (train214), `chunks214.txt`, `--fixed-eval 20000` (so the data order is the same, order sha1 `b9c122bbd7e2`), seed 0, normal batch,
  warmup 8, trust clamp 0.3, arm A (no A8), **eager with w214's flags** (no `--compile-*`; lead 11-21, so the no-eps checkpoints are exact controls; a first-step check against w214 is in the report).
  Budget 0.5 M rows, a 0.25 M WSD branch (`--budgets 2.5e5,5e5`). Ledger: role `train`, use `qad217` (= w214's documents; committed before any run).
  The only change: `--eps` (Adam's eps on the log multipliers; all else Adam as w214: betas 0.9 / 0.99).

## Step 1b: gradient noise (done; `data/qad217/gradnoise-*.json`)
At each unhealed release (start state, scales frozen, no optimizer step), the log-multiplier gradient of 40 different training batches of the normal size (the first 40 plan steps of w214's
order, 0.85 M gen rows): per scale, mean and std (ddof 1) across the 40 batches. Over all scales (r = suh side, c = svh side; gauge-fixed ones excluded):

| size | scales | median \|mean grad\| | median across-batch std | ratio | median \|mean\| if pure noise | share of scales with \|mean\| < 1.96 se | median std r / c | batch grad norm median / max (batches > 3x median) |
|---|---|---|---|---|---|---|---|---|
| 2.50 | 5,668,864 | 7.276e-07 | 5.527e-06 | 0.132 | 5.895e-07 | 0.896 | r 6.20e-06 / c 5.19e-06 | 0.0175 / 0.1262 (2) |
| 3.00 | 5,668,864 | 7.094e-07 | 5.724e-06 | 0.124 | 6.104e-07 | 0.925 | r 6.86e-06 / c 5.27e-06 | 0.0147 / 0.1493 (7) |
| 4.00 | 5,668,864 | 3.959e-07 | 3.345e-06 | 0.118 | 3.567e-07 | 0.936 | r 3.97e-06 / c 3.08e-06 | 0.0083 / 0.0856 (6) |

(`|mean|` of a pure-noise gradient would read 0.6745 std / sqrt(40) = 0.107 std; the measured ratios are to be read against that floor.)

## Arms (step 2): 18 = 3 sizes x 2 lr x 3 eps
eps = 0.3x, 1x, 3x that size's **median across-batch std** (all scales), rounded to two significant figures: (torch's default is 1e-8)

| size | median across-batch std | 0.3x | 1x | 3x |
|---|---|---|---|---|
| 2.50 | 5.5271e-06 | 1.7e-06 | 5.5e-06 | 1.7e-05 |
| 3.00 | 5.7240e-06 | 1.7e-06 | 5.7e-06 | 1.7e-05 |
| 4.00 | 3.3451e-06 | 1.0e-06 | 3.3e-06 | 1.0e-05 |

| size | lr | eps | arm names |
|---|---|---|---|
| 2.50 / 3.00 / 4.00 | 1e-3, 3e-4 | the three above | `<25 / 30 / 40>E-l<lr>-x<0.3 / 1 / 3>` |

All 18 run to 0.5 M (no early stop; w214's 4.00 lr-1e-3 divergence is itself a result to compare). Four cards (box2 HIP 0-3), one job a card, queues from `mkqueue217.py`.
**Order (lead, 11-21):** wave 1 = the middle eps (1x) at 2.50 lr 1e-3, 3.00 lr 1e-3, 3.00 lr 3e-4, 4.00 lr 3e-4 (one a card); then the other 1x arms (2.50 lr 3e-4, 4.00 lr 1e-3);
then the 0.3x and 3x arms. Each arm's 0.25 M checkpoint is read before its 0.5 M one and added to the report table, with the matching w214 control, as it lands (a card trains one arm
to its end, then reads it: the card lock is the training's).

## Readout (the same for every size)
Each arm's 0.25 M and 0.5 M checkpoints on j217: ratio of summed gen-row KL vs the size's unhealed release, paired per document, 95 % per-document bootstrap CI (2,000 resamples, seed 0;
`table217.py` = `cmp214.py`'s arithmetic), documents better / worse; the 0.25 -> 0.5 M step (0.5 M vs 0.25 M of the same arm). Rows of each size's table: lr x eps, with w214's no-eps
arm first. Descriptive differences inside the paired CI (or ~1.5 %) are noise.
**Blow-up companions (fixed now, after the controls' j217 reads showed them, before any eps arm):** the w214 no-eps checkpoints on j217 contain single chat documents whose KL / row jumps
from ~0.02 to ~2 (doc 3820 under 40A / 40A-lr3e-4 at 0.5 M; doc 1893 under 25A at 0.5 M), which alone move the sum-of-KL ratio by +1 to +3 (40A-lr3e-4 at 0.5 M reads 3.15 on j217, 0.95 on j214).
The headline ratio stays all-documents (the brief's), and every table row also gives: the number of **blow-up documents** (arm KL / row >= 0.3 and >= 5x the control's, at 0.25 / 0.5 M), the 0.5 M
ratio over the other documents, and the median per-document ratio at 0.5 M. The three answers below use the headline ratio; where a companion disagrees, the report says so. The three answers use these definitions, fixed now:
1. **Bigger gain at 0.25 M** (per size and lr): the eps arm's 0.25 M ratio vs the same-lr no-eps arm (paired per document, eps arm / no-eps arm) has its 95 % CI wholly below 1.
2. **Stops the drift**: the arm's 0.25 -> 0.5 M step ratio has a point estimate <= 1.000 while the same-lr no-eps arm's is > 1.000 (or, where the no-eps arm does not drift, the eps arm's is
   not worse than it by more than its CI half-width). A size without a no-eps drift at an lr is reported "no drift to stop".
3. **Makes lr 1e-3 usable at 3.00 / 4.00**: an lr-1e-3 eps arm whose 0.5 M ratio has its CI wholly below 1 **and** whose 0.25 -> 0.5 M step point estimate is <= 1.000. (Report, per eps, at both sizes.)
The report's headline names, per size, the best lr / eps by the 0.5 M point estimate — a descriptive pick on j217, **not an adoption** (nothing is decided or tuned on it; a pick would need a fresh
slice). The report also gives the training curves (running train KL, fixed eval on held-out calib documents: report only), card-hours, and the first-step check against w214.
Not decided here: any recipe, longer budgets, schedules (cosine / linear decays, anneals), a full run, adoption, publication. Verify and public data are never trained on or tuned against.

## Cards / hosts
box2 HIP cards 0-3 (`/data/q27/locks/card<N>.lock`; card lock, then ram.lock shared; MemAvailable >= 35 GB at a student build; `/data` >= 100 GB free). No iceman.
