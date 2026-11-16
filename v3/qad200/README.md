# v3/qad200: QAD noise: pull-back (AdamW) and big batches (#200, RULES200)

Seven served arms (A4M, P20-4M, B4, P50-4M, B4P20; the 1 M seeds s2 and s3) and one soup (S4seed) of the #191 / #196
scale-QAD recipe on the 2.5 incumbent (`vl/inc2.5m`, #181 data mix, lr 1e-3, served judging on x181g + fe38, pooled
gen rows, per-document bootstrap). Rules: `RULES200.md` (commit 2ad3f6ba, before any run). Tracker: INDEX / LOG #200.
Runtime: none (scales fold into the format; the arms change the trained values only).
Reference arms: D0 = #191's L0B0T0Q0D0 (w170, served), A = #196's arm A (w182, served), A-s1 = w186's seed-1 arm A (#198).

## Files

| file | what |
|---|---|
| `qad200.py` | #196's trainer (`qad196.py`, copied) + the #200 switches. All off = #196 arm A (checked bitwise, `../../data/qad200/checks200.md`). |
| `test200.py` | CPU checks of the switches (schedule in plan / optimizer steps, parser, AdamW closed form, W = 0, accumulation). `python3 test200.py`: 15 checks. |
| `cmp200.py` | bitwise comparer of two trainer runs (state.pt: scales and optimizer; train.jsonl records). |
| `run200.sh` | one training job on a box2 card (card lock held, flock -n, exit 75 when busy, exit 76 when MemAvailable < 35 GB). |
| `q200.sh` | one card's queue: `NAME\|BUDGET\|CKPT\|FLAGS\|DEP` lines; training, then the served read (eval196.py). Restartable. |
| `chk200-one.sh` | the unit check runs (first 3 plan steps, one card). |
| `env200.sh` | box2 env (copied to `/data/q27/w187/env.sh`): w182's #196 env with W = `/data/q27/w187`; shares w182's build lock. |
| `eval196.py`, `patchnorms196.py` | the served path (#196's, unchanged): pack, mix, census, scores on both judge sets. |
| `soup198.py` | soups of trained log multipliers (w186's, unchanged): `build OUT mean CKPT... --start CKPT`. |
| `analyse198.py` | per-arm served KL, per-document bootstrap contrasts (w186's, unchanged). |
| `jobs-smoke.txt`, `jobs-card{1,2,3}.txt` | the queues as run on box2 (card 2 waits for w186's analysis.json). |

## The switches (qad200.py)

- `--wd W`: AdamW (decoupled weight decay) on the trunk log multipliers: before each Adam update they are multiplied
  by (1 - lr W) toward 0 (= the stored scales). W = 20 is 2 % of the offset per step at lr 1e-3; W = 50 is 5 %.
  The RMSNorm gains (N arms only) are not decayed. W = 0 keeps `torch.optim.Adam`, #196's code path.
- `--accum N`: N consecutive plan steps per optimizer step. Each micro-batch is one plan step (24 x 1024 positions, as
  before); the streams carry their states across micro-batches; each micro-batch's KL sum is divided by the optimizer
  step's total rows. One Adam update per N micro-batches. N = 1 is #196.
- `--rows T`: the budget in gen rows (overrides `--budgets`). The plan is built on the whole #181 order (42.1 M rows), so
  the 4 M run is the same stream past 1 M rows: the 1 M run is its prefix. Chunk list: `data/qad200/chunks200.txt`.
- `--order-seed S`: the data-order seed (an alias of `--seed`; w186's A-s1 is `--seed 1`, the #191 switch, so no new code
  was needed for it).
- `--warmup-frac 8/49`: warmup = round(F x optimizer steps): 8 of 49 steps as #191, stretched to 32 of 193 plan steps at 4 M
  (and 8 of 49 optimizer steps for the 4 x batch arms). The decay (last 20 % of the budget) is as #191.

## Running (box2, `/data/q27/w187`)

```
source env.sh; code/v3/qad200/run200.sh NAME CARD ROWS [flags]      # one job (card lock; exit 75 busy / 76 low RAM)
code/v3/qad200/q200.sh CARD code/v3/qad200/jobs-cardN.txt           # a card's queue (train + served read; restartable)
python3 code/v3/qad200/test200.py                                    # CPU checks (any machine with torch)
/data/q27/venv/bin/python code/v3/qad200/cmp200.py runs/A chk/B      # bitwise comparison of two trainer runs
```
