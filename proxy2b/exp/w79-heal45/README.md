# w79-heal45 · #89-06c · settle healing at K4 / K5 on the 0.8B stacks (4070; ON HOLD)

Worker w79, 2026-11-08. Task `local/workers/instr3/w79.md` item 1. Follows d1105 (`../d1105/README.md`: rule
#89-06 "heal a size on the 27B if its K's healing ratio vs its stack ≤ 0.95 on both seeds", amendment k45lr: lr
screen on seed 0, the best by held-out gets seed 1).
**Status: not run.** The 4070 went off limits again at 11-08 1:20 AM CST (lead note) before any healing started.
Only the confounder check below ran (07:17–07:19 UTC = 1:17–1:19 AM CST; the note file was written 07:19 UTC and I read it at 07:25; 4 evals of ~17 s); the follow-up `check2.sh`
(07:22 UTC) died at once with CUDA OOM beside the user's vLLM server and produced nothing.

## What d1105 k45lr left (seed 0 only; `/data/yb/d1105/result.txt` on the 4070)

| arm | held-out (search docs, training path) | g08-search1 vs its stack (fold path) |
|---|---|---|
| K4 lr 3e-4 | −5.41% | **0.9494** (z −14.4) |
| K4 lr 1e-4 | −4.44% | 0.9605 (z −13.4) |
| K5 lr 1e-4 | −2.42% | 1.0119 (z +2.9) |
| K5 lr 3e-5 | −0.80% | **1.1441** (z +22.0) |

K4: training path and fold path agree (−5.4 / −5.1%, −4.4 / −3.9%). K5: they disagree, by far the most at lr 3e-5,
the arm with the *smallest* multipliers (max |log| 0.0046).

## Confounder check (ran 11-08 07:17 UTC, `check.sh`, `data/check-result.txt`)

- The stack's score reproduces exactly (0.0054978).
- K5 lr 3e-5 rescored: 1.1441 again; **without its folded norms 1.1302**, so the linear folds carry it, not the norms.
  K5 lr 1e-4 without norms: 1.0091.
- The folded wq files equal stack × e^c × e^r to f16 rounding (rel 2.1e-4 on every one of the 150 tensors): no
  corrupted file, no transposed multiplier.
- The +14% is spread evenly over the golden (top 5 documents carry 10% of it; every source 1.07–1.55× except
  swe-smith), not an outlier.
Open: whether the in-memory student (heal.py's wrappers, no weight re-rounding) scores the golden like the fold. In
the fold, every weight is re-rounded f16 → bf16 (`evalq.py`); at lr 3e-5 that re-rounding (rel 1.3e-3) is larger than
the healing change itself (rel 5e-4). w06's jitter control (re-rounding alone) cost 0.44% at K3, where the KL is ~10× K5's.
Not a mechanism yet: step 0 measures it.

## Rule (written 2026-11-08 ~1:25 AM CST (07:25 UTC), before any data of the steps below)

**Step 0, confounder check (gates the reading):** `evalm.py` scores stack and healed arms on g08-search1 through
heal.py's own student (multipliers in memory) for K5 lr 3e-5 / 1e-4 and K4 lr 3e-4, seed 0; jitter folds
(SD 0.002, `fold.py --jitter`) at K4 / K5.
- If the student path gives K5 healed / stack within ±1% of the fold path's ratio, the fold is not the confounder:
  the K5 failure is real (held-out vs golden disagree at K5; reported), the d1105 numbers stand.
- If not (student path ratio < fold ratio − 0.01, or jitter at K5 costs > 2%): the fold path's re-rounding is the
  confounder at these error levels; every K4 / K5 arm is then read on the student path (stack through the same
  wrappers at identity), and the jitter at K4 sets the noise band for K4.
**Steps 1–2 (K4):** seed 1 heals at lr 3e-4 and 1e-4 (d1105's recipe otherwise: qad windows, held-out early stop on
the 48 search docs). The lr is **picked by mean held-out gain over both seeds** (never by a golden). The picked lr's
two seeds are read on g08-search1 (vs their stacks) and checked once on **g08-verify1 + g08-verify2 pooled**
(verify pool; routine before/after check, not iterated).
- **Heal K4** (when 27B healing starts): search ratio ≤ 0.95 on both seeds **and** verify ratio ≤ 0.97 on both seeds.
- **Small:** otherwise, if some seed is ≤ 0.97 on search: healing at K4 is worth ≤ ~5%; the lead weighs it against
  the 27B healing cost.
- **Do not heal K4:** otherwise.
**K5:** seed 0 at the held-out-best lr (1e-4) is 1.012: K5 fails #89-06 unless step 0 flags the fold path; then K5
lr 1e-4 seed 0 / 1 are re-read on the student path with the K4 rule.
**Runtime cost: none** (multipliers fold into the stored suh / svh). 0.8B only; 27B healing stays deferred (user).

## Cost

Step 0 ~5 min; 3 heals × ~30–45 min (seed 1: K4 × 2 lr, K5 × 1); ~10 evals × 20 s. ≈ 2–2.5 h of the 4070 at ≤ 7.2 GB.

## How to run (when the 4070 is back)

`bash /workspace/rad-exl3-mix/proxy2b/exp/w79-heal45/run.sh` (idempotent; stop: `touch /data/yb/w79/STOP`). Queue
registration: copy `queue.d-70-w79heal.sh` to `/data/queue.d/70-w79heal.sh`. Readout: `readout.py /data/yb/w79` (run
by run.sh at the end). Files: `check.sh` (the check that ran), `check2.sh` (step 0's first half, OOM'd), `evalm.py`,
`docsplit.py` (per-document / per-source split of a paired difference), `run.sh`, `readout.py`.
