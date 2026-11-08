# w90-a4heal · healing-pilot prerequisites (merge 11-08 A4 items 2–3): top-512 teacher loss, data starvation (0.8B, 4070)

Worker w90, 2026-11-08 (lead note `local/workers/notes/w90.md`: "4070 after #89-06c, rules written before data").
Runs after #89-06c (`../w79-heal45/`) on the 4070 and is reported with it. Code `heal90.py`, `run90.sh`, `readout90.py`;
outputs `/data/yb/w90/` on the 4070.

**Rules written 2026-11-08 ~12:40 PM CST (18:40 UTC), before any data of these arms.** Not edited after data;
amendments get a date and go below.

### Amendment 1 (2026-11-08 ~1:15 PM CST, lead note 1:10 PM CST / user; before any data of these arms: none had started)

**`t512` (#89-07) is dropped** (our goldens are top-512 already; low value) and **`q4` / `q8` / `rr` (#89-08) are
deferred until after the trunk quant starts** (user). The 4070 queue now runs only #89-06c's student-path reading
(`run79s.sh` → `w79s.sh`, queue `queue.d-80-w79s.sh`). `run90.sh` / `heal90.py` / `readout90.py` stay as written for
the deferred arms (to run #89-08 later without t512: drop the t512 loop in `run90.sh`).

## What is compared

Stacks and controls are d1104's (`/data/yb/d1104`): the K2 / K3 edy → swj → asym ρ 0 stacks
`q-k{K}edyswjasym_r0[_s1]/wqall` (seeds 0 / 1 = quantizer seed offsets) and their **heal0** healing (w06 `heal.py`
defaults: 1,024 qad windows of ≤ 2,048 tokens, 2 epochs = 512 steps of 4 windows, lr 1e-3, trust 0.3, held-out early
stop on 48 search-pool docs not in the golden, data seed 0), already scored (`k{K}edyswjasym_r0[_s1]_heal.f32`). Every
arm below is heal0 with exactly one change, scored the same way (fold path: `fold.py` + `evalq.py`, full-vocabulary
KLD on `/data/g08-search1`, paired with `pcmp.py`, document bootstrap).

| arm | change vs heal0 | K × seed |
|---|---|---|
| `t512` | loss = KL of the teacher's **top-512 tokens + one lumped tail bin** (the challenge golden's stored form), in training **and** in the held-out early stop (`heal90.py`, `W90_TOPK=512`): everything the student learns from comes from the truncated target | K3, K2 × s0, s1 |
| `q4` | **¼ of the data**: the first 256 of heal0's 1,024 windows (same shuffle; nested), 8 epochs → the same 512 steps | K3, K2 × s0, s1 |
| `q8` | **⅛**: the first 128, 16 epochs → 512 steps | K3, K2 × s0, s1 |
| `rr` | none: heal0 rerun (K3 s0), the run-to-run noise floor (GPU nondeterminism) | K3 × s0 |

Design choice (before data): the data arms keep the step count (compute) equal, so only the amount of distinct
data changes; the held-out early stop protects them from overfitting as it does heal0.

## Rules

Per arm and seed: r = KLD(arm) / KLD(heal0) on g08-search1 (paired), and the **gain kept**
g = (KLD(stack) − KLD(arm)) / (KLD(stack) − KLD(heal0)) (1 = all of heal0's gain).
Noise floor: n = |log r(rr)|; reported, and the bands below are widened to max(band, 2n) if the rerun moves more.

**Item 2 (top-512 target):**
- **Same:** mean r over the two seeds ≤ 1.01 at both K and every seed ≤ 1.02 → a stored top-512 + tail teacher is
  enough for 27B healing (no full-logit teacher pass needed).
- **Loses:** mean r > 1.03 at either K → the 27B pilot needs the full-vocabulary teacher (online, or more stored mass).
- Otherwise: **small loss**, reported with g; the lead weighs it against the full-teacher cost.

**Item 3 (data starvation):**
- **Saturated:** q4's mean g ≥ 0.95 at both K → 1,024 windows (~2M tokens) are more than enough at the 0.8B.
- **Data-limited:** q4's mean g < 0.90 at either K → the gain still grows with data at 1,024 windows; with 4–8× more
  parameters per scored row at the 27B, its pilot should use the whole qad pool (and more).
- Otherwise: in between, reported. q8 gives the second point of the curve (reported, not deciding).

Reported, not deciding: per-seed r, z, g; KL99 and top-1 (pcmp); the held-out curves (train.jsonl: best step, max
|log|); the top-512 arm's own coarsened held-out KL vs heal0's full one is **not** comparable and is not reported as a
gain.

**Runtime cost: none** (multipliers fold into the stored suh / svh; the question is the 27B healing pilot's data and
teacher, not the format). 0.8B only, f16 path (no A8).

## #89-06c's student-path branch (`w79s.sh`, `readout79s.py`)

#89-06c's step 0 flagged the fold path (K5 lr 3e-5: student path 0.9930 vs fold path 1.1441; 18:42 UTC), and w79's
rule then says every K4 / K5 arm is read on the student path, K5 lr 1e-4 included, with the K4 rule. w79's `run.sh` /
`readout.py` implement only the fold-path reading, so `w79s.sh` (run first, once `W79 HEAL DONE`) scores the stacks
and healed arms through `evalm.py` on g08-search1 and g08-verify1 / 2 (into `/data/yb/w90/w79s`) and
`readout79s.py` applies w79's K4 rule unchanged (search ≤ 0.95 and verify1+2 ≤ 0.97 on both seeds → heal; some seed
≤ 0.97 on search → small; else not). w79's own fold-path readout is reported beside it. ~26 evals × ~20–30 s.

## Cost

13 heals × ~16–17 min (heal0 took 16 min at 512 steps) + 13 fold / evals (~30 s) ≈ 3.6 h of the 4070 at
`VRAM_CAP_GB=7.2`, one unit at a time under `/data/yb/gpu.lock`.

## Run

Queue entry `queue.d-80-w90a4.sh` → `/data/queue.d/80-w90a4.sh`; `run90.sh` waits for `W79 HEAL DONE` (#89-06c)
unless `/data/yb/w90/NOWAIT` exists, is idempotent (score files, `params.pt`, `wq/.done`), stop: `touch
/data/yb/w90/STOP`. Readout: `readout90.py /data/yb/w90` (run by run90.sh at the end, `W90 A4 DONE`).
