# v3/qadq — the global QAD job queue (w222)

One queue per machine, one dispatcher per card; every card pulls the next ready job. Trainer `../qad220/qad220.py` (unchanged), data / order / students / judge = w214's / w217's (j217),
so a job gives the same numbers on either machine (gate below). Raw results: `../../data/qadq/`. Report: `rad-local/workers/out/w222.md`.

| machine | queue root `QROOT` | cards | card lock | RAM floor | trainer env |
|---|---|---|---|---|---|
| box2 | `/data/q27/queue` | 0 1 2 3 | `/data/q27/locks/card<N>.lock` (+ `ram.lock` shared) | 35 GB MemAvailable | `/data/q27/venv`, inputs `/data/q27/w214`, j217 `/data/q27/w217/slices/golden-j217` |
| iceman | `/data/queue` | **0 only** (card 1 serves the user's Qwen: never) | `/root/card0.lock` (+ `/root/ram.lock`) | 30 GB | `/data/venv-exl3`, inputs `/data/queue/w214`, j217 `/data/queue/slices/golden-j217`, radiance = w218's V3 build |

`QROOT/`: `jobs.txt` (the queue) · `claims/` · `done/` · `failed/` · `cancel/` · `runs/<ARM>/` (trainer outputs) · `scores/` (served-read outputs `<ARM>-<ckpt>.json(.rows)`) · `q.log` (milestones, CT) ·
`run.log` · `logs/` · `results.tsv` · `code/` (this tree; `deploy.sh` on the VM ships `v3/qadq`).

## Job lines (`jobs.txt`, shlex syntax, `#` comments, one job per line, taken **in order**)
```
T ARM SIZE BUDGETS READS flags...     train, then the served reads.  BUDGETS = comma list (as qad220 --budgets); READS = comma list of ckpt dir names (ckpt-2e+05,...) or -
R ARM SIZE CKPTDIR                    read-only: the served read of an existing checkpoint dir (absolute, or relative to runs/). Score name ARM-<basename CKPTDIR>
B ARM SIZE flags...                   short run (gate / bench / gradnoise: budgets 2.5e5,5e5, e.g. --max-steps 2), no reads
M ARM SIZE BUDGETS TAPDIR flags...    two-card job (the teacher tap pass, `--make-taps TAPDIR`): needs `@cards=N` (the dispatchers allowed to take it) and `@partner=P`; the claimant takes P's card lock too (waits 45 min at most, else releases the claim), the partner's dispatcher takes no new job while the claim is live, the trainer sees both cards; needs >= 200 GB free under TAPDIR; done when TAPDIR/.done exists
directives (anywhere after the fixed fields):  @needs=PATH   the job waits until PATH exists (absolute or relative to runs/)      @ctl=ARM   control arm: the row gets the paired ratio vs ARM-<same ckpt>
{runs} in a field = QROOT/runs.    SIZE = 2.50 / 3.00 / 4.00 (the released picks; student = w214's rad2level214 of the HF release)
```
A budget of `1.5e6` names its checkpoint `ckpt-2e+06` (python's `%.0e`): READS must use the trainer's names. `--lr-steps` / `--decay-shape` runs take ONE budget (qad220's rule).
Flags are the trainer's: compiled runs = `--compile-scales --compile-ew`. A job is skipped when its outputs exist (final `ckpt-*/.done` and every `scores/<ARM>-<ck>.json` for T; the score for R).

## Editing the queue = editing `jobs.txt`
- **Add:** append a line (dispatchers poll every 60 s). **Reorder:** move lines (an unclaimed job is chosen by position at each pull; a running one is unaffected). **Remove an unstarted job:** delete its line.
- **Cancel a job:** `touch QROOT/cancel/<KEY>` (KEY = `T-ARM`, `R-ARM-<ckpt>`, `B-ARM`; `qq.py status QROOT` lists keys). A queued job is then skipped; a running training: also `touch QROOT/runs/<ARM>/STOP`
  (the trainer saves `state.pt` and exits; the job is marked cancelled, no reads). Never `kill` a training.
- **Claims** (`claims/<KEY>` = `CARD PID HOST TIME`, written under `queue.lock`): a claim is live while its dispatcher pid is alive. A dispatcher that died leaves a *stale* claim; the next dispatcher takes it over
  and resumes the job (trainer from `state.pt`, finished checkpoints / reads skipped). A job that exits non-zero gets `failed/<KEY>` and is **not** retried: delete that file (and `claims/<KEY>`) to rerun.
  `done/<KEY>` marks finished jobs. `qq.py status QROOT` prints every line with its state (done / failed / cancelled / running / stale / waiting / ready).
- A job whose `@needs` / checkpoint is missing is skipped *in place* (later ready jobs run first; it runs when the file appears).

## Dispatcher
```
setsid nohup QROOT/code/v3/qadq/qcard.sh MACHINE CARD < /dev/null > QROOT/logs/qcard-CARD.out 2>&1 &     # MACHINE = box2 | iceman
qcard.sh: card lock held for the whole job (training + reads) -> qrun.sh (trainer) -> qread.sh per read (swap214.py on the released host-emb container, qscore.sh = plain radiance --kld-ref on j217,
          row appended to results.tsv by qrow.py) -> qimport.sh (box2: w221's / w220's legacy reads) -> next job. Idle: polls jobs.txt.
touch QROOT/stop | QROOT/stop-card<N>          # stop between jobs (the running job finishes), then start the dispatcher again
qwait.sh MACHINE CARD PID...                    # start qcard when the old dispatchers (PIDs) have exited (the hand-over of w217's / w221's cards)
```
Restart-safe at every level: lost dispatcher -> stale claim -> resume; trainer restarts from `runs/<ARM>/state.pt`; outputs written `.part` then renamed. RAM floor not met: the dispatcher waits (exit 76 of qrun, retry every 2 min).

## Results
`results.tsv` (one row per read): when (CT), machine:card, arm, ckpt, size, KL / gen row, ratio vs the unhealed release on j217 (`ctl-<size>` rows) + paired per-document bootstrap CI (cmp214's arithmetic), ratio without
w217's blow-up documents (calib ids 1893, 3820), docs better / worse, control, ratio vs control + CI, s / step (median wall of main steps >= 2), score name. `python3 readout.py box2-results.tsv iceman-results.tsv [--size S] [--arm RE]`
prints the markdown table across machines. `readout.py --gate A/train.jsonl B/train.jsonl` compares the first two main-step records bit for bit.

## iceman card 0 (`/data/queue`) and the gate
Inputs identical to box2's: `iceman-pull.sh` (VM; goldens from Thor over the LAN with a forwarded ssh agent, small files + j217 + ctl rows from box2), HF download of the three host-emb releases + `iceman-setup.sh`
(`rad2level214.py` per size, `--check 4`), `inputs-digest.sh` (md5 of every input on both machines, diffed). Gate (2.50, the 25C flags, `--max-steps 2`): see the report.

## The int8 batch and the `a8int8.ok` handshake (11-22 night)
Every line of `jobs.box2.int8.txt` / `jobs.iceman.int8.txt` runs with `--a8-int8` (w223's real int8 forward; implies `--a8`) and carries `@needs=<QROOT>/a8int8.ok`, so no dispatcher starts one until that file exists.
**w223 / lead: create `/data/q27/queue/a8int8.ok` (box2) and `/data/queue/a8int8.ok` (iceman) only after (1) w223's gates passed and (2) the int8 trainer (the `--a8-int8` flag in `v3/qad220/qad220.py` or the copy `qrun.sh` runs: `TRAINER=$QROOT/code/v3/qad220/qad220.py`) and the kernel build / extension it needs are installed in THAT machine's `QROOT/code` and import under that machine's python (`/data/q27/venv`, iceman `/data/venv-exl3`; iceman has the system ROCm 7.14, box2 `/opt/rocm-10.1` libs).
Install a new copy and rename it into place (the running jobs have loaded the old file). Test the flag on the card with `B` bench line first if in doubt: `B i8-gate 2.50 --lr 1e-3 --compile-scales --compile-ew --a8-int8 --max-steps 2`.**
A line is skipped in place (later ready lines run) until its file exists. To run the batch without the handshake (e.g. the int8 path is dropped), delete the `@needs=` tokens or create the file. `python3 hours.py FILE...` prints the card-hour estimate (w220 speeds + 10 % A8, +7 min a read; the int8 kernel makes the training part cheaper).
Budgets are capped at 2e6: the plan holds 2.52 M gen rows (a 3 M / 4 M budget asserts "the data hold ... rows < budget"; ~21 k rows a step up to ~2 M, then the tail steps are sparse).

## w222 follow-up 3 additions (11-22 ~4 AM CST)
- **Restart a dispatcher between jobs** (to pick up a new `qcard.sh`): `touch QROOT/stop-card<N>` then `setsid nohup qwait.sh MACHINE N $(cat QROOT/dispatch/card<N>.pid) …`; qwait removes the stop file when the old dispatcher is gone and starts the new one. Only a dispatcher started from the current `qcard.sh` takes `M` jobs (`qq.py claim … --caps M`).
- **swap222.py** (used by `qread.sh` since 4:10 AM CST; = swap214.py for a checkpoint with only trunk scales, md5-identical output checked): also folds `scales/lm_head.safetensors` (H: `r` on output.weight's suh plane, `c` on svh) and `norms.safetensors` (N: the bf16 RMSNorm weights as `write_ckpt` stores them, written over the container's `blk.N.attn_norm / ffn_norm / attn_q_norm / attn_k_norm / ssm_norm.weight` and `output_norm.weight` planes, which hold the checkpoint's bf16 weights as they are). `test-swap222.py` builds an identity and a gain checkpoint for it (identity differs from the plain read only in 26 norm planes by <= 6e-8: the `(1 + w) * g - 1` form's float rounding).
- **qad220**: `--head-train none|in|both` (H: `in` = the 5,120 hidden-side `r` of the served head only; exported as `scales/lm_head.safetensors`), `--local-mse LAYERS,lambda,block` (the listed layers end blocks: each block trains jointly on the MSE at its end layer from the student's own detached state; the KL reaches only the layers after the last listed one; `test220.py` checks it).
- **readout**: `qrow.py refresh ROOT --golden G --ctldir C` recomputes every row incl. the new columns; D1 tools `blowup222.py`, `blowup222-detail.py`; token-frequency bins `tokfreq222.py` -> `QROOT/tokbins.npz`; `gn222.py` (gradient-noise significance + lambda); `qhold.py` (hold / release lines); `hours.py`.

## Checkpoint rule (user 11-22, all training lines from now on)
Every training line gives a **curve, not a point**: a decayed checkpoint read on j217 at **each of 0.25 M / 0.5 M / 1 M / 2 M (/ 4 M) rows up to the line's budget**.
- Plain lines: `BUDGETS` = those points up to the final budget, each a WSD branch (a cheap decay forked off the main run), `READS` = their `ckpt-<rows>` dirs. Example: `T X 2.50 2.5e5,5e5,1e6,2e6 ckpt-2e+05,ckpt-5e+05,ckpt-1e+06,ckpt-2e+06 --lr 1e-4 ...`.
- Lines without a constant phase or with a piecewise lr (`--decay-shape cosine-all|linear-all`, `--lr-steps`; qad220 takes one budget for them): the run's end is the real number; the intermediate points are **short-decay branches** (`--short-decay 2.5e5,5e5,1e6`: at each row count a branch forks off the main run and decays linearly from the run's own lr to ~0 over 50 k rows, `--short-decay-rows`; checkpoints `ckpt-sd-<rows>`). `readout.py` labels these "short-decay". They are not the mid-schedule (undecayed) states.
- **`--accum > 1` takes the same branches** (qad220 `branch_plan`, 11-22): a branch's fork point sits on an optimizer-step boundary (plan step g0 x accum, g0 = min(ceil(fork step / accum), the branch's last group)), the branch runs the standalone run's remaining optimizer steps with the standalone lr (`group_lr` with last = the branch's end step, fork_last = its decay start; the last group may be partial), so a branch at budget T equals a standalone `--accum` run of budget T with the same flags (gate: bit for bit, see the report). WSD only (no `--lr-steps` / `--decay-shape` / `--short-decay` / `--warmup-frac` with accum > 1 and branches) and the main run's constant phase must reach the fork. At 4x batch a branch whose decay window (20 % of its budget, in rows) is shorter than one optimizer step (85 k rows) does not decay at all: the 0.25 M and 0.5 M checkpoints of a 4x run are undecayed.
- Not covered: forks (`--fork-from`: stage 2 of the staged MSE runs, F1-*; a branch needs its fork point after the state's step).
- **New job kinds need a dispatcher restarted from the current `qcard.sh`** (`touch QROOT/stop-card<N>` + `qwait.sh`): `qq.py claim` hands `M` / `U` jobs only to dispatchers that pass `--caps` for them (an old dispatcher would claim them and never run them).
- `python3 qrule.py QROOT [--apply]` checks / rewrites every not-yet-started `T` line to the rule (backup `jobs.txt.pre-qrule`); run it after adding lines.
## Extended data plan (qad214-ext)
`--plan-ext {ext}/ext-plan.json --ext-root {ext}` (QROOT/ext: `gold/<new goldens>`, `genmask/`, `ext-plan.json`) points a job at the plan: w214's 1,835 documents in w214's order (first 84 plan steps = 1.76 M rows identical to w214's), then 5,326 new documents (xcorp / trace goldens xc8k-b1, xc8k-b2-c2, tr32k-b1 deduplicated, + calib qad-pool documents), one epoch, 10.04 M gen rows, calib 20 %. Budgets up to 1e7 (all 24 streams busy to 8.07 M). Add `@needs={ext}/gold/xc8k-b1/.done,{ext}/gold/xc8k-b2-c2/.done,{ext}/gold/tr32k-b1/.done`. Built by `plan222-count.py` + `plan222-build.py` (metadata only), shipped by `ext-pull.sh`. The plan file lists per-document ids of the user's traces: it stays on our machines (not in the repo).
## Fragility tests (D1 follow-up)
`U ARM BPW` (released container read as is), `R` reads of `mkpert222.py` checkpoints (random log-multiplier noise at a derailing checkpoint's rms / per-family revert); `fragility222.py` tabulates onset and KL / row after it per document.

## Trainer installs (11-22)
New code copies are installed by patching a fresh copy of `QROOT/code` and renaming it into place only for files that no running job reads again (python has the old file loaded): `qad220.py` carries `--head-train`, `block` / `joint` local MSE, Adam-state mapping for forks into a bigger trainable set, `--plan-ext`, `--short-decay`, accum branches, `--dynamo-limit`. Gate runs of a candidate trainer: `gate222.sh MACHINE CODEDIR ARM BUDGETS flags` (waits for a free card via the card locks, never preempts).

**Undecayed snapshots (lead 11-22, user: "checkpoint everything"; the norm since ~5:40 PM CST):** qad220 `--snap-at` defaults to `auto` (0.25 / 0.5 / 0.75 / 1 M, then every 0.5 M before the final budget; `none` = off; a fork skips points at / before its start; the list goes to `runs/<ARM>/snaps.json`), and `qq.py` adds the matching `ckpt-raw-*` reads to every T line automatically (only for runs the new trainer makes: no run dir yet, or a snaps.json) — job lines need nothing. `--snap-at r1,r2,..` writes the main run's state as it is (no decay, no branch) as `ckpt-raw-<rows>` (exact labels: `ckpt-raw-1.25e+06`) when the run passes each row count; nearly free (a checkpoint write + a 7-min read). Every new training line carries them on a grid (every 0.25M to 1M, every 0.5M after) next to its decayed WSD / short-decay points: the raw curves are the like-for-like plateau comparison across runs (same main-run states, no decay differences). `addsnap.py QROOT 'ARM=r1+r2;..'` adds them to unstarted lines. `--exact-ckpt-names` (off by default) gives budgets exact labels too (1.5e6 -> ckpt-1.5e+06 instead of python's ckpt-2e+06).
