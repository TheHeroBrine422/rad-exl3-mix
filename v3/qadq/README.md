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
B ARM SIZE flags...                   short run (gate / bench: budgets 2.5e5,5e5, e.g. --max-steps 2), no reads
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
