# v3/qad236 — V3.1 finals prep (w236)

Report: `rad-local/workers/out/w236.md`. Summaries (no document ids): `../../data/qad236/`. The plan JSON and the id lists stay on box2
(`/data/q27/w236/`) and in `rad-local/workers/w236/`. They list the user's traces and calib doc ids.

| file | what |
|---|---|
| `inv236.py QROOT CALIB POOLS_V3 V31_SPLIT OUTDIR` | Source inventory and held-out checks. Trainable calib = qad / fit / search + the 432 released verify docs. Deep traces = the 6 released whole calib traces + xg-fin's 2 longest agent sessions. Overlap checks: calib ids vs verify_v31 / golden-v31ver / v31tr; token-window hashes (first 256 tokens and a 256-token window at the middle) of every training doc vs verify_v31, v31ver, v31tr, v31x, jgap, jt1, jt2, public; xg_id / src_doc overlaps. |
| `mkspec236.py QROOT INVDIR OUT_SPEC [--margin --tail-rows --budget --last-start]` | The plan231 SPEC (one plan for every size). Budget = all the rows / 1.01, rounded down to 0.1M (= 14.8M). A calib-heavy end (the last 2.5M assigned rows at calib 0.8 / xgap 0.1 / tset2 0.1). Shares before the end are set so every source runs out together. Carried bound 500k tokens a rank, max_long 8, trace_hold, 8 trace starts evenly over steps 0..880. |
| `sum236.py PLAN [STEPS_TSV]` | Plan numbers: budget step, wall proxy (sum of the slower rank's cost), carried peak to the budget step, trace starts, deep traces at once per rank, calib share per phase, the deepest 20-step window. |
| `smoke236.sh` | 4-card smoke (box2, setsid nohup). Waits for an idle queue, then takes card locks 0-3. (A) 110 steps from scratch with the finals' 2.50 line. (B) A resumed from its state.pt for 10 steps. (C) 20 steps at the plan's deepest window (mkstate233). It runs the queue's deployed trainer (read-only) into `/data/q27/w236/{smoke25,deep25}`. |
| `smk236.py RUN_DIR [FROM TO]` | Smoke numbers per window: s / step (trainer wall and log-line elapsed), rank imbalance (rank_wall max / mean), card reserved peak / ooms / retries, host RSS, pool misses, evictions, MemAvailable min. |
| `finals-jobs.txt` | The 5 finals' queue lines (T4, r6 flags, flat lr ladder) + JV31 / JT31 / JX31 final reads and the unhealed controls (U). **Not queued.** |

plan231 changes (v3/qad231, separate commits, tests in `testplan231.py`): `share_windows` (shares per assigned-row window) and `trace_hold`
(while a due trace waits, its rank takes no new long document). Plans that don't use them build byte-identically.

## Rerun (box2)
```
W=/data/q27/w236; Q=/data/q27/queue; P=/data/q27/venv/bin/python   # code copied to $W/code (v3/qad236, qad231/plan231.py, qad181/data181.py, qad233/mkstate233.py)
$P $W/code/v3/qad236/inv236.py $Q /data/q27/challenge/golden/calib /data/q27/w217/pools-v3.json $Q/v31-verify-split.json $W/inv
echo '[1433, 4247]' > $W/inv/calib-exclude.json; echo '[1375]' > $W/inv/tset1-exclude.json        # near-duplicates of held-out docs (report)
$P -c "import json; json.dump(json.load(open('$Q/v31-verify-split.json'))['verify_v31'], open('$W/inv/verify_v31.json', 'w'))"
$P $W/code/v3/qad236/mkspec236.py $Q $W/inv $W/spec236.json
OMP_NUM_THREADS=1 $P $W/code/v3/qad231/plan231.py $W/spec236.json $W/plan236.json --steps-tsv $W/steps236.tsv
$P $W/code/v3/qad236/sum236.py $W/plan236.json $W/steps236.tsv
setsid nohup $W/code/v3/qad236/smoke236.sh > $W/smoke236.out 2>&1 < /dev/null &
```
