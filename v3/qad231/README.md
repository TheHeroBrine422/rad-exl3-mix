# v3/qad231 — deep-context QAD trainer for the finals (w231)

Report: `rad-local/workers/out/w231.md`. Raw results: `../../data/qad231/`. The trainer changes live in `../qad220/qad220.py`, the queue changes in `../qadq/`.

| what | where |
|---|---|
| `--pipe-xfer host` (default) / `p2p` | qad220 `run_pipe`: the pipeline's hidden state / gradient cross through pinned host buffers (`to_host` / `from_host`, event-ordered; each card's copies issued by its own thread) instead of peer copies; same values |
| data-parallel ranks with `--split-layers` | qad220 main: each rank sees only its own card pair (ROCR_VISIBLE_DEVICES); gloo host all-reduce of the gradient vector each step (as before for one-card ranks); streams s % world; `rank_wall` / `rank_rss_gib` logged |
| `--plan231 PLAN` | qad220 build_data: a mixed-length single-epoch plan's documents + per-stream queues (built by `plan231.py`) |
| `T4` queue kind | `../qadq/qq.py`, `qcard.sh`, `qrun.sh`: `T4 ARM SIZE BUDGETS READS flags @cards=A @partner=B,C,D` = two ranks on A,B and C,D; `qrun.sh ... A,B+C,D ...` launches the ranks |

## Files
- `plan231.py SPEC OUT [--steps-tsv F]` — the plan builder (spec format in its docstring). Prints a summary: shares by rows / tokens / cost, trace starts,
  rows per depth band and lr phase, per-rank step cost and carried state, warnings when traces do not finish by the budget's step or the carried bound
  was exceeded. Run it on box2 (the goldens' metadata), keep the plan JSON there (it lists trace ids).
- `memguard231.sh ARM FINAL_CKPT [FLOOR_GB] [ONLY_CARDS]` — STOP a deep run cleanly when MemAvailable drops below the floor (T2 / T4).
- `gate231.sh LIST CARD...` + `gpu231a.txt` / `gpu231b.txt` — the GPU runs (stress test, 300-step deep run, DP gate) under the card locks.
- `read231.sh NAME CKPT CARD GOLDEN` — a served judge read without a results.tsv row; compare with `../qad226/jl226.py QROOT CTL NAME --golden G`.
- `cmpck231.py A B` — two checkpoints' scale difference (log multipliers).
- `p2p231.py MODE SECONDS` — two-card transfer stress test (peer copies vs host buffers), no model.
- `h2d231.py` — is a pageable host→card copy asynchronous on this ROCm? (no: dad1, see the report).
- Tests: `testhf231.py` (box2 CPU: `HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= QAD_PIPELINE=/data/q27/w231/code/pipeline /data/q27/venv/bin/python testhf231.py`),
  `testplan231.py` (VM or box2, CPU), `test-t4.sh` (VM, the queue's test machine). `ship231.sh` copies the code to box2's `/data/q27/w231/code`.

## Running a 4-card deep line
```
T4 ARM 2.50 BUDGETS READS --split-layers 32 --pipeline --attn-gqa --state-card-gb 4 --cap 262144 --plan231 /data/q27/queue/ext/plan231-X.json \
   --lr ... --compile-scales --compile-ew --a8-int8 @cards=0 @partner=1,2,3
```
Dispatchers need a restart from the new `qcard.sh` to claim T4 (`touch QROOT/stop-card<N>` + `qwait.sh`). Start `memguard231.sh ARM ckpt-<final> 10` beside it.
