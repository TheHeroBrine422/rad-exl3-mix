# #187 — move prediction for the #183 search: replay, partial pooling, reranker

Rules (written and committed before any arm was run): `search/v3/RULES-187.md`. Results: `data/rerank187/`
(`table.md` the replay table, `table.json`, `rows.csv`, `winner.json`, `boot.json` in `table.json`). LOG: `docs/experiments/LOG.md` #187.

## Files

| file | what |
|---|---|
| `replay.py` | reads a frozen snapshot of the live driver run (never writes to it); `cands` (candidate table), `jobs` (refit point × arm list), `job` (one fit + its predictions) |
| `pool187.py` | arm (c): partial pooling, a subclass of `search/v3/fit.py`'s SModel (shared fit + per-band deviations of theta / q / kappa, shrinkage prior sd 0.1 / 0.3 / 1.0) |
| `run187.sh` | runs every job with `xargs -P` (restartable: finished outputs are skipped) |
| `metrics.py` | the replay table (Spearman, slope, hit rate, overstatement, audit and per-chunk rho), paired bootstrap, the winner rule; `metrics.py base` picks the base arm for (d) |
| `rerank.py` | arm (d): xgboost reranker on the base arm's out-of-sample predictions (training rows: whole-slice, result index <= N) |
| `test_187.py` | checks: add_obs reproduces the driver's fit0000 input; pool with zero deviations == fit.py; band tables == band row predictions; the candidate counts |

## Rerun (CPU only, colo)

Inputs: the snapshot (`snap/`: `driver.json`, `registry.json`, `menu.json`, `prior-obs-w1a8.jsonl`, `drv/state.jsonl`,
`drv/assign/*.json`, and the lead's `fit0000-all.in.json` for the test). The snapshot used for the 11-15 table is
the live driver at 2026-11-15 00:22 UTC (state.jsonl 1915 lines, sha256 `ec82d3cecc5892d0…`).

```bash
CODE=/data/work/w165/code/search/v3/rerank187; SNAP=/data/work/w165/snap; RUN=/data/work/w165/out
python3 -I $CODE/test_187.py $SNAP                                  # checks
python3 -I $CODE/replay.py cands $SNAP $RUN/cands.jsonl             # candidate table
$CODE/run187.sh $SNAP $RUN 6                                        # 121 jobs (arms a, b, c0.1, c0.3, c1.0), ~30 min on colo
python3 -I $CODE/metrics.py $RUN $RUN/cands.jsonl $RUN/table a,b,c0.1,c0.3,c1.0
BASE=$(python3 -I $CODE/metrics.py base $RUN $RUN/cands.jsonl)     # best of a, b, c* by S
python3 -I $CODE/rerank.py $SNAP $RUN/cands.jsonl $RUN $BASE       # arm d
python3 -I $CODE/metrics.py $RUN $RUN/cands.jsonl $RUN/table a,b,c0.1,c0.3,c1.0,d
```

Refit points are N = 20, 40, … (result events so far, all kinds). A job for (N, arm, band) fits on the rows the driver
had after its N-th result and predicts the candidates whose results are in (N, N+20].

## Notes

- Rows: the driver's own `add_obs` run on a bare `Driver` object (no state writes); reproduces fit0000's input exactly.
- Scale: predictions are log(KL(c)/KL(ref)) at the fit's own scale (no xscale), as the driver's `pred_logr`.
- `fit.py` is not edited; the driver patch for a winner (if any) is a separate `.patch` file, to be applied by the lead.
- Warnings `overflow encountered in power` come from the optimiser's trial steps; every predicted value is checked for finiteness (`n_nonfinite` in each job file).

## Driver hook (default off; `driver-187.patch`)

Lead's note (11-14 7:25 PM CST): w165 and w166 merge in one switch, so this change is kept small and behind its own CONF key.

- `moves.py`: `generate(..., rerank=None, rerank_k=8)`. `rerank=None` runs the original body unchanged. With a scorer, it
  draws up to `rerank_k` distinct new candidates of the kind and returns the one with the lowest predicted log r.
  `solve` is one deterministic mix and is never reranked.
- `driver.py`: `CONF "rerank": {"model": "<static reranker JSON>", "k": 8}`. Absent = off. `rerank_for(size, a, pr)` builds
  the scorer from the size's fitted Pred (base = log KL(cand)/KL(ref)) and the static model.
- `rerank_hook.py`: the feature function on the driver's own objects, and a pure-python evaluator of xgboost's JSON model
  (no xgboost on the VM). `rerank_static.py` trains the model; `data/rerank187/model/b-static.json` is the model trained on
  all replay rows (base = arm b). It is **not out-of-sample**: a live use would retrain it (fit_cmd variant not written).

Apply: `git apply search/v3/rerank187/driver-187.patch` (checked against HEAD; not applied). Enable: CONF `"rerank": {"model": ".../b-static.json"}`.

Tests (on colo, patched tree): `test_hook.py SNAP CANDS RUN MODEL ORIG_MOVES` (features == rerank.py; Booster == xgboost to 2.5e-8;
default path == original moves.generate for every kind and 3 seeds; reranked path valid on a real fit: `data/rerank187/checks/test_hook.log`);
`test_picks.py` 16/18 on both the patched and the unpatched copy (2 FileNotFoundError from input files missing in the reduced copy: same before and after).
