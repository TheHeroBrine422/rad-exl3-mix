# #187 replay data (move prediction for the #183 search)

Code: `search/v3/rerank187/` (README there has the rerun commands). Rules: `search/v3/RULES-187.md`. Result and reading:
`docs/experiments/LOG.md` #187 (7:30 PM CST entry). Outcome: no arm passes the rule; baseline kept.

- `cands.jsonl`: one line per candidate with a decision (258): tag, size, band, kind, slice, ref (the incumbent it was
  measured against), result index (N, every result counted), `stopped` (early-stopped), `audit`, the decision and its
  measured `logr`, the driver's live `pred_logr`, bpw / bytes, the size's b at submit.
- `replay/<arm>/<N>-<group>.json`: one refit point. `N` = results so far; the arm fitted on the rows of results <= N
  (plus the prior obs) and predicted the candidates whose results are in (N, N+20]. `preds` maps tag -> predicted log r
  (log KL(cand)/KL(ref)); the rest is the fit report (rows, labels, params, rmse, cost, nfev, `n_nonfinite`).
  Arms: `a` per-band hard fits (group = band), `b` joint, `c0.1` / `c0.3` / `c1.0` partial pooling (shrinkage sd),
  `d` xgboost reranker on arm `b` (`rerank.py`; `mode` = base where fewer than 30 training rows).
- `table/table.md` (the replay table), `table/table.json` (every number, incl. the paired bootstrap and the gate per
  arm), `table/rows.csv` (one line per candidate with every arm's prediction), `table/winner.json`.
- `jobs.txt`: the job list (N, arm, group).

Snapshot: the live driver's state at 2026-11-15 00:22 UTC (state.jsonl 1915 lines, sha256 `ec82d3cecc5892d0…`); not in
the repo (it is the driver's own run folder, read-only for us).

- `model/b-static.json`: the static reranker for the driver hook (xgboost JSON, 60 trees, 28 features; trained on all 104
  whole-slice rows with arm b's predictions; not out-of-sample). Built by `search/v3/rerank187/rerank_static.py`.
- `checks/`: `test_hook.log` (the hook checks), `test_picks_*.log` (the existing test_picks on the patched and unpatched copies: 16/18 both; the two failures are missing input files in the reduced copy).
