# #189-02 — one-shot re-selection of past incumbents on a fresh slice (RULES-confirm rule 9, w168)

Rule: `search/v3/RULES-confirm.md`, amendment 2026-11-14 9:40 PM CST (rule 9), committed `ddfaced8` before any code.
The design choices the rule leaves open are listed in the header of `driver-reselect.patch`. Experiment: `docs/experiments/LOG.md`
#189-02. **Nothing is applied to the live run.**

## Files

| file | what |
|---|---|
| `driver-reselect.patch` | `search/v3/driver.py` + `search/v3/slices.py` + `search/convplot4.py`, against HEAD `6d0ec11d` (w167's two-stage accept live). CONF `reselect` absent = today's driver. |
| `test_reselect189.py` | 12 tests: candidate set (from_line inclusive, dedupe by assignment hash, retags, members that left the pool), switch + pool top 3 + rescore on the active slice + no decision event, ranking = `PR.reseed_pick` with the z rule, rotation mid-re-selection (RULES-pool §5 held, then taken as done), go-live first + tentatives wait + confirmed vs the new incumbent, restart mid-re-selection, failed evals, registry role / rotation / refusals / pairs without e/f, plot level reset (nothing chained, re-anchor on the model's own read, held without one), identity with CONF `reselect` absent on the live state (on_result replay and `Driver.step` replay) |
| `step_events.py` | runs `Driver.step()` of a tree on a copy of a live run (no results, no fits, queue to a temp file): the step-path identity check and the dry run (`--conf-set`, `--role`) |
| `tests-reselect.txt` | the run on a fresh `git archive HEAD` + patch: 12/12, test_confirm189 15/15, pool188 13/13, test_picks 18/18; plus the mutation check |

Candidate config: `local/workers/w168/driver-189-02.candidate.json` (= the live `driver.json` of 11-14 9:47 PM CST with
`confirm.pairs` = a/b, c/d, `reselect` added, `about` extended). Dry run: `local/workers/w168/dry/`.

## Run the tests

```
cd /workspace/rad-exl3-mix            # after git apply -p1 search/v3/confirm189/driver-reselect.patch
python3 -I search/v3/confirm189/test_reselect189.py     # 12 / 12, ~35 s (identity: RXM_RS_SNAP, default local/workers/w168/snap2)
python3 -I search/v3/confirm189/test_confirm189.py      # 15 / 15
python3 -I search/v3/pool188/test_pool188.py            # 13 / 13
python3 -I search/v3/test_picks.py                      # 18 / 18
```
The identity tests compare `git archive HEAD` (or `RXM_POOL_BASE`) with the working tree: run them before the commit, or set
`RXM_POOL_BASE` to an archive of `6d0ec11d` afterwards.

## Switch procedure (lead)

1. **Code** (VM): `git apply --check -p1 search/v3/confirm189/driver-reselect.patch && git apply -p1 search/v3/confirm189/driver-reselect.patch`,
   the four test files above, then commit `search/v3/driver.py search/v3/slices.py search/convplot4.py` (+ the confirm189 files).
   The plot loop (`local/plots/convplot183-loop.sh`) runs the repo's convplot4, so it picks up the level reset by itself.
2. **Stop the driver** (`touch run/drv/STOP`) and **wait until its PID is gone** (`pgrep -f 'search/v3/driver.py'`; kill it if a fit
   blocks it for > 60 s, the 9:30 PM lesson), then `rm run/drv/STOP`.
   Back up: `mkdir -p run/pre189-02 && cp -a run/driver.json run/registry.json run/pre189-02/`.
3. **Registry** (driver stopped): `python3 -I search/v3/slices.py role run/registry.json s186e reselect`; check with
   `slices.py show run/registry.json`: `s186e staged reselect`, `s186f staged confirm`, s186e decisions `{}`.
4. **Config**: `cp local/workers/w168/driver-189-02.candidate.json run/driver.json`. It differs from `run/pre189-02/driver.json` only
   in `about`, `confirm.pairs` (e/f dropped) and `reselect`; if the live config changed since 9:47 PM, make those three edits by hand.
5. **Start** with the same command. It refuses (`CONF reselect: registry not ready`) if step 3 was missed.
6. **Check** `drv/driver.log`:
   - `replayed:` with the usual counts, no traceback;
   - (go-live already done) at once: `RESELECT on s186e (models since state line 2): 2:8 2.5:9 3:18 3.5:13 4:16 4.5:9 5:8 6:11 = 92
     models, 92 evals to submit` (9:47 PM state; a few more if incumbents / joins changed since);
   - per eval `re-selection eval of <tag> (<size>) scored on s186e: KLD ...`;
   - per size `RESELECT <size>: on s186e vs <incumbent> (n scored, 0 failed, b ...): incumbent kept | -> <eval tag> (of <old tag>);
     pool [...]; ranking (size-adjusted, * = z < -1): ...`, then `RESELECT done on s186e: 2:kept 3:switched ...`;
   - the new models' `incumbent of <size> scored on s186c` rescore lines, then the search resumes;
   - `status.txt`: `| reselect k/8 on s186e` until done. Waiting tentatives (`confirm pending`) are decided after their size.
7. **Cost:** 92 whole s186e evals (169 docs each) + up to ~24 rescores on s186c + one s186b / s186d reference eval per waiting
   tentative whose incumbent changed. The queue holds > `queue_depth`, so the search generates nothing until the s186e evals
   drain: ~45–60 min at ~120 whole evals / h.
8. **Rollback:** stop, restore `run/pre189-02/*`, `git revert` the commit (or `git apply -R`), start. The old code ignores the
   `reselect_*` events; incumbent / pool / submit events are standard, so it lands on the same incumbents and pools. It may run
   RULES-pool §5 once on the active slice for the re-selected sizes. s186e results arriving after the rollback are fit rows.

## Hosts

golden-s186e is on **ds8 and box2** at the registry alias `/data/q27/w164/slices/golden-s186e` (checked 11-14 ~9:50 PM CST):
same file sizes, identical md5 on both for `kld.json` (`993feda8…`), `top.bin`, `rows.f32`, `tokens.i32`, `gen.u8`;
169 docs, doc order / parts / src / tokens / strata = the registry's.
