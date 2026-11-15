# #189 — two-stage accept (confirmation on a fresh slice) for the #183 search driver (w167)

Rule: `search/v3/RULES-confirm.md` (user 11-14 9:05 PM CST, committed `29e7398e` before any code). Implemented exactly as
written; the open points and how they were decided are listed in the header of `driver-confirm.patch`.
Experiment: `docs/experiments/LOG.md` #189. **Nothing is applied to the live run.**

## Files

| file | what |
|---|---|
| `driver-confirm.patch` | `search/v3/driver.py` + `search/v3/slices.py`, against HEAD `5f4225e7` (driver = `4e089db2`: strata + pool K=3 + cross). Header = the design choices. CONF `confirm` absent = today's driver. |
| `test_confirm189.py` | 15 tests: tentative → confirm → accept / reject (K=3 and K=1), pool join confirmed as `join`, incumbent changed mid-confirmation (rule 3, cached), rotation mid-confirmation (retag = same model, K=3 waits for the rescore), restart mid-confirmation, failed confirm eval, paused size untouched, go-live keep / revert / same + one-shot, go-live restart, rotation skips confirm slices (+ refusal without roles, `slices.py role`, draw), convplot4 chains the stage-2 log r, and identity with CONF confirm absent on the #183 replay and on a copy of the live K=3 state |
| `tests.txt` | the run on a fresh `git archive HEAD` + patch: 15/15, pool188 13/13, test_picks 18/18 |

Candidate config: `local/workers/w167/driver-189.candidate.json` (= the live `driver.json` of 11-14 9:20 PM CST + `confirm` + `about`).

## Run the tests

```
cd /workspace/rad-exl3-mix            # after git apply -p1 search/v3/confirm189/driver-confirm.patch
python3 -I search/v3/confirm189/test_confirm189.py      # 15 / 15, ~30 s (identity: RXM_POOL_SNAP, RXM_CONF_SNAP; base = git archive HEAD or RXM_POOL_BASE)
python3 -I search/v3/pool188/test_pool188.py            # 13 / 13
python3 -I search/v3/test_picks.py                      # 18 / 18
```
After a commit of the patch, `git archive HEAD` is the patched tree, so the identity tests then compare the tree with itself.
Run them before the commit, or set `RXM_POOL_BASE` to an archive of `5f4225e7`.

## Switch procedure (lead)

1. **Code** (VM, repo clean for the two files): `git apply --check -p1 search/v3/confirm189/driver-confirm.patch && git apply -p1 search/v3/confirm189/driver-confirm.patch`,
   then the three test files above. Commit `search/v3/driver.py search/v3/slices.py` after review.
2. **Stop the driver** between decisions: `touch /workspace/rad-local/workers/lead-srch/run/drv/STOP` (it exits at its next loop),
   then `rm` that STOP file. Back up: `mkdir -p run/pre189 && cp -a run/driver.json run/registry.json run/pre189/`.
3. **Registry roles** (driver stopped, it rewrites the file):
   `for s in s186b s186d s186f; do python3 -I search/v3/slices.py role run/registry.json $s confirm; done`
   (status stays `staged`; only `role` is added). The next search slice is s186c: set `"status": "ready"` on **s186c only**
   when you want the next rotation (s186a was at 187 of 200 decisions at 9:23 PM CST; without a ready slice the driver logs
   "rotation due but no ready slice" and stays on s186a). Tentatives from s186a confirm on s186b even after the rotation.
4. **Config**: `cp local/workers/w167/driver-189.candidate.json run/driver.json`. Re-check: differs from `run/pre189/driver.json`
   in `confirm` and `about` only (if you edited the live config since 9:20 PM CST, add the `confirm` key to your file instead).
5. **Start** with the same command (`python3 search/v3/driver.py <run>/driver.json` from the repo). It refuses to start
   (`CONF confirm: registry not ready`) if step 3 was missed.
6. **Check** in `drv/driver.log`:
   - `replayed:` with the same counts as before, no traceback;
   - `GO-LIVE check on s186b (incumbents at state line 3271): 5 sizes to confirm (3 3.5 4 4.5 6), 3 unchanged` (dry run on the
     9:21 PM state; the list follows the incumbents at restart), then 2 `confirm` submits per listed size;
   - per size `GO-LIVE <size>: <cur> vs switch-time <inc-006xx> on s186b: ... -> kept | reverted`, then `GO-LIVE done: ...`;
   - stage-1 accepts appear as `TENTATIVE ...; confirmation on s186b`, later `ACCEPT (confirmed)`, `JOIN (confirmed)` or
     `reject (confirm)` with both stages' numbers;
   - `status.txt`: `| confirm pending N (size:n ...)` and `| go-live k/8` until done;
   - hosts: confirm evals run on s186b (no missing golden: s186b/d/f checked on ds8 and box2, md5 of kld.json identical to each
     other and doc order = registry).
7. **Rollback:** stop, restore `run/pre189/*`, `git apply -R -p1 ...` (or revert the commit), start. The old code replays the
   state: `tentative` / `confirm` / `golive*` events are ignored, a stage-2 `decision` counts as a decision, the incumbent / pool
   events are the usual ones. Pending tentatives are dropped (never applied). A confirm eval landing after the rollback is
   logged as a `stale` decision by the old code (harmless; drain the queue first if you want none).

## What it costs (live rates, 11-14 8:18–9:22 PM CST)

Stage-1 accepts + joins at 2.0–6.0: **~21 / h** (8–9.5 incumbent changes, 12–13 joins), with ~170 evals / h ≈ 120
whole-slice equivalents / h (search evals stop at 73 % of the slice on average). Each tentative needs one whole confirm eval
(174–180 docs ≈ one s186a whole eval); the reference is usually cached (a confirmed candidate's own confirm eval is the next
reference), so add ~1–3 / h for references not yet scored on the slice. **≈ 22–24 whole evals / h ≈ 18–20 % of the current
eval capacity**, more than the rule's estimate (10–15 %) because joins are more than half of today's stage-1 accepts.
The queue is shared (`queue_depth` 22 counts confirm evals), so in effect ~16–17 % fewer search decisions per hour (≈ 168 → ~140).
One-time: go-live 10 evals; at each rotation to a new pair, the first tentatives per size pay one reference eval each (≤ 8
incumbents, plus worst members for joins).

## Notes for the lead

- **The plot already chains pool joins today (K = 3 without confirm).** convplot4 chains every `decision: accept`, and a
  join that does not change the incumbent is logged as `accept` with its log r against the *worst member*. Since the switch,
  11 of 21 accepts were such joins, so the live training line overstates by their log r. Confirm mode writes them as
  `decision: "join"` (not chained). The past events are unchanged; a convplot4 fix for the history (skip accepts with no
  following `incumbent` event of that tag) is yours to decide.
- Offline readers of `decision` events (`rerank187/replay.py`, `a8172/a8gap.py`) see a confirmed tag's stage-2 numbers; the
  stage-1 numbers are in its `tentative` event (and `logr1` / `z1`).
- Stage-2 rows are fit rows (labels `s186b|...`): the fit gets new slice offsets for s186b/d/f.
