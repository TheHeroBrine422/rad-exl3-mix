# #188 — elite pool (K per size), crossover and re-seed for the #183 search driver (w166)

Rules: `search/v3/RULES-pool.md` (committed before any pool code ran on data; amended before any run, dated).
Procedure for the switch: `search/v3/SWITCH-186-188.md`. Experiment: `docs/experiments/LOG.md` #188.

## Files

| file | what |
|---|---|
| `driver-pool.patch` | the code: `search/v3/driver.py`, `search/v3/moves.py` (additive: `cross`, `generate_cross` at the end; no HEAD function changed), new `search/v3/poolrule.py` (pure rules). Against HEAD b9c58dd5. The live tree is not edited. Merge with w165's `rerank187/driver-187.patch`: see `SWITCH-186-188.md`. |
| `test_pool188.py` | 13 tests: pure rules; join (add / replace / reject); the worst member as the early-stop reference; rotation rescore, restart, re-selection; re-seed ranking; cross move (units, byte window); cap null with strata; state restart; K = 1 identity on the recorded #183 replay |
| `replay_events.py` | replays the recorded #183 state (read only) through one driver tree and dumps the events it emits; the identity test runs it on two trees |
| `replay188.py` | the sanity check of RULES-pool §7 (whole-slice results; K = 1 reproduces the live accepts; K = 3 pool members and in-sample promotions); writes `summary.json`, `per_slice.csv`, `rows.json` |
| `smoke_dry.py` | dry end-to-end run of the driver on a copy of a live run's state (no evaluations): rotation to a slice, re-seed, pools, rescore, re-selection, restart. Fabricated per-document results. **CPU-heavy once the queue is deep** (the generation with a real fit): run it on colo, not the VM |

Data: `data/pool188/` (`sanity-k1-k3/`, `smoke/`, `tests.txt`).

## Run (on a copy of HEAD with the patch applied)

```
git archive HEAD search eval pipeline | tar -x -C /tmp/w166-check        # or a clean checkout
cd /tmp/w166-check && git apply -p1 <repo>/search/v3/pool188/driver-pool.patch
cp <repo>/search/v3/pool188/*.py search/v3/pool188/
python3 -I search/v3/pool188/test_pool188.py          # 13 / 13 (identity baseline: RXM_POOL_BASE or git archive HEAD)
python3 -I search/v3/test_picks.py                    # 18 / 18
python3 -I search/v3/pool188/replay188.py --snap <snapshot dir> --out <dir>    # the sanity check
```

The identity test and the sanity check read the frozen #183 snapshot: `RXM_POOL_SNAP` (default
`/workspace/rad-local/workers/w165/snap`; its `state.jsonl` sha256 starts `ec82d3cecc5892d0`). Without it the identity test is skipped.

## What is verified, and what is not

- **Verified:** the patch applies to HEAD; the pure rules; the driver's join / replace / reject, rotation, re-seed and
  re-selection on synthetic per-document sums; the early-stop reference is the worst member; the state restarts with the
  same pools; K = 1 emits the same events as HEAD on the 275 recorded decisions of the #183 snapshot; cap null with strata
  has no weights; the cross move keeps each layer from one parent and returns children inside the byte window.
- **Verified on real menu and fits, no evaluations (dry run, depth 6, 60 steps):** rotation to s186a, the re-seed on 8 sizes
  (24 candidates), 4 re-selections at the next rotation, pools of 3, restart invariants (no violations).
- **Not verified:** K = 3 on any live slice; the join path through `generate()` at full queue depth (the deep dry runs were CPU
  heavy and stopped); the statistic on stratified slices with real stratified documents beyond the unit tests.
- **Sanity check, not a gate:** in-sample only (see LOG #188). The history cannot show whether a pool member would win after a
  rotation; the live rotation rescore is that test.
