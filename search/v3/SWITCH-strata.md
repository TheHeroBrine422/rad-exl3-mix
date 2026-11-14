# Switching the live search to the stratified rule (#186, w164) — procedure for the lead

w164 did not touch the live search (driver, coordinator, guard, hosts). This file is what the lead runs, in this order.

## What is ready (verified, not deployed)
| file | for | verified |
|---|---|---|
| `search/v3/earlystop-strata.patch` | the hosts: `pipeline/rxrun/hostd.py` (strata in the early-stop block), `pipeline/rxrun/earlystop.py` (= `seqrule.py`), `search/v3/seqrule.py`, `search/v3/pickstat.py`, tests | `git apply --check -p1` on a copy of the ds8 tree (`/data/q27/srch183/code`); on the patched copy: `test_picks.py` 18/18, `test_hostd_es.py` ALL PASS, `test_strata186.py` (pure) ALL PASS |
| `search/v3/driver-strata.patch` | the driver: `search/v3/driver.py` (against ds8's copy, which keeps its #183 decode / monitor lines), `slices.py`, `README.md`, `strata186.py`, `test_strata186.py` | `git apply --check -p1` on ds8's driver / slices / README; the patched copy parses |

The same change is in the repo at HEAD (#186-02), so the VM's driver tree (`/workspace/rad-exl3-mix`, which the running
driver imports from) already has it. Old slices are unchanged by it: `test_picks.py` `old_slice_unchanged_recorded`
pins the HEAD numbers, and an unstratified 36k-step comparison gave zero mismatches.

## Why the order matters
- The driver writes `"strata"` into each early-stop block and `"rule": {"strata": true}`. An **old host** ignores both
  and would stop stratified candidates on unstratified statistics. So: **hosts first, driver second**.
- An **old driver** ignores the registry's `stratum` fields and would decide stratified slices with the old statistic.
  So the driver must be restarted **before** any s186 slice becomes `ready`.
- The running driver (PID 370668, started about 5:00 PM CST on 11-14) still runs the pre-#186 code from memory; it loads the new
  modules on its next start. Its `state.jsonl` / registry resume as before.
- The live registry is written by the driver on every decision (`Registry.save`). Never copy a file over it while the
  driver runs: its in-memory copy would overwrite the copy at its next save. Use step 4.

## Steps
1. **Backup** (ds8): `cd /data/q27/srch183/code && mkdir -p ../code.pre186 && cp -a pipeline search ../code.pre186/` ;
   note `md5sum` of `pipeline/rxrun/hostd.py pipeline/rxrun/earlystop.py search/v3/{seqrule,pickstat,slices,driver}.py`.
2. **Apply the hosts patch** to `/data/q27/srch183/code` (`git apply -p1 earlystop-strata.patch` from that directory, or copy
   the six files from the verified copy: they are the patch's post-image). Check with `md5sum` against the verified copy.
3. **Hosts test on ds8** (CPU, `nice -n 19`): `cd /data/q27/srch183/code && python3 -I search/v3/test_hostd_es.py pipeline` →
   `ALL PASS`. (It needs `pipeline/rxq` and `pipeline/config`, which the tree has.)
4. **Restart the hosts' daemon** (`python -m rxrun.hostd /data/q27/srch183/host.json`, ds8, pid from `pgrep -af "rxrun.hostd /data"`
   with the **daemon** line, not the `--unit` children). Take the command line and cwd from `/proc/<pid>/cmdline` and
   `/proc/<pid>/cwd`. Eval units already running keep their code; new units import the new tree. Restart when no unit is in
   flight, or accept that in-flight units are lost (they are restartable). The lead decides the moment.
5. **Driver** (VM): when the driver is between decisions (`status.txt` quiet, outstanding evals drained or accepted as
   lost), stop PID 370668 and start it again with the same command from `/workspace/rad-exl3-mix`:
   `python3 search/v3/driver.py /workspace/rad-local/workers/lead-srch/run/driver.json`. It resumes from `state.jsonl`.
   Check `driver.log` for the first decision lines after the restart.
6. **Registry** (driver stopped, between steps 5 and 7): add the six staged entries to the live `registry.json` without copying
   the whole file. The staged file is `/workspace/rad-local/workers/w164/registry-staged-186.json` (a copy of the registry as of
   5:52 PM CST plus `s186a`–`s186f` with `"status": "staged"`). Merge only the `s186*` keys:
   ```python
   import json
   live = json.load(open(LIVE)); staged = json.load(open(STAGED))
   for k in [k for k in staged["slices"] if k.startswith("s186")]:
       assert k not in live["slices"]; live["slices"][k] = staged["slices"][k]
   json.dump(live, open(LIVE + ".part", "w"), indent=1)
   import os; os.replace(LIVE + ".part", LIVE)
   ```
   Keep `doc_decisions` as they are in `live` (the staged copy's are older).
7. **Start the driver again** (step 5 again) so it loads the registry with the staged slices. Staged slices are not
   activated: the driver rotates only to `ready` slices.
8. **Activate** (lead's decision, after the checks above): set `"status": "ready"` on the slice you want next (one at a
   time, s186a first). The driver rotates to it when the active slice reaches its rotation (`rotate_after` decisions, 25),
   and rescores the incumbents on it (RULES-strata §8; s173 decisions stand).
9. **Verify after the first stratified eval**: its line in `evals.jsonl` has `early_stop.strata` (183 entries) and
   `early_stop.rule.strata == true`; the host's `eval.log` has no `ValueError`; the driver's decision events carry
   `parts[*].strata` (per-stratum logr / docmean). `status.txt` shows the slice name.
10. **Rollback**: stop the driver, restore the backed-up files in `/data/q27/srch183/code` (step 1), restart the hosts'
    daemon and the driver; remove the `s186*` keys from the registry (their `stratum` fields are then ignored).

## Notes for the lead
- The stratified fit rows are labelled `S|p|s|full` / `S|p|s|pre<k>`; `fit.py`'s `part_of` still returns `p`, so the x-part
  lambda and the per-part offsets work as before. The fit (kfit3 on colo) needs no change.
- `pickstat.capped` changed for rows with fewer than 1/cap copies (was nan, now uncapped, as seqrule). It affects the
  monitor's and finalpick's bootstrap only in degenerate rows (none in any slice so far).
- The checks in the table were run on a copy of the ds8 tree (pipeline/, search/, eval/docstat/) with the two patches applied in
  that order (`git apply -p1`); the same commands are in the table's rows.
