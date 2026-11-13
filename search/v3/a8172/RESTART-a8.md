# Switching the V3 search to A8 containers (w132, #172) — prepared, NOT applied

**11-13: not executed.** The user approved the switch at 9:00 AM CST, then paused the search at 10:11 AM CST before any
rotation (lead note 10:15). Steps 1–2 below are done: hostd staged on ds8, patch applied in the repo. Step 3 was
undone: `driver.json` is back without `act`, and no `act` event was ever written. Use this plan if a search runs again
on the same driver.

The search (#171) packs and mixes without `act` lines, so every container it scores runs the f16-activation prefill path.
V3 ships int8-activation (`[knap].modes = "a8"`, decision 11-09). The patch makes every eval container an A8 container:
radmix `act <w> a8g128` lines on all 336 trunk weights. These lines are metadata only: radmix writes the same bytes
(measured: 2.5 / 3.5 / 5.0 / 6.0 incumbents, same file size with and without them).

The lead or the user decides whether to apply it; the numbers are in `README.md`.

## What the patch does (`hostd-driver-a8.patch`, against 98c231d)

- **`pipeline/rxrun/hostd.py`:** an eval request's `mix` may carry `"act": MODE` (applied to every trunk weight) or
  `{weight: MODE}`. hostd appends `act <w> <mode>` lines to the radmix mix file. Modes other than f16 / a8t / a8g<N>
  are refused before radmix runs. A request without `"act"` writes byte-identical mix files, so the patch can be staged
  on ds8 while the search runs (`test_hostd_act.py` "none").
- **`search/v3/driver.py`:**
  - CONF `"act": "a8g128"` sets the mode. Every eval line then carries it, and every submit event records `"act"`.
  - The mode changes only at a slice rotation (state event `act`). Until then, a restart with the new CONF keeps
    submitting f16. At the rotation, every incumbent is rescored on the new slice (whole slice, as at every
    rotation), now in A8. Candidates are therefore only ever paired with an incumbent scored in the same mode.
    Results still in flight on the old slice are stale (fit rows only), as at every rotation.
  - Monitor: at the switch, every incumbent is read once on ver3-8k in A8 (`act_rescore`; CONF
    `act_monitor_rescore: false` turns this off). Improvements and alerts compare reads within one mode. A
    `convplot3 --events` marker "act a8g128" is written at the switch.
  - The status line shows the mode in force. Submits and monitor reads carry `"act"` only when not f16 (convplot4
    splits its series on it; the switch event carries `act` too).
- **Fit: no code change.** Rows are labelled per slice, so a new slice's rows are all A8. The per-label offsets
  absorb a uniform shift, but the A8 shift is **not** uniform: ×0.99–1.01 at 2.5 / 3.5 and ×1.05–1.07 at 5.0 / 6.0
  (README). The f16 labels (fit129, s171a/b) therefore overstate the high-bpw gains by ~0.05 log. Recommended:
  rescore w129's 7 uniform rows at K ≥ 4.5 in A8 on fit129 and replace them in prior_obs (~0.6 card-h); the other 83
  rows are below K4.5, where A8 ≈ f16. The f16 search rows age out as A8 slices accumulate.

## Tests (all on the patched tree)

| test | what | result |
|---|---|---|
| `test_hostd_act.py PIPELINE` | mix file without act (unchanged), "a8g128" (336 lines), per-weight dict, bad mode refused | 4/4 PASS |
| `../test_hostd_es.py PIPELINE` | the early-stop checks on the patched hostd | ALL PASS |
| `test_driver_act.py V3 WORK` | stand-in with an A8 executor (KL × a bpw-dependent factor): CONF act + restart mid-slice → no act until the rotation; one act event at a rotation; all later evals A8; incumbents + monitor rescored in A8; 0 decisions across modes; monitor within mode; climbs; replay | 7/7 PASS |
| `../standin.py WORK` | the stock stand-in (no act: the f16 path) | 7/7 PASS |

## Restart plan: "switch at the next slice rotation, rescore incumbents on A8"

The switch needs no exact timing: the driver holds the new mode until the rotation itself.

1. **Stage hostd on ds8 (safe while the search runs).** Copy the patched `pipeline/rxrun/hostd.py` into
   `/data/q27/stage/code/pipeline/rxrun/`; keep the old one as `hostd.py.bak-w132`. Every new eval unit is its own
   process, so new units pick it up. Requests without `act` behave exactly as before.
   `env PYTHONPATH=/data/q27/stage/code/pipeline /data/q27/venv/bin/python -m rxrun.hostd /data/q27/multi/host.json --stamp`
   now gives a new tree stamp. The coordinator checks only the settings stamp, which is unchanged.
2. **Apply the patch in the repo and commit it.**
   `git apply search/v3/a8172/hostd-driver-a8.patch` (paths `pipeline/...`, `search/...`; `-p1`).
   The running driver has already loaded its code, so this does not touch it.
3. **Set the mode and restart the driver.** Add `"act": "a8g128"` to `workers/w131/run/driver.json` (and optionally
   `"act_monitor_rescore": true`, the default). Then `touch $W/drv/STOP`, wait for "STOP file: exit" in driver.log,
   `rm $W/drv/STOP` and start it again with the README's command. The coordinator and guard171 keep running.
   Outstanding evals stay queued, in f16, on the current slice.
4. **At the next rotation** (s171b → s171c; `rotate_after` 25 decisions a size), driver.log shows
   `ACT MODE f16 -> a8g128 from slice s171c`. The ~13 incumbents are scored on s171c in A8, plus ~13 A8 monitor reads
   on ver3-8k. That is ~13 × 5.5 min + 13 × 4 min ≈ 2 card-h, ~20 min of wall time on ds8's 7 eval slots. Candidates
   resume as each size's incumbent lands.
5. **Check the first A8 units.**
   - `grep -c '^act ' /data/q27/mixes/<tag>.mix` = 336.
   - The radiance log has `exl3i_actq` in "host-site execution" and 1394 declared ops (f16: 1186).
   - The container bytes equal the menu bytes as before (no WARN on bytes in driver.log).
6. **Monitor curves.** Read the A8 monitor reads as a new curve that starts at the switch (the events marker).
   Compare with V2 as before: the V2 refs stay the shipped f16 V2 containers.
7. **Rollback.** Remove `"act"` from driver.json and restart the driver: f16 again from the next rotation. The staged
   hostd can stay.

Not needed: repacking (act is mix metadata, so the level packs are unchanged), new goldens, or a refit before the switch.
