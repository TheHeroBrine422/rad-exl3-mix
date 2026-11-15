# Switching the live search to the s190c / s190d pair (#190-02, w176) — procedure for the lead

w176 did not touch the live search (driver, registry, config). Nothing below has been run. It follows `SWITCH-190.md` (the s190a / s190b switch,
done) with the names changed. Data and checks: `data/strata190cd/README.md`.

## What is ready
| file | what | md5 at staging |
|---|---|---|
| `/workspace/rad-local/workers/w176/registry-staged.json` | the live registry (`lead-srch/run/registry.json`, md5 `fa61256a…`, 11-15 ~11:50 AM CST) + `s190c` (staged, search) + `s190d` (staged, `"role": "confirm"`); nothing else changed | `78a25b97…` |
| `/workspace/rad-local/workers/w176/driver-staged.json` | the live `driver.json` (`f01e3797…`) with `confirm.pairs` = `[[s186a,s186b],[s186c,s186d],[s190a,s190b],[s190c,s190d]]`; nothing else changed | `b096bfec…` |

Data: `/data/q27/w176/slices/golden-s190{c,d}` (+ `-c`, `-x`), on ds8 and box2 (36 files, md5 identical: `data/strata190cd/md5-golden-s190cd.txt`).
The ledger already has the 175 calib rows (`eval/pools/ledger.jsonl`, role `select`).

## Steps
1. **Staleness check.** `md5sum …/lead-srch/run/registry.json`. If it is not the staged base, re-run
   `search/v3/run190/stage190cd.py` on the current copy (it refuses if a pair document sits in another slice, other than an N
   document in a retired slice). Same for `driver.json` (the staged config differs from the live one only in `confirm.pairs`).
2. **Stop the driver** between decisions (`status.txt` quiet, outstanding evals drained or accepted as lost). PID:
   `pgrep -f "[d]river.py /workspace/rad-local/workers/lead-srch"`.
3. **Wait for the PID to exit** (`kill -0 <pid>` until it fails).
4. **Merge the s190c / s190d keys** into the live registry (never copy the whole file over it: the driver writes it on every decision):
   ```python
   import json, os
   LIVE = "/workspace/rad-local/workers/lead-srch/run/registry.json"
   STAGED = "/workspace/rad-local/workers/w176/registry-staged.json"
   live = json.load(open(LIVE)); staged = json.load(open(STAGED))
   for k in ("s190c", "s190d"):
       assert k not in live["slices"]; live["slices"][k] = staged["slices"][k]
   json.dump(live, open(LIVE + ".part", "w"), indent=1); os.replace(LIVE + ".part", LIVE)
   ```
   Keep `doc_decisions` as in `live`. Both slices are `staged`: the driver rotates only to `ready` search slices.
5. **Config.** Back up (`cp driver.json driver.json.pre190cd`), then either copy `driver-staged.json` over it (if `driver.json` still
   equals `f01e3797…` or differs only in keys the staged copy has) or append `["s190c","s190d"]` to `confirm.pairs` by hand.
6. **Activate s190c** (the lead's decision, RULES-cooldown 2'': rotation to s190c as soon as it is built and checked): set
   `"status": "ready"` on `s190c` in the live registry (the same way as step 4). The driver rotates to the first ready search slice when
   the active slice reaches its `rotate_after` (200 decisions; s190a is already past it, so the rotation is at the next decision point).
   s190d (role confirm) is never rotated to; it confirms s190c's tentatives (RULES-confirm). s190b keeps confirming s190a's pending tentatives.
7. **Start the driver** with the same command as before, from `/workspace/rad-exl3-mix` (background, `< /dev/null`, log to a file):
   `python3 search/v3/driver.py /workspace/rad-local/workers/lead-srch/run/driver.json`.
8. **Verify.** `driver.log`: no CONF error (each pair's confirmation slice has `role: confirm`, and no search slice has one); after the rotation
   `status.txt` shows s190c; the first s190c eval lines carry the slice name; the pair s190c → s190d is in the config.

## Rollback
Stop the driver, restore `driver.json.pre190cd`, remove the `s190c` / `s190d` keys from the live registry (unused without the pair, unless s190c was
already rotated to: then rotate back to s186c / s190a first), start the driver.

## Notes for the lead
- **Amendment premise (checked in the driver log):** the registry still marks s186b / d / e / f as `staged`, and the config still lists
  `[s186a,s186b]` and `[s186c,s186d]`. The last confirmation on s186b was 11-14 21:51 CST and on s186d 11-15 00:20 CST; no tentative waits on either,
  so the amendment's "pairs finished" holds in practice. The N documents of s190c / s190d come from those slices, as the amendment allows.
- Five N documents per part; s190d's 4th script (CYRILLIC) carries 709 N gen; s190c's GREEK is 243 gen (small, passes the ≥ 4 scripts rule as written).
- A later cooldown step needs a new N-rich xcorp batch first (RULES-cooldown amendment 2').
