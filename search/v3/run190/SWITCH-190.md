# Switching the live search to the s190 pair (#190, w169) — procedure for the lead

w169 did not touch the live search (driver, registry, config). Everything below is what the lead runs, in this order.
Nothing has been restarted, activated or registered live.

## What is ready
| file | what | md5 at staging |
|---|---|---|
| `/workspace/rad-local/workers/w169/registry-staged-190.json` | the live registry (`lead-srch/run/registry.json`) at 11:46 PM CST + `s190a` (staged, search) + `s190b` (staged, `"role": "confirm"`); live slices and `doc_decisions` unchanged | `cb43e78e…` (copy of the live registry `8366d096…`) |
| `/workspace/rad-local/workers/w169/driver-190.json` | the live `driver.json` (`4a87755c…`) with `confirm.pairs` = `[[s186a,s186b],[s186c,s186d],[s190a,s190b]]`; nothing else changed | `55426b03…` |

Data: the slices are on ds8 and box2 at `/data/q27/w169/slices/golden-s190{a,b}` (+ `-c`, `-x`; md5 identical on both: `data/strata190/md5-golden-s190.txt`).
The ledger already has the 207 calib rows (`ledger.jsonl`, role `select`).

## Steps
1. **Staleness check.** `md5sum /workspace/rad-local/workers/lead-srch/run/registry.json`. If it is not `8366d096…`, the live registry changed
   since staging: re-run `search/v3/run190/stage190.py` on the new copy (it refuses if a calib document of s190 is in another slice; the x
   identity was checked against the goldens' `xdoc`; a new slice with x pieces would need the same check).
2. **Stop the driver** when it is between decisions (`status.txt` quiet, outstanding evals drained or accepted as lost). Its PID:
   `pgrep -f "[d]river.py /workspace/rad-local/workers/lead-srch"`.
3. **Wait for the PID to exit** (`kill -0 <pid>` until it fails).
4. **Merge the s190 keys** into the live registry (never copy the whole file over it, the driver writes it on every decision):
   ```python
   import json, os
   LIVE = "/workspace/rad-local/workers/lead-srch/run/registry.json"
   STAGED = "/workspace/rad-local/workers/w169/registry-staged-190.json"
   live = json.load(open(LIVE)); staged = json.load(open(STAGED))
   for k in ("s190a", "s190b"):
       assert k not in live["slices"]; live["slices"][k] = staged["slices"][k]
   json.dump(live, open(LIVE + ".part", "w"), indent=1); os.replace(LIVE + ".part", LIVE)
   ```
   Keep `doc_decisions` as they are in `live`. Both slices are `staged`: the driver rotates only to `ready` search slices.
5. **Roles.** The ledger rows are in. For a future `mkledger.py` rebuild, merge `search/v3/run190/ledger-roles-190.json` into
   `eval/pools/ledger-roles.json` (not done: the lead's call).
6. **Config.** Back up `driver.json` (`cp driver.json driver.json.pre190`), then copy `driver-190.json` over it. Only `confirm.pairs` differs.
7. **Activate s190a** (the lead's decision): set `"status": "ready"` on `s190a` in the live registry (the same way as step 4). The driver rotates
   to the first ready search slice when the active slice (s186c) reaches its rotation (`rotate_after`, 200 decisions), and rescores the
   incumbents on it (RULES-cooldown §3). s190b (role confirm) is never rotated to; it confirms s190a's tentatives (RULES-confirm).
8. **Start the driver** with the same command as before, from `/workspace/rad-exl3-mix`:
   `python3 search/v3/driver.py /workspace/rad-local/workers/lead-srch/run/driver.json` (background, `< /dev/null`, log to a file).
9. **Verify.** `driver.log`: no CONF error (the driver checks that each pair's confirmation slice has `role: confirm` and that no search
   slice has a confirm role); after the rotation `status.txt` shows s190a; the first s190a eval lines carry the slice name; the confirm
   pair s190a → s190b is in the driver's config.

## Rollback
Stop the driver, restore `driver.json.pre190`, remove the `s190*` keys from the live registry (they are unused without the pair), start the driver.

## Notes for the lead
- Doc ids: x ids are `x:<source>:<orig>`. Five s190 x ids (e.g. `x:fineweb-edu:474`) equal strings in s173c–f for different pieces (xdoc
  differs). The x identity is the xdoc; `doc_decisions` for those x ids are shared counts only (no draw reads x counts).
- The c N share is low (6.0 % / 6.7 % of gen): the pair's 15 free non-Latin calib documents hold 12,775 gen positions in all, split between
  the two parts. The x N share is 9.4 % / 10.2 %. The amendment accepts this (it falls to what the supply gives).
- Staging a second pair (or more) needs a new N-rich xcorp batch first (RULES-cooldown amendment 2'); w169 built nothing beyond s190a/b.
