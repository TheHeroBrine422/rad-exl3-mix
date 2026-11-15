# Switching the live search to the s190e / s190f pair (#190-03, w179) — procedure for the lead

w179 did not touch the live search (driver, registry, config). Nothing below has been run. It follows `SWITCH-190cd.md` (w176, the
s190c / s190d switch) with the names changed. Data and checks: `data/strata190ef/README.md`. Rule: RULES-cooldown amendments
2026-11-15 4:25 PM CST (pair, sources, fill) and 4:40 PM CST (Latin reuse for s190e).

## What is ready
| file | what | md5 at staging |
|---|---|---|
| `/workspace/rad-local/workers/w179/registry-staged.json` | the live registry snapshot (`lead-srch/run/registry.json`, md5 `32b691db…`, 11-15 ~4:30 PM CST) + `s190e` (staged, search) + `s190f` (staged, `"role": "confirm"`); nothing else changed | `6a6153e3…` |
| `/workspace/rad-local/workers/w179/driver-staged.json` | the live `driver.json` (`2c384964…`) with `confirm.pairs` = `[[s186a,s186b],[s186c,s186d],[s190a,s190b],[s190c,s190d],[s190e,s190f]]`; nothing else changed | `7e8f7133…` |

Data: `/data/q27/w179/slices/golden-s190{e,f}{,-c,-x}` on ds8 and box2 at the same path (36 files, md5 identical on both:
`data/strata190ef/md5-golden-s190ef.txt`). The ledger has the 164 new calib rows (`eval/pools/ledger.jsonl`, role `select`, line v3,
2026-11-15; 28,105 → 28,269 lines, old prefix byte-identical).

Shape (from `data/strata190ef/README.md`): s190e (search) = 104 calib c docs + 237 xcorp pieces, 341 docs in the mix; s190f (confirm) =
60 calib c docs + 313 pieces, 373 docs in the mix. Both c parts have 4 non-Latin scripts; both x parts have 7–8.

## Steps
1. **Staleness check (the driver rewrites the live registry on every decision).** `md5sum …/lead-srch/run/registry.json`. If it is
   not the base above, re-stage on the current copy (the staged files are a function of the copy; the pair is not in the live file):
   ```
   python3 search/v3/run190/stage190ef.py LIVE_REG_COPY C_KLD C_DIR D_KLD D_DIR OUT_REG LIVE_DRIVER_COPY OUT_DRIVER S190B_PENDING
   ```
   with `C_KLD` = `/data/q27/w179/slices/golden-s190e/kld.json`, `C_DIR` = `/data/q27/w179/slices/golden-s190e`, `D_KLD` / `D_DIR` the same
   for `golden-s190f`, and `S190B_PENDING` = `/data/q27/w179/in/s190b-pending.json` (ds8). It refuses if a pair document sits in a slice
   it may not (live s190c / s190d, t183, t186n, final750, s171, s173f, verify). Check that the pending file still says 0 (below).
2. **Confirm s190b has no pending confirmations at the switch.** `data/strata190ef/pending190b.json` says 0 (checked 11-15 ~4:20 PM CST:
   90 tentatives on s190a, 90 confirms on s190b, no unresolved candidate submits). Repeat the count on the live `drv/state.jsonl` if
   the switch is later than that: tentatives with slice `s190a` without a `confirm` event with the same tag. If there is one, s190b's
   documents must go out of the pair: the draw must be rerun (`draw190ef.py` reads the pending file), so do not switch first.
3. **Stop the driver** between decisions (`status.txt` quiet, outstanding evals drained or accepted as lost).
   `pgrep -f "[d]river.py /workspace/rad-local/workers/lead-srch"`.
4. **Wait for the PID to exit** (`kill -0 <pid>` until it fails).
5. **Merge the s190e / s190f keys** into the live registry (never copy the whole file over it: the driver writes it on every decision):
   ```python
   import json, os
   LIVE = "/workspace/rad-local/workers/lead-srch/run/registry.json"
   STAGED = "/workspace/rad-local/workers/w179/registry-staged.json"
   live = json.load(open(LIVE)); staged = json.load(open(STAGED))
   for k in ("s190e", "s190f"):
       assert k not in live["slices"]; live["slices"][k] = staged["slices"][k]
   json.dump(live, open(LIVE + ".part", "w"), indent=1); os.replace(LIVE + ".part", LIVE)
   ```
   Keep `doc_decisions` as in `live`. Both slices are `staged`: the driver rotates only to `ready` search slices.
6. **Config.** Back up (`cp driver.json driver.json.pre190ef`). The staged config differs from the base only in `confirm.pairs`; if the
   live `driver.json` still equals the base (`2c384964…`), copy `driver-staged.json` over it; otherwise append `["s190e","s190f"]` to
   `confirm.pairs` by hand.
7. **Activate s190e** (the lead's decision; RULES-cooldown 2'': rotation as soon as it is built and checked): set `"status": "ready"` on
   `s190e` in the live registry (the same way as step 5). The driver rotates to the first ready search slice when the active slice
   reaches its `rotate_after`. s190f (role confirm) is never rotated to; it confirms s190e's tentatives (RULES-confirm). s190d / s190b keep
   confirming their pending tentatives (s190c / s190a).
8. **Start the driver** with the same command as before, from `/workspace/rad-exl3-mix` (background, `< /dev/null`, log to a file):
   `python3 search/v3/driver.py /workspace/rad-local/workers/lead-srch/run/driver.json`.
9. **Verify.** `driver.log`: no CONF error (each pair's confirmation slice has `role: confirm`, and no search slice has one); after the
   rotation `status.txt` shows s190e; the first s190e eval lines carry the slice name; the pair s190e → s190f is in the config.

## Rollback
Stop the driver, restore `driver.json.pre190ef`, remove the `s190e` / `s190f` keys from the live registry (unused without the pair, unless
s190e was already rotated to: then rotate back to s190c first), start the driver.

## Notes for the lead
- **Latin (4:25 + 4:40 PM rules):** the calib Latin pool was short (53 fresh documents, 65,290 gen, against 2× the s186c / s186d c parts
  of 85,736 / 93,668). s190f (confirm) took all 53 fresh documents first (1.39× of its target). s190e (search) took 95 documents from the
  reuse pool (retired search slices and V2-era fit documents, least used first by V3 decisions), 2.10× of its target. Each slice's
  shortfall or overshoot is in its x part, so the slice total L gen is 2× the matching s186 slice's c + x (s190e 373,214 vs 369,940;
  s190f 420,458 vs 420,438). The c parts are therefore smaller than 2× the s186 c parts (positions: s190e c 110k against 120k for 2×
  s186c's c part; s190f c 86k against 127k for 2× s186d's).
- **N:** each c part has 9 / 7 calib N documents (7,255 / 7,581 N gen, 4 scripts each, max script 45 % / 28 %); each x part has 15 / 19
  reused N pieces (26,509 / 23,778 N gen; 7 / 8 scripts; max script 26 % / 34 %). The N share of the c parts is higher than s190c/d's
  (7.5 % / 10.4 % against 6 % / 7 %), because the calib Latin is short; the whole mix is 8.3 % / 6.9 % (s190c/d 9 % / 9 %).
- **N source (my reading):** the 4:25 text ("N reused from retired search / confirm / reselect slices, now also s190a and s190b") is
  applied to the x side as well as the calib side. The search-only reading (retired search slices only) leaves 60 xcorp N pieces
  (64,437 gen, 8 scripts) against the 50,240 gen two x parts need; both readings can be drawn, the wide one is built. Choose and say.
- **Matching:** s190e ↔ s186c, s190f ↔ s186d (the second s186 pair), not s186a / b as for s190c / d. Redirect if the old match is wanted.
- **s190b documents are used** (N and Latin reuse) because s190b has no pending confirmation (step 2).
- **Dates:** the 4:40 PM amendment text is dated later than its commit (`eef5c24f`, 4:23 PM CST by the commit clock). I took the text as
  the rule.
- The live registry and driver were only read. Their md5 at the snapshot are in step 1.
