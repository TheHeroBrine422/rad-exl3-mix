# w146 rules: qtf2 eighth steps K5.125–K7.875 (w140's patch) and the 14 blocked R16 head donors (#162-03)

Written 2026-11-13 ~4:35 PM CST, before the patch is applied anywhere and before any quant or score of this work.
Committed alone before launch. Task: `local/workers/instr3/w146.md`. Parents: `v3/yaqa/w140/RULES.md` (amendment 1: the
donor list, recipe, scores, price table) and `v3/yaqa/w122/RULES.md` (#162); nothing there is changed.
Machine: ds8 only (`/data/q27/heads/`, locks `/data/q27/locks/card<N>.lock`, `ram.lock`).

## The patch (no decision)

`v3/yaqa/w140/qtf2-eighths-ka5-7.patch` adds 12 qtf2 template instances (KA 5, 6, 7 × masks 0x8080, 0xa4a4, 0xdada,
0xfefe = n 2, 6, 10, 14 sixteenths) to `v3/e3/exl3frac2/` (instance units a5 / a6f / a7f, `qtf2_table.inc`, `mkqtf2.py`'s
MENU). It touches the run-time-built side extension in this repo, not the exllamav3 fork, so no fork commit and no
`quantize/exl3-patches/` entry (AGENTS.md: run-time builds live in the repo directly). ourq's code stamp does not hash
`v3/e3/exl3frac2/` (its glob is `quantize/exl3frac2/`, absent), so the stamps of new donors equal those a pre-patch run
would get; the identity gate below is what establishes that existing levels are unchanged.

## Identity gate (decides whether the patched build may be used)

ds8, w119's tree (`/data/q27/w119`, the stamp path) with the patch applied in place (originals kept in
`/data/q27/heads/w146/orig/`), JIT rebuild into run122's `TORCH_EXTENSIONS_DIR` (both variants, i8 and mul1). Requant with
`run122.sh worker` (unchanged recipe, same flags) into `q/w146chk-<ref>/K<k>`:
- `R16s003/K4` (i8, integer K4: qtf2 MASK 0; ref built on box2, ds8-reproduced by w122's ident gate),
- `R16s003m1/K6` (mul1, integer K6: qtf2 a6 MASK 0; ref built on ds8 by w140),
- `R16s003/K6.25` (i8, an existing quarter instance in the edited unit a6f; ref built on ds8 by w140).

**PASS** iff for all three `q-head.safetensors` and `wq/lm_head.rowerr.npz` have the same sha256 as the reference
(plus `wq/lm_head.safetensors` where the reference still has it), and the stamp in `stamp.json` equals the
reference's. FAIL: revert the ds8 tree from `orig/`, rebuild, build nothing, report. PASS: no stamp change for
existing levels; the three check dirs are deleted after the log.

## Build (only after PASS; no decision)

`run122.sh worker CARD QUEUE` on the 14 donors w140 marked BLOCKED (their `.claim/BLOCKED` dirs removed first):
i8 `R16s003` K5.125, 5.375, 5.625, 5.875, 6.125, 6.375; mul1 `R16s003m1` K6.125, 6.375, 6.625, 6.875, 7.125, 7.375,
7.625, 7.875. Free ds8 cards only (card locks), MemAvailable guard as in run122.

## Readouts (report; no gate)

- Scores: `run122.sh score CARD fit` and `... search` (score119; only unscored donors, so existing rows are untouched).
- Price table: `rep140.py table` → `v3/yaqa/data/w140/rep/hcurve140.json` regenerated = the present 48 plans with
  **identical values** (checked) + the 14 new keys (`u<K>` i8, `u<K>m` mul1). Report-only flag as w140's: a donor whose
  fit-gen KL is above that of the next-lower K of its codebook.
- Disk: as w140 amendment 1, drop each new donor's `wq/lm_head.safetensors` once both score files exist.

**Runtime class:** none for existing levels (instances only, identical output); the new levels' decode price is w140's
fractional-K head price (~bytes / 525 GB/s), not re-measured here.
