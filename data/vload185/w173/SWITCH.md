# #185 w173 · switch procedure (for the lead; nothing here is applied)

## What the switch does

The search's hostd serves each mix unit from the **hybrid** rule (`hostd-hybrid.patch`): a level with `final_root/LEVEL/READY`
is read from its `final/`, every other level from its pack, as now. A switch needs no level to be ready: the rest of the copy
can finish after it, level by level. A READY level's pack is then no longer read, so the lead can delete it (the user OK'd
deleting ds8 packs after the switch).

## State at this report (live numbers: `rollout.tsv`, `dry/`)

| part | where (ds8) | state |
|---|---|---|
| finals copied and READY (planes IDENTICAL 336/336 against the pack) | `/data/q27/finals-mad/LEVEL/{final,final.md5,READY}` | see `rollout.tsv`; each READY level is checked by md5 at copy and its planes at the rollout |
| hybrid hostd (patch) | `data/vload185/w173/hostd-hybrid.patch` (applies to the live file, md5 84d0bab8…) | tests ALL PASS on the candidate tree (`tests/test_hostd_hybrid.py`, 17 checks) |
| census (hybrid manifests) | `data/vload185/w173/census_hybrid.py` (+ `census_manifest.py`, w162's) | tests ALL PASS (22 checks); real manifest PASS |
| candidate radiance build | `/data/q27/w173/rad10/install/bin/radiance` (+ `home`) | fresh build of fork `64735054` (the vload branch: live `qc-exl3-v3` 120132ab + four #185 commits; no rebase). `format_test` 2271 checks / 65 cases pass |
| candidate host.json | `host.final.candidate.json` (ds8 `host.final.json`) | generated from the live file; NOT applied; `make_candidate_host.py --switch` makes the switch file (below) |
| dry identity run | `dry/v3s183-6-pair-01568/` (summary there) | see the table below |

## Gates

1. **Identity (dry run, card 7 under its lock).** One live mix (`v3s183-6-pair-01568`; live rows on golden-s190a exist on ds8):
   A = the live rows (rad10-e157b); B = the mix path through the candidate build; C = the hybrid manifest (READY levels as
   `final` lines, the rest as pack donors), checked by `census_hybrid.py`, then scored. Rows must be byte-identical:
   A == B (the build change alone keeps the rows) and B == C (the hybrid path keeps the rows).
   **Result: see `dry/v3s183-6-pair-01568/summary.txt`.**
2. **Census (decide).** A hybrid unit runs `census_hybrid.py` before it is scored, and fails closed: the final lines against each
   level's READY marker and `final.md5`; the pack lines by `census_manifest.py` (`census.py --fast`). The live host has no census
   today (`census_cmd` unset on ds8). The candidate sets one, and hostd refuses a `final_root` unit without it. Cost: the first unit
   that touches a level md5s its ~20 GB (2–3 min under load); later units only stat the files. Recommended: keep it.
3. **Hostd tests** (offline, on the candidate tree): `tests/test_hostd_hybrid.py` ALL PASS; `tests/test_census_hybrid.py` ALL PASS.

## Steps (lead)

1. **Backups:** `cp /data/q27/srch183/code/pipeline/rxrun/hostd.py …/hostd.py.pre-185`; `cp /data/q27/srch183/host.json host.json.pre-185`.
2. **Patch the live tree (once; the live file must still be md5 84d0bab8…):**
   `cd /data/q27/srch183/code && patch -p1 --dry-run < hostd-hybrid.patch && patch -p1 < hostd-hybrid.patch`
   (`hostd-hybrid.patch` is the complete change against the live file: it replaces w162's `hostd-final.patch`; do not apply both.)
3. **The switch file:** `python3 data/vload185/w173/make_candidate_host.py --switch LIVE_host.json OUT_host.json --tree /data/q27/srch183/code`.
   It keeps the live `root`, `eval.mixes`, `eval.scores` and `tree`; it changes `eval.final_root` (`/data/q27/finals-mad`),
   `final_codebook`, `census_virtual_cmd` (census_hybrid.py), `virtual_bytes_cmd`, and `score_cmd` / `score_cmd_progress` (the
   candidate build). Check the diff against the live file: only those six `eval` keys. The census and the candidate build are in
   `/data/q27/w173/` (`tools/`, `code/store/census.py`, `rad10/`): keep that folder while the switch is in use.
4. **Restart the search's hostd** with the patched code and the switch file at a clean point (your usual restart).
5. **First units:** each unit's manifest `<tag>.mix` is `base BASE`, then one line per trunk weight (`final` for READY levels,
   a pack line for the rest), then the donor lines, then the act lines. Check two or three units: the census line says PASS, the
   count of `final` lines equals the trunk lines of the READY levels, and the rows equal the live rows for the same mix (as in the
   dry run). Watch a few more before going on.
6. **Rollback:** put back `hostd.py.pre-185` and `host.json.pre-185`, restart hostd. The packs are still there, so the mix path works.
7. **Pack deletion (after the switch, your call; the user OK'd it):** delete the packs of READY levels (`packs-mad/LEVEL-s6c57a3d1.rad`
   or the unstamped `LEVEL.rad`) only after units on finals have scored. A READY level no longer needs its pack; a non-READY level
   still does (its pack is built from the ladder if missing).
8. **Copy continues:** after the deletions, restart the rollout driver on ds8: `setsid nohup /data/q27/w173/bin/start-rollout.sh
   > /dev/null 2>&1 < /dev/null &` (it resumes: the finished levels are skipped, a half copy resumes). It stops at the free-space
   floor (150 GiB) with exit 3. Each level that reaches READY is served from final/ from the next unit on; no restart of hostd is needed for that.

## Not covered

- Releasing the synthesized planes after the loader's pass (w162's known limit): a hybrid unit's radiance holds the planes of its
  final levels for the life of the process (about 7.6 GB per k2.5-size level set; a mix with many final levels uses more). Eight cards
  in parallel: check ds8's RAM after the first hour.
- The fill of a final level happens twice per unit (`rad-info --bytes` for the size, then the scoring): a few seconds per level set.
- The stat-signature shortcut in the census: a file rewritten in place with its old size and mtime is missed until `--full` (or a
  change of size or mtime) is run. The planes of a level are already checked by the rollout; the census guards later changes.
