# #185 · w173 · finals to ds8, plane checks, hybrid candidate host, dry identity run

Hybrid rollout: a search unit serves each level from its `final/` once that level is READY on ds8, and from its pack until
then. The switch therefore does not wait for all 49 levels; the lead frees a level's pack after the switch.

Scripts here run on ds8 under `/data/q27/w173/` (`bin/`, `tools/`, `tests/`, `results/`, `planes/`, `dry/`, `rad10/`)
unless marked VM. Nothing here changes the live search: no driver, coordinator, registry, hostd process, host.json or pack
of the search is touched. box2's address and the ds8 key for it are passed in the environment at launch and are not written
to any file.

| file | where | what |
|---|---|---|
| `order.txt`, `make_order.py` | VM | the copy and plane order: K ≥ 6 first (least used in the newest 300 live mixes first), then sub-1, then the middle (least used first). |
| `rollout_level.sh` | ds8 `bin/` | one level: rsync box2 `ladder-src-mad/LEVEL/final/` → `finals-mad/LEVEL/final.part` (resumable; free-space floor checked first); md5 of every file on both ends; rename to `final`; `final_mix.py` (w162's, unchanged) makes the 336-name manifest; `rad-info --weights-dump` of the pack and of the manifest stream into two FIFOs (nothing written to disk); `cmp_dumps.py` (w162's, unchanged). READY: writes `finals-mad/LEVEL/final.md5` (the verified list) and `finals-mad/LEVEL/READY` (`READY`, then the result line). Exit 0 READY, 1 REJECTED (moved to `rejected/`), 2 error (a rerun resumes), 3 at the floor. |
| `rollout.sh` | ds8 `bin/` | the driver over `order.txt`. Stops at the floor (exit 3); halts after 2 levels in a row that are not identical (exit 4); any other error halts (exit 2). A rerun skips levels with a result line. |
| `hostd-hybrid.patch` | VM | the hostd change for the live `pipeline/rxrun/hostd.py` (supersedes w162's `hostd-final.patch`; apply this one only). Checked: `git apply` onto the live file (md5 84d0bab8…) gives the hybrid file. |
| `census_hybrid.py` | VM, ds8 `tools/` | the census of a hybrid manifest (see "Census"). |
| `test_hostd_hybrid.py`, `test_census_hybrid.py` | VM, ds8 `tests/` | offline tests (no GPU). |
| `make_candidate_host.py`, `host.final.candidate.json` | VM | the candidate host.json (generated from the live one; not applied). |
| `dryrun.sh` | ds8 `bin/` | one live mix, three ways (see "Dry run"). |
| `collect.py` | VM | pulls `results/`, `planes/`, `rollout.out`, `dry/` from ds8 into this folder (`rollout.tsv`, `planes/LEVEL.txt`, `rollout/LEVEL.md5`, `rollout.log`, `dry/`). |
| `SWITCH.md` | VM | the switch procedure for the lead, with the gates and the rollback. |

## hostd: the hybrid rule (`hostd-hybrid.patch`)

With `eval.final_root = DIR` (and `eval.census_virtual_cmd` set; without it a unit fails before any pack or mix step):

- a level with `DIR/LEVEL/READY` is served from `DIR/LEVEL/final`: its 336 trunk lines (or however many of the mix's trunk
  weights name it) are `<weight> final DIR/LEVEL/final codebook=mad`. No pack is built or read for it;
- every other level is served from its pack, built as before (`pack_cmd` from the ladder's recipe, or reused), so
  `H.current(LEVEL)` is still required for it;
- READY but no `final/index` is an error (`final/ not on this host ... (READY, so no pack)`), not a silent fallback;
- the manifest (`base`, the trunk lines in trunk order, donor lines, act lines) is served as a virtual manifest, as before.

The rows should not depend on which path a level takes: a READY level's final lines serve the same planes as its pack (the
rollout's check, 336 of 336), and the dry run gives the same rows for one mix with 145 final lines and 191 pack lines (B == C).
A unit scored before a level's READY and one scored after should give the same rows; the first units after a switch check it.

## Census (`census_hybrid.py`)

`census_virtual_cmd` runs on every hybrid manifest before its scoring. It fails closed:

- **final lines:** each level must have `READY` (its result line: VERIFIED, status READY, file count, bytes and digest),
  `final.md5` (the list the rollout verified; its md5 must be the marker's digest), and `final/` must hold exactly those
  files, with that total size and each file's md5. Cost: a level's md5 is computed only when its stat signature (size, mtime,
  inode of each file) differs from the one recorded at its last full pass (`final/../census.stat.json`). The first unit that
  touches a level pays its md5 (~20 GB, about 2–3 min under load); later units only stat. A file rewritten in place with its
  old size and mtime is missed by the stat path; `--full` recomputes everything.
- **pack lines:** `census_manifest.py` (w162's, unchanged) with `census.py --fast` on the manifest with the final lines taken
  out (the final weights keep their act lines). Each trunk weight is either a pack line or a final line, so both parts together
  cover the trunk.
- the mixfile must be exactly the manifest's body after `base` (the candidate is what the mix says).

## Tests (offline, ds8 `/data/q27/venv/bin/python -I`, no GPU)

```
tests/test_hostd_hybrid.py /data/q27/w173/code/pipeline     # ALL PASS (17 checks): hybrid mix, all-pack, no census, missing final, unbuilt, census fail
tests/test_census_hybrid.py                                  # ALL PASS (22 checks): pass / stat cache / md5 / bytes / extra / missing / symlink / manifest
```
Real-data census (ds8, the dry-run manifest, 53 trunk lines on k7 and k7.5): `CENSUS HYBRID PASS: 2 final levels (53 trunk
lines; 136 files: 0 by stat signature, 2 by md5), 285 pack/donor lines (census_manifest rc 0)`.

## Dry run (`dryrun.sh MIX`; card 7 under `card7.lock`, RAM shared under `ram.lock`)

Mix `v3s183-6-pair-01568` (live rows on golden-s190a; chosen because it uses READY levels). In the run: C had 145 trunk lines
from final/ (k6.25 54, k6.5 38, k6.75 42, k7 11) and 191 from packs. Three ways:

- **A** the live rows (rad10-e157b, the search's own file; not re-run);
- **B** the mix path, candidate build: radmix writes the `.rad`, radiance scores it (A == B shows the candidate build keeps the
  rows of the live build);
- **C** the hybrid path, candidate build: `subset_mix.py` makes the manifest with the READY levels' lines as `final` lines and
  the rest as pack donors; `census_hybrid.py` checks it; radiance scores it (B == C shows the final path keeps the rows).

Rows must be byte-identical. Result (10:06 UTC, run 2): `A == B: SAME`, `B == C: SAME`, 6,756,448 bytes each, sha256
`d1275e33…`; `dry/v3s183-6-pair-01568/summary.txt` and the logs beside it. (B and C were scored on card 7 under its lock, at
normal priority as the search's own scoring is; radmix at nice 19.)

## Rerun (as run, ds8)

Build (the candidate, fresh tree from `git archive 64735054`; resumable: a killed build is resumed by the same command after
deleting its outputs newer than the kill):

```
cd /data/q27/w173 && J=8 nice -n 19 bash bin/rad10build.sh /data/q27/w173/rad10
```

Rollout (box2's address and the ds8 key for it come from the environment only):

```
BOX2_HOST=<box2 10GbE address> BOX2_KEY=<ds8 key for box2> setsid nohup nice -n 19 bash /data/q27/w173/bin/rollout.sh \
  >> /data/q27/w173/rollout.out 2>&1 < /dev/null &
```

Dry run: `bash /data/q27/w173/bin/dryrun.sh v3s183-6-pair-01568`.

## Freeze notes (11-15)

- Run 1 (`out/w173-run1.md`): ds8 stopped answering at 08:08 UTC (the build and the copy were cut there, with a zero-byte md5 list
  for k7). The reboot at 09:06 UTC came later. On resume: 251 build outputs newer than the build's start were deleted (ninja
  rebuilds them); the two zero-byte md5 lists were removed; the k7 copy resumed from 18.8 GB of 21.3 GB (rsync, md5 checked).
- `rollout_level.sh` now writes the READY marker and `final.md5` (its k7.5 result was finalized by a rerun: "done before: READY").
