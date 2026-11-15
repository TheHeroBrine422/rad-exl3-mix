# #185 · w162 · radiance serves a search candidate without a mix file

Today every #183 candidate is written as a mix file (radmix, 8–24 GB), censused, loaded, scored, and deleted. Two ways
to serve it instead, both in the fork `radiance-hero`, branch `qc-exl3-v3-vload` (off `120132a`; worktree
`/workspace/wt/w162-radiance`; not pushed; the live branch `qc-exl3-v3` is untouched):

- **(A) packs as donors** (stopgap, commits `719dae65`, `64e4487c`): a manifest (`base BASE` + radmix's mixfile lines) opens
  as the file radmix would write, each weight mapped from its pack in place. Needs the level packs (729 GB on ds8).
- **(B) final/ as the trunk** (the required end state, commits `e29c765b`, `64735054`): a manifest line
  `<weight> final DIR codebook=mad` serves a trunk weight straight from a level's `final/` (ourq's per-tensor safetensors):
  the exl3i planes are filled at open from the trellises. No level pack, no mix file. The non-trunk donors (head, embedding,
  a/b, norms: `base-T4.rad` and the small `.rad` donors) stay as they are.

`radiance --model X.rad` is unchanged; the new input is the manifest (`--model X.mix` or any mixfile behind `base`).

## Status (2026-11-15)

| piece | where | state |
|---|---|---|
| (A) manifest reader | fork `719dae65`, `64e4487c` | KLD rows byte-identical to the mix path on the 4 scored #183 mixes; census equivalence on 3 mixes + negative |
| (B) `final` lines | fork `e29c765b` (`core/format/final_trunk.{h,cpp}`, `radfile.{h,cpp}`) | planes byte-identical on all 336 trunk weights of levels k0.125, k2.5, k7.5; KLD rows byte-identical on 4 mixes with 3 levels from `final/`; full k2.5 model served 3 ways: same rows |
| `rad-info --weights-dump` | fork `64735054` | the per-weight identity check (`cmp_dumps.py`) |
| hostd `eval.final_root` | `hostd-final.patch` (A + B; applies to the live `pipeline/rxrun/hostd.py`) | `test_hostd_final.py` ALL PASS; `test_hostd_virtual.py` ALL PASS; **not applied** (lead) |
| hostd `eval.virtual` (A) | `hostd-virtual.patch` | `test_hostd_virtual.py` ALL PASS; not applied |
| profile: rad-convert (step 1) | `data/vload185/step1-profile.md` | superseded by the (B) profile |
| profile: exl3i import from final/ (step 3) | `data/vload185/b-profile.md` | 2.3 s / model (8 threads), 13.8 s (1 thread) |
| identity runs, timing | `data/vload185/identity-b-2026-11-15/` | logs, JSON reports, sha256 of every rows file (rows stay on ds8) |

## How (B) works (`core/format/radfile.cpp`, `core/format/final_trunk.cpp`)

- The line is read like a donor line plus the form `final DIR [codebook=CB] [rate=K]`. The weight's shape, rank, layer,
  expert and slot come from the base's entry of that name (so the base must have the weight, as for a donor).
- `final_plan` checks the trellises the way the exl3_quant plugin does (I16 `[k/16][n/16][m]`, the `.mul1` marker, no bias,
  F16 suh / svh, the rate and the stamp's codebook) and builds the encoding with the plugin's own exl3i encoding header
  (`plugin/exl3import/include/exl3i_encoding.h`). The entry's options text is `codebook=mad,rate=2.5,src=DIR`, as a pack's.
- The planes are written at the writer's offsets (256-aligned, in a 4 KiB-aligned entry). They live in the image's own
  anonymous pages; `open` fills them with `final_fill` (the plugin's record loop) on min(8, cores) threads. `read_at` /
  `read_direct` copy them; `drop_range` does nothing for them. The file tier still refuses a manifest.
- No file is written. A candidate's (B) cost is the open (fill) plus the loader's pass over the image.

## Rerun (exact commands; ds8, as run)

Build (the rad10 incremental build; `SRC` is the fork tree at the commit, copied as `git archive` / the changed files):

```
# the src tree is a git archive of the fork commit (e29c765b; COMMIT = e29c765b), or the changed files copied in
cd /data/q27/w162/rad10/build && export ROCM_PATH=/opt/rocm-10.1 HIP_PATH=/opt/rocm-10.1 PATH=/opt/rocm-10.1/bin:/data/q27/venv/bin:$PATH
nice -n 19 cmake --build . -j 8 --target rad_core radiance rad-info format_test && nice -n 19 cmake --install .
```

Unit (host, no GPU) and the hostd tests (offline; on a copy of the pipeline with the patch applied):

```
cd /data/q27/w162/rad10/build && LD_LIBRARY_PATH=/data/q27/w162/rad10/install/lib:/opt/rocm-10.1/lib ./tests/format_test
mkdir -p /tmp/chk && cp -r pipeline /tmp/chk/ && cd /tmp/chk && git apply /workspace/rad-exl3-mix/search/v3/run183/vload/hostd-final.patch   # paths are a/pipeline/...
python3 /workspace/rad-exl3-mix/search/v3/run183/vload/test_hostd_final.py /tmp/chk/pipeline     # ALL PASS
python3 /workspace/rad-exl3-mix/search/v3/run183/vload/test_hostd_virtual.py /tmp/chk/pipeline   # ALL PASS (the (A) patch: git apply hostd-virtual.patch)
```

Plane identity of one level (no GPU; copies that level's `final/` from box2 to ds8, dumps the 336 trunk weights of its pack and
of the manifest, compares, deletes the copy):

```
bash search/v3/run183/vload/plane_level.sh OUT q27v3-635bac02-k2.5-mad      # IDENTICAL (identical weights: 336 of 336)
```

KLD identity of the mixes with three levels from `final/` (card 7 under its lock; ~2.5 min per mix; needs the three finals
under `FIN`, copied with `rsync` from box2 `ladder-src-mad/<level>/final/` (k1, k3.75, k7 here)):

```
bash search/v3/run183/vload/identity_final.sh /data/q27/w162/idb      # four "KLD ROWS SAME" lines, then the control
```

Timing of one candidate, three ways (k2.5, all 336 trunk weights; card 7; rows must match):

```
bash search/v3/run183/vload/timing_k25.sh /data/q27/w162/timing      # rows of mix / packs / final: same sha256
```

Per-weight identity of any two containers (the census of a `final` candidate; see below):

```
rad-info --weights-dump A.dump --only NAMES A.rad ; rad-info --weights-dump B.dump --only NAMES B.mix
python3 search/v3/run183/vload/cmp_dumps.py A.dump B.dump
```

`final_mix.py FINAL_DIR BASE HEAD EMB PREFIX [--pack PACK]` writes the 336 names, the all-`final` manifest and (with
`--pack`) the all-pack manifest; `subset_mix.py MIXFILE BASE FINALS_ROOT OUT LEVEL...` turns the trunk lines of some levels
of a search mix into `final` lines.

## Results (2026-11-15, ds8; raw in `data/vload185/identity-b-2026-11-15/`)

**Profile (B), `b-profile.md`:** one stacked gate_up (56 MB of trellis): 53.7 ms read cold, 12 ms reorder. The whole level
(336 weights, 7.6 GB): 13.8 s on one thread, 2.3 s on eight.

**Plane identity (336 trunk weights each, header + every plane sha256 vs the pack):** k0.125 (m = 2, sub-1 bit) **336/336**;
k2.5 **336/336** (1408 planes); k7.5 **336/336**. Other 46 levels: not run (`plane_level.sh`, ~3 min each, box2 copy).

**KLD identity (card 7, golden-s173a, rows byte-identical; sha256 in `subset/rows.sha256`):**

| mix | trunk lines from final/ | rows vs the mix path | max per-row difference |
|---|---|---|---|
| pair158 | 40 (k1, k3.75, k7) | SAME (3,491,568 bytes) | 0 |
| thompson86 | 251 | SAME | 0 |
| block13 | 231 | SAME | 0 |
| pairact (ssm_ab donor + act) | 40 | SAME | 0 |
| control: pair158 `.rad` path through this build | — | SAME as the previous build | 0 |

The exact four mixes need 41 levels' finals at once (456 GB; ds8 had ~513 GB free and the floor is 150 GB), so the test
takes the same four mixes with the trunk lines of three levels (34 GB) from `final/` and the rest from packs. Head, embedding,
the a/b donor and every act line are as in the original mixes.

**Census:** (A): `census.py` on radmix's `.rad` and `census_manifest.py` on the manifest give the same per-weight rows and
mismatch lists on pair158, thompson86, block13 (CENSUS PASS, both) and on a negative expectation (384 mismatches, both).
(B): `census_manifest.py` refuses a `final` line (it would otherwise skip it), so the (B) census is the weights-dump check
above (every header and plane of every trunk weight).

**Timing, full k2.5 candidate, three ways (`identity-b-2026-11-15/timing/`):**

| path | serve setup | loader pass (container read) | score wall | rows |
|---|---|---|---|---|
| mix: radmix `.rad` then radiance | radmix writes 8.06 GiB (not timed in this run; step 1 measured 25 s for an 11.8 GB mix); `rad-info --bytes` 0.03 s | 8.7 s for 8.04 GiB (1.0 GB/s, disk) | 165.5 s | ref (sha256 `e16e7303…`) |
| packs: manifest, pack donors | `rad-info --bytes` 0.96 s (mmap) | 8.9 s for 8.04 GiB (1.0 GB/s, disk) | 157.9 s | SAME (sha256 `e16e7303…`) |
| final: manifest, final/ trunk | `rad-info --bytes` 3.7 s (fill of 336 weights, 8 threads) | 4.4 s for 8.04 GiB (2.0 GB/s, from memory) | 161.7 s | SAME (sha256 `e16e7303…`) |

The score wall is the KLD compute (~150 s on every path); the setup differs by ~25–30 s (radmix) and the loader pass by
~4 s. The (B) numbers are warm-page-cache for the fill (the level was just copied); cold, the first open took 18 s
(`rad-info --bytes`, the same 336 weights: 18.0 s on the first run, 6.2 s on the second, 3.7 s inside the timing run).

## Known limits (in the code, not hidden)

- The synthesized planes stay in the process for its lifetime: about 7.6 GB anonymous memory for a k2.5-size candidate, on
  top of the usual ~2.5 GB. `rad-info --bytes` on it showed a 15 GB maxrss (the 8.6 GB image plus the mapped trellis pages the
  fill read). Releasing the pages after the loader's pass is a small follow-up (madvise after the sweep); not done.
- The file tier refuses a manifest (as in (A)). Every weight of this model is resident, so the tier is never planned.
- `final` lines take `codebook=` (required) and `rate=` (optional, checked). The stamp's codebook is checked when the level has a
  `stamp.json`. The ssm_ab trellis is not a `final/` projection (it stays a donor, as the lead asked).
- `census_manifest.py` refuses `final` lines; `census.py` itself reads the packed container only, so for (B) the census is the
  per-weight dump check.
- The live search's `hostd.py` is not changed. `hostd-final.patch` is the (A) patch plus (B).

## For the lead to decide

1. Roll out: `hostd-final.patch` (contains (A)); a new radiance build for the search (the `rad10` build here is not the search's
   `rad10-e157b`); `eval.final_root` pointing at finals on the search host.
2. Disk: the 49 levels' `final/` total 451 GiB (box2 `ladder-src-mad`); the packs on ds8 are 729 GB. Copying the finals
   over is box2 → ds8 direct on 10 GbE (7.1 GB in 13 s); dropping the packs then frees ~280 GB net. ds8 has 513 GB free now
   (3 finals of 34 GB are in `/data/q27/w162/finals-sub`; delete them after your review, they are rebuilt in ~1 min each).
3. The exact four-mix KLD identity needs all 41 levels' finals at once (456 GB): needs the packs gone or more disk. The subset
   runs above cover the same code path, and the installed build was re-checked on pairact (rows SAME) and k0.125 (336/336).
4. Plane identity of the other 46 levels: `plane_level.sh` is ready (about 3 min each with the box2 copy and the two dumps).
5. Whether to release the synthesized pages after the load (see Known limits).
7. The base is not small: `base-T4.rad` (8.2 GB) is a whole T4 model (394 exl3i entries). Every candidate overrides all 336 trunk
   names, so none of its trunk bytes are served; it still supplies the directory (shapes, layers), the norms and the a/b trellis
   of the layers not overridden. One shared file, not a per-candidate cost. A base with only those (a repack) would make the
   donor set small; not done.
6. census: keep the dump check for (B), or port `census.py`'s checks to `final` lines (`census_manifest.py` would need
   the trunk's expected encodings from the final).
