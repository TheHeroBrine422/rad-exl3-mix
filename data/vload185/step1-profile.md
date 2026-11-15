# #185 w162 · step 1: profile and the choice (A) vs (B)

> Superseded: the choice (A) below was the stopgap. The (B) profile (`b-profile.md`, 2026-11-15) measured the exl3i import from
> final/ at 2.3 s per model (8 threads), under the 15 s bar, so (B) is the end state (lead, 11-15). The mix numbers stand.

Date 2026-11-14. Hosts: ds8 (live search host, cards and RAM shared), box2 (a floor test only). Raw numbers below.

## Mix path, as the search runs it today (ds8, measured with the search running)

| step | what | number | how |
|---|---|---|---|
| radmix | `v3s183-3.5-pair-00158.mix` (674 lines: 338 donor lines, 336 act lines, 0 add / drop) on `base-T4.rad`: writes 11.77 GB | **25.4 s wall** (sys 12.6 s, user 1.0 s) | `/usr/bin/time`, `nice 19`, `ram.lock` shared (ds8 `ram_mode`), output removed after |
| census | `census.py --fast --expect-act a8g128 --expect-emb trellis --max-version 4` on that mix | **0.21 s** | `CENSUS PASS` |
| radiance open | `rad-info` (open, mmap, validate, index) of that mix | **0.08 s** | |
| radiance load | 21.90 GiB of weights to VRAM (one card) | **7–8 s** (lead's LOG 11-14 4:55 PM: 7 s / 8 s) | score logs: `load: 8 GiB of 21.90 GiB read, 1.5 GB/s`, `16 GiB ... 2.0 GB/s` |

So a candidate's mix overhead is radmix's ~25 s plus ~0.2 s of census. The ~100 s the LOG attributes to mix + census
was mostly the exclusive `conv.lock` queue, which `eval.conv_lock = null` already removed. Re-measure the per-eval
split after the change; the saving to expect is about 25 s and 11–12 GB of writes per candidate.

## rad-convert (the level pack), what can be measured

- Existing level packs (ds8 `packs-mad/*.rad.log`, 49 logs with the timing line): weights stage **mean 169 s, range
  127–208 s** (`pack-times-ds8.txt`). Output 6.0–6.7 GiB of weights per level.
- Write floor, box2, O_DIRECT: **6 GiB in 4.48 s (1.4 GB/s)**. So the write is about 3 % of a pack. The rest is read
  plus import / convert CPU (the bf16 kept weights are read too).
- **Not measured:** the read floor and the CPU split of one level. Two blockers, both recorded:
  1. rad-convert needs a visible GPU to declare the arch. With no device the host backend is used and
     `qwen35_exl3` refuses (`rad_arch_declare returned container format error`, box2 run, `ROCR_VISIBLE_DEVICES` empty).
     box2's cards are w160's, so the pack can only run on ds8, on card 7 under its lock.
  2. A level's finals (`ladder-src-mad/<level>/final`, 7.1 GB at K2.5) are on box2 only. The box2 → VM → ds8 path ran at
     1.2 MB/s (box2 read) and 4.7 MB/s (VM → ds8). A full copy would be about 1–2 h. Stopped; nothing left on ds8 or box2.
- `prof_pack.sh` + `floor_read.py` (`search/v3/run183/vload/`) are ready: they time the floor, the pack, and sample the
  pack's CPU and I/O once a second. Run them on ds8 once a level's finals are there.

## Choice: (A), the virtual container

(B) needs the import at load time. The import's cost is the pack's weights stage, 127–208 s measured (mean 169 s), plus
the bf16 reads. That is about 10× the 15 s bar, and the write is only a few seconds of it. Even a much faster
in-loader import would miss the bar, and it would need the exl3 import ported into radiance, bit-identical to the
pack. So (B) is out on the numbers.

(A) reads the packs in place, as radmix does, through an in-memory directory built from a manifest with radmix's own
rules (base, donor, act, add, drop). The bytes served are the bytes the mix would hold. The expected load cost is the
same as today's, with the packs cold in the page cache rather than the freshly written mix. Step 3 measures it.

## Design sketch (A)

- Input: a manifest with the radmix mixfile syntax, first line `base PATH`. Detected by the first bytes. `--model X.mix`
  keeps `--model X.rad` unchanged.
- One address space: base at slot 0, each donor at its own 64 GiB slot (`MAP_FIXED` over a reserved region), the new
  tables (header, directory, planes, encodings, string blob, metadata with the act keys) in an anonymous slot. Every
  `base_ + offset` access in the reader then works unchanged.
- Reads split per slot: `RadFile::read_at`, `drop_range`, `drop_blob_cache`, and the loader's `sweep_container` (one
  DirectFile per source). The file tier of the mover is refused in virtual mode (the search's model is fully resident).
- Identity: the directory and the metadata must equal radmix's output (checked by a dump), and the KLD rows must match
  byte for byte.
