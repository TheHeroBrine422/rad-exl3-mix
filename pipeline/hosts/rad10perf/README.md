# rad10perf — the staged qc-exl3-v3-perf build and the R16 head end to end (#164-02, w136, 2026-11-13)

w124 put the R16 head into the plugins (radiance-hero `qc-exl3-v3-perf` 8d3f594: exl3i encoding v3 `/r16`, importer
`rows=r16`, the store un-permutes). This directory gives the staged build, the rad-convert recipe line for a real
quantized R16 head, and the checks against the bf16-spliced head the #171 search scores.

## Build

Fork: `qc-exl3-v3-perf` **5c9956c** = 374503f + a merge of `qc-exl3-v3` (81bad51 `rad-convert --no-error`, ca5491d
`--kld-progress`): the two core commits rad10 carried as `core-42960d5..ca5491d.patch` are now in the branch. Its tree
equals w130's `rad10/r374/src` (374503f + that patch).

```
git -C radiance-hero archive --prefix=src/ 5c9956c | gzip > rad-5c9956c.tgz       # on the VM; copy to the host
mkdir -p /data/q27/rad10-perf && tar xzf rad-5c9956c.tgz -C /data/q27/rad10-perf && echo 5c9956c > /data/q27/rad10-perf/src/COMMIT
bash /data/q27/tools/radbuild-perf.sh            # ~8 min on box2; log /data/q27/rad10-perf/logs/
```

`radbuild-perf.sh` = w130's `rad10build.sh` (ROCm 10.1, VOPD-repeatable shadow clang, gfx1201, our five plugins + host
tests; FFmpeg from `/data/q27/rad10/ffmpeg` when present) into `OUT=/data/q27/rad10-perf`, then the pack home. Needs
`/opt/rocm-10.1`, `/data/q27/rad10/{rocm-repeatable,scripts/rad10build.sh,src-1aa18de,ffmpeg}` (both box2 and ds8
have them). Out:

| dir | use |
|---|---|
| `install`, `home` | serving / KLD: the branch head (qwen35_exl3 with the vision tower + DFlash2, R16 heads) |
| `home-pack` | rad-convert: `home` with qwen35_exl3 from 1aa18de (text-only declare). The head's qwen35_exl3 (639ad2f) declares the bf16 vision tower at convert, so its packs carry `model.visual.*` (+0.92 GB at K4, w130); exl3quant / exl3i (R16) are the head's |

**What differs from rad10** (`/data/q27/rad10`, hostd's build): source a3799b5 + patch → 5c9956c (= r374's source:
ae909bf + 639ad2f, the vision gate fix, + PERF.md); with FFmpeg (media requests work; rad10 is FFmpeg off). Against
`rad10/r374`: every plugin `.so` byte-identical; `libr4d.so`, `libquant.so` and the binaries differ only by embedded
source paths (`__FILE__`); libr4d's `.hip_fatbin` identical. `home-pack`'s qwen35_exl3.so = rad10's byte for byte.

## Recipe line for an R16 head

Append to the mix's recipe (the trunk lines unchanged), pack with `home-pack` and `--no-error`:

```
output.weight exl3i src=/data/q27/heads/q/R16s003/K4 codebook=mul1i8 rate=4 rows=r16       # u4 (R16s003 donors: i8)
output.weight exl3i src=/data/q27/heads/q/R16s003/K2.5 codebook=mul1i8 rate=2.5 rows=r16   # u2.5
output.weight exl3i src=/data/q27/heads/q/R16s003m1/K7 codebook=mul1 rows=r16              # u7 (m1: mul1)
```

```
cd /data/q27 && RADIANCE_HOME=/data/q27/rad10-perf/home-pack RADIANCE_HOST_DEVICES=1 /data/q27/rad10-perf/install/bin/rad-convert \
  --no-error models/qwen38-27b-bf16 --recipe MIX.recipe --set quantization_config.quant_method=exl3 -o OUT.rad
```

`codebook` must match the level's `stamp.json` (i8 → mul1i8, mul1 → mul1), `rows=r16` is required (the importer refuses
a stamp with `head_perm` without it) and is checked against the stamp's `/data/q27/w119/perms/R16.npy` (present on box2
and ds8; when unreadable it warns and trusts the recipe). Serve / score the pack with `install` + `home`.
Release packs: pin gpubox's tokenizer config / chat template first (box2 / ds8 hold HF's 11-12 revision; notes/box2.md).

## Scripts

| script | what |
|---|---|
| `radbuild-perf.sh [TGZ]` | the build above (box2 copy: `/data/q27/tools/radbuild-perf.sh`) |
| `r16check.sh` | A = K4 trunk (ladder k4 final.recipe) + R16s003 K4 / K2.5 head by rad-convert; B = A with the search's bf16 donor `hd-u<h>.rad` spliced by radmix; ents, KLD on golden-ver3-8k, kldpair, serve A / B (ask.sh text + 8k prefill, dstep.sh decode step C 1); `ROUNDS="3 4 5"`: timing rounds only |
| `trunc.sh [trunc\|stoch]`, `donortrunc.py` | yardstick: B4t / B4s = A4 with the same head narrowed to bf16 by truncation / stochastic rounding (the search's donors round to nearest) |
| `mkf16ck.py` | checkpoint view with the f16 dequant head (unused for packs: the plugin declares output.weight bf16 only, f16 is refused at declare); also checks that a donor = bf16(f16 dequant) value for value |
| `radents.py A [B]` | container directory: entries, output.weight's stored bytes; with B, which payloads differ |
| `kldpair.py GOLDEN RES A B ...` | pairs of KLD rows: \|Δ mean\|, per-document max \|Δ mean\|, per-position \|Δ\|, top-1 flips |
| `f16ref.sh` | the f16-reference attempt (stopped at declare, see mkf16ck) |

## For the lead: switching the search's packs (or only the release packs)

Measured (data README): the quantized R16 head scores like the bf16 donor the search splices (|Δ mean KLD| 9e-7 at K4,
2e-6 at K2.5; per-document differences ~1e-4 = the noise between any two bf16 roundings of the same head), greedy text
identical, decode −3.1 ms / step, containers −1.91 GB. So the search's scores stand as they are; no rescoring needed.

- **Release packs: use the real head** (recipe line above; `home-pack`, `--no-error`). Bytes = the menu's head price
  − 293,376 B (mkmenu prices trellis + 1 MiB; real is trellis + 3 B × vocab + 2 B × 5120; optional fix there).
- **Search packs: optional, no change needed for correctness.** If wanted (smaller packs, faster scoring, VRAM as
  shipped): radmix splices a quantized head from any container byte for byte (C4: rows identical to the rad-convert
  pack), so build one quantized-head donor per plan (`hq-u<plan>.rad` = rad-convert of the K0.125 level's recipe +
  the plan's head line, ~3.5 GB, like hd129.sh) and point the menu's head `src.path` at them between rounds. hostd's
  current builds already serve R16 heads (rad10 = a3799b5 has 8d3f594; w130's K4H check), so no build switch is
  needed. Incumbents' scores move by ≤ ~1e-5 (below the paired SE); rescore them only if the protocol wants
  like-for-like rows.
- **Build switch (rad10 → rad10-perf) is independent:** same device code and plugins as r374; only matters for media
  requests (FFmpeg) and matching the branch head. ds8 has the tree at the same path (sha256 identical).

Results: `data/rad10perf/w136/` (README there); LOG 2026-11-13 #164-02.
