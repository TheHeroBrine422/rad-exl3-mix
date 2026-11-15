# #185 w162 · step 3: profile of the exl3i import from a level's final/ (the (B) path)

Date 2026-11-15 (VM clock 00:14–00:20 UTC). Host: ds8, CPU only (no GPU, nothing written). The ds8 load average was 35–65
during the runs (the live search), so the parallel numbers include contention.
Harness: `search/v3/run183/vload/prof_trunk.cpp`, built from the fork's own `plugin/exl3quant/exl3_src.cpp` and
`plugin/exl3import/include/exl3i_encoding.h` (`git archive 120132ab`), g++ 15.2 `-O3 -march=native`.
Input: level `q27v3-635bac02-k2.5-mad`'s `final/` (7.6 GB of trellises, 67 shards, 336 exl3i weights; the stacked gate_up is one
weight of two parts). "plan" = the source's header lookups and `exl3i_encoding_make`; "read" = `pread` of the trellis bytes;
"reorder" = the plugin's per-tile memcpy into the record; "scales" = F16 suh / svh.

## One weight, phase by phase (cold = page cache dropped first; warm = the next two passes)

| weight | trellis | plan | read cold | read warm | reorder | scales |
|---|---|---|---|---|---|---|
| `layers.3.mlp.gate_proj+up_proj` (stacked gate_up, 2 parts) | 56 MB | 0.35 ms | 53.7 ms (1.04 GB/s) | 9–10 ms (5.7–6.3 GB/s) | 12.4 / 34.3 / 19.7 ms | 3.1 ms cold |
| `layers.3.mlp.down_proj` | 28 MB | 0.83 ms | 32.1 ms (0.87 GB/s) | 4.5–4.7 ms (6.0–6.2 GB/s) | 14.0 / 8.4 / 11.0 ms | 0.9 ms cold |
| `layers.3.self_attn.k_proj` | 2 MB | 0.47 ms | 3.4 ms (0.48 GB/s) | 0.26–0.27 ms | 0.25 / 0.12 / 0.12 ms | 0.2 ms cold |

`Source::open` (config + index + every shard header), first call: 186 ms (1.8 ms once the process has it).

## The whole level, 336 weights (`prof_trunk DIR all 8`: each pass drops the level's pages first, i.e. cold)

| threads | wall | summed over threads: plan | read | reorder | scales | read | record written |
|---|---|---|---|---|---|---|---|
| 1 | 13.76 s | 0.27 s | 10.50 s | 2.03 s | 0.32 s | 7.62 GB | 7.60 GB |
| 8 | 2.29 s | 0.23 s | 12.10 s | 3.28 s | 1.01 s | 7.62 GB | 7.60 GB |

## Decision

- The import is a read plus a block transpose of the trellis. A whole model's read + reorder is **2.3 s on 8 threads, 13.8 s on
  one**; both are under the lead's ≤ 15 s bar. The VRAM upload (the loader's 7–8 s for 21.9 GiB) is the same as today.
- (B): the planes come from the trellises, not from a pack. The record planes are the pack's bytes (checked on every weight of
  k2.5, k0.125 and k7.5, see `identity-b-2026-11-15/`).
- Footprint: the 49 levels' `final/` total 451 GiB (box2); the packs on ds8 are 729 GB. Copying the finals to the search host
  (box2 → ds8 direct on 10 GbE: 7.1 GB in 13 s, ~550 MB/s) frees about 280 GB net once the packs are dropped.
