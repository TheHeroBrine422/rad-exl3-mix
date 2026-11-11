# #142 (w108) data: qtf2 sub-1 0-bit-step fusion

Rule `../../SUB1FUSE.md` (a99c06d, committed before data); code `../../sub1fuse/` (README there). Card: dad1's RX 9070 XT
(gfx1201, the R9700's chip; ratios on one card), 2026-11-11 ~8:55-9:25 AM CST, unless noted.

| file | what |
|---|---|
| `dad1-final-cmp-i8.jsonl`, `dad1-final-cmp-mul1.jsonl` | judge (a), final source vs HEAD's kernel: one line per (instance, LUT, batch), determinism, summary |
| `dad1-fc-cmp-nofuse-i8.jsonl` | the `QTF2_NOFUSE` build of the new source vs HEAD's kernel (i8) |
| `dad1-final-speed.jsonl` | tiles/s, ref (HEAD) vs new, i8 at batches 32-256, mul1 at 128 (`speed.py`, best of 3, 1,024 Gaussian tiles) |
| `dad1-step2-split.jsonl` | the step-2 rule's timing split: new vs `QTF2_F0NOGLOB` (no global costs / barriers; output invalid) |
| `gpubox/` | the real-layer check on gpubox card 0 (`gpubox.sh`): `logs/gpubox.log` (verdict, per-tensor / job seconds), `out/wall-ms.txt`, every job's `job.json` / `log.jsonl` (ref = today's qtf2, new = this one), `jobs/jobs.jsonl` |

## Judge (a): PASS

0 differing indices / values in 80 comparisons per codebook (7 KA = 0 instances × LUT 1 / 0 × batch 32 / 128 = 28,
plus 51 KA ≥ 1 instances at batch 128 as a regression), 2,456 tiles each (qtf2check's 1,432: Gaussian 1,024, Laplace 256,
half-integer ties 128, 8 zero, 8 × 3.5, 8 × 12-sigma; plus 512 Gaussian × 0.5 and 512 × 2). The rule text said 1,448 +
1,024 = 2,472 tiles: a counting slip in the rule; the set is the one it defines. Determinism: 4 runs per instance, 0
differing. `QTF2_NOFUSE` = HEAD's kernel. Real-weight tiles: the fallback set (the rule's "else") on dad1; the gpubox run
checks real layers end to end.

## Speed (tiles/s, old → new)

| cb | batch | K0.125 | K0.25 | K0.375 | K0.5 | K0.625 | K0.75 | K0.875 |
|---|---|---|---|---|---|---|---|---|
| i8 | 32 | 6.4k → 38.7k (×6.10) | 6.8k → 35.2k (×5.15) | 7.1k → 32.0k (×4.48) | 7.3k → 30.0k (×4.11) | 8.1k → 28.6k (×3.50) | 8.9k → 27.2k (×3.08) | 9.9k → 26.1k (×2.64) |
| i8 | 64 | 9.1k → 44.8k (×4.92) | 9.4k → 40.8k (×4.35) | 9.7k → 38.1k (×3.93) | 10.1k → 36.1k (×3.58) | 11.2k → 31.9k (×2.85) | 12.0k → 29.5k (×2.45) | 13.3k → 28.7k (×2.17) |
| i8 | **128** | 12.0k → 45.3k (**×3.77**) | 12.6k → 41.2k (×3.26) | 13.3k → 38.4k (×2.89) | 14.1k → 36.5k (**×2.60**) | 14.8k → 31.3k (×2.11) | 15.7k → 28.4k (×1.80) | 16.5k → 28.3k (×1.71) |
| i8 | 256 | 12.1k → 48.3k (×4.01) | 12.7k → 41.2k (×3.25) | 13.3k → 38.4k (×2.88) | 14.1k → 37.0k (×2.63) | 14.8k → 31.0k (×2.09) | 15.8k → 27.4k (×1.74) | 16.6k → 27.5k (×1.65) |
| mul1 | 128 | 11.4k → 48.8k (×4.28) | 11.9k → 43.9k (×3.69) | 12.7k → 38.6k (×3.02) | 13.7k → 36.7k (×2.68) | 14.7k → 30.1k (×2.05) | 15.9k → 28.8k (×1.81) | 17.2k → 28.5k (×1.65) |

Batch 128 is the production launch batch (`get_temp_buffers_frac`). Run-to-run spread of the new kernel at K0.125 / 128:
45-49.5k across runs of the same build (the last runs: 48.9k, 45.3k).

**Step-2 split (rule: > 30% at K0.125 → attempt step 2):** global costs + barriers = 1 − full / NOGLOB = **5.0%** at
K0.125 (10.6 / 14.5 / 16.0 / 11.5 / 16.7 / 15.7% at K0.25-0.875) → step 2 skipped.

## Real layer on the R9700 (gpubox card 0, 11-11 9:38-9:43 AM CST)

`rxq job` with today's settings (524bc8684c7fda53), K0.125 i8, layer 10 (4 group jobs: in_proj_qkv + z, out_proj,
gate + up, down), the same specs and inputs, `RXQ_RXM` = HEAD's tree (code stamp a3d17268c48e3f62) vs the patched tree
(07061095fcb9940e). **All 6 tensor files bit-identical.**

| tensor | ref s | new s | ratio |
|---|---|---|---|
| in_proj_qkv | 22.1 | 8.6 | 2.57 |
| in_proj_z | 12.5 | 4.3 | 2.91 |
| out_proj | 15.3 | 6.0 | 2.55 |
| gate_proj | 37.2 | 13.2 | 2.82 |
| up_proj | 35.8 | 11.7 | 3.06 |
| down_proj | 44.7 | 18.2 | 2.46 |
| **Σ tensor** | **167.6** | **62.0** | **2.70** |
| job wall (4 processes) | 186.2 | 82.1 | 2.27 |

w95's K0.125 L10 (11-09, settings 80b5e6d7): job seconds 34.2 / 16.1 / 76.3 / 43.8 (this run's ref: 34.6 / 16.4 / 73.1 /
44.7). With the trellis at 90% of the ref's tensor time (PROFILE.md §0), the implied R9700 kernel ratio is ×3.33 at
K0.125 (dad1 ×3.77).

**Projection, R9700-calibrated** (dad1's ratios × 0.884): job wall per level 1.48 / 1.56 / 1.63 / 1.69 / 1.87 / 2.00 /
2.01 h, the seven KA = 0 levels **21.8 → 12.2 card-h (−9.6)**.

**Projection (dad1 ratios as measured):** PROFILE.md §0's K0.125 level = 3.26 h Σ tensor wall (90% trellis) /
3.53 h job wall. Trellis at the other sub-1 levels scaled by the old kernel's tile rate, the rest (0.60 h a level)
assumed K-independent: job wall per level 3.53 / 3.38 / 3.24 / 3.11 / 2.97 / 2.84 / 2.73 h → 1.37 / 1.45 / 1.51 / 1.56 /
1.72 / 1.84 / 1.84 h; **the seven KA = 0 levels 21.8 → 11.3 card-h (−10.5)**. K1 runs on the KA = 1 kernel (unchanged).

## Development path (i8, batch 128, K0.125 / K0.5 / K0.875 tiles/s; each step identical where checked)

| version | K0.125 | K0.5 | K0.875 | change |
|---|---|---|---|---|
| HEAD | 12.0k | 14.1k | 16.5k | per-step global costs |
| v10a | 21.9k | 17.2k | 11.6k | registers between branching positions, 65,536 LUT (identity PASS) |
| v10a procedural | 27.8k | 19.4k | 12.1k | procedural 0-bit reads (faster than the LUT once fused) |
| + byte-sum table | 36.2k | 27.6k | 21.9k | 1,024-entry LDS table, (1, 1) step in the fused path |
| + no 32-bit multiplies | 42.1k | 31.6k | 22.7k | uniform terms scalar, static LDS table, weights in LDS |
| lanes 8 states apart | 22.4k | 21.5k | 19.8k | vector global accesses, but 2× LDS bank conflicts (simulated 2.8 vs 1.1) |
| lanes on bits 10-14 | 45.9k | 17.3k | 14.2k | conflict-free table reads, uncoalesced global |
| + swizzled scratch | 38.2k | 36.1k | 29.3k | `qtf2_adr0` layout: vector + coalesced (occupancy 7 at K ≤ 0.375) |
| + per-MASK step kinds | 45.3-49.5k | 36.5-37.0k | 28.3-28.8k | final (identity PASS, both codebooks) |
