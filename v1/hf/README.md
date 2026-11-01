---
license: apache-2.0
base_model: turboderp/Qwen3.8-27B-exl3
tags:
- radiance
- exl3
base_model_relation: quantized
---
# Qwen3.8-27B — turboderp's EXL3 ladder as radiance containers (all sizes in one repo)

**New: [EXL3-V2](https://huggingface.co/Hero422/qwen3.8-27b-exl3-v2-rad), our mixed-K quants: 10–31% lower KLD than these containers at the same size, on the same plugins.**

Every size of [turboderp's EXL3 conversion](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) (self-calibrated
"SC" and plain branches), read in **bit for bit** (trellis, codebooks and scales as published, nothing
re-quantised), with the token embedding quantised to a Lloyd codebook under a 128-point Hadamard rotation.
Collected here for scoring in the radiance quantization challenge. 

These are baselines. EXL3-V2 is our own quantization on the challenge Hessians, with each tensor at its own bit width
(16 sizes from 1.5 to 9.0 bpw; about 24% lower KLD than these on average from 2 to 5 bpw, leaderboard scores). Unless
you need turboderp's exact tensors, use EXL3-V2.


**Needs radiance 1.13.0 plus the [`qc-exl3-ladder` plugins](https://codeberg.org/StillDeadcode/radiance/pulls/292)**
(`qwen35_exl3` architecture, `exl3_dense` / `exl3r4d` kernels, `embed_tq` gather).

| file | branch | GiB | real bpw | leaderboard KLD | leaderboard top-1 |
|---|---|---|---|---|---|
| `qwen3.8-27b-exl3-1.40bpw.rad` | self-calibrated (SC) | 4.94 | 1.578 | **0.58332** | 72.0% |
| `qwen3.8-27b-exl3-1.60bpw.rad` | self-calibrated (SC) | 5.51 | 1.759 | **0.39054** | 76.8% |
| `qwen3.8-27b-exl3-1.80bpw.rad` | self-calibrated (SC) | 6.07 | 1.940 | **0.25844** | 81.0% |
| `qwen3.8-27b-exl3-2.00bpw.rad` | self-calibrated (SC) | 6.64 | 2.121 | **0.15402** | 85.8% |
| `qwen3.8-27b-exl3-2.00bpw-plain.rad` | plain | 7.09 | 2.263 | **0.14763** | 86.0% |
| `qwen3.8-27b-exl3-2.20bpw.rad` | self-calibrated (SC) | 7.21 | 2.302 | **0.10247** | 88.1% |
| `qwen3.8-27b-exl3-2.50bpw-plain.rad` | plain | 8.50 | 2.715 | **0.07784** | 90.0% |
| `qwen3.8-27b-exl3-3.00bpw.rad` | self-calibrated (SC) | 9.77 | 3.120 | **0.03488** | 93.1% |
| `qwen3.8-27b-exl3-3.00bpw-plain.rad` | plain | 10.07 | 3.215 | **0.03481** | 93.4% |
| `qwen3.8-27b-exl3-3.50bpw-plain.rad` | plain | 11.48 | 3.666 | **0.02005** | 95.0% |
| `qwen3.8-27b-exl3-4.00bpw.rad` | self-calibrated (SC) | 12.75 | 4.072 | **0.00952** | 96.4% |
| `qwen3.8-27b-exl3-4.00bpw-plain.rad` | plain | 12.90 | 4.119 | **0.00901** | 96.6% |
| `qwen3.8-27b-exl3-5.00bpw.rad` | self-calibrated (SC) | 15.88 | 5.071 | **0.00260** | 98.2% |
| `qwen3.8-27b-exl3-5.00bpw-plain.rad` | plain | 15.88 | 5.071 | **0.00281** | 98.0% |
| `qwen3.8-27b-exl3-6.00bpw.rad` | self-calibrated (SC) | 18.86 | 6.023 | **0.00106** | 98.7% |
| `qwen3.8-27b-exl3-6.00bpw-plain.rad` | plain | 18.86 | 6.023 | **0.00104** | 98.8% |

Real bpw = file bytes × 8 / 26,895,998,464 (embedding and head included). Leaderboard KLD / top-1: the challenge
leaderboard's scores on its private corpus (595 documents, 1.05M scored positions, bf16 KV), as published 2026-11-01.

## What we changed: the token embedding

Everything except `token_embd.weight` is turboderp's tensors, read in bit for bit, including the `output.weight`
head at its branch's precision. turboderp's branches store the token embedding unquantised in bf16
([248320, 5120], 2.54 GB, ≈0.76 bpw of the model). We quantise it with radiance's own `rtn` quantiser, set as
follows:

- **Rotation:** each row is split into 128-column blocks, and each block is rotated by a 128-point Walsh–Hadamard
  transform (`transform=fwht128`). This spreads outliers before quantisation.
- **Codebook:** the rotated values are coded against a 2^B-level Lloyd–Max codebook (`table=` in the recipe). The
  codebook is fitted to the table's own values after rotation, with each group normalised by its max magnitude.
  Every group shares the one codebook.
- **Scales:** each group of 128 values gets one bf16 scale (`group=128 scale=bf16`). The scale is found by search
  for the least squared error (`rule=search`), rather than taken from absmax.
- **Bits:** we use 3 bits up to 2.50 bpw, 4 bits at 3.00–4.00, 5 bits at 5.00 and 6 bits at 6.00. B bits cost
  B + 0.125 bits per value; at 4 bits the table is ≈0.20 bpw of the model, down from 0.76.
- **Decode:** the `embed_tq` gather in the `exl3dense` plugin decodes it at lookup time (codebook level × group
  scale, then the inverse rotation), so the bf16 table never exists in memory.

| embedding | used at | bits per value | table size (GB) | bpw of the model | saved vs bf16 |
|---|---|---|---|---|---|
| bf16 (turboderp's) | — | 16 | 2.54 | 0.756 | — |
| 3-bit Lloyd | 1.40–2.50 | 3.125 | 0.50 | 0.148 | 0.609 bpw |
| 4-bit Lloyd | 3.00–4.00 | 4.125 | 0.66 | 0.195 | 0.561 bpw |
| 5-bit Lloyd | 5.00 | 5.125 | 0.81 | 0.242 | 0.514 bpw |
| 6-bit Lloyd | 6.00 | 6.125 | 0.97 | 0.290 | 0.467 bpw |

The exact codebook for each file is the `table=` line in its recipe under `recipes/`.

**Calibration data.** The codebooks use no text at all: they are fitted to the embedding weights themselves
(every 160th row). The bit widths were chosen by hand from a small sweep of embedding formats and widths at the
2.00 and 4.00 bpw sizes, scored by KLD against the bf16 golden. The sweep ran on a 102k-position subset of the
**challenge's public corpus** (52 documents, one per source, cut to 4096 tokens), which is disjoint from the
leaderboard's private corpus. Measured cost against a bf16 embedding: 6 bits was effectively free, 4 bits cost
about +0.6% KLD at 2.00 and +2% at 4.00, and 2 bits was clearly bad (+14% / +54%). The widths are a size-against-KLD judgement on those
numbers: 3 bits at the smallest sizes, rising to 6 at 6.00.

## Use one file

```
hf download Hero422/qwen3.8-27b-exl3-turboderp-ladder qwen3.8-27b-exl3-4.00bpw.rad --local-dir .
radiance --model ./qwen3.8-27b-exl3-4.00bpw.rad --kv-cache-dtype bf16 --max-model-len 262144
```

Recipes are in `recipes/` (`EXL3_SRC` = the turboderp branch):
`rad-convert Qwen/Qwen3.8-27B --recipe recipes/<name>.recipe --set quantization_config.quant_method=exl3 -o <name>.rad`
