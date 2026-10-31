---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- radiance
- exl3
base_model_relation: quantized
---
# Qwen3.8-27B — EXL3-V2: mixed-K EXL3 radiance containers (all sizes in one repo)

EXL3 quants of Qwen3.8-27B for the [radiance](https://codeberg.org/StillDeadcode/radiance) inference engine, where
**every tensor gets its own bit width**. We quantised the model ourselves at every whole bit width from 1 to 8 bits a
weight (EXL3 trellis codes, on the radiance quantization challenge's calibration Hessians). For each target size a
fitted sensitivity model and an exact knapsack solver picked, per tensor, which version to use. The candidate mixes
were then measured, refitted and searched over about 1,450 evaluations (800+ of them whole mixes), and the final choice per size was made on
held-out data.

Compared with turboderp's EXL3 ladder at the same size, KLD is roughly 5–30% lower (most at 2–4 bpw)
(table below).

**Needs radiance 1.13.0 plus the [`qc-exl3-ladder` plugins](https://codeberg.org/StillDeadcode/radiance/pulls/292)**
(`qwen35_exl3` architecture, `exl3_dense` / `exl3r4d` kernels, `embed_tq` gather). The same plugins serve
[EXL3-V1](https://huggingface.co/Hero422/qwen3.8-27b-exl3-turboderp-ladder).

**Text only:** these containers have no vision tower and no speculative drafter (DFlash2); both are planned for
EXL3-V3.

## Files

| file | GiB | real bpw | KLD (held-out calib) | est. leaderboard KLD | top-1 |
|---|---|---|---|---|---|
| `qwen3.8-27b-exl3-v2-1.50bpw.rad` | 4.68 | **1.494** | 0.49221 | ~0.49 | 72.7% |
| `qwen3.8-27b-exl3-v2-1.75bpw.rad` | 5.44 | **1.738** | 0.28036 | ~0.29 | 79.0% |
| `qwen3.8-27b-exl3-v2-2.00bpw.rad` | 6.22 | **1.988** | 0.15471 | ~0.16 | 84.8% |
| `qwen3.8-27b-exl3-v2-2.50bpw.rad` | 7.81 | **2.494** | 0.06250 | ~0.067 | 90.0% |
| `qwen3.8-27b-exl3-v2-3.00bpw.rad` | 9.39 | **2.999** | 0.02726 | ~0.030 | 93.5% |
| `qwen3.8-27b-exl3-v2-3.50bpw.rad` | 10.95 | **3.498** | 0.01517 | ~0.018 | 95.3% |
| `qwen3.8-27b-exl3-v2-4.00bpw.rad` | 12.52 | **3.998** | 0.00712 | ~0.0081 | 96.6% |
| `qwen3.8-27b-exl3-v2-4.50bpw.rad` | 14.09 | **4.500** | 0.00374 | ~0.0043 | 97.6% |
| `qwen3.8-27b-exl3-v2-5.00bpw.rad` | 15.65 | **5.000** | 0.00215 | ~0.0025 | 98.2% |
| `qwen3.8-27b-exl3-v2-5.50bpw.rad` | 17.20 | **5.492** | 0.00122 | ~0.0015 | 98.6% |
| `qwen3.8-27b-exl3-v2-6.00bpw.rad` | 18.73 | **5.981** | 0.00076 | ~0.00093 | 98.9% |
| `qwen3.8-27b-exl3-v2-6.50bpw.rad` | 20.34 | **6.495** | 0.00047 | ~0.00058 | 99.1% |
| `qwen3.8-27b-exl3-v2-7.00bpw.rad` | 21.92 | **7.001** | 0.00035 | ~0.00043 | 99.3% |
| `qwen3.8-27b-exl3-v2-7.50bpw.rad` | 22.91 | **7.316** | 0.00031 | ~0.00038 | 99.3% |
| `qwen3.8-27b-exl3-v2-8.00bpw.rad` | 25.05 | **8.001** | 0.00026 | ~0.00032 | 99.4% |
| `qwen3.8-27b-exl3-v2-9.00bpw.rad` | 28.18 | **9.001** | 0.00021 | ~0.00026 | 99.4% |

- **Real bpw** = file bytes × 8 / 26,895,998,464 (the challenge's parameter count; embedding and head included).
  The 7.50 file came out at 7.32 bpw: no mix closer to 7.5 measured better.
- **KLD (held-out calib):** mean KL divergence vs the bf16 model on a slice of the challenge's calibration corpus
  that the search never used (244 documents, 316k scored positions, documents cut to 8k tokens, bf16 KV cache).
  The winner per size was picked on this slice from 2–6 candidates (1–3 finalists), so the values carry a small selection
  optimism (well under the ±1.5% measurement noise).
- **Est. leaderboard KLD:** the held-out KLD scaled by the ratio leaderboard / held-out of turboderp's EXL3
  containers measured on both (×1.00 at 1.6 bpw rising to ×1.24 at 6 bpw; held at ×1.24 above 6, where we have no
  reference). An estimate until the board scores these.
- **1.50 and 1.75** are research points: at these sizes the model is not practically usable.

### Against IQR-V3 and EXL3-V1 (estimated, at the same size)

| size | EXL3-V2 (est.) | nearest IQR-V3 | EXL3-V1 (turboderp's ladder) |
|---|---|---|---|
| 2.0 | 0.16 | — | 0.154 at 2.12 bpw |
| 2.5 | 0.067 | IQ2R-V3 0.068 at 2.46 | 0.078 at 2.71 |
| 3.0 | 0.030 | IQ3R-S-V3 0.026 at 3.15 | 0.035 at 3.12 |
| 3.5 | 0.018 | IQ3R-V3 0.017 at 3.46 | 0.020 at 3.66 |
| 4.0 | 0.0081 | IQ4R-S-V3 0.0076 at 4.13 | 0.0095 at 4.07 |
| 4.5 | 0.0043 | IQ4R-V3 0.0051 at 4.46 | — |
| 5.0 | 0.0025 | IQ5R-S-V3 0.0025 at 5.06 | 0.0026 at 5.07 |
| 6.0 | 0.00093 | IQ6R-S-V3 0.0011 at 6.05 | 0.0011 at 6.02 |
| 7.0 | 0.00043 | IQ6R-L-V3 0.00072 at 6.94 | — |
| 8.0 | 0.00032 | IQ8R-V3 0.00047 at 8.04 | — |

IQR-V3 and EXL3-V1 values are their leaderboard scores (sizes from the leaderboard's byte counts). Roughly level with IQR-V3 from 2.5 to 3.5 bpw and at 5.0, ahead at
4.5 and from 6.0 bpw up. IQR-V3's GEMMs are faster (int8 compute); EXL3-V2 uses the same f16-activation kernels as
EXL3-V1. Speed is EXL3-V3's focus.

## How the mixes were made

- **Options per tensor:** our own EXL3 quants of every linear at K = 1…8 bits a weight (exllamav3's quantiser, LDLQ
  over the trellis, `mul1` codebook, on the challenge's published calibration Hessians); at ≥ 5 bits also
  turboderp's published tensors where they measured better (see below). Embedding: Lloyd codebook under a 128-point
  Hadamard rotation (as in EXL3-V1), bf16 at 8.00 and 9.00. Head: ours at K2–K8 (turboderp's 5-bit head at 3.50).
- **Whose tensors:** 9 sizes are entirely our own quants (1.50–3.00, 4.00–5.00, 7.50). The others include some of
  turboderp's ≥ 5-bit tensors, chosen by measurement: 3.50 (6 of 337), 5.50 (104), 6.00 (52), 6.50 (11), 7.00 (4),
  8.00 (42), 9.00 (2). 9.00 also keeps 33 tensors in bf16.
- **Sensitivity model:** per-tensor Hessian proxy error (tr(ΔW H ΔWᵀ)) scaled by a fitted per-group, per-depth
  sensitivity with error compounding along depth; fitted to measured KLDs of whole mixes and group swaps.
- **Solver:** exact dynamic programming over byte buckets: the lowest predicted KLD at exactly the target size.
- **Search:** streaming. Every few evaluations the model is refitted and the queue refilled with the mean model's
  mix, Thompson draws from a per-document bootstrap of the fit, local-search mutations of the best measured mix,
  and tilted / random-direction solves for exploration.
- **Data discipline:** the calibration corpus is split into fixed pools. The model is fitted on one pool, a separate
  held-out slice tracked overfitting during the search, and the final pick per size was made on another held-out
  slice that nothing else touched, by a paired per-document bootstrap among 2–6 candidates (ties go to the
  mix built entirely from our own quants). The challenge's public corpus was not used.

## Long context

The search and the held-out picks used documents cut to 8k tokens. To check that the mixes hold up at long context,
each release file was scored on the calibration corpus's two held-out agent traces, **whole** (250k and 253k
tokens, 275k scored positions, bf16 KV), next to turboderp's EXL3-V1 containers on the same traces.

| size | EXL3-V2 | EXL3-V1 nearby | ratio, whole traces | ratio by position: 0–8k / 8–32k / 32–64k / 64–128k / 128k+ |
|---|---|---|---|---|
| 1.50 | 0.405 | — | | |
| 1.75 | 0.252 | — | | |
| 2.00 | 0.131 | 0.0938 at 2.12 bpw | | |
| 2.50 | 0.0511 | 0.0721 at 2.72 | 0.71 | 0.58 / 0.49 / 0.73 / 0.70 / 0.71 |
| 3.00 | 0.0217 | 0.0281 at 3.12 | 0.77 | 0.87 / 0.72 / 0.78 / 0.77 / 0.77 |
| 3.50 | 0.0126 | 0.0177 at 3.67 | 0.71 | 0.62 / 0.64 / 0.72 / 0.69 / 0.72 |
| 4.00 | 0.00604 | 0.00786 at 4.07 | 0.77 | 0.81 / 0.78 / 0.78 / 0.76 / 0.77 |
| 4.50 | 0.00347 | — | | |
| 5.00 | 0.00192* | 0.00199 at 5.07* | 0.96* | |

\* first 160k tokens of each trace (one card's KV limit at that size); both containers on the same positions.

Our advantage holds steady from the first 8k tokens out to 250k: no sign that searching on 8k-token documents
overfit to short context. 6.0 bpw and up were not checked at full length (mostly K6–K8 tensors).

## Use one file

```
hf download Hero422/qwen3.8-27b-exl3-v2-rad qwen3.8-27b-exl3-v2-4.00bpw.rad --local-dir .
radiance --model ./qwen3.8-27b-exl3-v2-4.00bpw.rad --kv-cache-dtype bf16 --max-model-len 262144
```

Files up to ~7.5 bpw run on one 32 GB card at moderate context; 8.00 and 9.00 need two (`--tp 2`). Use prefill
chunks of ≥ 2048 tokens (`--max-num-batched-tokens`).

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) for the model (Apache-2.0).
- [turboderp / exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format and quantiser, and the
  published EXL3 ladder some tensors come from.
- [radiance](https://codeberg.org/StillDeadcode/radiance) (Deadcode) for the engine and the quantization challenge;
  [radiance-exl3](https://github.com/Dyluhn/radiance-exl3) (Dyluhn) and PR #284 (Code000) for the EXL3 kernels our
  plugin builds on.
- Tooling (quantiser driver, mixer, search, evaluation): [rad-exl3-mix](https://github.com/TheHeroBrine422/rad-exl3-mix).
