---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- radiance
- exl3
base_model_relation: quantized
datasets:
- StillDeadcode/qwen3.8-27b-quant-challenge
---
# Qwen3.8-27B — EXL3-V3: all-mad trellis quants, integer compute, vision + DFlash2 drafter (16 sizes)

EXL3 quants of Qwen3.8-27B for the [radiance](https://codeberg.org/StillDeadcode/radiance) inference engine,
where **every tensor gets its own bit width**. Like EXL3-V2, each size is the result of a per-tensor allocation:
we quantise the model ourselves at 49 trellis levels (1/8-bit steps, our own integer "mad" codebook), a fitted
sensitivity model + an exact knapsack solver pick the per-tensor mix for a target size, and a streaming search
over whole mixes (thousands of evaluations on held-out slices) finds the best one. V3 changes three things:

- **Integer compute.** The GEMMs run on int8 activations (A8, f16 below the gate) instead of f16 — the same
  trellis codes, faster prefill. (This is the "EXL3-V3" step of the V1 → V2 → V3 series; V1/V2 remain served by
  the older plugins.)
- **One codebook.** Every tensor — trunk, head, embedding — uses the same integer trellis codebook (mad). V2
  mixed three codebooks by tensor class; for this model one of them was enough, and one codebook is one set of
  kernels to maintain.
- **The full model in the file.** Each container carries the bf16 vision tower and the DFlash2 speculative
  drafter: images work out of the box, and decoding runs ~4.5–5.1 tokens per step (measured, greedy) with the
  drafter on by default. Neither is counted in the quality numbers below.

Sixteen sizes, 1.5–7.0 bits a weight (trunk + head):

| file | GiB | bpw (trunk+head) | public KLD | public top-1 | long 245k traces |
|---|---|---|---|---|---|
| `vram/qwen3.8-27b-exl3-v3-1.50bpw.rad` | 7.62 | 1.501 | 0.226589 | 83.2% | 0.173 / 0.190 |
| `vram/qwen3.8-27b-exl3-v3-1.75bpw.rad` | 8.47 | 1.757 | 0.129496 | 87.0% | 0.103 / 0.112 |
| `vram/qwen3.8-27b-exl3-v3-2.00bpw.rad` | 9.31 | 2.020 | 0.093233 | 89.5% | 0.061 / 0.068 |
| `vram/qwen3.8-27b-exl3-v3-2.50bpw.rad` | 10.99 | 2.532 | 0.041709 | 92.9% | 0.027 / 0.030 |
| `vram/qwen3.8-27b-exl3-v3-2.75bpw.rad` | 11.81 | 2.770 | 0.031004 | 93.9% | 0.020 / 0.022 |
| `vram/qwen3.8-27b-exl3-v3-3.00bpw.rad` | 12.67 | 3.026 | 0.021467 | 94.9% | 0.014 / 0.015 |
| `vram/qwen3.8-27b-exl3-v3-3.25bpw.rad` | 13.51 | 3.282 | 0.015052 | 95.8% | 0.010 / 0.010 |
| `vram/qwen3.8-27b-exl3-v3-3.50bpw.rad` | 14.35 | 3.532 | 0.011459 | 96.4% | 0.007 / 0.007 |
| `vram/qwen3.8-27b-exl3-v3-3.75bpw.rad` | 15.19 | 3.788 | 0.008565 | 96.9% | 0.005 / 0.005 |
| `vram/qwen3.8-27b-exl3-v3-4.00bpw.rad` | 16.03 | 4.032 | 0.006398 | 97.3% | 0.004 / 0.004 |
| `vram/qwen3.8-27b-exl3-v3-4.50bpw.rad` | 17.71 | 4.532 | 0.004223 | 97.8% | — |
| `vram/qwen3.8-27b-exl3-v3-5.00bpw.rad` | 19.39 | 5.019 | 0.002527 | 98.3% | — |
| `vram/qwen3.8-27b-exl3-v3-5.50bpw.rad` | 21.07 | 5.544 | 0.001667 | 98.7% | — |
| `vram/qwen3.8-27b-exl3-v3-6.00bpw.rad` | 22.75 | 6.051 | 0.001126 | 99.0% | — |
| `vram/qwen3.8-27b-exl3-v3-6.50bpw.rad` | 24.43 | 6.526 | 0.000972 | 99.1% | — |
| `vram/qwen3.8-27b-exl3-v3-7.00bpw.rad` | 26.12 | 7.001 | 0.000805 | 99.2% | — |

- **bpw (trunk+head)** = trunk + head bytes × 8 / 25,625,844,224 params (the embedding, and the vision tower +
  drafter carried in the file, excluded). The file itself is 0.77 bpw larger than that (the 2.58 GB tower +
  drafter); the challenge's container-bytes formula gives 2.268–7.768 across the files.
- **public KLD / top-1**: KLD vs the bf16 model on the challenge's *public* corpus (596 docs, 1.69M tokens;
  1.17M scored positions — tool outputs and user messages are not scored; bf16 KV, 262k context). These are our
  own read of a published corpus; the official board scores on the *private* corpus will be reported when the
  entries are submitted. 1.50 / 1.75 are research points, not practical sizes.
- **Long 245k traces** (1.50–4.00 only): the corpus's two long agentic traces (245k tokens each), scored whole.
  They read *below* the short-document mean at every size — the mixes, which were picked on documents ≤ 16.4k
  tokens, show no long-context degradation. At 4.50–7.00 the two traces don't fit a single 32 GB card's KV pool
  next to the weights, so those sizes are scored on the other 594 documents; the trace numbers will be added
  once tensor-parallel serving of these containers lands.

**Needs radiance 2.1.0 plus the EXL3-V3 plugins** (branch `qc-exl3-v3` of the radiance fork: `exl3i` kernels,
`exl3import`, `exl3quant`, `qwen35_exl3` arch). The upstream PR is in preparation; until it lands, build from
the branch.

### EXL3-V3 vs EXL3-V2, same corpus

_(placeholder: V2's 5.00 / 5.50 / 6.00 / 7.00 files scored on the same 594-document subset, numbers incoming)_

### Against IQR-V4 on the leaderboard

At the sizes where IQR-V4's board entries are published: 2.5 bpw — IQR iq2r 0.0611, here **0.0417** (−32%, and
this number includes both long traces); 4.5 bpw — IQR iq4r-s 0.00643, here **0.00422** (−34%). On our own
2.8M-position evaluation slices the gap over the full IQR-V4 range was 14–27%.

## Use one file

```
hf download Hero422/qwen3.8-27b-exl3-v3-rad vram/qwen3.8-27b-exl3-v3-4.00bpw.rad --local-dir .
radiance --model ./qwen3.8-27b-exl3-v3-4.00bpw.rad --kv-cache-dtype bf16 --max-model-len 262144
```

The DFlash2 drafter is declared in the container and runs by default (disable with
`--num-speculative-tokens 0`); image prompts work as-is. Files up to ~7 bpw run on one 32 GB card at moderate
context (262k context fits up to ~4.5 bpw; 5.5–7.0 need two cards once tensor parallelism is available). Use
prefill chunks of ≥ 2048 tokens (`--max-num-batched-tokens`).

## How the mixes were made

- **Options per tensor:** our own EXL3 quants of every linear at 49 trellis levels (K 0.125–7, 1/8 steps, the
  mad codebook, on the challenge's published calibration Hessians; the head and the trellis embedding are
  quantised separately at the levels the mix needs).
- **Allocation:** a per-tensor sensitivity fit (weight error × fitted depth-compounding) priced with measured
  A8 compute costs; an exact knapsack solve gives the mix at each target size.
- **Search:** streaming local search over whole mixes on a held-out pool (thousands of evaluations,
  per-document bootstrap, slice rotation), ending in a two-phase final selection: ~130 candidates per size on a
  fresh xcorp slice, then the survivors — including fresh consensus models — on a second 2.8M-position slice
  drawn like the private corpus. The pick per size comes from that second slice alone; the public corpus above
  never entered any decision.
- **One codebook, verified end to end:** after the picks were frozen, every non-mad head / embedding was
  re-tested against its mad replacement on the selection slices; the all-mad mixes were adopted (small cost at
  3.25/3.75, small gain at 1.5–2.75) and the other codebooks removed from the release plugin.

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) for the model (Apache-2.0).
- [turboderp / exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format and the quantizer core
  this work is built on.
- [StillDeadcode](https://huggingface.co/StillDeadcode) for the challenge and the calibration Hessians.
- z-lab for the DFlash2 drafter.
