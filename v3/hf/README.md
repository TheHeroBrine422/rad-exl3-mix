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

**Needs radiance 2.1.0 plus the [EXL3-V3 plugins (PR #401)](https://codeberg.org/StillDeadcode/radiance/pulls/401)**
(branch `qc-exl3-v3` of the radiance fork: `exl3i` kernels, `exl3import`, `exl3quant`, `qwen35_exl3` arch).
Until the PR lands, build from the branch; EXL3-V1/V2 remain served by the older plugins.

EXL3 quants of Qwen3.8-27B for the [radiance](https://codeberg.org/StillDeadcode/radiance) inference engine,
where **every tensor gets its own bit width** and the GEMMs run on integer compute. Each file is a single
`.rad` container carrying the whole model — the bf16 vision tower and the DFlash2 speculative drafter are
merged in.

| | |
|---|---|
| Weights | EXL3 trellis codes, one integer "mad" codebook, per-tensor allocation (sensitivity fit + exact knapsack + streaming search — [how](#how-these-quants-were-made)) |
| Speculator | DFlash2, on by default; `--num-speculative-tokens 0` turns it off |
| Vision | the 27-block vision tower, bf16: images in chat requests |
| Context | 262,144 tokens |

Sixteen sizes, 1.5–7.0 bits a weight:

| file | GiB | bpw (LM: trellis + trunk + head) | public KLD | public top-1 |
|---|---|---|---|---|
| `vram/qwen3.8-27b-exl3-v3-1.50bpw.rad` | 7.10 | 1.501 | 0.226589 | 83.2% |
| `vram/qwen3.8-27b-exl3-v3-1.75bpw.rad` | 7.88 | 1.751 | 0.129496 | 87.0% |
| `vram/qwen3.8-27b-exl3-v3-2.00bpw.rad` | 8.67 | 2.001 | 0.093233 | 89.5% |
| `vram/qwen3.8-27b-exl3-v3-2.50bpw.rad` | 10.23 | 2.501 | 0.041709 | 92.9% |
| `vram/qwen3.8-27b-exl3-v3-2.75bpw.rad` | 11.00 | 2.746 | 0.031004 | 93.9% |
| `vram/qwen3.8-27b-exl3-v3-3.00bpw.rad` | 11.80 | 3.001 | 0.021467 | 94.9% |
| `vram/qwen3.8-27b-exl3-v3-3.25bpw.rad` | 12.58 | 3.251 | 0.015052 | 95.8% |
| `vram/qwen3.8-27b-exl3-v3-3.50bpw.rad` | 13.36 | 3.501 | 0.011459 | 96.4% |
| `vram/qwen3.8-27b-exl3-v3-3.75bpw.rad` | 14.15 | 3.751 | 0.008565 | 96.9% |
| `vram/qwen3.8-27b-exl3-v3-4.00bpw.rad` | 14.93 | 4.001 | 0.006398 | 97.3% |
| `vram/qwen3.8-27b-exl3-v3-4.50bpw.rad` | 16.49 | 4.501 | 0.003739 | 98.0% |
| `vram/qwen3.8-27b-exl3-v3-5.00bpw.rad` | 18.06 | 5.001 | 0.002252 | 98.5% |
| `vram/qwen3.8-27b-exl3-v3-5.50bpw.rad` | 19.63 | 5.501 | 0.001432 | 98.8% |
| `vram/qwen3.8-27b-exl3-v3-6.00bpw.rad` | 21.19 | 6.001 | 0.001058 | 99.0% |
| `vram/qwen3.8-27b-exl3-v3-6.50bpw.rad` | 22.75 | 6.501 | 0.000828 | 99.2% |
| `vram/qwen3.8-27b-exl3-v3-7.00bpw.rad` | 24.31 | 7.001 | 0.000780 | 99.2% |

- **bpw (LM: trellis + trunk + head)** = the language model's bytes × 8 / 26,895,998,464 (the challenge's
  parameter count) — the number the board scores, as on the IQR-V4 card. The **vision tower + DFlash2 drafter
  are not in it**: they ride in the file (2.58 GB of the GiB column) and come on top, and are not required
  (the drafter turns off with `--num-speculative-tokens 0`).
  The trunk + head alone is what the `host-emb/` files put on the card (the VRAM column
  below).
- **public KLD / top-1**: KLD vs the bf16 model on the challenge's *public* corpus (596 docs, 1.69M tokens;
  1.17M scored positions — tool outputs and user messages are not scored; bf16 KV, 262k context, scored whole).
  4.50–7.00 were scored at tensor parallelism 2 (two 32 GB cards): at those weights a single card's KV pool
  cannot hold the 245k-token traces beside the weights. TP2 is the same model to within bf16 rounding
  (one ulp per position; ≤ 1e-4 nats absolute on the corpus means) — the scores below are the containers'
  own numbers. These are our own read of a published corpus; the official board scores on the *private*
  corpus will be reported when the entries are submitted. 1.50 / 1.75 are research points, not practical sizes.

### Host-embedding variant (`host-emb/`)

The same final picks with the token table replaced by the checkpoint's **bf16** embedding (248,320 ×
5,120, 2.37 GiB). Every file is verified weight-by-weight **byte-identical to its `vram/` twin except
`token_embd.weight`** — same trunk, same head, a strictly more accurate table (the `vram/` tables are
1.5–7.0-bit trellis; here bf16). Quality is at least the `vram/` file's by construction; the measured
read is the board's.

| file | VRAM (GiB) | vs `vram/` (GiB) |
|---|---|---|
| `host-emb/qwen3.8-27b-exl3-v3-1.50bpw.rad` | 4.48 | −0.23 |
| `host-emb/qwen3.8-27b-exl3-v3-1.75bpw.rad` | 5.24 | −0.25 |
| `host-emb/qwen3.8-27b-exl3-v3-2.00bpw.rad` | 6.02 | −0.25 |
| `host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad` | 7.55 | −0.28 |
| `host-emb/qwen3.8-27b-exl3-v3-2.75bpw.rad` | 8.26 | −0.34 |
| `host-emb/qwen3.8-27b-exl3-v3-3.00bpw.rad` | 9.03 | −0.38 |
| `host-emb/qwen3.8-27b-exl3-v3-3.25bpw.rad` | 9.79 | −0.39 |
| `host-emb/qwen3.8-27b-exl3-v3-3.50bpw.rad` | 10.54 | −0.43 |
| `host-emb/qwen3.8-27b-exl3-v3-3.75bpw.rad` | 11.30 | −0.45 |
| `host-emb/qwen3.8-27b-exl3-v3-4.00bpw.rad` | 12.03 | −0.51 |
| `host-emb/qwen3.8-27b-exl3-v3-4.50bpw.rad` | 13.52 | −0.58 |
| `host-emb/qwen3.8-27b-exl3-v3-5.00bpw.rad` | 14.97 | −0.69 |
| `host-emb/qwen3.8-27b-exl3-v3-5.50bpw.rad` | 16.54 | −0.69 |
| `host-emb/qwen3.8-27b-exl3-v3-6.00bpw.rad` | 18.05 | −0.75 |
| `host-emb/qwen3.8-27b-exl3-v3-6.50bpw.rad` | 19.47 | −0.89 |
| `host-emb/qwen3.8-27b-exl3-v3-7.00bpw.rad` | 20.88 | −1.04 |

- **VRAM (GiB)** = the trunk + head — what is on the card when the bf16 table is served from host RAM
  (`--embedding-placement ram`; it pins in the host pool, `--host-pool-mib` needs to cover the 2.37 GiB;
  without the flag it stays in VRAM, +2.37 GiB on the card). **vs `vram/`** = how much less VRAM this file
  uses than its `vram/` twin, which is byte-identical everywhere else: it is the `vram/` file's trellis
  embedding table (0.23–1.04 GiB). The tower + drafter are in neither number (optional, as above); the
  files themselves are 9.24–25.62 GiB on disk.

### EXL3-V3 vs EXL3-V2, same corpus

The four published V2 sizes, re-scored on the same 594-document public subset (883,911 scored
positions, gen mask) with each release's own engine (V2: radiance 1.x; V3: 2.1.0, TP1):

| bpw | V2 KLD | V3 KLD | Δ | V2 top-1 | V3 top-1 |
|---|---|---|---|---|---|
| 5.0 | 0.00317 | 0.00253 | **−20 %** | 98.2 % | 98.3 % |
| 5.5 | 0.00213 | 0.00167 | **−22 %** | 98.6 % | 98.7 % |
| 6.0 | 0.00134 | 0.00113 | **−16 %** | 98.9 % | 99.0 % |
| 7.0 | 0.000646 | 0.000805 | +25 % | 99.2 % | 99.2 % |


At 5.0–6.0 V3 is 16–22 % below V2 on the same corpus — the V2 board numbers at those sizes
(0.00256 / 0.000914) were scored on different slices and made the comparison look the other way.
At 7.0 V2 keeps a real edge on quality: its 7.0 mix (V2-era per-tensor allocation) sits below V3's
all-mad 7.0. V3's 7.0 is still the release file — its integer-activation GEMMs run substantially
faster at the same size.

### Long context

The corpus's two long agentic traces (`flask` and `tqdm`, 245k tokens each) were scored whole at every
size (262k context, bf16 KV):

| size | long traces (flask / tqdm) | corpus KLD |
|---|---|---|
| 1.50 | 0.172516 / 0.189598 | 0.226589 |
| 1.75 | 0.102521 / 0.111927 | 0.129496 |
| 2.00 | 0.060879 / 0.067555 | 0.093233 |
| 2.50 | 0.027042 / 0.030276 | 0.041709 |
| 2.75 | 0.019552 / 0.021533 | 0.031004 |
| 3.00 | 0.013546 / 0.015073 | 0.021467 |
| 3.25 | 0.009578 / 0.010279 | 0.015052 |
| 3.50 | 0.006973 / 0.007470 | 0.011459 |
| 3.75 | 0.005170 / 0.005429 | 0.008565 |
| 4.00 | 0.003698 / 0.003943 | 0.006398 |
| 4.50 | 0.002106 / 0.002261 | 0.003739 |
| 5.00 | 0.001276 / 0.001360 | 0.002252 |
| 5.50 | 0.000837 / 0.000841 | 0.001432 |
| 6.00 | 0.000585 / 0.000593 | 0.001058 |
| 6.50 | 0.000458 / 0.000457 | 0.000828 |
| 7.00 | 0.000415 / 0.000415 | 0.000780 |

Both traces score **below the full-corpus mean at every shipped size** — the picks (chosen on documents of
≤ 16.4k tokens) do not degrade at 245k context, from 1.5 to 7.0 bpw.

Against EXL3-V2 on the same traces — V2's published files, same full 596-doc golden, the V2-era engine
(the short-doc comparison above used a 594-doc subset that excluded these traces):

| size | V3 long traces | V2 long traces |
|---|---|---|
| 2.5 | 0.027042 / 0.030276 | TBD |
| 3.0 | 0.013546 / 0.015073 | TBD |
| 3.5 | 0.006973 / 0.007470 | TBD |
| 4.0 | 0.003698 / 0.003943 | TBD |
| 4.5 | 0.002106 / 0.002261 | TBD |
| 5.0 | 0.001276 / 0.001360 | 0.001604 / 0.001660 |
| 5.5 | 0.000837 / 0.000841 | 0.001073 / 0.001070 |
| 6.0 | 0.000585 / 0.000593 | 0.000630 / 0.000641 |
| 7.0 | 0.000415 / 0.000415 | 0.000329 / 0.000329 |

V3 is below V2 on both traces from 2.5 to 6.0 — the same direction as the short-doc result, with a narrower
gap; at 7.0 V2's mix keeps its edge in long context as well, as it does in short context.

### Against IQR-V4 on the leaderboard

Against IQR-V4's nearest board entries: iq2r (2.47 bpw) 0.0611 → here at 2.5 bpw **0.0417** (−32 %);
iq4r-s (4.47 bpw) 0.00643 → here at 4.5 bpw **0.00374** (−42 %). Both numbers include the two long traces.
On our own 2.8M-position evaluation slices the gap over the full IQR-V4 range was 14–27 %.

## Use one file

```
hf download Hero422/qwen3.8-27b-exl3-v3-rad vram/qwen3.8-27b-exl3-v3-4.00bpw.rad --local-dir .
radiance --model ./qwen3.8-27b-exl3-v3-4.00bpw.rad --kv-cache-dtype bf16 --max-model-len 262144
```

The DFlash2 drafter is declared in the container and runs by default (disable with
`--num-speculative-tokens 0`); image prompts work as-is. Files up to ~7 bpw run on one 32 GB card at moderate
context (262k context fits up to ~4.5 bpw; 5.5–7.0 need two cards, `--tp 2`). Use
prefill chunks of ≥ 2048 tokens (`--max-num-batched-tokens`).

## How these quants were made

V3 changes three things versus EXL3-V2:

- **Integer compute.** The GEMMs run on int8 activations instead of f16 — the same
  trellis codes, faster prefill. 
- **mad codebook** instead of mul1. This reduces dequant overhead and 
- **The full model in the file.** Each container carries the bf16 vision tower and the DFlash2 drafter: images
  work out of the box, and decoding runs with the drafter on by default. Neither is counted in the size or quality numbers on this card.

Like EXL3-V2, each size is the result of a per-tensor allocation:

- **Options per tensor:** our own EXL3 quants of every linear at the 49 menu levels (K 0.5–6.0 in 1/8
  steps, then 6.25–7.0 in 1/4 steps; the mad codebook, on the challenge's published calibration Hessians; the
  head and the trellis embedding are quantised separately at the levels the mix needs).
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
