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
| `vram/qwen3.8-27b-exl3-v3-1.50bpw.rad` | 7.10 | 1.501 | 0.227 | 83.2% |
| `vram/qwen3.8-27b-exl3-v3-1.75bpw.rad` | 7.88 | 1.751 | 0.129 | 87.0% |
| `vram/qwen3.8-27b-exl3-v3-2.00bpw.rad` | 8.67 | 2.001 | 0.0932 | 89.5% |
| `vram/qwen3.8-27b-exl3-v3-2.50bpw.rad` | 10.23 | 2.501 | 0.0417 | 92.9% |
| `vram/qwen3.8-27b-exl3-v3-2.75bpw.rad` | 11.00 | 2.746 | 0.0310 | 93.9% |
| `vram/qwen3.8-27b-exl3-v3-3.00bpw.rad` | 11.80 | 3.001 | 0.0215 | 94.9% |
| `vram/qwen3.8-27b-exl3-v3-3.25bpw.rad` | 12.58 | 3.251 | 0.0151 | 95.8% |
| `vram/qwen3.8-27b-exl3-v3-3.50bpw.rad` | 13.36 | 3.501 | 0.0115 | 96.4% |
| `vram/qwen3.8-27b-exl3-v3-3.75bpw.rad` | 14.15 | 3.751 | 0.00856 | 96.9% |
| `vram/qwen3.8-27b-exl3-v3-4.00bpw.rad` | 14.93 | 4.001 | 0.00640 | 97.3% |
| `vram/qwen3.8-27b-exl3-v3-4.50bpw.rad` | 16.49 | 4.501 | 0.00374 | 98.0% |
| `vram/qwen3.8-27b-exl3-v3-5.00bpw.rad` | 18.06 | 5.001 | 0.00225 | 98.5% |
| `vram/qwen3.8-27b-exl3-v3-5.50bpw.rad` | 19.63 | 5.501 | 0.00143 | 98.8% |
| `vram/qwen3.8-27b-exl3-v3-6.00bpw.rad` | 21.19 | 6.001 | 0.00106 | 99.0% |
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

## Use one file

```
hf download Hero422/qwen3.8-27b-exl3-v3-rad vram/qwen3.8-27b-exl3-v3-4.00bpw.rad --local-dir .
radiance --model ./qwen3.8-27b-exl3-v3-4.00bpw.rad --kv-cache-dtype bf16 --max-model-len 262144
```

The DFlash2 drafter is declared in the container and runs by default (disable with
`--num-speculative-tokens 0`); image prompts work as-is. Files up to ~7 bpw run on one 32 GB card at moderate
context (262k context fits up to ~4.5 bpw; 5.5–7.0 need two cards, `--tp 2`). Use
prefill chunks of ≥ 2048 tokens (`--max-num-batched-tokens`).

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

- **VRAM (GiB)** = trunk + head — what is on the card when the bf16 table is served from host RAM
  (without `--embedding-placement ram` it stays in VRAM, +2.37 GiB). **vs `vram/`** = the saving over the
  byte-identical `vram/` twin: its trellis embedding table (0.23–1.04 GiB). The tower + drafter are excluded
  (as above); the files are 9.24–25.63 GiB on disk.

Serve with the table in host RAM:

```
hf download Hero422/qwen3.8-27b-exl3-v3-rad host-emb/qwen3.8-27b-exl3-v3-4.00bpw.rad --local-dir .
radiance --model ./qwen3.8-27b-exl3-v3-4.00bpw.rad --kv-cache-dtype bf16 --max-model-len 262144 \
    --embedding-placement ram --host-pool-mib 4096
```

### Against IQR-V4

Both scored on our 2.8M-position evaluation slice. Will be replaced with leaderboard numbers when scored.

| size | EXL3-V3 | IQR-V4 | V3 vs IQR-V4 |
|---|---|---|---|
| 2.5 | 0.0501 | 0.0633 | **−21 %** |
| 2.75 | 0.0362 | 0.0445 | **−19 %** |
| 3.0 | 0.0254 | 0.0311 | **−18 %** |
| 3.25 | 0.0180 | 0.0221 | **−19 %** |
| 3.5 | 0.0133 | 0.0162 | **−18 %** |
| 3.75 | 0.00973 | 0.0115 | **−16 %** |
| 4.0 | 0.00725 | 0.00847 | **−14 %** |
| 4.5 | 0.00397 | 0.00482 | **−18 %** |
| 5.0 | 0.00230 | 0.00264 | **−13 %** |
| 5.5 | 0.00141 | 0.00165 | **−15 %** |
| 6.0 | 0.000920 | 0.00117 | **−21 %** |
| 6.5 | 0.000700 | 0.000870 | **−19 %** |
| 7.0 | 0.000610 | 0.000710 | **−14 %** |

### EXL3-V3 vs EXL3-V2

The four published V2 sizes, re-scored on the same 594-document public subset (883,911 scored
positions, gen mask) with each release's own engine (V2: radiance 1.x; V3: 2.1.0, TP1):

| bpw | V2 KLD | V3 KLD | Δ | V2 top-1 | V3 top-1 |
|---|---|---|---|---|---|
| 5.0 | 0.00317 | 0.00253 | **−20 %** | 98.2 % | 98.3 % |
| 5.5 | 0.00213 | 0.00167 | **−22 %** | 98.6 % | 98.7 % |
| 6.0 | 0.00134 | 0.00113 | **−16 %** | 98.9 % | 99.0 % |
| 7.0 | 0.000646 | 0.000805 | +25 % | 99.2 % | 99.2 % |

### Long context

V3 vs V2 at 245k context (the corpus's two long agentic traces, scored whole) and on short docs —
negative means V3 is below V2:

| size | short docs | long traces |
|---|---|---|
| 2.5 |  | −43 % |
| 3.0 |  | −30 % |
| 3.5 |  | −38 % |
| 4.0 |  | −29 % |
| 4.5 |  | −27 % |
| 5.0 | −20 % | −19 % |
| 5.5 | −22 % | −22 % |
| 6.0 | −16 % | −7 % |
| 7.0 | +25 % | +26 % |

The long-trace column tracks the short-doc column at every size — no long-context degradation. V2 was
scored on the public corpus only at 5.0–7.0 (hence the blank short-doc cells); both columns are the same
596-doc golden, V2's published files on the V2-era engine.

## How these quants were made

V3 changes three things versus EXL3-V2:

- **Integer compute.** The GEMMs run on int8 activations instead of f16. Doubles prefill speed.
- **The mad codebook** instead of mul1: reduces dequant overhead and speeds up decode
- **The full model in the file.** Each container carries the bf16 vision tower and the DFlash2 drafter:
  images work out of the box, and decoding runs with the drafter on by default. Neither is counted in the
  size or quality numbers on this card.

### The quantisation method (for the technically curious)

Like EXL3-V2, each size is the result of a per-tensor allocation — with a larger menu and a longer,
better-tested allocation than V2's:

- **Options per tensor:** our own EXL3 quants of every linear at the 49 menu levels (K 0.5–6.0 in 1/8
  steps, then 6.25–7.0 in 1/4 steps — V2's menu was integer bits only) on the challenge's published
  calibration Hessians, with a joint SwiGLU scale refit (swj) on the frozen codes for the gate/up pairs
  (27B-verified: −6.5 % KLD at K2.5 over the base fit, two seed pairs). The head and the trellis embedding
  are quantized separately at the levels the mix needs.
- **Allocation:** a per-tensor sensitivity fit (weight error × fitted depth-compounding) priced with
  measured A8 compute costs; an exact knapsack solve gives the mix at each target size.
- **Search:** streaming local search over whole mixes on a held-out pool (thousands of evaluations,
  per-document bootstrap, slice rotation), ending in a two-phase final selection: ~130 candidates per size
  on a fresh xcorp slice, trimmed to finalists, then the survivors — including fresh consensus models — on
  a second 2.8M-position slice drawn like the private corpus. The trim trades a little signal for much
  less selection noise; the pick per size comes from that second slice alone, and the public corpus above
  never entered any decision.
- **One codebook, verified end to end:** after the picks were frozen, every non-mad head / embedding was
  re-tested against its mad replacement on the selection slices; the all-mad mixes were adopted (small cost
  at 3.25/3.75, small gain at 1.5–2.75) and the other codebooks removed from the release plugin.

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) for the model (Apache-2.0).
- [turboderp / exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format and the quantizer core
  this work is built on.
- [StillDeadcode](https://huggingface.co/StillDeadcode) for the challenge and the calibration Hessians.
- z-lab for the DFlash2 drafter.
