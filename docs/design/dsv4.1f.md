# DeepSeek-V4.1-Flash-1017 in EXL3-V3: design notes

Status: draft, 2026-11-20 (lead + user discussion). Target: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (the post-tuned release, not the
original V4.1-Flash), served by radiance on 4× R9700 (box2, 128 GB VRAM) with host-RAM expert offload. Engine base and phases:
`rad-local/TODO.md` § DSV4.1-Flash-1017. Calibration (dense + per-expert Hessians, expert-input dumps, routing counts):
HF `Hero422/deepseek-v4.1-flash-1017-quant-calib` (`ds41pipe/WHERE.md`).

## Scope

- **In:** the main model's language path.
- **Out (separate work):** the DSpark / MTP drafter (3 layers, 13.6B MXFP4 experts), the vision tower + aligner, the Engram tables.
- **Engram tables** (2 layers, 196.6B params, 202.8 GB FP8): a hashed n-gram lookup, a few rows per token → **memory-mapped from SSD**,
  never in VRAM or RAM budgets. Their size does not matter.

## The model (1017 checkpoint headers, box2 `/data/ds41ck-full`; `data/ds41/inventory-1017-groups.json`)

Main model only (no Engram tables / drafter / vision): **551.9B params, 298.6 GB.** "Read / token" = weight bytes read to decode one
token at batch 1 at native precision (routed experts 6 of 384 per layer; embedding one row). KV cache not included.

| part | native | bpw | params | bytes | % bytes | read / token | % of decode read | plan |
|---|---|---|---|---|---|---|---|---|
| **Routed experts** (40 × 384 × gate/up/down, 5120 × 2304) | MXFP4 (fp4 + e8m0/32) | 4.25 | 543.6B | 288.8 GB | 96.7 % | 4.51 GB | 34.6 % | **codebook frontier** (below) |
| **Attention projections** (wq_a/b, wkv, wo_a/b) | FP8 e4m3, 32×32 block scales | 8.01 | 5.07B | 5.07 GB | 1.7 % | 5.07 GB | 38.9 % | trunk ladder ~K5–6.5 A8 |
| **Shared experts** (1 / layer) | FP8 | 8.01 | 1.42B | 1.42 GB | 0.5 % | 1.42 GB | 10.9 % | trunk ladder |
| **LM head** | BF16 | 16 | 0.66B | 1.32 GB | 0.4 % | 1.32 GB | 10.2 % | ~K6 (R16 mad head path) |
| Engram projections | FP8 / BF16 | 8.01 | 0.32B | 0.32 GB | 0.1 % | 0.32 GB | 2.4 % | keep (later: trunk ladder) |
| Router | BF16 (+F32 bias) | 16 | 0.08B | 0.16 GB | 0.05 % | 0.16 GB | 1.2 % | keep (routing is precision-sensitive) |
| Hyper-connection mixers | F32 | 32 | 0.04B | 0.16 GB | 0.05 % | 0.16 GB | 1.2 % | keep |
| Indexer / compressor (sparse attention) | FP8 / BF16 | ~10.6 | 0.06B | 0.08 GB | 0.03 % | 0.08 GB | 0.6 % | keep (block selection is precision-sensitive) |
| Token embedding | BF16 | 16 | 0.66B | 1.32 GB | 0.4 % | ~0 | 0 % | keep, in VRAM (1.3 GB; not worth host placement) |
| Norms / biases | BF16 | 16 | ~0 | ~0 | — | ~0 | 0 % | keep |
| **total** | | | **551.9B** | **298.6 GB** | | **13.0 GB** | | |

Other facts (config.json): 40 layers (20-layer causal encoder + 20-layer decoder), hidden 5120, 384 routed experts top-6 + 1 shared,
expert width 2304, vocab 129,280, sliding window 128 + sparse (indexer top-512) attention, FP4 main KV cache (890 B/token).

**Focus: four targets.** Routed experts set the container size and the VRAM / PCIe placement; attention + shared experts + head
(~7.8 GB read every token at native precision; ~76 % of decode reads once experts are at ~2.5 bpw) set the VRAM bandwidth per token.
Everything else stays native (~0.7 GB, ~3 % of reads).

## Deployment: experts offloaded to host RAM

4× R9700 = 128 GB. At ~2.5 bpw routed experts are ~170 GB: ~100 GB fit in VRAM, ~70 GB stay in host RAM and stream over PCIe
(box2: EPYC 7443, PCIe 4.0 → ~25 GB/s per card expected, **to be measured**) and are decoded on the GPU (CPU decode of trellis codes is
far slower than streaming).

Routing is skewed but not extremely (d100b, 100M tokens, `ds41pipe/data/routing/`: usage 0.1×–10× of uniform, flat middle):

| experts resident per layer (most-used) | 40 % | 50 % | ~59 % (≈ 100 GB at 2.5 bpw) | 70 % | 80 % |
|---|---|---|---|---|---|
| share of expert hits served from VRAM | 65 % | 74 % | ~82 % | 89 % | 94 % |

Rough decode estimate at 2.5 bpw, batch 1 (independent routing per token = pessimistic for one stream; to be replaced by a per-layer
timeline measurement):

| term | bytes / token | time / token |
|---|---|---|
| trunk at native precision (attention + shared + head + small) | ~8.5 GB | ~3.4 ms (4 × ~640 GB/s) |
| resident routed experts | ~2.2 GB | ~0.9 ms (more if K2–3 decode is ALU-bound) |
| missed routed experts over PCIe (~18 % of hits) | ~0.48 GB | ~5 ms (~100 GB/s aggregate) |

**Overlap:** within a layer the router must run before misses can be fetched; resident-expert compute overlaps the miss transfer;
missed-expert compute waits for its data; the next layer waits for this one. So the miss transfer is serialized and the resident
experts' ALU (codebook decode included) hides under it.

**Consequences.**
1. For **routed experts** under offload, decode ALU cost barely matters (link-bound); quality per bit becomes speed (fewer link bytes).
   ALU matters again only for a fully VRAM-resident configuration (≤ ~1.8 bpw experts, or 8 cards).
2. **Compaction** of experts (codebook, bit allocation across hot / cold experts, placement co-optimized with bits) is the main lever.
3. The **trunk** is the second-biggest decode term: quantize attention + shared experts + head (~K5–6.5 A8, head ~K6).
4. Engine levers (later, not quant): a dynamic expert cache over the static pins, routing prediction / prefetch, DSpark-informed fetch.

## Routed-expert codebook frontier (round 1, running 11-20)

Axes: **ALU cost** (measured on gfx1201; decode GEMV M = 1–8 and a prefill tile; w8a8 and w4a4) · **size** (bpw incl. tables; tables
≤ 8 KB per tensor) · **quality** (i.i.d. effective bits + H-weighted error on real 1017 experts, hot / medium / cold, gate_up and down).
Gate: **fractional K** (1/8-step shift patterns with salted 0-bit steps) — every codebook must be a pure function of the 16-bit state.
Also tracked: encode cost, prefill vs decode, robustness across expert classes. Rates: K0.5–3 for experts; K5–6.5 for the trunk (w8a8;
degradation seen above that with the int8 grid).

Candidates: mad (production, int8), mad4 (int4 variant), mul1_i8, CPTC Q6 (64 B, folded) / Q5 full (63 entries, paper's GPU form) /
minimal Q = K+1 in INT(K+2) (int4 output at K ≤ 2) / one-multiply / state-table L10–12 (1–4 KB, cheapest decode), QTIP 1MAD / 3INST,
int4grid. CPTC uses the paper's unified coefficient pair (one pair for every K, needed for fractional patterns).

w4a4: weight side is ~free at K ≤ 2 (int4 grid ×1.01 at K2, ×1.07 at K3, ×2.7 at K4 — LOG 10-31; paper Table 18: INT(K+2) tables lose
≤ 0.02 bits). The cost is the activation side (0.8B #46-02 failed at K2: KLD ×1.14–1.16); stage 2 measures it offline on the real
expert-input dumps. Candidate use: cold / unimportant experts at low K.

Work: w208 = ALU bench (dad1); lead = quality harness on real experts; w207 = MoE routed-expert ops on the general `exl3-v3` branch
with ONE codebook hook (decode) — the encoder should share the same header.

Evidence so far (27B, LOG 11-20): CPTC beats mad by 2.9 / 4.4 / 4.7 % H-weighted error at K2 / 3 / 4 (22 / 22 tensors), i.i.d. CPTC
1.947 / 2.940 / 3.928 vs mad 1.930 / 2.907 / 3.869 effective bits; at K5 / K6 CPTC 4.903 / 5.807 vs mad 4.770 / 5.556 (mad's int8
grid binds at high K: relevant for the K5–6.5 trunk). Headroom beyond CPTC for any scalar trellis code: ≤ 0.05–0.06 bits
at K2–3 (CPTC's gap to the rate-distortion limit).

Not for experts: CPTC matmul-aware tables (need correlated operand pairs, ρ > ~0.3; weights × RHT activations ≈ 0). LittleBit-2
(2603.00042): needs full QAT, worse than 2-bit methods at 1 bpw, a different kernel family; at most a later option for experts below
1 bpw.

## Encoder

Target: a native fused encoder in radiance **libquant** (C++ / HIP, rad-convert), not the PyTorch / exllamav3 path. Base:
`libquant/lq_tcq_gpu.hip` (GPTQ over a trellis on the card). Batch across experts (15,360 routed experts × 3 matrices, independent →
the GPU stays full although each matrix's LDLQ is sequential). CPTC paper's fused encoder: ~60M values/s at L=16 on an L40S, FP16 path
costs packed with the predecessor index into one 32-bit word → ~40 min of trellis search for 544B values on 4 cards, plus LDLQ.
Build after the frontier picks the codebook.

## Open measurements

- box2 host→GPU bandwidth with all 4 cards copying at once; host memory bandwidth.
- Per-layer decode timeline of an offloaded container (transfer / wait / compute).
- w208's ALU table; the quality table on real experts; A4 activation error on the expert-input dumps.

## Final note (user, 11-20)

Once everything else is well optimized: try the **Engram projections at K4–7** (they are 2.4 % of decode reads at native FP8, 0.32 GB
per token — the largest of the parts kept native).
