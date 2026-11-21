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

Evidence so far (27B, LOG 11-20 / 11-21): CPTC beats mad by 2.9 / 4.4 / 4.7 % H-weighted error at K2 / 3 / 4 (22 / 22 tensors) and
4–7 % at K5; i.i.d. with mad's scale searched: CPTC +0.016 / 0.022 / 0.028 / ~0.03 bits at K2 / 3 / 4 / 5, all codebooks tie at K1.
**At K6 mad / mul1_i8 beat CPTC** (CPTC's 127-value alphabet caps it; real tensors +2–6 % E2M5, +10–14 % int8): trunk at K ≥ 6
stays mad. Headroom beyond CPTC for any scalar trellis code: ≤ 0.05–0.06 bits
at K2–3 (CPTC's gap to the rate-distortion limit).

**Candidate results so far (i.i.d. Keff, scale searched / int8 tables fitted; decode = w208 factor vs the shift-and-mask base at
K1, M = 1; 11-21), split by output type — int4 output feeds the w4a4 path (cheaper GEMM from M ≥ 4, weights limited to 16 levels),
int8 the w8a8 path:**

int4 output (useful to ~K2; 16 levels cap it above):

| codebook | K1 | K2 | K3 | K4 | decode |
|---|---|---|---|---|---|
| madpop4 (popcount of 15 mad-hash bits) | 0.952 | 1.930 | 2.774 | 3.058 | ≈ mad (3 instr; untimed) |
| madpoplut (popcount → 16-entry table) | 0.952 | 1.930 | 2.781 | — | ≈ mad + load |
| int4grid (mul1 hash → 16 levels) | 0.941 | 1.931 | 2.889 | 3.214 | 2.2× |
| cptcmin (Q = K+1, int4 at K ≤ 2) | 0.951 | 1.916 | (int8 above K2) | | 1.71× |
| mad4 (3+3+1-bit fields) | 0.923 | 1.883 | 2.824 | 3.194 | 1.52× |

int8 output:

| codebook | K1 | K2 | K3 | K4 | K5 | K6 | decode |
|---|---|---|---|---|---|---|---|
| CPTC unified pair (cptc6u, 64 B) | 0.951 | 1.948 | 2.939 | 3.928 | 4.886 | 5.743 | 1.77× (packed form under test) |
| cptc5 (63 B) | 0.951 | 1.947 | 2.938 | 3.920 | | | 1.77× |
| mul1lut (1 KB) | | 1.939 | 2.930 | 3.911 | 4.886 | 5.824 | > mul1 |
| mul1_i8 | 0.942 | 1.937 | 2.924 | 3.905 | 4.880 | 5.843 | 2.13× |
| madlut (253 B) | 0.952 | 1.935 | 2.923 | 3.907 | 4.855 | 5.781 | ≈ mad + load |
| mad (production) | 0.951 | 1.932 | 2.917 | 3.900 | 4.866 | 5.830 | 1.41× |
| madpop31 | 0.945 | 1.923 | 2.893 | 3.448 | | | ≈ mad |
| lc1 (64 B, one multiply) | 0.950 | 1.919 | 2.871 | 3.781 | | | 1.29× |
| cptcst10 (1 KB state table) | 0.913 | 1.808 | 2.684 | 3.566 | | | 1.71× |

Regimes: int4 — madpop4 at K ≤ 2 (= mad's quality at mad's cost); int8 — CPTC K2–5 (+0.016–0.028 bits), mad / mul1_i8 at K ≥ 6,
anything at K1. All are pure functions of the 16-bit state (fractional K works; CPTC needs the unified pair; its fractional-K table
is untested).

**Decision (user, 11-21): mad everywhere for now (simplicity: one codebook), w8a8.** Upgrade path kept for later (V4): lc1 at
K ≤ 2 (same quality, 10–15 % cheaper decode where dequant-bound) and packed cptc7 at K2–5 (~3–5 % lower weight error), both on one
per-tensor table plane added to the format without touching mad records. Previous candidate plan, for reference:
**w8a8 everywhere; codebook by K: packed cptc7 at K2–5 (one 255 B int8 table, glb, packed
decode), mad at K ≤ 1.5 and K ≥ 6** (K1: all codebooks tie; lc1 searched is a cheaper K ≤ 2 option if decode at K1 matters). A4 /
int4 compute stays out (27B: +~4.7 % KLD per band at K2; cold experts at K1 would tolerate it but int4 saves no storage and cold
experts are rarely computed). madpop4 kept as the int4 codebook if a better A4 appears. To confirm: cptc7 on the 22 real 27B
tensors, its fractional-K table, whole-model KLD mad-only vs frontier.

Not for experts: CPTC matmul-aware tables (need correlated operand pairs, ρ > ~0.3; weights × RHT activations ≈ 0). LittleBit-2
(2603.00042): needs full QAT, worse than 2-bit methods at 1 bpw, a different kernel family; at most a later option for experts below
1 bpw.

**Menu step size (user + lead, 11-20).** Error falls ~4× per bit (convex), so splitting a group of tensors between two rates is
worse than giving each its best rate — the number of experts does not average a coarse grid out. Expected extra error from a
step Δ ≈ 0.96 · Δ²/12: integer 8 %, 1/2 2 %, 1/4 0.5 %, 1/8 0.13 %. The 27B agrees: integer → quarter steps −5…−7 % KLD (#36,
`design/v3.md`), extra 1/8 levels ≤ 0.8 % (11-01, "extra trunk levels don't pay"). Menu (not exact):

| range | step | for |
|---|---|---|
| K0.125–1 | 1/8 | cold routed experts (cheap to quantize; the 27B's sub-1 curve #90) |
| K1–4 (experts up to ~K4–5) | 1/4 | routed experts — the bulk of the quantization work; about half the levels of 1/8 steps |
| K4–7 | 1/4 (trunk is ~7B params, so the extra levels are cheap) or 1/2 | attention, shared experts, head |

The format and encoder keep 1/8 everywhere (free); add 1/8 levels only where the knapsack piles up. Codebook gate: supports
fractional patterns (1/8 granularity itself is not needed for experts).

**Lazy ladder, not a full expert ladder (user + lead, 11-20).** Menu: 20 expert levels to K4, 24 to K5 → ~1M knapsack options
(easy). But a full ladder = 20–24 encodes of 544B values (~13–15 h Viterbi on 4 cards at the paper's ~60M values/s/card) and
**~2.5 TB (to K4) / ~3.7 TB (to K5)** of records: too big for any box. Instead:
1. full ladders on a sample of experts (a few %, across layers, hot / cold) → the curve shape (27B: fractional levels sit at 0.99–1.00
   of the interpolated curve, #36);
2. every other matrix at 2–3 levels → its fitted curve (keep only the error numbers, delete the records);
3. knapsack per target size on the fitted curves;
4. quantize only the (matrix, K) pairs the sizes pick, into a cache keyed by (tensor, K, codebook, quantizer version); a record is
   identical whichever size picks it, and neighbouring sizes share most picks.
Cost for 5 sizes (1.5–3.5): ~8 passes (~5 h) and **~0.8–0.9 TB cache** (+ ~50–100 GB sample ladders); a later size adds ~1 pass
and ~0.15–0.2 TB. Containers: **radiance assembles the mix in the engine from the store** (manifest container pointing at the cached records;
`quantizer-native.md`), so no container copies exist — storage is the cache only. (box2 `/data` is ext4: no reflinks anyway.)

Healing knob (user, 11-20; BACKLOG): a table codebook lets each tensor (or expert) carry its own 64 B table that post-mix
healing / QAD can train — a nonlinear correction that scales can't express. Mad has no such knob. Counts for CPTC in the choice.

## Serving base (scout, lead 11-21; user: option 1 = port onto our live branch)

bakon's `deepseek41-wip` (`/workspace/forks/radiance-ds41`, 9a963c2, shallow) is NOT on 1.17 any more (old note): it sits on
radiance `main` between 2.1.0 and 2.2.0. v2.2.0 already carries part of it (the `F8_E8M0` safetensors reader,
`rad_deepseek41_chat.h`). What a port onto `exl3-v3` (rebased to 2.2.0) needs, all small or new files:
- `arch/deepseek41/` (2.8k lines, new) + `ds41_walk_geom.h`;
- 29 new `libr4d` files: `r4d_ds41_sparse*` (compressed / sparse attention + indexer), `r4d_ds41_walk*`, `r4d_ds41_elem*`,
  `r4d_ar_mhc_pcn.hip` (mHC), `r4d_ds41_mxfp4*` (DSpark drafter, out of scope), `r4d_ds41_ggml*` (IQ1_S / IQ2_XXS — only for the
  Unsloth baseline containers);
- `abi/rad_kv_fp8.h` (189), `arch/common/rad_expert_split.h` (60), `rad_ggml.h` (49), `core/runtime/rowprefetch.*` (row read-ahead
  of the mapped tier), the weight sidecar, `RAD_MAP_MUL` (Engram gate) in the reader, the imatrix-name fix, scheduler lever D7;
- `rad_block_moe_fp8.h`: 43 lines differ from 2.2.0 — w207 also edited this file on `exl3-v3` (merge point);
- his `arch/common/exl3` + `exl3/` (radiance-exl3 import, ABI 40) are replaced by our exl3i MoE ops.
Estimate: one focused worker session (port + build + boot on box2 at TP4 with the release's MXFP4 experts), then exl3i experts.
His upstream PR plan splits the same pieces (imatrix fix #267, gather read-ahead, release reader, model support, kernels).

## KV cache (w216 measurement; user 11-21: i8 KV deferred)
deepseek41 bf16 KV: 3,200 B a token a card at TP4 + ~93 MiB fixed → 122 MiB / card at 9k, 294 MiB at 64k, 3.2 GiB at the 1M max,
~6.3 GiB at 2M. The main compressed KV is the model's own FP8 / FP4 (exact, `rad_kv_fp8.h`); i8 would only touch the bf16 window /
tail / indexer caches, and indexer error can change which 512 tokens are selected (discrete, high-risk). Not worth it at this size:
**deferred** (needs kernels anyway; the plugin takes bf16 / fp8 only).

## Scoring (user 11-21)
KLD is read only on the rows the golden marks in its own `gen.u8` (from the HF parts, copied byte for byte into every slice; `eval/ds41/w212/score41.py`). Never rebuild the mask; prompt-side rows are diagnostic only (they read high KLD by nature: golden G6 check).

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
