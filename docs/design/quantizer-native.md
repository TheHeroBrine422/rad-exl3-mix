# Native fused EXL3-V3 quantizer (radiance libquant): planning

Status: **planning draft, 2026-11-20** — open questions for the user marked **[Q]**; nothing started. Intended builder: one Fable
(low or medium) agent after the plan is agreed. First targets: 27B equivalence → Qwen3.6-35B-A3B (real MoE, one card) →
DS4.1-F-1017 routed experts + trunk.

## Why

Our quantizer today is Python + the exllamav3 fork (`quantize/ourq.py`, `quantize/exl3-patches/`) plus our own CPU tools (`v3/e3/`,
`viterbi3.c`). It is slow (the 27B ladder took days of card time) and lives outside radiance. DS4.1-F has 543.6B routed-expert
weights (15,360 experts × 3 matrices) plus ~7B trunk weights to quantize, at several K each for the mix search. The CPTC paper's
fused GPU encoder runs ~60M values/s at L=16 on an L40S → ~40 min of trellis search for one K over all experts on 4 cards.

## What radiance already has (v2.2.0; read 11-20)

- **libquant** quantiser plugins behind a **recipe** (`--recipe`, `--quant 'PATTERN=QUANTISER:key=value'`, first matching rule
  wins; declare refuses an unservable recipe).
- **`lq_tcq_gpu.hip`**: GPTQ over a trellis grid on the card (IQR's): a workgroup per run of rows, 128-column blocks, tail-biting
  search per column, error feedback, host picks the next block's dyn scales while the card finishes the tail.
- **`lq_hess`**: calibration Hessians from the engine's calib tap (`gram.r<rank>.<stem>.bin`, per rank), with a recorded
  transform domain (raw or block-Hadamard of width N; converted on read). A routed expert uses its own Hessian when present
  (the tap records one for `down`), else the layer's pooled one.
- `--imatrix` (diagonal importances, expert popularity), GPU device lists, multi-file checkpoints, fp8 / MXFP4 source readers
  (DS4.1-F converts already).

## What our format needs (EXL3-V3 / exl3i, frozen in `qc-exl3-v3`)

| feature | ours | libquant today |
|---|---|---|
| rotation | random Hadamard per 128 block with per-tensor sign seeds on both sides (suh / svh) | Hadamard domain exists (calib side) |
| tiling | 16×16 tiles, 256-value tail-biting trellis per tile, L = 16 | 128-column GPTQ blocks, per-column trellis |
| feedback | LDLQ with 16-column blocks | GPTQ column order |
| codebook | mad hash (today), table codebooks (CPTC) later; **one header shared by encoder and decoder** | grid codebooks |
| rates | K 0.5–8 incl. **fractional 1/8-step patterns** (salted 0-bit steps), sub-1 K | integer |
| scale | per-tensor scale factor search {1, 1.15, 1.3}, damp 0.01 | dyn scales per block |
| output | exl3i records (`qc-exl3-v3` ABI), per-tensor K from a **mix** | one container per convert |
| A8 / A4 | decode-time mode; no encoder change | — |

## Open questions

1. **[Q] Which radiance features are missing that you have in mind?** My list of likely gaps: the ladder / mix flow (below),
   exl3i records as a libquant output, fractional-K patterns, per-expert Hessians from our HF format, out-of-core streaming at
   DS4.1-F size, resumability, error reports for the knapsack.
2. **[Q] Ladder vs container.** Our pipeline quantizes every tensor at every K level (a "ladder" store), then the knapsack picks a
   mix and `store/assemble.py` builds the container. rad-convert makes one container per run. Options: (a) a libquant
   "ladder mode" writing per-(tensor, K) records + error stats into a store, assembly stays ours; (b) rad-convert takes a mix
   file and quantizes only what the mix picks (needs the sensitivities from elsewhere); (c) both. Recommendation: (a) first.
3. **[Q] Upstream or fork-only.** Built as a libquant quantiser plugin on `exl3-v3` (our live branch), upstreamable later, or kept
   as our own tool linking libquant? Recommendation: plugin on `exl3-v3`, PR later.
4. **Equivalence bar.** New encoder vs ours on the 27B: same codebook (mad), same rotation seeds, same Hessians → per-tensor proxy
   error within noise on the 22 C3 tensors, then whole-model KLD within the usual bar on one size. Byte identity is not
   expected (search order / float sums differ).
5. **Hessians.** 27B: published calib Hessians (raw domain, sum x xᵀ). DS4.1-F: our d100b collection (dense G1, per-expert G3 h13
   [5120²] + h2 [2304²], f64 safetensors, HF quant-calib) — a reader, or a converter into `gram.*.bin`. Rare experts have few
   rows: fallback to the pooled layer Hessian below a row count (as libquant does) **[Q] threshold**.
6. **Batching and scale.** One launch over many experts (same shape, independent); 4 cards; host RAM 113 GB on box2 vs 289 GB
   MXFP4 source + Hessians → stream per layer from disk; per-layer shards, `.part` then rename, restartable.
7. **Codebook interface.** One C header (decode of a 16-bit state → value, per codebook) used by the encoder's Viterbi and by the
   w207 / exl3i kernels; mad first, CPTC tables (global or per tensor, int8 or E2M5) by header swap. Table codebooks need the
   table stored in the record (≤ 8 KB budget).
8. **Search settings.** TB = 1 (TB16 costs 16× for ~0.3 %), viterbi3's fused form as the CPU reference.
9. **Later, not in v1:** trainable tables / healing hooks, A4-aware encoding, QAD.

## Proposed order (one agent, staged, each stage reviewed by the lead)

1. Codebook header + GPU tile Viterbi (tail-biting, fractional patterns) vs `viterbi3.c` on random tiles: bit-identical codes.
2. LDLQ + rotation + scale search on one tensor vs `ourq.py` / `tensor_check.py`: proxy error within noise.
3. Ladder mode over a whole 27B (all levels), into the store; KLD check of one assembled size.
4. MoE batching; Qwen3.6-35B-A3B end to end (convert, serve with w207's kernels, KLD vs bf16).
5. DS4.1-F: MXFP4 / fp8 source, d100b Hessians, out-of-core, 4 cards.
