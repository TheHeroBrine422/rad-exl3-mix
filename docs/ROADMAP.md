# Roadmap

Current as of 2026-11-05. Design detail for V3: `docs/design/v3.md`; experiments: `docs/experiments/` (INDEX, LOG,
BACKLOG).

**Names (user, 2026-11-04):** EXL3-V1 = the turboderp-ladder port (released); EXL3-V2 = mixed-K EXL3 on the
f16-activation kernel (released); **EXL3-V3** = the next release: integer compute, a new quantizer stack and a finer
ladder; **V3 extras** = optional add-ons built on V3's tooling; **V4** = the release line after V3 (QAD on V3
checkpoints, then KV / LSA / MoE work as it lands). Older notes' "V3.5" and "V5" labels map onto V3 extras / V4;
most V3.5 items moved into V3 on 2026-11-04.

## EXL3-V1 and EXL3-V2 (released)
- V1: turboderp's EXL3 conversions in radiance 1.13.0 (`exl3r4d`, Lloyd `embed_tq`), HF
  `Hero422/qwen3.8-27b-exl3-turboderp-ladder`.
- V2: released 2026-10-31, 16 sizes 1.5–9.0 bpw, HF `Hero422/qwen3.8-27b-exl3-v2-rad`. Our K1–8 quants plus
  turboderp's ≥ 5-bit tensors where they measured better; the per-tensor mix from a fitted sensitivity model and an
  exact knapsack DP; streaming mix search on calib fit slices, final pick on a held-out verify slice; the public
  corpus was not used. Board `gemm_us` ~8,050 µs (9.0: 7,331). No further V2 work (user).

## EXL3-V3 (in progress)

**Name (tentative, 2026-11-05): KTQ (Knapsack Trellis Quantisation).** V3 is no longer loadable by exllamav3
(int8 / int4-rounded codebooks, own encoding), so it ships under its own name, as a QTIP-family trellis format derived
from EXL3 (credit: turboderp's exllamav3; Cornell RelaxML's QTIP and YAQA).

**Sizes and variants (user, 2026-11-04 / 11-05):**
- Release sizes 2.0, **2.5–4.5 (the focus of the quality work)**, 5.0, 6.0 and **7.0** (11-05: cheap, since the
  ladder holds its levels, and a prefill speed gain for TP2 / 32 GB users; V2 7.00 runs 8,105 µs vs IQR-V4's 4,006).
  **8.0** decided after the 7.0 results and a floor split on V2 8.0 (built only if predicted ≥ 10% below V2 8.0's
  0.00034). 7.5 stays V2.
- Each size in an **accurate** (f16 wherever it lowers KL) and a **fast** variant (int8 path where cheap, under a speed
  budget); a third, **A4** variant at ≤ 3 bpw only if it is ≥ 15% faster than that size's fast variant.
- Research points 1.5 / 1.75, then a **low-bpw research ladder** in 0.1 bpw steps below 1.5 (1.4, 1.3, ...) until KLD
  blows up (user, 2026-11-05) (scored on our own held-out slices; a few points go to the board); saturation-aware allocation for these.

**Format and kernels:**
- Per-tensor weights × compute modes (the W/A table in `docs/design/v3.md` §1): i4 codebook K0.125–3 (A4 / A8 / A16),
  i8 K2–7 (A8 / A16), mul1 K6–8 (A16), bf16. Sub-1 K (K0.125–0.875, salted 0-bit read) is in V3 (#90 prices it).
- **A4 on i4 weights** as a per-tensor knapsack option for the fast variant at ≤ 3 bpw (#46-03; d1106's all-MLP A4
  failed as a global setting). The int4-plane kernel is built only if its rule passes.
- **Decode stays on 16-bit activations.** Low-bit activations only run in prefill steps of ≥ 512 rows: at decode
  (1–16 rows a step) the GEMV is bound by reading the weights, not by arithmetic, so int8 / int4 activations would
  save nothing and only add error. The fast and accurate variants therefore decode identically; they differ in
  prefill. (The 512 threshold is checked by the kernel timing's M = 16–4096 sweep.)
- **Prefill: two-pass kernels** (decode each weight once per step into an int8 / f16 (/ int4) plane, then radiance's
  tuned tile; IQR's route). Est. ~4,000–4,300 µs a layer for the fast path vs V2's ~8,050.
- Engine: radiance 1.17 (fork branch `qc-exl3-v3`, rebased on each release; plugins `qwen35_exl3`, `exl3i`, ...).
- Embedding: trellis-coded (one row decodes alone, H = I), quarter steps, with the menu extended at both ends:
  sub-1 (t0.25–0.75) for the smallest sizes and t6.25–t8 for 7.0 (the solver already sits on t6, the old menu top).
  Head: row-slab K (rows sorted by the head's Fisher diagonal) vs a vocabulary-weighted head (IQR-V4). Per-row
  (vocabulary-tiered) embedding: dropped (2026-10-31).
- Vision tower and DFlash2 drafter bundled in bf16 (outside the bpw count).

**Quantizer stack** (each item proven on the 0.8B proxy needs its 27B confirmation, BACKLOG "27B confirmation
queue"): diagonal output Hessian H_out (Σdy², block-normalized; 27B swap test #33-28) → joint SwiGLU scale refit
(swj) → sibling-sequential asym writers (ρ0, last) → frozen-code scale healing (#89): **deferred, not dropped** (user, 2026-11-05): no 27B test before the settings
freeze; 27B work starts with the ladder generation, at the latest after V3; may still reach V3's releases if ready.
0.8B healing tests continue. Plus: per-tensor damping, scale gain per K and family (larger for attn_output / ssm_out at ≥ 5 bits), turboderp's tensors as options, and to test:
IQR-V4's act order in the trellis, bf16 (row, 128) scales on attn_output / ssm_out from K5, a vocabulary-weighted
head vs our row-slab head; A1 (down_proj predicted from up) as a per-layer option if it is on the frontier;
self-calibrated statistics (#91). The cubic i8c codebook is likely dropped (no gain on top of H_out).

**Ladder (66 levels, ~180 card-hours):** i4 K0.125–3 in 1/8 steps (24), i8 K2–5 in 1/8 (25) + K5.25–7 quarters (8),
mul1 K6–8 quarters (9). Quantized in order of use (2.5–4.5 first), once the settings above are frozen.

**Selection:** knap4 with per-tensor compute modes and a speed table; streaming search with fresh eval draws
(`docs/design/eval-data.md`: xcorp 8k + a share of private agentic text for search accept / reject; calib verify
for finals and published numbers). Public only for reporting at the end.

**Order:** freeze the quantizer settings (swap test, confirmation queue) and build the kernels in parallel → ladder
→ knap4 / stream per size and variant → finals on calib verify, long-context check, **decode timing** (end-to-end engine step time on the finals: batch 1 and a few
concurrencies, short and long context; how Deadcode measures it; reported, not solved for: decode tracks size and
prefill, user 2026-11-05) → release. 7.0 after the focus
sizes and 6.0.

**Hardware:** 2xR9700, plus various 12-16GB consumer cards. 

## V3 extras (optional add-ons on V3's tooling)
- **Finer release sizes below 2.5 bpw** (2.25 at least; 1.875 if it earns its place): KLD changes fastest there
  (0.067 → 0.164 → 0.297 on the board from 2.5 to 1.75).
- **7.5 rebuilt** with V3's embedding and head, if the 7.0 / 8.0 results make it worth it.
- **Decode-optimized variants** (after the decode kernels, see "Later").
- **Disaggregated quantization (DQ, arXiv 2609.26333):** different compute formats (and optionally weights) for
  prefill and decode. The shared-weight form is what V3 does (low-bit activations only for prefill steps of M ≥ 512);
  the separate-prefill-weights form ("ODP", streamed from SSD) is a serving variant, not a board entry.
- **Quantized DFlash2 drafter and vision tower** (low priority, user 2026-11-05): needs drafter acceptance and an
  image eval as metrics.
- **REAL-Q (arXiv 2609.00049)** as a final pass on 2–3 shipped sizes, if the 0.8B test (below) pays: it is per-mix
  (sequential), so it cannot enter the per-level library (~40–50 GPU-hours each at 27B; its gains shrink with model
  size: −49% at 1.7B, −5…−9% at 8–70B).
- **Speed gap to IQR, for the record:** decode (~30 vs ~100 tok/s at TP1) is mostly speculative decoding (DFlash2,
  bundled in V3); prefill (~1,500 vs ~2,500 tok/s) is the GEMM time V3's int8 path targets.

## V4: QAD and after
- **QAD** on the published V3 checkpoints, after scale healing (if healing did not already ship in V3: see the V3
  quantizer stack). 0.8B results to start from (each needs a 27B check): frozen-code healing
  K2 −38% (#89); field report: QAD on an EXL3 quant gained ~6–13% KLD at a 1,024 window, ~22% training only layers
  48–63 (trellis codes interfere earlier in the stack). Ideas to test on a small model first: QAD before the trellis
  step, other rotations than the 128-point Hadamard (e.g. ParoQuant's), QAD on a separate corpus.
- **KV-cache work** (V-mean, Hadamard KV, per-layer KV recipe, KV-QAT); may come from other radiance work on our published checkpoints
  instead (2026-11-04): our part is keeping qwen35_exl3's attention on
  radiance's standard KV groups.
- **LSA**, then **MoE models** (QFN / GLM-5.3-Flash / Mistral Large 4; `docs/research/moe-kld.md`): measured on our
  own corpus first; need bf16 reference logits and input Hessians of the full model (SSD-streamed bf16 weights or
  cloud hardware); MoE needs per-expert handling.

## Later (no version yet)
- **Decode speed as its own axis** (user, 2026-11-05: later): decode kernels after radiance 1.17 (block-per-tile GEMV
  with K-split waves, biased-int fold, barrier-free 16-token tile; BACKLOG); profile decode time per K and
  fractional pattern (measured nearly flat in K below ~4 bpw: 32.5 tok/s at 1.40 vs 24 at 6.00), then
  decode-optimized variants. Longer term the frontier is 4-D (size, KLD, decode, prefill): for the dense 27B decode
  tracks size, for MoE models it can differ a lot.
- **Search-algorithm improvements** for the tensor search: in parallel with the quantization runs.

## Research method and infrastructure
- **Small-model proxies:** Qwen3.5-0.8B on a 12 GB consumer GPU for most screening (anything with gradients or
  output-side statistics; `proxy2b/`, goldens, fake-quant evals through our quantizer), the 2B where width matters
  (its calibration study `proxy2b/CALIBRATION.md` is paused), the 9B (untied embedding / head, closest shapes) when a
  test needs it (an R9700 slot). Same family and block design as the 27B; nothing transfers as numbers, method
  rankings should. 27B single-layer tests also run there (`v3/yaqa/hconv/`).
- **0.8B tests queued now:** robust (heavy-tail) Σdy² estimator, sequential (quantized-prefix) Hessians, REAL-Q,
  self-calibration (#91, running).
- **Corpora:** calib pools for every decision (`AGENTS.md`); xcorp v1 (96M tokens from the calib's upstream
  sources, #61) for training / statistics and a share of search accept / reject; private agentic sessions
  (never published) the same way; turboderp's self-sampled calibration trace for this model.
  Preference for training / statistics data: generated by the model itself from calib-like prompts, then more of
  the calib's upstream sources, then generic text.
- **Donated GPU time** (volunteer v0 built, parked until the ladder): per-layer quantization jobs for the ladder and
  `eval_mix` jobs for the search (public slices only; mixes assembled locally from cached levels), a central queue,
  points by verified work; DDP / DiLoCo-style training only later.
- **YAQA:** full Sketch A / B lost on the 0.8B (#33-08); its diagonal H_out alone wins (#33-10…#33-16), which is the
  V3 candidate above. **27B: only −2.8% at K2.5 (#33-28), a third of the 0.8B's gain; not part of the pipeline for
  future / larger models** (user, 2026-11-05) unless a cheap estimator (≤ 384 sequences, bf16 hi/lo backward instead
  of fp32, a proper multi-GPU backward) brings it to a few GPU-hours and a pilot on that model shows it pays. Fuller (mixed) H_out only if the diagonal clearly helps on the 27B (BACKLOG).

## Data (after QAD)
- Clean held-out text runs out once QAD trains on the qad pool. Options: reuse documents by their ledger record
  (`eval/pools/ledger.jsonl`; measure the bias of reused search-pool docs first), new agentic traces generated in
  the corpus's style, openly licensed text matched per category, or asking the challenge author for more held-out
  text.
