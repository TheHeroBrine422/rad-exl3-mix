# Roadmap

Current as of 2026-11-06. Design detail for V3: `docs/design/v3.md`; experiments: `docs/experiments/` (INDEX, LOG,
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
  i8 K1.125–7 (A8 / A16), mul1 K6–8 (A16), bf16. Sub-1 K (K0.125–0.875, salted 0-bit read) is in V3 (#90 priced it:
  it pays only below ~1.3 bpw, so it serves the low-bpw research ladder). **i8 at K7 is not free (2026-11-06, #94):**
  ×1.27 error vs mul1 (~0.18 bit) on real tensors, not the ×1.02 a Gaussian-tile smoke showed; the int8 grid tops
  out near mul1 K6.85, so there is no i8 level above K7 and the knapsack prices i8 K6.5–7 from measured curves.
- Format checks that left it unchanged (2026-11-06): chained tail-biting (#102) and down_proj block act order (#103)
  negative. Still open: the IQR-style hash codebook (#100; a CPU re-check and a small real run on the 0.8B + gfx1201).
- **A4 on i4 weights** as a per-tensor knapsack option for the fast variant at ≤ 3 bpw (#46-03; d1106's all-MLP A4
  failed as a global setting). The int4-plane kernel is built only if its rule passes. **[superseded 2026-11-05]**
  The A4 path was built at once (user; ×1.31 faster than A8 on a K2 layer at M 2048); #46-03's rule decides whether
  it ships. Per-block A8 exceptions for A4 and an i4 plane for speed alone: negative (11-06).
- **Decode stays on 16-bit activations.** Low-bit activations only run in prefill steps of ≥ 512 rows: at decode
  (1–16 rows a step) the GEMV is bound by reading the weights, not by arithmetic, so int8 / int4 activations would
  save nothing and only add error. The fast and accurate variants therefore decode identically; they differ in
  prefill. (The 512 threshold is checked by the kernel timing's M = 16–4096 sweep.)
- **Prefill: two-pass kernels** (decode each weight once per step into an int8 / f16 (/ int4) plane, then radiance's
  tuned tile; IQR's route). Est. ~4,000–4,300 µs a layer for the fast path vs V2's ~8,050. Two-pass A8 and A4 are
  merged (fork branch `qc-exl3-v3-merged`, 2026-11-06); `qc-exl3-v3` is fast-forwarded to it after the i4 importer.
- Engine: radiance 1.17 (fork branch `qc-exl3-v3`, rebased on each release; plugins `qwen35_exl3`, `exl3i`, ...).
- Embedding: trellis-coded (one row decodes alone, H = I), quarter steps, with the menu extended at both ends:
  sub-1 (t0.25–0.75) for the smallest sizes and t6.25–t8 for 7.0 (the solver already sits on t6, the old menu top).
  Head: row-slab K (rows sorted by the head's Fisher diagonal) vs a vocabulary-weighted head (IQR-V4). Per-row
  (vocabulary-tiered) embedding: dropped (2026-10-31).
- Vision tower and DFlash2 drafter bundled in bf16 (outside the bpw count).

**Quantizer stack (status 2026-11-06;** each item proven on the 0.8B proxy needs its 27B confirmation with two seed
pairs, BACKLOG "27B confirmation queue"; order H_out → swj → writers, last):
- Diagonal output Hessian H_out (Σdy², block-normalized; #33-28): K2.5 pooled 0.982 over two seed pairs, **not yet a
  pass**; a third pair and the K4 pairs are running. If adopted it makes a level ~5× slower to quantize.
- Joint SwiGLU scale refit (swj): **adopted** (#104: 27B −6.5% KLD at K2.5, no requant); stacking with H_out and
  the K4 transfer are being tested.
- Sibling-sequential asym writers (ρ0, last): 27B test #105 queued.
- Frozen-code scale healing (#89): **deferred, not dropped** (user, 2026-11-05): no 27B test before the settings
  freeze; 27B work starts with the ladder generation, at the latest after V3; may still reach V3's releases if ready.
  0.8B healing tests continue.
- Tile coordinate descent after LDLQ on attn_qg only (#77): −0.5–1.2% on those tensors on the real quantizer (the
  CPU proxy overstated it ~3×), ~+5 min a level: the user decides.
- Kept: per-tensor damping (#38), scale gain 1.0 (#39: flat; the old "larger at ≥ 5 bits" was a proxy artifact),
  fractional-K drift as is (#107 null), turboderp's tensors as options. Open: a vocabulary-weighted head vs our
  row-slab head; A1 (down_proj predicted from up) as a per-layer option if it is on the frontier.
- Dropped: input Hessians blended with xcorp (#101: 0.8B −2%, 27B pooled ~0.993; user 11-06), self-calibrated
  statistics (#91), REAL-Q (#92), IQR-V4's act order and bf16 (row, 128) scales (#95), the cubic i8c codebook (no gain
  on top of H_out).

**Ladder (73 levels):** i4 alone K0.125–1 (8), i8 K1.125–5 in 1/8 steps (32) + K5.25–7 quarters (8), i4 twins
K1.125–3 in 1/8 steps (16; K2 / K3 first, the other 14 only if A4 pays), mul1 K6–8 quarters (9). (Was "66 levels,
~180 card-hours" before the 11-05 amendments.) Measured: ~37 min a level on one R9700 without H_out, ~3.2 h with it.
Quantized in order of use (2.5–4.5 first), once the settings above are frozen.

**Production (2026-11-06):** the quantizer pipeline lives in `pipeline/` (package `rxq`, merged in 2026-11-07;
`config/v3.toml` pins the freeze settings, each with its evidence); the ladder run uses donated compute via
`rad-quant-at-home` (per-tensor jobs, inputs from Hugging Face pinned by revision, results verified by score rather
than by replication; on small cards every family but ffn_down fits in ≤ 4.1 GB). Where the time goes: the trellis
search is ~80% of quant time.

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
- ~~**REAL-Q (arXiv 2609.00049)** as a final pass on 2–3 shipped sizes, if the 0.8B test pays~~ **[dropped
  2026-11-06, #92]**: on the 0.8B on top of our stack it failed every rule (healed K3 ×1.10, K2 ×1.04).
- **D-LSA runtime option (user 2026-11-05):** latent sparse attention (Prefrontal-AI/latent-sparse-attention,
  arXiv 2610.42147, Apache-2.0) on the 16 full-attention layers, off by default (`--lsa MAP`). It doesn't change the
  weights: a released 6.3 MB fp32 map (`qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt`, release
  `d-lsa-weights-v1`) picks which keys each query reads (bucket centroids + sink + window). Kept fp32, not quantized.
  Shipped with V3 only (user 2026-11-05; not added to V2), as one sidecar file per V3 repo (`lsa/`), not as a variant;
  developed and tested on V2 4.00 weights until V3 exists (same runtime path). Target use: 256k–750k context with fp8 KV
  (KV is 32 KiB/token fp8, so at 512k attention is most of a decode step and ~2/3 of prefill): decode and
  incremental prefill both. Work: a ROCm/radiance port of the key clustering, centroid scoring and
  sparse selection (the reference is SGLang/CUDA). Gate: KLD of quant+LSA vs the same quant with full attention, fp8 KV,
  256k / 512k / 750k (YaRN), at 2.0 / 4.0 / 7.0, and prefill/decode speed; a low-bpw failure means re-fitting the map
  on the quantized q/k (their `lsa distill`). Not in board entries: it approximates full attention.
- **Speed gap to IQR, for the record:** decode (~30 vs ~100 tok/s at TP1) is mostly speculative decoding (DFlash2,
  bundled in V3); prefill (~1,500 vs ~2,500 tok/s) is the GEMM time V3's int8 path targets.

## V4: QAD and after
- **QAD** on the published V3 checkpoints, after scale healing (if healing did not already ship in V3: see the V3
  quantizer stack). 0.8B results to start from (each needs a 27B check): frozen-code healing
  K2 −38% (#89); field report: QAD on an EXL3 quant gained ~6–13% KLD at a 1,024 window, ~22% training only layers
  48–63 (trellis codes interfere earlier in the stack). Reference recipe (2026-11-06): NVIDIA's QAD of Qwen3.8-Flash-Next to
  NVFP4 (`nvidia/Qwen3.8-Flash-Next-FP4-Local`): bf16 teacher generates the responses (prompts only from Nemotron
  post-training sets), loss = total variation on token probabilities + hidden-state matching + separate
  end-of-thinking / end-of-response boundary losses, 500M tokens × 3 epochs (2,500 trunk + 1,500 joint updates,
  attention frozen); GPQA 92.2 → 91.5, ToolEvalBench 89.4 → 88.7. Its latent-weight STE does not map onto frozen
  trellis codes; the loss terms and the boundary losses do (healing / V4 QAD), and the data choice is the opposite
  of the board's metric (#91: on-policy text costs calib KLD), so it fits the real-use variant, not board entries. Ideas to test on a small model first: QAD before the trellis
  step, other rotations than the 128-point Hadamard (e.g. ParoQuant's), QAD on a separate corpus.
- **KV-cache work:** done upstream by Code 000 (radiance PR #416, 2026-11-06, open): per-layer `--kv-recipe` (bf16 / i8h /
  fp8 / fp8_v4 / k6_v4 / k5_v4 / k4h_v4 / k4h_v2h), Hadamard rotation, V mean, V prediction, and `rad-autoquant` (a
  per-layer KL knapsack with device verify rounds and a measured noise floor). Rows at head 256 / gqa 12 and 6, so
  Qwen3.8-27B at TP1 should fit. Our part: keep qwen35_exl3 on radiance's standard KV groups, run rad-autoquant for
  the 27B once merged, and teach D-LSA's `attn_paged_list` the new packed layouts. KV-QAT and any further KV work are parked (user 2026-11-06): model quant first; later we may iterate on #416.
- **N-LSA / LSA beyond D-LSA's released map**, then **MoE models** (QFN / GLM-5.3-Flash / Mistral Large 4; `docs/research/moe-kld.md`): measured on our
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
- **0.8B tests (2026-11-05/06):** REAL-Q (#92) failed; self-calibration (#91) did not help on calib; the xcorp H_in
  blend passed on the 0.8B (#101) but shrank to ~1% at 27B and was dropped. The 0.8B's effect sizes shrink at 27B
  (H_out −8% → ~−2%, xcorp −2% → < 1%; swj the exception, −6.5% at 27B): every setting gets its 27B seed pairs.
  Listed on 11-05, no INDEX row yet: a robust (heavy-tail) Σdy² estimator (BACKLOG; moot unless H_out passes) and
  sequential (quantized-prefix) Hessians beyond the writers' sibling-sequential form (#87).
- **Corpora:** calib pools for every decision (`AGENTS.md`); xcorp v1 (96M tokens from the calib's upstream
  sources, #61) for training / statistics and a share of search accept / reject; private agentic sessions
  (never published) the same way; turboderp's self-sampled calibration trace for this model.
  Preference for training / statistics data: generated by the model itself from calib-like prompts, then more of
  the calib's upstream sources, then generic text. **[2026-11-06]** For V3's quantizer statistics neither paid on the
  board's metric (#91 self-generated, #101 xcorp): they stay calib (published H_in, H_out from qad windows).
- **Donated GPU time** (`rad-quant-at-home` 0.3, built and tested, not deployed; runs the ladder once the settings
  freeze): per-tensor and per-layer quantization jobs for the ladder and `eval_mix` jobs for the search (public
  slices only; mixes assembled locally from cached levels), a central queue, points by verified work. Results are
  verified by score: the server decodes the codes and checks the tensor's Hessian-weighted error against the band
  expected at that K (no re-runs). DDP / DiLoCo-style training only later.
- **YAQA:** full Sketch A / B lost on the 0.8B (#33-08); its diagonal H_out alone wins (#33-10…#33-16), which is the
  V3 candidate above. **27B: only −2.8% at K2.5 (#33-28), a third of the 0.8B's gain (two seed pairs pooled 0.982,
  2026-11-06: not yet a pass); not part of the pipeline for future / larger models** (user, 2026-11-05) unless a cheap estimator (≤ 384 sequences, bf16 hi/lo backward instead
  of fp32, a proper multi-GPU backward) brings it to a few GPU-hours and a pilot on that model shows it pays. Fuller (mixed) H_out only if the diagonal clearly helps on the 27B (BACKLOG).

## Data (after QAD)
- Clean held-out text runs out once QAD trains on the qad pool. Options: reuse documents by their ledger record
  (`eval/pools/ledger.jsonl`; measure the bias of reused search-pool docs first), new agentic traces generated in
  the corpus's style, openly licensed text matched per category, or asking the challenge author for more held-out
  text.
