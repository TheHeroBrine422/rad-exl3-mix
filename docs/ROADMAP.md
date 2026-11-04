# Roadmap

Names: EXL3-V1 = the turboderp-ladder port (released); EXL3-V2 = mixed-K EXL3 on the f16-activation kernel
(our K1–8 quants, the per-tensor mix chosen by a fitted sensitivity model and an exact knapsack DP);
EXL3-V3 = integer compute.

## Scope decisions (user, 2026-11-04; supersede the version sections below where they differ)
- **Into V3 now:** sub-1 K (K0.125–0.875); the low-bpw research ladder (2.0 → 1.0 in 0.1 steps, then down until
  KLD blows up); research sizes 1.25 / 1.5 / 1.75; A4 built and measured end to end (fast variant at ≤ 2.25 bpw if
  d1106 passes); saturation-aware allocation for the research sizes; sub-1 trellis embedding and t6.25–t8 embedding
  levels (menu work; releasing 7+ bpw is optional); all three IQR-V4 ideas (act order in the trellis, bf16
  (row, 128) scales on attn_output / ssm_out from K5, vocabulary-weighted head vs our slab head); fast + accurate
  variants per size; vision tower + DFlash2 in the containers.
- **0.8B tests now:** robust (heavy-tail) Σdy² estimator; sequential (quantized-prefix) Hessians; REAL-Q.
- **Search-algorithm improvements** for the tensor search: after / in parallel with the quantization runs.
- **Later:** decode speed as its own axis (profile per K, decode-optimized variants; maybe not V3); 7.0–8.0 rebuilds
  (optional extra, V3.5 at most); disaggregated quantization; KV-cache work (V-mean, Hadamard KV, per-layer KV
  recipe, KV-QAT); LSA; MoE models (QFN / GLM-5.3-Flash / Mistral).
- **QAD:** after V3, starting from the published V3 checkpoints.

## EXL3-V2 (current)
- Released 2026-10-31 (16 sizes, 1.5–9.0 bpw). Streaming mix search on calib fit slices; final pick of measured
  mixes on a held-out verify slice. The public corpus was not used.
- Turboderp's ≥5-bit tensors allowed where they measure better (most at 5.5, 6.0 and 8.0 bpw).

## EXL3-V3
- int8 / int4 activations per tensor (`v3/`, design in `docs/design/v3.md`, experiments in
  `docs/experiments/`), a solver trading KLD against measured kernel time.
- An int8-native trellis code (the Viterbi searches the exact integers the int8 tile multiplies) and a row-scale
  fold in the prefill tile.
- Quantizer settings swept per tensor family and K during the V3 bring-up, before the requant (`quantize/damptest.py`,
  `quantize/gaintest.py`): GPTQ/LDLQ damping (lower for the attention
  and delta-net input projections), scale gain (larger for attn_output / ssm_out at ≥5 bits).
- Finer K (2026-10-31), all from one generalised trellis whose per-weight shift follows a repeating
  pattern (2,3,2,3 = K2.5; 2,3,3,3 = K2.75; any rational K = a/b):
  - quarter steps K1 … K5, half steps K5 … K6.5 (sub-1 K: V3.5);
  - the same quarter steps for the embedding and, if row slabs do not pay, the LM head;
  - row-by-row K for the LM head (rows sorted by sensitivity), under test.
  Expected (knapsack discretisation, exponential D(R)): whole → half steps ~−5% KLD, half → quarter ~−1.5% more,
  quarter → continuous < 1% more; row-slab K ~0.05 bit on large tensors (E3c). Each extra K is a quant run;
  kernels take the shift pattern as a parameter.
- Per-row (vocabulary-tiered) embedding: **dropped** (2026-10-31). Early measurements: low-bit tails cost more
  than token frequency predicts and hurt multilingual text (+40% KLD on the guard slice), for an estimated 0–3% gain.
- Backlog: 2-tier head (~2%), residual re-ranker with more data.
- Vision tower and DFlash2 drafter in the containers (V2 is text-only).

## EXL3-V3.5 (between V3 and QAD)
- **Finer sizes below 2.5 bpw.** The released steps 2.5 → 2.0 → 1.75 are wide where KLD changes fastest (0.067 →
  0.164 → 0.297 on the board). Add release sizes in between (2.25 at least; 1.875 if it earns its place), and the
  trunk levels they need: K1.25 / 1.5 / 1.75 in the ladder (each ~3 card-hours; today they are interpolated from
  V2's K1, the level the fit under-predicts most). Measure K1.5 with a few swaps first.
- Sub-1 K (K0.125–0.875 at 1/8 steps, a salted 0-bit read in the quantizer and every decode kernel), for the < 2 bpw
  sizes. No K0 (dropping a tensor): it saves only 0.125 bit a weight over K0.125 on that tensor, for a far larger
  error (user, 2026-11-01). Before QAD because low-bit quants are where healing should gain
  the most, so they are the natural QAD targets and their menu should exist when QAD (V5) starts.
- Embedding menu at both ends: sub-1 trellis embedding (t0.25–0.75) for the smallest sizes, once t1 / t1.25 show
  whether the < 2 bpw solves sit at the menu floor; t6.25–t8 for the 7.0 / 8.0 releases (at 7.0 bpw the solver
  already sits on t6, the menu top). CPU sweeps (`v3/e3/emb/run4.sh` pattern).
- **Low-bpw research ladder:** release-style solves from 2.0 down to 1.0 in 0.1 bpw steps, then 0.9, 0.8, ... until
  KLD blows up (1.2 is possible with V3's K1 + 1/8 levels alone; ~1.0 and below needs sub-1). Mechanical floor with
  K0.125 everywhere ~0.15-0.19 bpw; the meaningful knee is expected around 0.8-1.0. Scored on our own held-out slices;
  only a few points go to the board. Quantizer support for sub-1 (0-bit steps) is built in V3's kernel pass.
- **A4 (int4 activations), measured in full.** Dropped from V3 on the solver's estimate (≤ 14% of FLOPs at
  x1.88 → x1.94 for +2–5% KLD), but never built or measured end to end. Build the iu4 path, measure real speed and
  KLD per size, and publish the frontier whatever it shows.

- **Sequential (quantized-prefix) Hessians** (somewhere V3.5-V5; user, 2026-11-01): calibrate each layer on inputs
  from the already-quantized prefix, as upstream exllamav3 does; most relevant at K2-3. Caveat: our own collections
  have far fewer tokens than the published 10M-token Hessians, so the test must beat that noise.

## EXL3-V4: variety pack
Variants that matter to some users but not to the main frontier, built on V3 / V3.5's tooling:
- 7.0 / 7.5 / 8.0 rebuilt with V3's embedding and head (t6.25–t8 embedding levels).
- **Decode speed as its own axis.** Decode time is not proportional to size: trellis dequant and other per-token
  costs add overhead (measured nearly flat in K below ~4 bpw: 32.5 tok/s at 1.40 bpw vs 24 at 6.00, design/v3.md).
  First profile where decode time goes per K and fractional pattern (and whether an int8-activation decode GEMV
  helps), then decode-optimized variants. Longer term, treat the frontier as 4-D (size, KLD, decode, prefill): for
  the 27B dense model decode tracks size apart from the embedding, but for MoE models they can differ a lot.
- Speed variants per size (fast / balanced from the final V3 fit), A4 variants if V3.5 shows a useful point.
- Vision tower and DFlash2 drafter in the containers, if not already in V3.
- **Disaggregated quantization (DQ, arXiv 2609.26333):** different compute formats (and optionally different weights)
  for prefill and decode. The shared-weight form is what V3 already does (int8 activations only for prefill steps of
  M >= 512, decode stays f16-activation); worth reading for their accuracy data. The separate-prefill-weights form
  (compute-native prefill weights, streamed from SSD: "ODP") is a serving variant, not a board entry (the board
  counts every stored tensor).
- **Speed gap to IQR-V3, for the record (user, 2026-11-01):** decode (~30 vs ~100 tok/s at TP1) is mostly
  speculative decoding: IQR-V3 ships DFlash2, V2 does not -- DFlash2 support (planned for V3) closes most of it.
  Prefill (~1,500 vs ~2,500 tok/s) is the GEMM time (`gemm_us`), which V3's int8 path targets.
- **REAL-Q (arXiv 2609.00049), tested before QAD:** first on the proxy. Its correction depends on the exact mix
  (sequential, per-mix), so it cannot enter the per-level library: at most a final pass on 2–3 shipped sizes
  (~40–50 GPU-hours each at 27B). Its reported gains shrink with model size (−49% at 1.7B, −5…−9% at 8–70B), so a
  proxy win overstates the 27B gain.

## Research method and scale (after V3)
- **Small-model proxies: Qwen3.5-0.8B and 2B for general testing, 9B when a test needs it.** Same family and block
  design as the 27B (3:1 Gated DeltaNet / gated attention, head_dim 256, same 248,320 vocab; Qwen3.8-27B is this
  architecture with different weights). Sizes (trunk / embedding + head share): 0.8B 0.50B / 34%, tied; 2B 1.37B /
  27%, tied; 4B 3.57B / 15%, tied; 9B 6.92B / 23%, **untied**, DeltaNet value heads 32 and attention 16/4 (closest
  shapes); 27B 24.35B / 9%, untied.
  - **0.8B and 2B:** most screening (codes, codebooks, LDLQ variants, damping, sequential Hessians, rotations, QAD
    recipes, the low-bit ladder's method ranking, search / stream runs, pipeline tests). The 0.8B where speed or
    memory decides (anything with gradients or output-side statistics on the 4070, e.g. the YAQA run, #33-08), the 2B
    where the extra width matters. Not embedding / head work: both are tied, the head *is* the embedding.
  - **9B (~3.5x smaller trunk than the 27B, untied):** embedding / head work, and the step between a small-model win
    and a 27B run when the test needs it (closest shapes; gradients need a 32 GB card).
  - Nothing transfers as numbers (allocation, sensitivities, kernel speed, KLD per bpw); method rankings should.
  - Calibration first, pass rule fixed beforehand: replicate known 27B results (mixed vs uniform at ~3 bpw,
    damping-sweep gain, int8 codebook cost ~0, the fractional-level curve, output scales off = worse, the YAQA
    gates) before trusting a screen (2B study: `proxy2b/CALIBRATION.md`, paused).
- **Hardware: the user's RTX 4070 (12 GB, CUDA)** for the 0.8B and 2B (built: `proxy2b/`, goldens, fake-quant evals
  through our quantizer on CUDA, input Hessians over all calib). The 9B does not fit there (18 GB bf16); it needs an
  R9700 slot or offload.
- **A larger corpus for proxy research** (the calib's 10.26M tokens bound Hessians and QAD). Built: xcorp v1, 96M
  tokens from the calib's upstream sources (#61; proxy research only). Evaluation stays on
  the calib pools (the private corpus is drawn like calib). Training / statistics data, in order of preference:
  (1) generated by the bf16 model itself from calib-like prompts (no overlap with any test set); (2) more documents
  from the calib's own upstream sources (could overlap the private corpus: fine for proxy research, but check the
  challenge rules before anything trained on it ships); (3) generic text (distribution shift). Target ~100-150M
  tokens. For QAD the teacher runs on the fly; stored reference logits only for eval slices.
- **YAQA on the 0.8B (2026-11-02): full YAQA loses, H_out alone wins.** The paper's Sketch B as a whole stays worse
  than one-sided LDLQ at 1K–8K sequences (#33-08), but the loss comes from its gradient-weighted H_in: with our calib
  input Hessian kept and only Sketch B's H_out added, K3 is −9.9% vs LDLQ and −4.3% vs our tuned quant (z −5.4,
  #33-10). On top of our tuned settings: K3 −9.7%, K2 −5.6% vs our tuned quant (#33-11). The diagonal of H_out alone
  keeps the gain (3-seed means: K3 −8.1%, K2 −5.4%, #33-16), so the 27B collection is diagonal-only (~10 MB of accumulators,
  one pass, 1K sequences suffice on the 0.8B: ~8.5 h fp32 on both cards, #33-13); then a 27B swap test before V3
  changes.
- **Donated GPU time:** quant creation is single-card and embarrassingly parallel (a container with the job
  hardcoded); the tensor search can run in batches (~100 evals, then a refit here), or against a coordination server,
  so contributors never give shell access.
- **Larger / MoE models (Qwen Flash, Qwen4):** need the bf16 reference logits and input Hessians of the full model --
  feasible with SSD-streamed bf16 weights (the challenge author did it that way) or cloud hardware; quantization itself
  can then run locally. Most of this pipeline applies to any recent LLM; MoE needs per-expert handling.

## Data (any time after V3)
- Clean held-out text runs out after QAD (V5, trains on the qad pool). Options: reuse documents by their ledger record
  (`eval/pools/ledger.jsonl`; measure the bias of reused search-pool docs first), new agentic traces generated in the
  corpus's style, openly licensed text matched per category, or asking the challenge author for more held-out text.

## EXL3-V5
- QAD / scale-only healing: not before V3 is done.
- **0.8B proxy results of 2026-11-03 (each needs a 27B check: BACKLOG "27B confirmation queue"):** sibling-sequential
  asym target for the writers (K3 −14%, K2 −18%, #87); fixed-code scale refits (K2 −9%, #85); frozen-code healing
  (K2 −38%, #89; needs the user); gen-masked H_out loss (#83). i8c's gain vanishes under H_out at K3 (#78-03).
- Field reports to start from: QAD on an EXL3 quant gained ~6-13% KLD overall at a 1,024-token window, ~22% when
  trained only on the last 16 layers (48-63; trellis codes interfere with training earlier in the stack); a 2,048
  window was out of memory. Ideas to test on a small model first: QAD before the trellis step (train continuous
  weights / scales, then quantize), and rotations other than the 128-point Hadamard (e.g. ParoQuant's).
- possibly do QAD on a completely separate corpus, likely test that on a small model too
