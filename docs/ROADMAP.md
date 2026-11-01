# Roadmap

Names: EXL3-V1 = the turboderp-ladder port (released); EXL3-V2 = mixed-K EXL3 on the f16-activation kernel
(our K1–8 quants, the per-tensor mix chosen by a fitted sensitivity model and an exact knapsack DP);
EXL3-V3 = integer compute.

## EXL3-V2 (current)
- Released 2026-10-31 (16 sizes, 1.5–9.0 bpw). Streaming mix search on calib fit slices; final pick of measured
  mixes on a held-out verify slice. The public corpus was not used.
- Turboderp's ≥5-bit tensors allowed where they measure better (most at 5.5, 6.0 and 8.0 bpw).

## EXL3-V3
- int8 / int4 activations per tensor (`v3/`, design in `docs/DESIGN-v3.md` (earlier notes: `docs/DESIGN-new-format.md`), log in
  `docs/EXPERIMENTS-new-format.md`), a solver trading KLD against measured kernel time.
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

## EXL3-V4: variety pack
Variants that matter to some users but not to the main frontier, built on V3 / V3.5's tooling:
- 7.0 / 7.5 / 8.0 rebuilt with V3's embedding and head (t6.25–t8 embedding levels).
- **Decode speed as its own axis.** Decode time is not proportional to size: trellis dequant and other per-token
  costs add overhead (measured nearly flat in K below ~4 bpw: 32.5 tok/s at 1.40 bpw vs 24 at 6.00, DESIGN-v3 §9).
  First profile where decode time goes per K and fractional pattern (and whether an int8-activation decode GEMV
  helps), then decode-optimized variants. Longer term, treat the frontier as 4-D (size, KLD, decode, prefill): for
  the 27B dense model decode tracks size apart from the embedding, but for MoE models they can differ a lot.
- Speed variants per size (fast / balanced from the final V3 fit), A4 variants if V3.5 shows a useful point.
- Vision tower and DFlash2 drafter in the containers, if not already in V3.
- **REAL-Q (arXiv 2609.00049), tested before QAD:** a measured trial on a few sizes, so V5 starts from the best
  non-trained quantizer.

## Data (any time after V3)
- Clean held-out text runs out after QAD (V5, trains on the qad pool). Options: reuse documents by their ledger record
  (`eval/pools/ledger.jsonl`; measure the bias of reused search-pool docs first), new agentic traces generated in the
  corpus's style, openly licensed text matched per category, or asking the challenge author for more held-out text.

## EXL3-V5
- QAD / scale-only healing: not before V3 is done.
