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
  - quarter steps K1 … K5, half steps K5 … K6.5 (sub-1 K and K0: later, < 2 bpw only);
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
- Sub-1 K (K0.25–0.75, a salted 0-bit read in the quantizer and every decode kernel) and K0 (drop the tensor or a
  small low-rank stand-in), for the < 2 bpw sizes. Before QAD because low-bit quants are where healing should gain
  the most, so they are the natural QAD targets and their menu should exist when V4 starts.
- First check (CPU): proxy-priced K0.5 / K0.75 options in knap4 at 1.5–2.5 bpw; build them only if they earn picks.

## EXL3-V4
- QAD / scale-only healing: not before V3 is done.
