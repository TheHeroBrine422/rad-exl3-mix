# w03-probe — A10: sqrt-Fisher logit probe instead of the sampled label (diagonal H_out collector, 0.8B)

**Question.** The collector's backward starts from a sampled label: dL/dlogits = p − e_y, y ~ p. Its covariance is the
Fisher diag(p) − ppᵀ, but each draw puts all its mass on one token, so a confident position contributes a very
spiky, high-variance gradient. A random probe d = √p⊙ξ − p(√pᵀξ) (ξ Gaussian or Rademacher) has exactly the same
covariance and spreads over the vocabulary. Does it give the same diagonal H_out with fewer sequences — enough that the
27B night (#33-13, ~8.5 h at 1K sequences on both cards) can be cut to 256–512 sequences? (ranked.md A10; INDEX #33-13,
#33-17 stopping rule, #33-18 dose.)

**Mechanism / what it can and cannot fix.** The single-sequence variance of a diagonal estimate has two parts: label
draws (removed in part by the probe) and window-to-window variation (untouched). `probevar.py` measures both on real
0.8B distributions (CPU); the effective-dose factor it reports, (ν_sample + ω)/(ν_probe + ω), predicts how many
sampled sequences one probe sequence is worth. The GPU run then measures the same thing on all 150 linears directly
(split halves, no quant noise) and in KL.

## Files
- `collect-probe.patch` — against `v3/yaqa/collect.py`: `YAQA_PROBE=gauss|rad` (with YAQA_DIAG or YAQA_FULL). A
  checkpointed `chunk_probe` whose loss is Σ z·stopgrad(d), so dL/dz = d; ξ comes from a `torch.Generator` seeded by
  (SEED, window, chunk), so checkpoint recompute and resume redraw the same ξ. In place, ~3 × the chunk's logits of
  transient memory (as the sampled path). The logged loss/tok becomes Σ z·d (not a CE). Off by default; no-op when off.
- `probetest.py` / `probetest.out` — CPU self-test (covariance, the patched function's gradient, checkpoint
  bit-identity, no-op diff analysis). `run.sh` step 0 adds the GPU no-op check (bit-identical hdiag, 8 windows).
- `probevar.py` / `probevar.json` / `probevar.txt` — the real-data CPU variance study (colo).
- `probecmp.py` — precision of the probe collection vs c18 (halves, cross-cosines, effective dose).
- `seedsumm.py` — 3-seed summaries (same as w03-swiglu-diag's).
- `run.sh` — the 4070 entry point.

## Arms (K3 / K2, rotation seed offsets 0 / 1 / 2, base settings + diagonal H_out, output damping 0.01, i8, g08-search1)
| arm | diagonal | source |
|---|---|---|
| `base` | none | existing (#33-16) |
| `eB`, `eB512`, `eB256` | sampled labels, 1024 / 512 / 256 seqs (c18) | yaqaFact7 (#33-18); built here if missing |
| `eP256`, `eP512` | probe, 256 / 512 seqs, **the same windows as c18** | new (12 quants); `PROBE_Q1024=1` adds `eP1024` |

## Draft pass rule (for the lead to commit before data)
- **P1 (precision, noise only, decisive for "the probe is cheaper"):** the probe's split-half median cosine at n = 512
  ≥ the sampled collection's at n = 1024 (`probecmp.py`: effective dose ≥ 2×), and its p5 ≥ the sampled p5 at 1024.
- **P2 (KL):** at both K, eP_N's mean gain vs base ≥ 85% of eB's (1K sampled) mean gain, and every eP_N seed beats every
  base seed.
- **P3 (no loss at equal dose):** mean KL(eP_N) ≤ mean KL(eB_N) + 1 base per-seed SD at both K.
- **Adopt** (27B collector runs with `YAQA_PROBE`, the 27B stopping rule of amendment 10 unchanged — it is a precision
  rule, so the run stops earlier by itself) if P1 holds and P2 + P3 hold at N = 512 (or 256). If P2 also holds for the
  sampled arm at the same N (#33-18's A5 answer), the probe is still adopted when P1 holds, because the 27B stopping
  rule is in halves units, where the probe gets there with fewer sequences.
- **Reject** (keep sampled labels) if P1 fails, or P2 / P3 fail at both N.
- **CPU pre-gate:** if `probevar.json`'s Sketch B effective-dose factor for the better probe is < 1.3, P1 is not
  expected to pass; the lead may skip the GPU run (then: A10 rejected on CPU evidence).

## What it changes
Adopt → 27B diagonal night ~2× (or more) shorter at the same precision (≤ ~4 h of both cards instead of ~8.5 h), at
no code cost beyond this patch. Reject → nothing changes. Either way the probe is the cheaper estimator for any later
collection (gen-masked A1, operating-point A12) if it passes.

## Running
`bash proxy2b/exp/w03-probe/run.sh` on the 4070 after #33-18's c18 collection. `PROBE=rad` (default; lower variance
than Gaussian, see probetest.out / probevar.txt) or `PROBE=gauss`. Idempotent; resumable (the collection resumes from
hdiag-state.pt); results in `/data/yb/w03-probe/result.txt`. Before the merge it runs a patched copy in
`/data/yb/w03-probe/src/`; after the merge it uses the repo's collect.py.

**GPU time:** no-op gate ~4 min; collection 1,024 windows ≈ 25 min (c18 ran 1.3 s/window at peak 4.8 GB);
12 quants + evals ≈ 50 min; plus eB256 / eB512 / eB controls (≤ 18 quants, ≈ 75 min) only if yaqaFact7 has not made
them. **≈ 1 h 20 min.**

`run.sh` stops after step 2 (no quants, ~30 min of GPU in all) when P1 fails; `FORCE_Q=1` runs the quants anyway.

## CPU results (2026-11-02): the probe is not expected to help
- `probetest.out` (synthetic): the probe has the right covariance, and **per position** it is much less noisy than a
  one-hot label on a peaky distribution (CV of vᵀddᵀv 1.2–1.4 vs 6.2; equal on a flat one).
- `probevar.txt` (real 0.8B, 8 calib qad windows × 1,023 positions, 8 draws per estimator, mean entropy 0.9–3.0
  nats): for **Sketch B's** per-sequence diagonal ‖Σ_t g_t[j] x_t‖², the label-draw variance is the same for all three
  estimators: ν = 0.586 sampled, 0.599 Gaussian, 0.563 Rademacher (relative to ‖D̄‖²). Window-to-window variance:
  ω = 0.335. **Effective-dose factor 0.99 (Gaussian) / 1.03 (Rademacher)**, far below the 1.3 pre-gate. The means
  agree (cos 0.991; the gap is the noise of 64 draws).
- Why: G_b = Σ_t dy_t x_tᵀ sums ~2,000 independent per-token draws. By the central limit theorem, each token's
  kurtosis (the one-hot label's problem) averages out of the sum, and only the per-token covariance is left. That
  covariance is the same for the label and the probe by construction. What remains is the chi-square noise of
  squaring a sum, which only more draws per window (each a backward pass) or more windows reduce. At ν/ω ≈ 1.75, an
  extra draw on the same window costs ~0.9 of a new window's time (0.8B: fwd 0.5 s, bwd 0.9 s) and buys less, so
  multi-draw is not worth it either. For the per-token sums (the xw / dy estimators) the label noise is negligible
  anyway: ν ≈ 0.005 vs ω ≈ 0.32.
- Caveat: the test linear is the last layer only (head gradient, RMSNorm Jacobian left out). The CLT argument does not
  depend on depth. It would fail only if a handful of positions dominated a sequence's G_b.
- **Recommendation:** A10 rejected on CPU evidence. If the lead wants the confirmation on all 150 linears, run
  `run.sh`: it stops after P1 (~30 min of GPU) unless P1 passes.
