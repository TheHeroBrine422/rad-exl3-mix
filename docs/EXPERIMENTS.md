# Experiment log — what we tried that did not work (Qwen3.8-27B, EXL3, radiance)

Negative and null results, kept so nobody repeats them. One entry each: what was tried, what happened, the
evidence, and what we do instead. Newest first within each section. Metric everywhere: mean KLD vs bf16 (challenge
golden, top-512); "calib-A" / "calib-B" / "ver1" are calib slices (pools: `eval/pools/`, `docs/METHOD.md` §6). Add to this file whenever an
idea is dropped.

## Quantization / format

- **YAQA output-side Hessians from sampled positions** — 2026-10-31. *Idea:* two-sided LDLQ with H_out = Σ w dy dyᵀ
  from the model's backward pass (YAQA), collected as rows r = ‖x‖·dy at 32 random positions per 2048-token
  sequence (fit pool). *Test:* blocks 22+23 requantized two-sided vs one-sided at K2/K3, spliced into V2 6.00, paired
  KLD on golden-ver1 (124k positions). *Result:* (a) bf16 backward, 200 seqs: two-sided +23% (K2) / +28% (K3) worse
  increment; the bf16 gradients are noisy toward early layers (down_proj per-row cosine vs fp32 p10: 0.999 at layer
  63, 0.02 at layer 20) and the bad rows are ~3x too large, dominating H_out. (b) fp32 backward, 200 seqs:
  +17%/+19% worse at sigma_reg_out 0.025, +6%/+10% at 0.3 (z ≥ 2); more output regularisation keeps helping, i.e.
  the trend is toward one-sided. *Conclusion:* bf16 backward is unusable; sampled-position H_out (6,400 rows) still
  hurts. Not yet tested: full-token sums (upstream sums every token; H_out is dominated by few high-gradient tokens)
  — `v3/yaqa/collect.py YAQA_FULL=22,23`. YAQA is dropped if that fails too. `v3/yaqa/`, tracker 2026-10-31.
  (c) **full-token fp32 H_out, 200 seqs: still worse** (+30%/+21% at 0.025, +7%/+5% at 0.3; worse than sampled).
  *Conclusion:* two-sided YAQA with exllamav3's ldlq_2hess does not help our trellis quants; dropped. Not tested:
  the paper's Sketch-B estimator with far more data (64K sequences), which we cannot afford.
  (d) **diagonal output scales** (gate 2d, exllamav3's sensitivity_out_scales from diag(H_out)): K2 +10-15%, K3 +4%
  (no auto-vs-always confound: D1, code reading). (e) **gradient-weighted input Hessian** (gate 2e, H_in =
  sum ||dy||^2 x x^T, 100 seqs, one-sided LDLQ, 2026-11-01): K2 +44% (z 10), K3 +31% (z 12); far worse than the plain
  Hessian, so the pre-set blend follow-up was skipped. **Every backward-based Hessian variant is dropped** (2026-11-01).
- **Best-of-N rotation seeds** — 2026-10-31. Pick each tensor's rotation sign seed from 4 by proxy. Seed spread 0.5%,
  not reproducible across row samples (corr +0.22), honest gain ~0. Dead. `v3/e3/seedtest.sh`.

- **Per-group gain scales (finer than EXL3's per-row + per-column scales)** — 2026-10-30. *Idea:* IQR-V2 uses
  E4M3 scales per (row, 128); maybe that is their low-bit edge. *Test:* in EXL3's rotated domain (128-Hadamard both
  dims, random signs) after per-row/col RMS normalisation, 0.5·log2(AM/GM) of group variances (= bits a per-group
  scale saves at high rate), sampling bias removed, 23 layers × all projections + head. *Result:* 0.000–0.0025
  bit/weight (MLP ≈ 0, worst tensor 0.006). *Conclusion:* dead; rotation + row/col scales already flatten the
  variance. Not tested: Hessian weighting, heavy tails, K ≤ 2. `quantize/amgm.py`, `logs/amgm.jsonl`.
  *Retest 2026-10-31* (`v3/e3/gscale.py`, Hessian-weighted, real trellis + LDLQ, K0.5–2, G64/G128 8-bit scales):
  raw gain 1.3–1.8% at equal K, so groups lose 8–20% at equal bits. Dead at low K too.
- **Trellis (EXL3-style) token embedding instead of Lloyd** — 2026-10-29. Measured 4-bit Lloyd at ≈ +2% KLD at
  4.00 bpw vs IQR's trellis embedding ≈ +0.6% → at most ~1.5% total to win, and it needs a row-wise trellis gather
  kernel. *Deferred*; only reconsider for vocabulary-tiered embeddings (TODO: simulate both first).
- **2-bit Lloyd embedding** — earlier embedding sweep: clearly bad vs 3-bit at normal sizes. Still offered to the
  optimizer for the research floor (1.2–1.5 bpw) only.
- **Head above K6** — 2026-10-29. Our K7/K8 heads have 5×/16× less proxy error than a 6-bit head but the KLD gain
  over a 6-bit head is within noise (~−5% vs the 5-bit head for all of them). Head bits past K6 are wasted except
  maybe at 8+ bpw.

## Quantizer speed (`ourq.py` on exllamav3 1.6.3, gfx1201)

- **Two quantizer processes per card** — +14% throughput only (GPU already at 100%), and memory fragmentation risk
  (head quants OOMed before). One process per card.
- **Bigger tile batches** — per-tile trellis cost is flat from 64 to 8,704 tiles per launch; nothing to gain.
- **Sharing work across K values in one run** — the K-independent part (Hessian prep, LDL, Hadamard) is ~5 s of
  ~35 s per layer → ≤ 15%. Not worth the restructure.
- **K7's slow trellis kernel** (~40 µs/tile vs ~15 for K4/K8) — not fixed: needs an exllamav3 extension rebuild.
  (What did work: dropping the diagnostic `block_trace`, 79 → 35 s/layer bit-identical; `EXL3_QT_OPTIMIZED=1`,
  identical indices, K1 2.2×, K6 1.26×.)
- **Head quantization OOM at K5/K6** — `refit_scales` allocated full-size temporaries; fixed by an in-place chunked
  rewrite (`/data/quantize.py.orig` is the original). Head quant still needs ~22 GB host RAM.

## Eval speed (`radiance-kldgold-fast`)

- **Bigger prefill steps** — `--max-num-batched-tokens 4096` is a no-op (chunks are capped at the 2,048 GDN
  checkpoint interval); `--prefill-span 2` and `--checkpoint-interval 4096` run 4,096-token chunks but take the same
  118 s with the same KLD. 92% of GPU time is the EXL3 GEMMs at ~40% of peak; the fix is the prefill GEMM rewrite.
- **rocprofv3 on radiance** — unusable: memory-copy tracing breaks the load, kernel tracing makes the recorded-launch
  replay fault. Use radiance's own `--profile-ops`.
- **Card 1 evals ~25% slower** (152 vs 119 s): the full-vocab f32 logits copy over its PCIe3 x4 link. Async copy +
  background scoring would fix it (~1–1.5 h engine work); not done yet.
- **TP2 for whole-corpus runs** — card 1's x4 link makes it slow; TP2 only for documents that do not fit one card.

## Mix allocation (knapsack / surrogate model)

- **First knapsack (`knap.py`): band sensitivities from big K=2 swaps, linear model, Lagrangian solver** —
  2026-10-29. The K=2 swaps overstate slopes ~4× (only relative values usable); a power-law (P) variant blew up and
  was dropped; a units bug in the head/embedding cost gave every mix a 3-bit head. Fixed version: ~−3.5% vs the
  reference containers on calib-B on average, and the 2.00 bpw mix was +68% *worse* (non-additivity at low bpw).
- **Symmetric "depth compounding" term** (kfit.py: cost_w·(1 + c·cost of earlier layers)) — it is algebraically a
  global quadratic link on the total, not a depth effect; it improved CV only by fixing the overall level, and its
  solver step biased against late layers. Replaced by the directional model (error amplified by *downstream*
  error, kfit2.py): CV log-RMSE 0.28 (additive) → 0.19 (link forms) → 0.11–0.13 (directional).
- **Lagrangian multiple-choice-knapsack sweep** — exact only on hull points, lands well below targets (hence the old
  "3.99" targets), cannot handle the directional term. Replaced by an exact byte-bucket DP. Pitfall in the DP's
  first version: 4 MiB buckets rounded *up* per tensor left ~0.1 bpw unused per mix → 1 MiB, nearest rounding,
  exact-size check.
- **Treating all quantizer families alike** — turboderp's "plain" exports cost ~1.15–1.23× the KLD of "SC" ones per
  unit of Hessian proxy error; the proxy is not quantizer-neutral. Fit a factor per family (and per K range for ours).
- **Measuring single-tensor swaps end to end** — paired SE on ~100k positions is ~1.5%, a single tensor's effect is
  ~0.1–0.5%. Use group swaps, whole mixes, or random multi-tensor perturbation designs.
- **Noise-injection sensitivity calibration** (HIGGS / Q-Palette / exllamav3 style) — not cheaper for us: real swaps
  are 0.2 s reflink mixes, and noise injection would need a bf16 forward that does not fit one card.
- **Shapley-value interactions (IMPQ, SLQ)** — literature puts interactions at ~1% for ~15–18 h per 8B model. Skipped.
- **RCO** (Riemannian constrained optimisation, arXiv 2605.00649) — needs a differentiable torch forward+backward
  of the full 27B on ROCm, 800–3,200 passes per size, 0–4% over measured-loss search. Skipped; the "one backward pass =
  in-context sensitivity of every tensor" idea is kept as an optional spike (`research-rco.md`).
- **Gradient boosting / generic black-box surrogates (XGBoost, full GP BO)** — not usable: the exact solver needs a
  per-tensor-decomposable model, and black-box BO over ~340 categorical dims needed ~10k evals in AMQ.

- **Per-layer edge offsets in kfit2** (KFIT_EDGE, 2026-10-31) — a free, shrunk log-offset per (group, layer) in
  layers 0–3 / 60–63 on top of the 8-layer depth curve, plus a variant with knots every 6. With all 47 single-layer
  K2 probes in the data, CV pair ranking 1614 / 1618 vs 1620 / 1898 for the plain model: no gain. The probe data itself
  (in the plain fit) carries the allocation signal, e.g. the L63 ffn_gate_up spike. Code kept behind the flag.
- **Residual re-ranker on top of kfit2** (2026-10-31, `search/rr/rr.py`) — learn log(measured / kfit2 out-of-fold
  prediction) from 102 summary features, re-rank candidates. Grouped 5-fold CV on 519 mixes: plain kfit2 log-RMSE
  0.118 / pair ranking 79.9%; ridge 0.157 / 76.4%; XGBoost (all measurements) 0.212 / 80.4%; XGBoost (mixes only)
  0.127 / 80.5% (+7 pairs of 1130: noise); a Gaussian process with ARD did not finish in 16 min. The residual (~±12%)
  is mostly slice / run noise, not learnable structure at this data size. Revisit with V3's larger dataset.
- **Layer 0 as a sensitivity spike** — the hypothesis was wrong. K2 on layer 0 ffn_down costs +0.00006 (≈ free); the
  spike is at the *end*: L63 ffn_gate_up K2 +0.0065 (80% of the base's KLD), L60–63 ffn_down +0.0006–0.0013.

- **Vocabulary-tiered embedding** (2026-10-31, `quantize/embtier.py`, data in the new-format log) — rows ranked by
  token frequency, top tiers at 3–4 bits, tail at 1–2 bits, fake-quantised on a bf16-embedding copy of sc4.00-e4,
  calib-A + a multilingual guard slice (67 fit-pool docs). Uniform: e2 +38% / e3 +8% / e4 +2% calib-A KLD over a
  bf16 embedding (multilingual +64 / +20 / +4%). Best tiered, 20k@4 + 30k@3 + rest@2 (2.41 bits a value): +11% /
  +28%: better than uniform at equal bytes, but against the e3 the mixes use it saves 0.034 bpw for +3% KLD, net
  ~1–2% at 3–4 bpw, while multilingual text pays more (+28% vs +20%). 1-bit tails are bad everywhere (+29–76%,
  multilingual +60–145%): rare-token rows matter far more than their frequency. A language-neutral ranking
  (min(calib rank, token id)) did not help the multilingual slice. Dropped (not worth hurting international
  text). The head version: simulated, ~2% (backlog).
- **GPTQ damping for our quants** (2026-10-31, `quantize/damptest.py`, `data/damptest-2026-10-31.jsonl`) — not a dead
  end, deferred: sigma_reg 0.025 → 0.005 changes per-tensor error (undamped H) by −2 to −2.7% on GDN in_proj_qkv / z,
  −3 to −6.5% on attention q/k/v (layer 3), −0.5 to −1.2% on o_proj, ~0 on ssm_out and the FFN (70% of weights).
  Est. ~0.5–1% whole-model KLD: not worth a V2 requant; use 0.005 for attention/GDN inputs in the V3 requant.

## Process pitfalls

- **Selecting on public** — never: public is held out (`docs/METHOD.md` §6). Verify-slice reuse also bakes in noise slowly.
- **`pkill -f` / `pgrep -f` with a pattern in our own ssh command** — killed the shell several times. Anchor patterns.
- **Queue claims by `mkdir` alone on the overlay filesystem** raced: two cards ran one eval. Claims now take a lock.
- **`golden_pick.py` with `splitlines()`** split documents on Unicode line separators; split on `\n` only.
- **fp8 KV cache** moves KLD; all measurements use bf16 KV.
- **Reference curve including dominated containers** (bigger *and* worse, e.g. plain 2.00/4.00/5.00/6.00) made mixes
  look better than they were; compare against the Pareto envelope only.

## R1 residual-stream rotation for int8/int4 activations (2026-10-31, dropped)
SpinQuant-style R1: one random orthogonal 5120² Q folded into the residual stream, the readers' norm gains (1 + w)
folded into their weights. CPU proxy on the K6 activation dumps (`v3/e1b/r1test.py`, all 208 residual-reader
linears, output-error energy vs today's block Hadamard with the quant's own suh): **A8-g128 6.6× worse**, A8
per-token 2.1×, A4-g64 5.7× (layers 0–15: 17× / 4.9× / 15×; 48–63: 1.3× / 0.66× / 1.3×). R1 does spread the
massive-activation channel (top channel's share of input energy 65% → 0.3% in layers 0–15), but this model's norm
gains shrink that channel, and R1 must fold the gains into the weights: the rotation then spreads a much larger
outlier over every 128-block, where today's per-block Hadamard keeps it inside one block's scale. Only per-token A8
in deep layers gains, and per-token is ~5× worse than g128 there to begin with. A learned Q (SpinQuant's Cayley
optimisation) would not close a 6.6× gap. Side result: the quant's learned input scales (suh vs signs only) lower
the A8-g128 error 5%, per-token 35%. Data: `v3/e1b/data/r1.jsonl`.

