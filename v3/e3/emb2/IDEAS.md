# emb2 · Better embedding (and head) representations for EXL3-V3/V4 — candidates, ranked (w83, #119, 2026-11-08)

Inputs read: #23, #32, #35, #64, #66, #97 (embedding), #34 + `v3/yaqa/data/{lcurve,fcurve-A}.json` (head), #113
(bigger trellis), #90 / `data/e3/sub1/r90knap.md` (low sizes), IQR-V1–V4 (V4 §2.4 head), the corpus digests on head /
embedding / transform coding (2608.12026 SoftWater, 2608.02703 ARCHead, 2609.31291 softmax shift, 2610.10385 OrBIT,
2610.31309 OPTIC, 2610.35985 GateRQ, 2608.14191 AATC, 2608.07915 SPECTRA, 2608.25188 transforms survey).
Rules for the screens: `RULES.md` (written before data, one amendment before the trellis runs).

## The fact that decides the ranking

The shipped embedding coder (fwht128 + one scale a row + mul1 trellis) already sits at **1.03–1.09× the iid-Gaussian
distortion-rate bound 2^-2R** on the real rows (#35 `embmenu.jsonl`: t1 0.257 vs 0.250, t2 0.0649 vs 0.0625, t3 0.0166
vs 0.0156, t4 0.00427 vs 0.00391; t0.25 0.681 vs 0.707). So **no better quantizer of the same source can buy more than
~0.03–0.06 bit/value** (that is why #113's 2^20 trellis is worth ~0.02 bit, and why VQ / lattice / LUT codes are not
listed as contenders). Real gains can only come from (a) structure in the source that a white-Gaussian model misses,
(b) a better **metric** (which errors the model actually feels), or (c) **where** the bits go (rows).

Measured today on all 248,320 rows (`specan.py`, `embnn.py`; data `data/emb2/`), (a) is closed:

| structure | measurement | best possible gain |
|---|---|---|
| covariance (KLT + reverse water-filling) | eigenvalue AM/GM of the unit-RMS rows **1.138** (centered 1.127); effective rank 2,448 of 5,120; top-256 directions hold 15% of the energy | **≤ 0.093 bit** (Gaussian bound, flat in R), minus an int8 5120² rotation = 0.165 bit: **net < 0** |
| per-128-block energy (what one scale a row misses) | CV 0.126 median vs 0.125 for iid rows | 0 |
| mean row | 1.0% of the energy | 0.007 bit |
| the head predicts the embedding (untied, but trained together) | least squares E ≈ Hd·B, full 5120² B: **85% of the energy left** | ≤ 0.12 bit before B's 0.165 bit: net < 0 |
| shared rows / clusters (row-level VQ, "parent" rows) | best other row: median cos² **0.05**, p99 0.33; best 8 rows by least squares leave 86% | ≤ 0.11 bit before pointers and the parents' own noise |

The table behaves as a white Gaussian matrix to second order and has no near-duplicate rows. That leaves the metric
and the row allocation, and for the head the existing row-Fisher route.

## Candidates, ranked

### 1. Embedding LDLQ against a measured sensitivity metric (H = M instead of H = I) — **screened: FAIL on the CPU proxy (the proxies disagree); needs the true Fisher (GPU)**
- **Mechanism.** Same format, same decoder, same bytes: the embedding rows are quantized like a trunk tensor (fwht128
  basis, 16×16 tiles, block-LDLQ feedback across columns) against a 5120×5120 metric M that says which directions of
  an embedding error the model reads, instead of H = I. Every row gets the same treatment, so the rare-row trap of
  #32 does not apply. Rows are independent, so the per-row scale stays.
- **Why it could win.** The layer-0 reader metric M0 = diag(g)·ΣWᵀW·diag(g) (layer 0's mixer and MLP input
  projections, norm gains folded) weights the embedding's top-256 principal directions **2.7×** the average, bottom
  ones 0.64× (correlation with log λ 0.39). Gaussian bound in M0: water-filling saves **0.31 bit**, of which LDLQ
  alone (no rotation, AM/GM of M0's eigenvalues 1.35) can take **~0.2 bit**. The measured diagonal Fisher of the
  residual at layer 0 (27B H_out `hdiag-dy`, layers.0 out_proj, qad pool) is less anisotropic on its diagonal (AM/GM
  1.14 ≈ 0.1 bit) and **disagrees with M0** there (M0's diagonal is flat, AM/GM 1.003; F has channels at 18× the
  mean): the off-diagonal part of the true Fisher is unknown.
- **Runtime.** None (quantizer only). `ourq.py` already does this given an H; the embedding just needs `embq`/`embgpu`
  to call LDLQ with a 5120² H.
- **Cheapest test.** CPU: arms H ∈ {I, M0, F, MF} × K {1, 1.5, 2, 3} on 4,096 rows, errors in all three metrics
  (`embldlq.py`; result in README: each of M0 / F wins only in its own metric, MF wins in both but F by only 3–5%). Decisive next step (GPU, lead): the **full** residual-gradient covariance at layer 0 (one 5120² Σgᵀg,
  sampled labels as collect.py, fit pool, ~1–2 card-h) — then H = F_full and a KLD swap at t1 / t2 on a 1.5-bpw base.
- **Avoids:** #32 / #35-run6 (no frequency or per-row weights), #23 / #64 / #113 (no code change).

### 2. Embedding rows by **measured** per-token sensitivity, with a floor (the IQR-V4 head rule moved to the embedding)
- **Mechanism.** Runs of 256 rows (sorted by measured sensitivity s_v = Σ over positions where token v is the input of
  ‖∂ log p / ∂h0‖² with sampled labels) take quarter-step K around a floor; unseen rows sit at the floor. Same trellis,
  a per-run K table (embed_tq trellis path: one K per 256-row run, ~1 KB).
- **Why it could win.** #32 failed with **frequency** as the proxy (1-bit tails +29–76%, multilingual +60–145%) because
  a rare row's error costs much more per occurrence than its frequency says; a measured s_v carries that per-occurrence
  cost, and the floor protects the never-seen half of the vocabulary (IQR-V4 needed one on the head, for the e^δz
  reason). GateRQ (2610.35985) is the closest evidence: per-lookup importance beat frequency by 3.5× in error removed
  (frequency was *worse than random*), on embedding-like tables with the same 1-bit trellis.
- **Runtime.** ~0 (row gather; K looked up per run).
- **Cheapest test.** Needs the GPU pass of #1 extended to log ‖g_t‖² with the input token id (same backward). Then a
  CPU simulation (weighted error at equal bytes, floors t1 / t1.5) and a KLD swap with the multilingual guard
  (`golden-ml`), as #32 did. Not CPU-screenable now: no per-row embedding Fisher exists anywhere.
- **Avoids:** #32's frequency ranking (by construction), its no-floor tails.

### 3. Low-K slab head at the 1.0–2.0 sizes (existing #34 data, new place in the menu)
- **Mechanism.** The row-slab head (head Fisher Σp(1−p) per row, 128-row slabs, K per slab above a floor) is offered to
  the knapsack only at K5 / 5.5 / 6 (`KN_HEADSLAB`). At the low sizes the head runs at K2–3, where the headsim curve is
  steepest: `lcurve.json` (32,768 held-out head inputs; uniform heads log-interpolated to the same bits) budget
  2.125 / floor 2: head KL **−37%** (guard KL +14%); 2.5 / f2 **−42%** (0.0101 vs 0.0173); 3.125 / f3 **−46% with the
  rare-token guard KL also lower** (0.0128 vs 0.0135); floor 2 at budget 3 instead nearly doubles the guard KL
  (0.0293 vs 0.0163): the floor must sit ~1 bit below the budget, as IQR-V4 found.
  SoftWater (2608.12026) and IQR-V4 §2.4 report the same order (one to 1.5 bits of head at equal KL).
- **Runtime.** D: a 3–4-class head GEMV; IQR's 4-class head is *faster* than an int8 head (624 vs 1,020 µs a rank).
  Prefill 0 (last token only).
- **Cheapest test.** CPU: knap5 with low-K slab options priced from `lcurve.json` (rule: floor ≥ budget − 1, guard KL
  ≤ uniform's) at 1.0–2.0 bpw; then the pending #34-02 whole-model check at one low budget.
- **Avoids:** #32's frequency tiers (Fisher-ranked, floor kept); not a new representation — the lever already exists.

### 4. KLT + water-filling for the embedding — **closed by the bound** (0.093 bit < 0.165-bit rotation)
Transform coding pays when the spectrum is spread (SPECTRA / AATC on K/V caches: top 25% of key channels hold 96%);
the embedding's top 25% hold ~45%. A structured (block / Givens) KLT without side cost would still be ≤ 0.09 bit.

### 5. Embedding predicted from the head (shared basis across the untied pair) — **closed** (85% residual)

### 6. Shared rows / row clusters / row-level VQ (OrBIT, OPTIC-style codes) — **closed** (median cos² 0.05)
OPTIC's 100× works because Gemma's per-layer embeddings are shared across 42 layers; OrBIT measured only
reconstruction error and keeps 21–26% of the variance. Neither finds structure our table does not have.

### 7. Head: KLT / water-filling in the H metric — **closed by the bound**
Under the calib `lm_head.input` Hessian the head rows are extremely anisotropic (H-metric eigen AM/GM 7.1, top
direction 61% of the energy, mostly the mean-row direction #34 already removes for free), but LDLQ captures that
(pivot AM/GM 1.02 in the signed fwht128 basis): water-filling beats the LDLQ model by only ×0.84 (0.12 bit), less
than a stored rotation.

### 8. Head low-rank residual correction (ARCHead, 2608.02703) — **not tested, low prior**
Its gain is against RTN group-INT4; the trellis + LDLQ error is near-white in the H-whitened space, which a rank-6–16
fit cannot capture. Revisit only if a measured error spectrum of a real K2–3 head shows a knee.

### 9. Better codes for the embedding (2^20 trellis, LUT VQ / E8 / Leech, entropy-coded rows) — **closed by the bound**
≤ 0.03–0.06 bit (1.03–1.09× the Gaussian bound), and #113 measured 0.02 bit for 2^20 states. Entropy coding of trellis
bits gains nothing (they are near-uniform); variable-length rows only matter with per-row rates (#2).

### 10. Softmax-shift search for the head (2609.31291) — **already have it** (mean-row, #34: −30%); the paper itself
finds no gain on Qwen3.5 / Gemma heads beyond centering.

## What is left after the screens

The representation of the embedding is at its information-theoretic floor for an H = I objective; the remaining levers
are a measured metric (#1, encode-only) and measured row sensitivities (#2), both of which need one GPU backward pass
that nobody has run yet. For the low sizes the bigger and cheaper wins are in the **menu and pricing**: the slab head
at low budgets (#3) and the real price of t0.25–t1 embeddings (see README §"Knapsack translation").
