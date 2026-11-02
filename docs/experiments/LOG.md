# Experiment log (newest first)

Append-only; template and rules in `README.md`. Entries before 2026-11-02 are in `archive/`.

### 2026-11-02 12:40 CST · #33-14 · Split-precision GEMMs on 0.8B: fails the pass rule (10× better than bf16)
- **Status:** planned → negative (as written) · **Machine:** 4070, allocator capped at 7.2 GB (peak 3.6 GB)
- **Code:** `proxy2b/prectest.py` @ 0ef4e99 · **Pass rule:** `proxy2b/YAQA08.md` amendment 7
- **Setup:** 16 xcorp windows, same sampled targets, trunk linear GEMMs fp32 / x2 / bf16, everything else fp32; the
  summed diagonal per linear vs fp32.
- **Result:** loss/token |diff| max: x2 8.2e-5, bf16 1.8e-3. Relative L2 error of the summed diagonal:

  | layers | x2 median | x2 max | bf16 median | bf16 max |
  |---|---|---|---|---|
  | 0–5 | 5.5e-3 | 2.5e-2 | 5.3e-2 | 1.0e-1 |
  | 6–11 | 4.5e-3 | 9.8e-3 | 3.4e-2 | 1.0e-1 |
  | 12–17 | 3.1e-3 | 5.0e-3 | 2.5e-2 | 4.3e-2 |
  | 18–23 | 1.2e-3 | 5.1e-3 | 1.3e-2 | 3.0e-2 |

  Time (4070, not informative for the R9700): fp32 1.60, x2 1.43, bf16 1.04 s / sequence.
- **Reading:** the products are ~1e-6 accurate (CPU self-test); the backward through 24 layers amplifies the
  difference, most in the early layers. Misses the median bound (1e-3) everywhere and the worst-linear bound in
  layers 0–5. Missing control: fp32's own spread under a harmless change (summation order), which would show
  whether 1e-3 was attainable at all.
- **Decision / next:** the 27B collector stays fp32 for now. Options if speed matters: the fp32 noise-floor control;
  a three-way split (~2^-24, 3 / 6 GEMMs); or judge x2 by its effect on KL (a quant from an x2-collected diagonal
  vs an fp32-collected one) instead of a matrix error bound — any of these needs a new pass rule first.
- **Files:** `data/yaqa08/prectest.jsonl`

### 2026-11-02 12:20 CST · #33-14 · Split-precision (bf16 hi/lo) GEMMs for the fp32 collector: prepared
- **Status:** planned · **Machine:** 4070 (check), iceman (use) · **Code:** `v3/yaqa/bf16x.py`, `proxy2b/prectest.py`,
  `collect.py` YAQA_PREC=x2 @ 0ef4e99
- **Question:** can the 27B collector's fp32 GEMMs (~30 s / sequence) run as bf16 hi/lo splits on the matrix cores
  (2 GEMMs per product, 3 for the weight gradient; weights exact in bf16) without losing the fp32 gradient quality?
- **Pass rule:** `proxy2b/YAQA08.md` amendment 7.
- **Setup:** CPU self-test on random data (emulated): x2 vs fp32 relative error 2.5e-6 (products), 6e-7 (diagonal);
  bf16 1.7e-3 / 3.8e-4. The model-level test (`prectest.py`, ~10 min on the 4070) is not run yet.
- **Decision / next:** run prectest + `yaqaFact4.sh` when the 4070 is free; on iceman, the gate run of `hdiag27.sh`
  with YAQA_PREC=x2 also reports one linear's precision against a whole fp32 G and the time per sequence. Then the
  two-stage pipeline (#33-13) if the GEMMs still dominate the profile.

### 2026-11-02 11:50 CST · #33-13 · 27B diagonal H_out: collector and 0.8B dose check prepared
- **Status:** planned · **Machine:** iceman (both cards) + 4070 · **Code:** `v3/yaqa/collect.py` YAQA_DIAG,
  `v3/yaqa/hdiag27.sh`, `quantize/ourq.py` (`--hout` reads `hout_diag` vectors), `proxy2b/yaqaFact4.sh` @ f0fb687
- **Question:** how many sequences the diagonal needs (0.8B), then the 27B collection and a 27B swap test.
- **Pass rule:** `proxy2b/YAQA08.md` amendment 6 (0.8B dose); the 27B swap test's rule is written before it runs.
- **Setup:** 27B: per sequence and trunk linear, diag(G_b G_bᵀ) with G_b = Σ_t dy_t x_tᵀ, sampled targets, fp32
  backward (~30 s / sequence on both cards, measured in gate 2b), qad-pool 2048-token windows (pools-v3), snapshots
  at 256 / 512 / 1024 / 2048; accumulators ~10 MB. 27B H_out sizes for reference: full 206 GB fp32 (gate / up 155),
  without gate / up 51 GB.
- **Decision / next:** run `yaqaFact4.sh` when the 4070 is free (~12 min); the 27B run needs both cards (user's
  arrangement, e.g. a night): 1K sequences ≈ 8.5 h. Faster near-fp32 backward (bf16 hi/lo split of activations,
  weights exact in bf16) is a possible speed-up, to be checked for precision first.

### 2026-11-02 11:15 CST · #33-12 · H_out for 27B sizing: diagonal is enough; ~4K sequences
- **Status:** done · **Machine:** 4070 · **Code:** `proxy2b/yaqaFact3.sh` @ 86f1233
- **Pass rule:** `proxy2b/YAQA08.md` amendment 5 (written before data).
- **Setup:** all arms on `base`'s settings + H_out at output damping 0.01, i8, g08-search1.
- **Result** (vs `base`: ratio, z; vs `bhout` = full H_out, 8K):

  | arm | K3 KL | vs base | vs full | K2 KL | vs base | vs full |
  |---|---|---|---|---|---|---|
  | full (`bhout`, #33-11) | 0.0893 | −9.7% (−12.4) | — | 0.506 | −5.6% (−6.3) | — |
  | full, gate/up diagonal (`bmix`) | 0.0891 | −9.9% (−13.5) | −0.2% (−0.4) | 0.494 | −7.7% (−11.0) | −2.2% (−5.0) |
  | diagonal only (`bdiag`) | 0.0910 | −8.0% (−13.5) | +1.9% (2.4) | 0.493 | −8.0% (−13.1) | −2.5% (−3.0) |

  Dose, full H_out at K3: 1K 0.0950 (−3.9%, 40% of the 8K gain), 2K 0.0922 (−6.8%, 70%), 4K 0.0890 (−10.0%,
  103%; vs 8K z −0.4).
- **Reading:** the diagonal keeps the gain by the rule (≥ 2/3 at both K, worse than full only at K3 by z 2.4 < 3), so
  it sets the 27B collection: **diagonal only** (~10 MB of accumulators). The full gate/up H_out *hurts* at K2 (the
  noisiest factors, #33-09). The mixed arm is the best at K3 (+~2% over diagonal) for 25 GB of accumulators at 27B.
  Full H_out needs ~4K sequences (2K reaches 70%); the dose of the diagonal alone was not measured (its two halves
  agree at ≥ 0.997, #33-09, so likely fewer).
- **Decision / next:** plan the 27B collection: per-sequence weight-gradient row norms (diag of Sketch B's H_out)
  with sampled targets, fp32 backward, calib H_in unchanged; ~4K sequences of 2048 tokens from the qad pool; then a
  27B K3 / K2.5 swap test against the ladder's levels before anything changes in V3. Optional on 0.8B: dose of the
  diagonal (1K/2K on `bdiag`).
- **Files:** 4070 `/data/yb/result-fact3.txt`; 27B H_out sizes per type in this entry's commit message context
  (full 206 GB fp32, without gate/up 51 GB, diagonal ~10 MB).

### 2026-11-02 10:30 CST · #33-11 · H_out follow-ups on 0.8B: holds at K2 and stacks on our settings
- **Status:** done (positive) · **Machine:** 4070 · **Code:** `proxy2b/yaqaFact2.sh` @ 6496e9e
- **Pass rule:** `proxy2b/YAQA08.md` amendment 4 (written before data).
- **Setup:** 8K snapshot, i8, g08-search1. `hout` / `hdiag`: paper damping 0.01, no scaled Hessian. `bhout`: our
  `base` settings (scaled Hessian, damping 0.025 / input 0.005) plus full H_out at output damping 0.01.
- **Result:**

  | K | arm | KL | vs ldlq (z) | vs base (z) |
  |---|---|---|---|---|
  | 3 | base | 0.0989 | −5.7% | — |
  | 3 | hout (#33-10) | 0.0946 | −9.9% (−13.8) | −4.3% (−5.4) |
  | 3 | **bhout** | **0.0893** | — | **−9.7% (−12.4)** |
  | 2 | base | 0.535 | −3.7% | — |
  | 2 | hout | 0.537 | −3.4% (−3.6) | +0.2% (0.2) |
  | 2 | hdiag | 0.532 | −4.2% (−7.9) | −0.6% (−0.9) |
  | 2 | **bhout** | **0.506** | — | **−5.6% (−6.3)** |

- **Reading:** all three rules pass: H_out holds at K2; it stacks with the scaled Hessian and tuned damping (bhout vs
  hout: K3 −5.6%, K2 −5.8%); bhout beats base at both K with z ≤ −2, which meets the user's switch bar **on the
  0.8B**. At K2 the diagonal alone does as well as the full H_out without our settings.
- **Decision / next:** plan a 27B check; its cost is set by two 0.8B tests first: (1) a mixed arm on base's settings,
  full H_out except gate/up (diagonal only), and the all-diagonal arm — the full gate/up H_out is ~155 GB of fp32
  accumulators at 27B, the diagonal a few MB; (2) dose: 1K / 2K / 4K snapshots on base's settings.
- **Files:** 4070 `/data/yb/result-fact2.txt`

### 2026-11-02 10:15 CST · #33-10 · YAQA factorial on 0.8B: the H_in swap was the loss; H_out with calib H_in wins
- **Status:** done (positive) · **Machine:** 4070 · **Code:** `proxy2b/yaqaFact.sh` @ 053a640
- **Pass rule:** `proxy2b/YAQA08.md` amendment 3 (written before data).
- **Setup:** K3, i8, damping 0.01 both sides, no scaled Hessian, 8K snapshot, g08-search1; reference `ldlq`
  (calib H_in, one-sided) and `base` (our tuned quant).
- **Result:**

  | arm | H_in | H_out | KL | vs ldlq (z) | vs base (z) |
  |---|---|---|---|---|---|
  | ldlq | calib | — | 0.1049 | — | +6.1% |
  | hin | Sketch B | — | 0.1257 | +19.8% (16.5) | +27.1% (20.2) |
  | yb8192 | Sketch B | full | 0.1123 | +7.0% (5.1) | +13.6% (8.1) |
  | **hout** | calib | full | **0.0946** | **−9.9% (−13.8)** | **−4.3% (−5.4)** |
  | hdiag | calib | diagonal | 0.0972 | −7.4% (−11.1) | −1.7% (−2.1) |

- **Reading:** the earlier YAQA losses (#33-08) came from Sketch B's gradient-weighted H_in, not from H_out. With
  the calib H_in, the full H_out beats paper LDLQ by ~10% and our tuned one-sided quant by ~4%, even without the
  scaled Hessian and tuned damping `base` has; the diagonal alone gets about three quarters of the gain over ldlq.
- **Decision / next:** H_out passes. Before anything at 27B, on the 0.8B: (1) the same arm at K2; (2) H_out on top of
  `base`'s settings (scaled Hessian, tuned damping) — the arm that would actually ship; (3) dose (1K/2K/4K
  snapshots) to size a 27B collection; (4) whether H_out damping matters. 27B collection only after those.
- **Files:** 4070 `/data/yb/result-fact.txt`

### 2026-11-02 10:25 CST · #33-09 · Sketch B factors: structure and noise (CPU)
- **Status:** done · **Machine:** 4070 box CPU (niced) · **Code:** `proxy2b/houtstruct.py`
- **Question:** is the 0.8B Sketch B H_out noise-limited, how is it shaped, and how far is Sketch B's H_in from the
  plain calib input Hessian?
- **Setup:** all 150 linears, snapshots 1K–8K (each the mean over its first n sequences). Noise: cosine between the
  two independent 4K halves (H(4096) vs 2·H(8192) − H(4096)). Spectrum of H(8192): participation ratio and entropy
  rank as a fraction of d, diagonal share of ||H||²_F, spread (CV) of the diagonal.
- **Result** (medians by type):

  | type | d | PR | diag share | diag CV | H_out halves cos | diag halves cos | H_in vs calib cos |
  |---|---|---|---|---|---|---|---|
  | GDN in_proj_qkv | 6144 | 0.053 | 0.22 | 1.69 | 0.903 | 0.998 | 0.764 |
  | GDN in_proj_z | 2048 | 0.054 | 0.81 | 3.96 | 0.960 | 0.998 | 0.756 |
  | GDN out_proj | 1024 | 0.49 | 0.50 | 0.18 | 0.770 | 0.999 | 0.891 |
  | attn q / k / v | 4096 / 512 / 512 | 0.04 / 0.04 / 0.20 | 0.27 / 0.13 / 0.43 | 2.3 / 1.5 / 0.5 | 0.958 / 0.997 / 0.905 | ≥0.998 | 0.79 / 0.54 / 0.73 |
  | attn o_proj | 1024 | 0.37 | 0.41 | 0.20 | 0.656 | 0.997 | 0.866 |
  | mlp gate / up | 3584 | 0.28 / 0.22 | 0.63 / 0.47 | 1.0 / 1.0 | 0.797 / 0.845 | 0.999 | 0.69 / 0.81 |
  | mlp down | 1024 | 0.60 | 0.63 | 0.17 | 0.944 | 1.000 | 0.633 |

  By depth (median over types): H_out halves cos 0.69 (layers 0–5), 0.80 (6–11), 0.92 (12–17), 0.98 (18–23).
  Sketch B's H_in is itself stable (halves cos ≥ 0.96) but differs from the calib Hessian (cos 0.35–0.98).
- **Reading:** (1) the **diagonal of H_out is precise** at 4K (≥ 0.997) and very uneven for input projections
  (CV 1.5–4) but flat for residual writers (out_proj, o_proj, down: CV ~0.2); (2) the **full H_out is noisy in the
  early half and for residual writers / gate / up** (0.66–0.80 at 4K sequences per half): two-sided LDLQ there rounds
  partly against noise; (3) residual writers have near-isotropic H_out (PR 0.4–0.6), where two-sided ≈ one-sided;
  (4) the YAQA arms also **replaced the calib H_in** (10M tokens, x xᵀ) with Sketch B's gradient-weighted H_in from
  xcorp, which is far from it (cos ~0.5–0.9), and gradient-weighted H_in alone was +31…+44% worse on the 27B (#33-07).
  So the YAQA loss mixes two changes; which one costs the KL is untested.
- **Decision / next:** a factorial on the 0.8B (GPU, ~4 min an arm) separating the two: calib H_in + Sketch B H_out;
  Sketch B H_in + identity H_out; calib H_in + diag(H_out); pass rule to be written before it runs.
- **Files:** `data/yaqa08/houtstruct.jsonl`

### 2026-11-02 09:10 CST · #33-08 · YAQA Sketch B on 0.8B: 2K–8K negative, 16K stopped
- **Status:** running → negative · **Machine:** 4070 · **Code:** `proxy2b/yaqaB.py`, `yaqaBrun.sh`
- **Pass rule:** `proxy2b/YAQA08.md` (amendment 2); reference arm `ldlq`, our tuned `base` beside it.
- **Setup:** as the 1K read; snapshots 2K/4K/8K from one bit-exact resumed collection (8,192 sequences, 207.6 min).
- **Result** (KL on g08-search1; ratio to ldlq, z paired):

  | seqs | K3 yb | vs ldlq | z | vs base | K2 yb | vs ldlq | z | vs base |
  |---|---|---|---|---|---|---|---|---|
  | 1K | 0.1184 | +12.8% | 9.4 | +19.7% | 0.690 | +24.1% | 19.6 | +28.8% |
  | 2K | 0.1158 | +10.3% | 7.1 | +17.1% | 0.677 | +21.8% | 13.5 | — |
  | 4K | 0.1121 | +6.9% | 5.1 | +13.4% | 0.675 | +21.4% | 13.4 | +26.0% |
  | 8K | 0.1123 | +7.0% | 5.1 | +13.6% | 0.642 | +15.5% | 11.8 | +19.9% |

  References: K3 ldlq 0.1049, base 0.0989; K2 ldlq 0.556, base 0.535.
- **Decision / next:** fails the pass rule at every snapshot. K3 is flat from 4K to 8K; K2 still improves at 8K,
  but extrapolating it would at best approach ldlq, still far from base. 16K collection stopped by the user
  (15:07 UTC); the 4070 is free. Snapshots 1K–8K kept for H_out research and a later diagnosis run. YAQA is not
  adopted for V3; the V3 design stays.
- **Files:** 4070 `/data/yb/result.txt`, `/data/logs/yaqaBrun.log`, `/data/yb/snap{1024,2048,4096,8192}`

### 2026-11-02 06:20 CST · #— · Tracker reorganised
- **Decision:** the two old trackers (`EXPERIMENTS-new-format.md`, `EXPERIMENTS.md`) are archived verbatim in
  `archive/`; every experiment so far was numbered #1–#62 in `INDEX.md`; open plans and the backlog moved to
  `BACKLOG.md`. Old paths keep a one-line pointer. From now on: one numbered row per experiment, one templated log
  entry per event.

### 2026-11-02 05:50 CST · #33-08 · YAQA Sketch B on 0.8B: two-sided objective optimised but does not track KL
- **Status:** running (unchanged) · **Machine:** 4070 · **Code:** `proxy2b/objcheck.py` (CPU)
- **Question:** is the 1K loss a quantizer bug (two-sided objective not optimised) or the objective itself?
- **Setup:** 14 linears of layers 0/5/11/19/23; per linear, one-sided error tr(E Hc Eᵀ)/tr(W Hc Wᵀ) with the calib
  input Hessian, and two-sided tr(E Hi Eᵀ Ho)/tr(W Hi Wᵀ Ho) with the 1K Sketch B factors; K3 arms base, ldlq, yb1024.
- **Result:** yb1024 has a 1.4–8× lower two-sided objective than base on every linear, and a 1.4–3.5× higher
  one-sided error (e.g. layer 5 in_proj_qkv: two-sided 5.9e-3 → 7.6e-4, one-sided 3.0e-3 → 8.8e-3).
- **Decision / next:** the quantizer path works; at 1K sequences and damping 0.01 the two-sided objective is the
  wrong target. Wait for 2K–16K before diagnosing; keep every snapshot for H_out research (`local/TODO.md`).
- **Files:** 4070 `/data/logs/objcheck.log`

### 2026-11-02 05:25 CST · #33-08 · YAQA Sketch B on 0.8B: 1K read negative
- **Status:** running (unchanged) · **Machine:** 4070 · **Code:** `proxy2b/yaqaB.py`, `yaqaBrun.sh`
- **Pass rule:** `proxy2b/YAQA08.md` (amendment 2 written before any data); `ldlq` is the one-sided reference,
  `base` (our tuned quant) reported beside it.
- **Setup:** 0.8B, i8 codebook, K3 and K2; arms base / ldlq (σ 0.01, no scaled Hessian) / yb1024 (per-linear
  H_in, H_out from 1,024 sequences, σ 0.01 both sides); golden g08-search1.
- **Result:**

  | K | base | ldlq | yb1024 | yb vs ldlq |
  |---|---|---|---|---|
  | 3 | 0.0989 | 0.1049 | 0.1184 | +12.8% (z 9.4) |
  | 2 | 0.535 | 0.556 | 0.690 | +24.1% (z 19.6) |

- **Decision / next:** collection continues; 2K/4K/8K scored ~09:00–09:15 CST, 16K ~13:20 CST. The 2K–8K reads
  decide whether to stop before 16K.
- **Files:** 4070 `/data/yb/result.txt`
