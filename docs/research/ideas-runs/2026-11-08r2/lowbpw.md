# Lens lowbpw, round 2 (2026-11-08): ideas for 1.0–2.5 bpw that must be settled before the freeze

**Scope.** Only ideas that change the per-level library (quantizer settings, codebooks, format, level menu) or the knapsack's inputs. Per-mix post-processing, healing/QAD and kernel speed work are left out. I built on the 11-07 lowbpw report and the 11-08 freeze review rather than repeating them.

**The main new finding concerns the 1.0–1.5 sizes.** On 17 real 27B tensors, an H-weighted low-rank factorisation (factor bits charged) beats our sub-1 trellis at equal bits:
- at K0.125, 13 of 17 (4-bit factors) or 10 of 17 (8-bit factors);
- at K0.25, 10 of 17;
- at K0.5, still 8 of 17.

It loses on every down_proj and on mid-layer attn_v. Separately, removing the least important half of the MLP neurons (by down_proj H-energy) costs relative error 0.10–0.33 while removing 50% of all three MLP tensors. In late layers that beats trellis K0.5 on down_proj outright. Both change what the bottom of the level menu should contain. Neither affects 2.5 bpw, where sub-1 levels are never picked (#90).

**For 2.5 bpw the cheap wins are elsewhere:**
- A code-reading finding: the post-LDLQ input-scale refit ignores H_out and fits in the *damped* Hessian.
- Quantizing the bf16 GDN a/b projections, still outside the knapsack.
- Applying #118's attention price offsets.
- Calib-part pricing of the head and embedding.

---

## 0. Conventions and conversion factors

- **KL per bpw.** w63 re-pricing gives 0.0787 → 0.0534 → 0.0364 at 2.25 / 2.5 / 2.75. That is about 1.55 ln-units per bpw, so **0.01 bpw ≈ 1.5% KL at 2.5** and ≈1.7% at 2.0. The V2 board between 1.75 and 2.0 gives ≈2.4% per 0.01 bpw. At 1.0–1.5 I assume **≈2.5% ± 1% per 0.01 bpw**. That is extrapolated: saturation (kfit2+sat C≈1.8) may flatten it.
- **Shrink rule.** Proxy or 0.8B noise-shaping gains are shrunk to ⅓ at 27B, the round-1 rule. Coherent or structural fixes are not shrunk, but are marked "unvalidated".
- **"Rel" metric.** "Rel" = tr(E H Eᵀ)/tr(W H Wᵀ), one-sided calib H_in, unless stated otherwise.
- **Fixed overheads by size.** All rows in bpw; percentages are shares of the size's budget.

| part | 2.5 bpw (knap5 main-acc) | 1.0 bpw (if left at the 2.5 settings) | 1.0 bpw (plausible) |
|---|---|---|---|
| head (1.27B params) | K4 0.189 (7.6%); slab 3.125 0.148 | 0.189 (19%) | K1 0.047 (4.7%) |
| embedding | t2 0.095 (3.8%) | 0.095 (9.5%) | t0.5–1: 0.024–0.047 |
| GDN a/b (bf16, outside knapsack) | 0.014 (0.56%) | 0.014 (1.4%) | 0.014 (1.4%) |
| suh/svh fp16 (Σ(m+n)≈6.8M values) | 0.004 (0.16%) | 0.004 (0.4%) | 0.004 (0.4%) |

---

## 1. What I measured this round (colo CPU, ~25 CPU-min, read-only on the model)

Scripts and results are in colo `/data/work/ideas-lowbpw2/` (`lr90.py`/`lr90.jsonl`, `chimp2.py`/`chimp.jsonl`).

### M1. H-weighted low-rank vs sub-1 trellis, equal bits

**Low-rank (LR) column.** Optimal rank-r in the one-sided H_in metric (SVD of W·H^½), with r = K·m·n / (b·(m+n)) so the factors cost exactly K bits per weight at b bits per factor entry. Factor quantisation error is not included; it is negligible at 8 bits and about +0.01 rel at 4 bits. Scale metadata is not included.

**Trellis column.** The #90 CPU proxy (`ldlq_cubic0.02_i8`, 128 rows, one-sided, no H_out, before σ0.2). It is free-start, so per #57 it is slightly *optimistic*.

LR4@K means 4-bit factors at twice the rank of LR8@K. Bold marks the better of trellis and the best LR at that K.

| tensor | trellis K0.125 | LR8@K0.125 | LR4@K0.125 | trellis K0.25 | LR4@K0.25 | trellis K0.5 | LR4@K0.5 | H-energy top-1 / top-64 |
|---|---|---|---|---|---|---|---|---|
| L2 ssm_qkv | 0.574 | 0.296 | **0.236** | 0.372 | **0.175** | 0.172 | **0.116** | 0.45 / 0.72 |
| L10 ssm_qkv | 0.472 | 0.309 | **0.254** | 0.304 | **0.200** | 0.167 | **0.145** | 0.43 / 0.71 |
| L10 ssm_z | 0.284 | 0.143 | **0.118** | 0.164 | **0.094** | 0.086 | **0.070** | 0.73 / 0.87 |
| L10 ssm_out | 0.632 | 0.458 | **0.402** | 0.490 | **0.335** | 0.311 | **0.254** | 0.38 / 0.57 |
| L43 attn_qg | 0.357 | 0.159 | **0.128** | 0.217 | **0.100** | 0.112 | **0.073** | 0.60 / 0.85 |
| L43 attn_o | 0.679 | 0.609 | **0.528** | 0.528 | **0.434** | 0.338 | **0.321** | 0.14 / 0.44 |
| L3 attn_k | 0.928 | 0.402 | **0.338** | 0.513 | **0.254** | 0.199 | **0.179** | 0.41 / 0.77 |
| L3 attn_v | **1.504 (worse than zero)** | 0.688 | **0.619** | 0.953 | **0.523** | 0.427 | **0.406** | 0.19 / 0.51 |
| L62 gate_up | 0.225 | 0.151 | **0.133** | 0.145 | **0.113** | **0.081** | 0.091 | 0.74 / 0.85 |
| L43 gate_up | 0.498 | 0.444 | **0.396** | **0.374** | 0.340 | **0.229** | 0.272 | 0.38 / 0.55 |
| L10 gate_up | 0.636 | 0.685 | **0.604** | **0.493** | 0.506 | **0.316** | 0.389 | 0.12 / 0.31 |
| L43 attn_k | 0.423 | 0.504 | **0.382** | **0.269** | 0.277 | **0.131** | 0.195 | 0.23 / 0.75 |
| L43 attn_v | 0.669 | 0.708 | **0.612** | **0.462** | 0.488 | **0.260** | 0.359 | 0.11 / 0.55 |
| L19 attn_v | **0.704** | 0.799 | 0.720 | **0.490** | 0.608 | **0.265** | 0.474 | 0.07 / 0.43 |
| L10 down | **0.632** | 0.748 | 0.693 | **0.496** | 0.614 | **0.327** | 0.506 | 0.14 / 0.26 |
| L30 down | **0.348** | 0.810 | 0.750 | **0.248** | 0.662 | **0.150** | 0.537 | 0.06 / 0.19 |
| L43 down | **0.698** | 0.839 | 0.786 | **0.557** | 0.706 | **0.371** | 0.589 | 0.04 / 0.16 |

How to read this:
- **Where low-rank wins.** It wins on the readers of the normalised residual (GDN qkv and z, attn_qg, early attn_k/v), on ssm_out and attn_o, and on late-layer MLPs. These tensors have concentrated H-weighted spectra, partly from the massive input channel 3994 and its late-layer equivalents.
- **Where it loses.** down_proj (flat spectrum, top-512 holds only 42–50%) and mid-layer attn_v.
- **Equivalent bits.** LR4@0.125 on ssm_z ≈ trellis at K≈0.37, on attn_qg ≈ K≈0.45, on L2 ssm_qkv ≈ K≈0.32. That saves 0.2–0.3 bit on those tensors at the bottom of the menu.
- **The cheapest floor.** For ssm_z, rank-1 alone (≈0.003 bit/weight) captures 0.725 of the H-energy against 0.716 for trellis K0.125. At the bottom of the menu the trellis is mostly paying for one direction.
- **The worse-than-zero cell.** L3 attn_v at K0.125 has rel 1.504 in this proxy. That is consistent with #114's "GPU search there is worse than all-zero (err 1.014)".

### M2. Transform-coding headroom (reverse water-filling with a free basis)

This assumes Gaussian rows with the tensor's own H-weighted covariance spectrum, and treats the basis as free. At K1, water-filling / trellis is:

| tensor | ratio at K1 |
|---|---|
| ssm_z L10 | 0.0107 / 0.034 (0.31) |
| attn_qg L43 | 0.0159 / 0.047 (0.34) |
| ssm_qkv L10 | 0.027 / 0.070 (0.39) |
| gate_up L10 / L43 / L62 | 0.70 / 0.71 / 0.73 |

**The basis is not free.** An n×n basis at 8 bits costs 0.15 (gate_up) to 0.83 (ssm_z) bits per weight, and still ≥1.2 bits/weight when shared by every reader of one input. So explicit KLT/transform coding is out. Low-rank, which pays only for r columns, is the honest version (A-ideas). Tensors with m < n (down, attn_k/v) are rank-deficient and the bound is meaningless for them.

### M3. down_proj input-channel (neuron) concentration, 5 layers

| layer | top-10% neurons' share of diag(H)·‖w‖² | top-50% share | exact rel error of dropping the bottom 50% (no compensation) | diag-sum / exact |
|---|---|---|---|---|
| L1 | 0.295 | 0.748 | 0.193 | 0.585 |
| L10 | 0.220 | 0.687 | 0.264 | 0.716 |
| L30 | 0.196 | 0.621 | 0.330 | 0.763 |
| L43 | 0.230 | 0.654 | 0.321 | 0.847 |
| L62 | **0.530** | **0.879** | **0.102** | 0.731 |

**The error is exact.** Dropping neuron j removes W_down[:,j]·h_j, so the down-side error *is* the full error. The gate/up rows of a dropped neuron contribute nothing beyond that.

**Comparison with trellis.** Trellis K0.5 down rel is 0.33 (L10), 0.15 (L30), 0.37 (L43) and 0.24 (L62). At L62, dropping half the neurons beats K0.5 trellis on down *and* frees half the bits of gate/up/down.

**Compensation.** diag-sum/exact is below 1 because neurons are correlated. LS compensation of the kept columns (OBS-style) would therefore lower the drop error further. Not run.

### M4. Head and embedding (from last round's `pca.out`, re-read)

- **lm_head.** The mean row holds 0.498 of the H-weighted energy (already handled by mean-row). Of the centred rest, top-16/64 hold 0.45/0.54.
- **Embedding.** H-flat (top-64 holds 0.062, mean row 0.0097). This confirms #119.

### M5. Code facts (read this round)

1. **The input-scale refit ignores H_out.**
   - `refit_scales` (fork `quantize.py:1730`, called at 1988/2283 with `H_orig` only) solves the input-scale step as ((QQᵀ)∘H)·r = rowsum(Q∘HW).
   - Output columns enter unweighted, so **the suh refit ignores H_out**, which has been adopted at every level.
   - The output-column (svh) step is unaffected by a diagonal H_out.
   - Its docstring: "the fits use H as given (**damped**)". The refit therefore fits in a metric damped by σ = 0.005/0.025 (≥K1), 0.05 (sub-1 inputs) or **0.2 (K0.125)**.
2. **Sub-1 LDLQ drift is clamped at 1.08.** rxq `engine.py` interpolates `LDLQ_DRIFT = {1: 1.08, 1.5: 1.035, 2: 1.018, 2.5: 1.009, 3: 1.004}`; every K ≤ 1 gets 1.08. The g_scale tile sample at K0.125–0.875 is inflated by the K1 drift.
3. **GDN a/b are not quantized.** `in_proj_a/b` do not appear in rxq's tensor lists or in v3.toml. They stay bf16 and outside the knapsack.
4. **Tile CD covers attn_qg only** (v3.toml `families = ["attn_qg"]`).

---

## 2. Ideas by theme

### A. The bottom of the menu: sub-1 codecs (feeds 1.0–1.25; ~0 at ≥1.5)

#### A1 · Low-rank sub-1 options (H-weighted SVD, int4/int8 factors)
**Mechanism.** For each tensor, add options W ≈ U·Vᵀ from the SVD of W·H^½, rescaled back (V = H^{-½}V′), with factors stored at 4 or 8 bits. Budgets: ranks equivalent to K0.0625, 0.125, 0.25 and 0.5. The knapsack picks trellis-K or LR per tensor at equal bytes. At inference this is two small GEMMs (y = U(Vᵀx)).
- **Source:** New as a menu option. Related: BiLoCo 2610.32243 (signed low rank), NanoQuant/ShamAN-Q 2609.38521, AF1 2609.06161 (genuine 1-bit factorisations). They replace the whole model's quantiser; here LR is used only where it beats the trellis, priced by the same proxy.
- **Why:** M1. Low-rank wins at K0.125 on 13/17 tensors and at K0.5 on 8/17, saving 0.2–0.3 bit per tensor on GDN readers, attn_qg, attn_o and late MLPs. #90 put the 1.25-bpw gain from *all* sub-1 trellis options at only −1.8%, a sign that the trellis floor is weak.
- **Fit:** Format change (a factored tensor type; side-car field in exl3i) plus per-level library. Building it is cheap: an SVD plus factor quantisation, minutes per tensor, no Viterbi.
- **Test:** Diagnostic 1 (full census, colo, ~3 CPU-h) → knap5 with LR options at 1.0/1.25 (CPU minutes). If the predicted gain is ≥3% at 1.0, run one 27B mix pair at 1.25 on card 0 (≈2 evals).
- **Gain/risk:** 2.5: 0. 1.5: 0–1%. 1.25: 1–4%. 1.0: 3–10% (wide). Risks:
  - LR error is *structured*: the dropped directions are coherent across tokens, so KL per unit of proxy may exceed the trellis's (needs G6).
  - Rank granularity: r is a multiple of 16; a 128 multiple if the U factor also takes the input Hadamard.
- **Conflicts:** K0 was dropped 11-01, but LR is not K0. #40/#96 are different. #120's "no exl3i field" decision would need reopening.

#### A2 · Trellis-coded factors (a factored exl3i tensor)
**Mechanism.** Store U and Vᵀ as ordinary exl3i trellis tensors at K_f 2–4 and run the branch on the existing GEMM ops (two launches). Quantise Vᵀ with LDLQ under H_in, and U under the exact r×r Hessian Vᵀ·H_in·V (free), with H_out rows on U.
- **Source:** New. CALDERA-like LDLQ on factors, but with our trellis.
- **Why:** At equal error a K3 trellis factor costs ~25% fewer bits than int4 RTN, which gives the A1 options about +30% rank.
- **Fit:** Format (a container entry = two exl3i tensors). No new GEMM kernel.
- **Test:** colo settle64 on the factors of 6 tensors from M1. Pass if trellis factors beat int4 factors by ≥5% rel at equal bits. 1–2 CPU-h.
- **Gain/risk:** +20–30% on top of A1's gain. 2.5: 0. Risk: r must suit the 128-Hadamard on U's input (pad, or input rotation off).
- **Conflicts:** None.

#### A3 · Binary double factorisation as a sub-1 codec for flat-spectrum tensors
**Mechanism.** W ≈ diag(a)·A±·diag(m)·B±·diag(b), with inner dimension r = K·mn/(m+n), sign factors and scale vectors only, fitted by H-weighted ADMM (optionally a KL-Shampoo metric). Offer it at K0.25–0.5 for the tensors where int-LR loses (down, mid attn_v).
- **Source:** AF1 2609.06161 reports Qwen3-8B at 1.00 bpw Wiki2 16.26 (BF16 9.00) and at 2.50 bpw 9.71. That includes its scale-only reconstruction (on LLaMA-3.2-3B, 41.34 → 30.47 from that step). Also BiLoCo: ±1 factors capture 47–53% of ‖W‖ vs 26–29% for bf16 rank-32 at equal storage. Differences here: no KD (out of scope), per-tensor use only.
- **Why:** Binary factors trade precision for rank, which suits flat spectra.
- **Fit:** Format plus kernel (binary GEMM or decode-to-dense). Per-level.
- **Test:** colo ADMM fit at K0.25/0.5 on the 6 tensors where low-rank lost in M1. Pass if it beats trellis on ≥3 of them. ~4 CPU-h.
- **Gain/risk:** 1.0: +1–4% on top of A1. 2.5: 0. High risk: fit cost and a new kernel.
- **Conflicts:** None.

#### A4 · Low-rank + trellis residual hybrid at K0.25–1.5
**Mechanism.** Remove a small H-weighted rank-r part first, then LDLQ-trellis the residual at K_R. Offer a few (r, K_R) pairs per family where the spectrum is concentrated.
- **Source:** CALDERA, LQER, SVDQuant. 2607.20434 finds low-rank and quantisation are non-orthogonal and that low-rank-then-quantise is the better order.
- **Why:** At K0.125 the trellis captures almost exactly the top direction (ssm_z: 0.716 vs rank-1's 0.725). Above K1, LDLQ already handles a rank-1 spike cheaply (only about one pivot is large), so the gain should fall with K. M2 bounds the headroom: ≤0.85 bit for GDN readers, ≤0.25 bit for MLPs, even with a free basis.
- **Fit:** Format (same field as A1) plus per-level hybrid levels, limited to r ∈ {16, 64} × K_R ∈ {0.5, 1}.
- **Test:** colo settle64 on residuals of ssm_z, attn_qg, L2 ssm_qkv and L62 gate_up at K0.5/1. Pass if hybrid ≤ 0.9× trellis at equal bits. 1 CPU-h.
- **Gain/risk:** 1.0–1.25: 1–3%. 1.5–2.0: 0–1%. 2.5: 0–0.3%.
- **Conflicts:** #120-02 (writers' row 3994 at K2.5, null) is a different setting (readers, sub-1).

#### A5 · K0 and "spike-only" as priced floors; exclude broken sub-1 cells
**Mechanism.** Give the knapsack a zero-byte K0 option (rel 1.0) and the A10 spike option. Forbid trellis cells whose measured rel is ≥ 0.9.
- **Source:** New. #114 found K0.125 worse than zero (err 1.014). M1 has L3 attn_v at rel 1.504 (K0.125) and 0.953 (K0.25), and L3 attn_k at 0.928. ProQuant 2610.30779 found MoE experts pruned beat experts at 1 bit ({0,2,3} 74.49 vs {1,2,3} 68.46).
- **Why:** A guard against proxy-optimistic sub-1 pricing at 1.0–1.25.
- **Fit:** Knapsack input only; K0 needs no quant run.
- **Test:** knap5 at 1.0/1.25 with K0 and the cell exclusion; count changed cells. CPU minutes.
- **Gain/risk:** 1.0: 0–1%. Mostly risk reduction.
- **Conflicts:** The user dropped K0 as a *level* (11-01). This is only a pricing floor.

#### A6 · Guarantee rel ≤ 1 at K≤0.25 (scale-grid floor and zero fallback)
**Mechanism.**
- Extend the g_scale coarse grid below 0.5 at K≤0.25: r90 hit the 0.5 floor in 6/49 cases, all at K0.125.
- After the refit, check that rel in the *undamped* metric is ≤ 1 per output column; otherwise set the column to 0 (c = 0 is always feasible for the LS step).
- **Source:** New.
- **Why:** M1/M5 show that the K0.125 failure plausibly comes from the σ0.2-damped refit metric (see D2).
- **Fit:** Quantiser setting (sub-1 levels).
- **Test:** Rerun 6 small early attn tensors at K0.125/0.25 through rxq's CPU stand-in or settle64 with grid and guard. Pass if rel drops below 0.9. 30 CPU-min.
- **Gain/risk:** 1.0: 0–1%. Trivial cost.
- **Conflicts:** #114-02 (σ0.2) addressed damping, not the scale grid.

#### A7 · Tile-sparse sub-1 (K1 on the most important tiles, zero elsewhere)
**Mechanism.** Code only the top 12.5–50% of 16×16 tiles (by H-weighted energy after rotation and LDLQ) at K1, with a 1-bit/tile mask (0.004 bit/weight).
- **Source:** New.
- **Why:** It concentrates bits where error matters. The prior is low: the Hadamard flattens tile energies.
- **Fit:** Format change.
- **Test:** CPU analytic tile-energy census on the M1 tensors. Pass if the top 25% of tiles hold ≥45% of the weighted energy. 20 CPU-min.
- **Gain/risk:** 1.0: 0–2%. Likely null.
- **Conflicts:** #96 (row slabs) suggests flatness.

#### A8 · Low-rate row water-filling with zero-rate rows (re-check #96 at sub-1)
**Mechanism.** At sub-1 rates, reverse water-filling gives the least important output rows (by Σdy²) *zero* bits. The gain then exceeds the high-rate AM/GM formula used for #96's 0.036-bit ceiling.
- **Source:** #96, recomputed in the low-rate regime.
- **Why:** At high rate the allocation gain is ½·log₂(AM/GM); at low rate, zero-rate rows give more.
- **Fit:** Format (row-slab K). Pre-freeze field question.
- **Test:** CPU: Σdy² row weights × M1 rel curves, low-rate water-filling ceiling at K0.25/0.5. Pass if ≥0.10 bit. 30 CPU-min.
- **Gain/risk:** 1.0: 0–3%.
- **Conflicts:** The user closed per-head K and row-slab K for V3 on 11-08. Only worth reopening if this ceiling exceeds 0.10 bit.

#### A9 · Shared right factor across siblings (gate/up; q/k/v; GDN qkv/z)
**Mechanism.** Siblings that read the same normalised input share one V: W_s ≈ U_s·Vᵀ. Vᵀx is computed once and V is stored once.
- **Source:** New.
- **Why:** Siblings' H-weighted top directions come from the same input directions (massive channels). M1's gate_up rows are already a *fused* SVD, which is the shared-V version.
- **Fit:** Format (as A1).
- **Test:** Add a separate-vs-shared SVD comparison to diagnostic 1. Pass if shared beats separate by ≥10% rank efficiency at equal bits.
- **Gain/risk:** +10–30% on A1's gain for these groups.
- **Conflicts:** None.

#### A10 · Spike-only option (exact columns for the top input directions)
**Mechanism.** Store a reader's columns for the top 1–4 H directions in bf16 (≈ channel 3994 early; another channel late) and zero the rest. The cost is m×16 bits per column, e.g. 0.003 bit/weight for ssm_z. It is a gather, not a GEMM.
- **Source:** New. A degenerate case of A1 with V = e_j.
- **Why:** M1: rank-1 holds 0.725 (ssm_z), 0.60 (attn_qg L43) and 0.74 (L62 gate_up) of the H-energy, roughly what trellis K0.125 achieves at 40× the bytes.
- **Fit:** Format (sparse-column side-car) plus knapsack floor option.
- **Test:** Diagnostic 1 add-on: is the top singular direction ≥0.9 aligned with a single input channel?
- **Gain/risk:** 1.0: 0–2%. 2.5: 0.
- **Conflicts:** #120-02 (writers' row restore, null at K2.5). Different case: readers, sub-1 floor.

#### A11 · Chained tail-biting for sub-1 encodings only
**Mechanism.** For sub-1 tiles, take the start window from the previous tile's end state (one wrap per n tiles) instead of tail-biting every tile.
- **Source:** #102 (negative at K2–3: n=4/n=1 ≈ 0.996). CurveTQ 2610.09212 (a stored start state was worth more than scale bits at 2-bit).
- **Why:** At K0.125 a tile holds 32 bits, against a 16-bit state window. The wrap constraint is a far larger share than the 512+ bits per tile at K2 where #102 was tested.
- **Fit:** Format change for the sub-1 variant only (a `mul1s`-style flag already exists). The decoder gains a cross-tile dependency.
- **Test:** colo `viterbi.c` (from #57/#102): free-start vs tail-biting vs chained at K0.125/0.25/0.5 on 6 tensors. Pass if chained ≤ 0.95× tail-biting. 1 CPU-h.
- **Gain/risk:** 1.0: 1–4% if free/TB at sub-1 is ≤0.85. 2.5: 0. Risk: decode parallelism.
- **Conflicts:** #102 tested only K2–3.

#### A12 · More wrap candidates (NTRY) in the sub-1 search
**Mechanism.** Raise exllamav3's wrap-candidate count to 16 on sub-1 levels only. Quantiser-only, no format change.
- **Source:** #57: 16 candidates recover 12/8/4% of the tail-biting gap at K1/2/3, so the gap likely grows below K1.
- **Fit:** Quantiser setting (sub-1 levels; quant time ×NTRY, on cheap levels).
- **Test:** Same run as A11.
- **Gain/risk:** 1.0: 0.5–2%.
- **Conflicts:** None.

#### A13 · Sparse-alphabet sub-1 codebook
**Mechanism.** At K≤0.375 the optimal reproduction is mostly zeros plus a few ±a values. Try a sub-1 codebook variant where most trellis states decode to 0 and the rest to wide levels, instead of the i4 grid's Gaussian-like spread.
- **Source:** New (entropy-constrained-VQ intuition).
- **Fit:** Format (codebook variant flag) plus quantiser.
- **Test:** colo `viterbi.c` with alternative state tables at K0.125–0.5 on 6 tensors. Pass if ≤0.95× the i4 grid. 1 CPU-h.
- **Gain/risk:** 1.0: 0–3%.
- **Conflicts:** #90-02 (i4 ≈ i8) compared grids, not sparsity.

#### A14 · Two-sided low-rank fit
**Mechanism.** Fit the factors in the adopted objective: SVD of diag(√h_out)·W·H_in^½.
- **Source:** New for us. Standard weighted SVD with one diagonal side.
- **Fit:** Per-level (A1 variant).
- **Test:** Diagnostic 1, both metrics.
- **Gain/risk:** +3–10% on A1's gain.
- **Conflicts:** None.

#### A15 · Post-quantisation low-rank correction of the trellis error
**Mechanism.** After trellis K, fit U·Vᵀ to the trellis error E in the H-metric (rank 4–32) and store it as a side-car.
- **Source:** ARCHead 2608.02703 (head rank 6: rel PPL 1.151 → 1.007 against plain INT4; only vs 1.0104 against a GPTQ head); QERA/LQER.
- **Why:** Only pays if LDLQ leaves structure in E·H^½.
- **Fit:** Format (as A1) plus per-level.
- **Test:** SVD of E·H^½ from settle64 outputs at K0.5/1/2.5 on 6 tensors. Pass if top-16 holds ≥15% of the error energy. 30 CPU-min.
- **Gain/risk:** 0–2% at 1.0–1.5; probably ~0 at 2.5.
- **Conflicts:** None.

#### A16 · Drop trellis K0.125/K0.25 for families where low-rank dominates
**Mechanism.** Do not build trellis K0.125/K0.25 for ssm_z, attn_qg, ssm_qkv and attn_o (and early attn_k/v) if diagnostic 1 confirms that low-rank or spike options dominate there.
- **Source:** M1.
- **Fit:** Level menu.
- **Test:** Diagnostic 1.
- **Gain/risk:** Saves roughly 20–30% of the sub-1 ladder hours (~2–7 card-h). No KL change.
- **Conflicts:** #90 (sub-1 kept for the research ladder). Here the menu changes, not its existence.

### B. Structured removal options (K0-type) for 1.0–1.5

#### B1 · Neuron-pruned MLP options for K≤1.5 levels
**Mechanism.** Per layer, remove the p% of intermediate neurons (p ∈ {12.5, 25, 50}) with the lowest down-proj H-energy and LS-compensate the kept down columns (OBS). Permute so the kept neurons are contiguous, then quantise the kept gate/up rows and down columns at each low K. The knapsack picks (p, K) per layer.
- **Source:** Wanda/SparseGPT family, F-WANDA 2608.00481, ProQuant ({0,2,3} beat {1,2,3}). New as a knapsack option on trellis levels.
- **Why:** M3. Dropping half the neurons costs rel 0.10 (L62) to 0.33 (L30) on down, adds no extra error on gate/up, and removes 50% of the bits of all three tensors. The MLP is 64% of the params.
  - At L62, drop-50 plus K1 on the kept neurons beats trellis K0.5 on both down (≈0.19 vs 0.24) and gate/up (≈0.036 vs 0.081) at equal bits.
  - At L30 the trellis wins on down.
- **Fit:** Format (per-layer intermediate size in the container; the MLP op takes N per layer) plus per-level pruned variants (MLP quant ×#p, low levels only).
- **Test:** Diagnostic 4 (drop census with OBS, all 64 layers) → knap5 with pruned options → one 27B pair at 1.0/1.25.
- **Gain/risk:** 2.5: 0. 1.5: 0–1%. 1.25: 1–4%. 1.0: 2–8%. Risks:
  - Structured error may cost more KL per proxy unit (G6).
  - #21: the L63 gate_up spike means late layers are sensitive. Concentrated energy is not the same as low sensitivity.
- **Conflicts:** None. #81's A1 (down predicted from up/gate) is a different idea.

#### B2 · GDN value-head pruning at ≤1.0 bpw
**Mechanism.** Drop the least important GDN v-heads per layer (48 per layer). This removes v rows of qkv, rows of z, columns of out_proj, and rows of a/b.
- **Source:** New for us.
- **Fit:** Format; the architecture plugin needs per-layer head counts and state shapes.
- **Test:** CPU exact head-drop error from out_proj's H (as M3). Pass if dropping the bottom 12.5% of heads costs ≤ rel 0.05.
- **Gain/risk:** 1.0: 0–2%. High plugin risk.
- **Conflicts:** Per-head K is closed (11-08); this is removal, not K.

#### B3 · q-head pruning in the 16 full-attention layers
**Mechanism.** Drop low-importance q heads (24 per layer; KV stays 4): q rows plus o_proj columns.
- **Fit:** Format.
- **Test:** CPU exact o_proj column-drop error.
- **Gain/risk:** 1.0: 0–0.5% (only 16 layers).
- **Conflicts:** None.

#### B4 · Depth pruning for the 1.0 size
**Mechanism.** Remove 1–3 middle GDN layers (each about 1.56% of params) and spend the bytes on the rest.
- **Source:** ShortGPT family. Usually costly in KL.
- **Fit:** Format (per-container layer list).
- **Test:** 27B bf16 with one layer skipped, scored on golden-search27b (1 eval per layer, card), compared with +0.016 bpw worth of KL.
- **Gain/risk:** 1.0: −3% to +2%. Low prior.
- **Conflicts:** None.

### C. Fixed overheads at 1.0–2.5

#### C1 · GDN a/b into the trellis menu
**Mechanism.** Quantise `in_proj_a/b` (48 layers × 96×5120 ≈ 23.6M params, bf16 today, 0.014 bpw) at i8 K3–4. The output Hadamard does not fit N=48: use input-side rotation only (svh scales on the output) or pad to 128.
- **Source:** 11-07 lowbpw idea #2 (not tested since). 2609.04098 on this model: a 2.1% and b 2.6% median GDN output error under NVFP4, the least sensitive projections (qkv 10.4%, out 12.7%), and a full NVFP4 W4A4 of all 496 linears at task parity.
- **Why:** At K4, a/b would cost 0.0035 bpw and free 0.0105 bpw.
- **Fit:** Per-level library (tiny tensors) plus format (an N=48 encoding variant).
- **Test:** CPU proxy curves of a/b at K2–4 (minutes), then one 27B swap (bf16 → K4) on golden-search27b. Pass if the ratio ≤ 1.003.
- **Gain/risk:** **2.5: ≈1.5% ± 0.7. 1.0: ≈3%.** Risk: a errors feed the decay gate. 2609.04098 found them benign.
- **Conflicts:** D6 (11-01) closed merging a/b *GEMMs*. This is weight quantisation.

#### C2 · Head menu for ≤1.5: K1/K1.5 donors and low floors
**Mechanism.** Add head donors at K1 and K1.5 (plus i4 K0.5–0.875 for the tail slabs). Extend w88's slab grid below its K2 floor, priced by headsim with the rare-token guard.
- **Source:** #34 (slab head BUILD), w88 (grid floors start at K2).
- **Why:** A K1 head is 0.047 bpw, 4.7% of a 1.0-bpw budget. Today the low sizes extrapolate the head price below K2.
- **Fit:** Per-level library (head donors, after the trunk) plus knapsack input (KN_HEADGRID).
- **Test:** headsim curves for the new donors (each head quant ≈1 card-h, ~22 GB host RAM), then knap5 at 1.0/1.25.
- **Gain/risk:** 1.0: 2–6%. 1.25: 1–3%. 2.5: 0.
- **Conflicts:** #34. This extends its grid downward.

#### C3 · H-weighted low-rank head core (generalise mean-row to rank-r)
**Mechanism.** Before the slab/trellis head, remove the top-r centred directions in the head-input metric. The head already carries a mean-row correction; rank-r extends it.
- **Source:** 11-07 idea #1 (untested). ARCHead 2608.02703: Qwen3-8B head at 25.6% of bf16 bytes, rel PPL 1.0058–1.0062 vs a GPTQ head 1.0104–1.0105.
- **Why:** M4: after mean-row, the centred top-16/64 hold 0.45/0.54 of the remaining H-energy. Rank 64 at int8 costs 0.0048 bpw.
- **Fit:** Format (mean-row field → rank-r) plus head donors.
- **Test:** headsim (exact head-only KL on 32,768 held-out head inputs) on deflated K1/K2/K3 donors at equal bytes. Pass if ≥15% lower head KL.
- **Gain/risk:** 2.5: 0.2–1% (head ≈5% of KL on calib-like text). 1.0: 1–4%. Risk: the energy captured is H-weighted, while the trellis error depends on plain RMS. headsim decides.
- **Conflicts:** #34 mean-row (adopted). This generalises it.

#### C4 · Low-rank-only head option for 1.0
**Mechanism.** A head stored only as mean-row plus rank-256 int8 (≈0.019 bpw), compared against trellis K0.5 (0.024 bpw).
- **Source:** New (from M4).
- **Fit:** Format (as C3).
- **Test:** headsim, same run as C3.
- **Gain/risk:** 1.0: 0–3%.
- **Conflicts:** None.

#### C5 · Real-KL pricing of the sub-1 embedding levels
**Mechanism.** Measure t0.25 / t0.5 / t1 swaps on a ~1.25-bpw base. Today KN_EMBT extrapolates chain1's t2–t4 swaps along the CPU curve.
- **Source:** #119's open "next (3)". #97 (all sub-1 embedding levels on the hull).
- **Fit:** Knapsack input.
- **Test:** 3 swaps on 27B golden-search27b, reading the calib part (C8).
- **Gain/risk:** 1.0–1.25: 0–3% (sign unknown). 2.5: 0.
- **Conflicts:** #119 closed the representation question, not pricing.

#### C6 · Metadata diet: suh/svh to 8-bit log scales
**Mechanism.** The per-channel scales are fp16, Σ(m+n) ≈ 6.8M values ≈ 0.004 bpw. Store them as 8-bit log (or e5m3) with a per-tensor offset, and refit after rounding.
- **Source:** New.
- **Why:** Saves ≈0.002 bpw.
- **Fit:** Format (scale dtype field).
- **Test:** CPU: round the scales of 64 decoded tensors to 8-bit log, refit, proxy delta. Pass if ≤0.1% rel worse. 20 CPU-min.
- **Gain/risk:** 2.5: ≈0.3%. 1.0: ≈0.5%. Risk: the scales carry the swj/C9/H_out refits.
- **Conflicts:** #110 went the other way (more scale bits).

### D. Quantiser settings for K0.125–2.5 levels (cheap; several are post-code refits)

#### D1 · Two-sided input-scale refit
**Mechanism.** Weight the suh step by the H_out diagonal: A = (Q·diag(h)·Qᵀ)∘H, b = rowsum(Q∘HW∘h). The svh step is unchanged.
- **Source:** New; from code reading (M5-1).
- **Why:** Today the post-LDLQ refit re-optimises suh in the *one-sided* metric, partly undoing the adopted two-sided objective. #118 measured H_out levels at ×1.076 one-sided while KL falls.
- **Fit:** Quantiser (per-level, closed form, free). Post-code, so with H6 it no longer blocks the freeze.
- **Test:** Diagnostic 2: w87 held-out two-sided proxy on 20 tensors at K0.5/1/2.5. Pass if ≥0.5% better. Then one 27B K2.5 pair (rule g ≤ 0.997).
- **Gain/risk:** 2.5: 0.1–0.7%. 1.0: 0.3–2%.
- **Conflicts:** None.

#### D2 · Refit in the undamped (lightly damped) H
**Mechanism.** The k+n-parameter LS refit needs no regularisation. Fit it with σ_refit ∈ {0, 0.001} instead of the LDLQ damping (0.2 at K0.125, 0.05 for sub-1 inputs, 0.025 for the rest).
- **Source:** New (M5-1). Consistent with #114's K0.125 GPU error of 1.014 despite a per-column LS refit, which cannot exceed 1 in its own metric.
- **Fit:** Quantiser (post-code).
- **Test:** Diagnostic 2, held-out split halves (w77/w87). Pass if ≥1% at sub-1 or ≥0.3% at K2.5.
- **Gain/risk:** 2.5: 0–0.3%. 1.0: 0.5–3%. Risk: overfitting the calibration H (the split halves guard against it).
- **Conflicts:** #114/#121 tuned the *LDLQ* damping, not the refit's.

#### D3 · Measured sub-1 LDLQ drift table
**Mechanism.** Measure the post-feedback RMS growth at K0.125–0.875 (with H_out blocks and block order) and extend `LDLQ_DRIFT` below K1, instead of clamping at 1.08.
- **Source:** New (M5-2).
- **Why:** At sub-1 the feedback is large, so a 1.08 inflation mis-scales the g_scale samples.
- **Fit:** Quantiser (sub-1 levels; pre-code, so it does block the freeze).
- **Test:** Read `drift`/`g_scale` from the w80 i4 GPU logs, or colo settle64. Pass if the measured drift is ≥1.2 at K0.5. 30 CPU-min.
- **Gain/risk:** 1.0: 0–2%. 2.5: 0.
- **Conflicts:** #107 tested interpolation at K≥1.

#### D4 · Weighted g_scale search
**Mechanism.** Choose g_scale by the two-sided, LDL-pivot-weighted tile error instead of the unweighted sample MSE.
- **Source:** H-Scale 2608.28113, Schur Replay 2609.36654 (scale scored by δᵀ·S·δ), ScaleSweep.
- **Why:** At low K the shrink matters most.
- **Fit:** Quantiser (pre-code).
- **Test:** colo settle64 at K0.25/1/2.5, held-out error. Pass if ≥0.5% better. 1 CPU-h.
- **Gain/risk:** 2.5: 0–0.3%. 1.0: 0.3–1.5%.
- **Conflicts:** #39 kept a global multiplier of 1.0. This changes the search objective.

#### D5 · Tile CD at K≤1.875 on all families except attn_k
**Mechanism.** Run cd3 on all families except attn_k, only on sub-1 and K1–1.875 levels (13 of 55).
- **Source:** #77 (attn_qg only; at K2.5 others gain 0.996–0.998 on GPU).
- **Why:** CD gains plausibly grow as K falls.
- **Fit:** Quantiser (pre-code). Quant time ×3.7 on cheap levels only, roughly +2–3 card-h.
- **Test:** colo settle64 (order-fixed) cd3 vs LDLQ at K0.25/0.5/1/1.5, 27 tensors × 2 samples. Pass if median ≤ 0.99 at ≥2 of those K. 2–3 CPU-h.
- **Gain/risk:** 1.0–1.5: 0.3–1%. 2.5: 0.
- **Conflicts:** Tile CD on attn_k is closed (11-08), hence excluded.

#### D6 · Soft-decision (BCJR) init plus tile CD
**Mechanism.** Per tile, run forward–backward over the trellis on the LDLQ targets, take marginal-MAP codes, then CD sweeps. Hard Viterbi with feedback commits early at low K.
- **Source:** MPQ 2610.32127 (Llama-3-8B 2-bit E8P12: LDLQ 15.01 → +CD 12.76 → MPQ 11.63 PPL). Their denoiser enumerates codewords; ours would use BCJR over 2¹⁶ states.
- **Fit:** Quantiser (pre-code, about 2× Viterbi time).
- **Test:** colo prototype on 6 tensors at K0.5/1/2. Pass if ≤ 0.98× LDLQ+CD. ~1 CPU-day to build.
- **Gain/risk:** 2.5: 0–0.5%. 1.0: 0.5–2% (⅓ shrink applied).
- **Conflicts:** #113 (larger state space) is different. #77.

#### D7 · Tempered metrics at sub-1 (H_out^p, H_in^p, trace-preserved)
**Mechanism.** Use p ∈ {0.5, 0.75} at K≤0.5, where ‖E‖ ≈ 0.3–0.6‖W‖ is far outside the quadratic regime.
- **Source:** ShamAN-Q 2609.38521 (1.7B at ~1 bpw: untempered 19.21 vs p=½ 17.46).
- **Fit:** Quantiser (pre-code).
- **Test:** A quadratic held-out proxy cannot judge this, so it needs KL: a 0.8B pair (the lead's 4070) or a 27B pair at a K0.5 level.
- **Gain/risk:** 1.0: 0–3%. Likely small.
- **Conflicts:** #114/#121 are additive damping shifts; tempering is spectral.

#### D8 · C9 α for v-type readers at K≤1.5
**Mechanism.** Extend the in-job C9 from writers to attn_v, the v slice of ssm_qkv, and ssm_z at low K.
- **Why:** q/k slices are norm-invariant (q_norm/k_norm; GDN L2-norm), so their shrink is harmless. v and z shrink is not, and r90's median scale factor is 0.85 at K0.125/0.25.
- **Source:** #85 (C9 K3 0.973 / K2 0.911 on 0.8B), #108 (junction refits failed at K2/K3 in the full stack), #112.
- **Fit:** Per-level fixed codes (post-code).
- **Test:** 27B fixed-code pair at a 1.25 mix, or 0.8B.
- **Gain/risk:** 2.5: 0–0.3%. 1.0–1.25: 0.5–3%.
- **Conflicts:** #108. Why it might work now: the sub-1 shrink is far larger than at K2/K3.

#### D9 · Attention temperature restoration per q/k level
**Mechanism.** q/k noise shrinks the logit cosine; q_norm/k_norm remove the norm but not the cosine loss. Scale the q_norm (k_norm) weight per head by 1/cos, estimated from the level's own calib error, and attach the delta to the q_proj (k_proj) level. Full-attention layers only: GDN outputs are RMS-normed, so there is no analogue.
- **Source:** 11-07 B4 (untested).
- **Why:** At K≤1, rel 0.05–0.2 gives cos ≈ 0.9–0.97, so logits run 3–10% too cool per side.
- **Fit:** Per-level (a norm-weight delta).
- **Test:** CPU estimate of the per-head cosine from calib activations, then 0.8B KL.
- **Gain/risk:** 1.0: 0.3–2%. 2.5: 0–0.2%.
- **Conflicts:** None.

#### D10 · Damped LDLQ feedback at sub-1
**Mechanism.** Scale the feedback by γ_i < 1 (decaying with normalised [H⁻¹]_ii) at K≤0.5, where full feedback inflates later columns.
- **Source:** QTEA 2609.00224 (error decay: +0.46 average zero-shot at 14B ternary).
- **Fit:** Quantiser (pre-code).
- **Test:** colo settle64, γ ∈ {1, 0.9, 0.8} at K0.125–0.5. Pass if ≥2% held-out.
- **Gain/risk:** 1.0: 0–2%.
- **Conflicts:** None.

#### D11 · Block order at sub-1
**Mechanism.** #117's heavy-last order was decided at K2.5. Check heavy-first vs heavy-last at K0.25/0.5.
- **Source:** #117 (geo 0.9968 at K2.5).
- **Fit:** Quantiser setting (per K band).
- **Test:** colo settle64. Pass if the other order wins by ≥1% at sub-1. 30 CPU-min.
- **Gain/risk:** 1.0: 0–1%.
- **Conflicts:** #117.

### E. Trellis, codebook and format at K1–2.5

#### E1 · Curvature-weighted Viterbi, gated on a Γ_b census
**Mechanism.** Weight the Viterbi branch metric by the LDL pivot D_ii within each 16-column tile.
- **Why:** The gain is proportional to the within-tile anisotropy Γ_b. With 128-point (not full) Hadamard blocks and the #117 order, Γ_b may exceed QTIP's.
- **Source:** CurveTQ 2610.09212 (Γ_b 0.038 → 0.00013 after a full RHT).
- **Fit:** Quantiser (pre-code; same format).
- **Test:** CPU: block-LDL of the rotated H for 60 tensors, Γ_b per family. Implement only if ≥0.01 bit. 30 CPU-min.
- **Gain/risk:** 2.5: 0–0.5%.
- **Conflicts:** None.

#### E2 · 1/16 K steps at K1–2
**Mechanism.** Add 8 levels (K1.0625, …, K1.9375).
- **Why:** The slope is steep there (≈×0.79 rel per ⅛ step below K2).
- **Source:** w21 (⅛ steps at K1.125–1.875: −12% predicted at 1.5–1.75). k4levels (extra steps at K3–5 ≤ 0.8%).
- **Fit:** Level menu.
- **Test:** knap5 with interpolated 1/16 levels (KN_FRACS_EXTRA) at 1.25–2.0. Pass if ≥1% predicted. CPU minutes.
- **Gain/risk:** 2.5: 0–0.2%. 1.5: 0–1%. About +4 card-h of ladder.
- **Conflicts:** #36/k4levels covered higher K.

#### E3 · Extra fp feedback columns inside LDLQ (GPTQ-intrinsic low rank)
**Mechanism.** Augment the Hessian with r unquantised directions (top eigenvectors of H) whose fp coefficients absorb rounding error during the sweep. This is a joint-solve version of A4.
- **Source:** 2606.01412 (Qwen3-1.7B 3-bit: GPTQ 50.48, GPTQ+OLrC r=64 27.71, intrinsic r=64 20.49; low-rank bits not charged there).
- **Fit:** Quantiser plus format (A1's field).
- **Test:** colo settle64 at K0.5/1 on low-rank-ish tensors, charging the bits. Pass if ≥5% over A4.
- **Gain/risk:** +10–30% on A4.
- **Conflicts:** None.

#### E4 · Joint code and scale re-search
**Mechanism.** After LDLQ and the refit, re-run Viterbi per tile against the refitted scales (with feedback), for 1–2 rounds.
- **Source:** CoS-VQ 2610.37336 (same start: scale-only 21.01 → joint 13.68 Wiki2, Qwen3-4B 2-bit); JARQ 2609.38599.
- **Fit:** Quantiser (pre-code; quant time ×2–3).
- **Test:** colo settle64 at K0.5/1/2.5. Pass if ≤0.98.
- **Gain/risk:** 2.5: 0–0.5%. 1.0: 0.5–2%.
- **Conflicts:** #77. Tile CD optimises codes before the refit, not after.

#### E5 · Partial activation-denoising target (re-test #88 with ρ < λ)
**Mechanism.** Use the pretransformed target W·F and metric G from Thm 1 of 2610.07522, with small propagated noise (γ ≈ 0.05) and partial denoising (e.g. λ = 0.4, ρ = 0.3). It is per-level and mix-independent.
- **Source:** 2610.07522 (closes 18–31% of the parallel→sequential gap at W3; QuIP# W2 Llama-3-8B 13.76 → 12.70, better than sequential).
- **Fit:** Quantiser. Needs per-linear C and N statistics from one noisy forward pass (≈12 card-h at 27B, forward-only).
- **Test:** 0.8B first (λ, ρ sweep), then 27B. BACKLOG lists #88 as depth-dependent.
- **Gain/risk:** 2.5: 0–1%. Cost is the collection.
- **Conflicts:** #88 (K3 1.031, K2 1.10) used full level-matched noise. Partial ρ < λ and small γ are the difference.

#### E6 · Per-input-block K for the massive-channel block
**Mechanism.** Give the 128-input block that holds the dominant H direction (early layers: 0.8–0.93 of the trace) extra bits.
- **Source:** New.
- **Why:** Likely small, because LDLQ concentrates the spike in about one pivot.
- **Fit:** Format (column-block K).
- **Test:** H1's per-input-block error maps (free once emitted), or a settle64 census. Pass if one block holds ≥15% of the weighted error at K≤1.
- **Gain/risk:** 0–1%.
- **Conflicts:** Per-head/row-slab K is closed. This is a different axis (input blocks).

### F. GDN-, attention- and channel-specific

#### F1 · Split fused ssm_qkv (q/k/v) and attn_qg (q/gate) into separate knapsack items
**Mechanism.** Quantise the slices as separate tensors (shared group Hessian and su draw), each with its own K.
- **Why:** The q/k slices are norm-invariant and enter the delta rule's erase step (k); v and gate are not. Their sensitivities likely differ, and granularity matters more at low bpw.
- **Source:** 11-07 idea #4 (untested).
- **Fit:** Per-level library (more jobs, no extra Hessian) plus format (three exl3i tensors; launches merge when K is equal).
- **Test:** Per-slice proxy curves from existing levels (decode, split rows), then 2–4 27B group swaps (q-slice only up or down) for KL factors.
- **Gain/risk:** 2.5: 0.3–1.5%. 1.0–1.5: 1–3%. Kernel launches increase.
- **Conflicts:** Per-head K is closed (11-08). This needs no per-row K field (separate tensors), which is why it is not the same decision.

#### F2 · Reader massive-column census before the side-car decision
**Mechanism.** Measure the share of the readers' LDLQ-weighted error in the input coordinate(s) of 3994 and the late-layer equivalents, at K0.25/0.5/1/2.5.
- **Why:** If ≥15% at sub-1, reserve the A10 side-car in exl3i.
- **Source:** #120-01/-02 measured writers only.
- **Fit:** Diagnostic for a format decision.
- **Test:** colo settle64 on 6 early and 2 late readers. 1 CPU-h.
- **Gain/risk:** Decides A10/A4.
- **Conflicts:** #120 (closed for writers).

#### F3 · GDN head-lifetime prior in the knapsack
**Mechanism.** Use each GDN layer's head-decay half-life (from A_log/dt_bias and calib a statistics) as a feature for GDN tensors' extrapolated (sub-1) prices.
- **Source:** STEPQuant 2609.38169 (half-life vs accumulated INT6 state error, ρ ≈ 0.80 over 2,304 heads of Qwen3.8-27B).
- **Fit:** Knapsack input.
- **Test:** Correlate with measured GDN group sensitivities from V2 probes. CPU minutes. Pass if |ρ| ≥ 0.4.
- **Gain/risk:** 1.0: 0–1%.
- **Conflicts:** None.

#### F4 · C9 α re-fit for writers at sub-1 and K1
**Mechanism.** Grid α per K band (gdn_out sits at the grid edge of 3 at K2/K3), especially at sub-1.
- **Source:** #112, #112-01 (α 4–4.75: −0.1 to −0.4%).
- **Fit:** Per-level fixed codes (post-code).
- **Test:** 27B fixed codes on sub-1 levels, or 0.8B.
- **Gain/risk:** 1.0: 0.3–1.5%. 2.5: 0.
- **Conflicts:** #112. The sub-1 shrink is larger.

#### F5 · Sub-1 floors for early attention k/v
**Mechanism.** In L0–L15, forbid trellis K < 0.375 for attn_k/v, or route them to low-rank (L3: low-rank 0.40/0.69 vs trellis 0.93/1.50 at K0.125).
- **Fit:** Knapsack constraint.
- **Test:** knap5 at 1.0.
- **Gain/risk:** 1.0: 0–1%.
- **Conflicts:** None.

### G. Knapsack inputs at low budgets

#### G1 · Apply #118's per-family offsets
**Mechanism.** Price attn_k ×0.73, attn_v ×0.81, attn_qg ×0.92 and ssm_qkv ×0.95 in knap5.
- **Why:** knap5 overprices attention error by 8–28%, pushing bits into small k/v tensors. The effect is larger at low budgets.
- **Source:** #118 (pending with the lead).
- **Fit:** Knapsack input.
- **Test:** knap5 re-solve at 1.0–2.5. Run a pair only if ≥1% of bytes move.
- **Gain/risk:** 2.5: 0.2–1%. 1.0–1.5: 0.5–2%.
- **Conflicts:** None.

#### G2 · Saturation-aware composition at ≤1.75
**Mechanism.** Fit KL = C·(1 − exp(−P/C)) on V2's low mixes (13 + the r9 files on gpubox), then on 6–10 V3 mixes at 0.75–1.5.
- **Source:** BACKLOG ("Saturation at 0.5–2 bpw"); kfit2+sat gave C ≈ 1.8.
- **Fit:** Knapsack input.
- **Test:** CPU refit of the existing rows; predicted allocation change at 1.0/1.25.
- **Gain/risk:** 1.0–1.25: 1–5% (allocation). 2.5: 0.
- **Conflicts:** 11-03 saturation check (null at ≥1.6 bpw).

#### G3 · Real-KL anchors for the sub-1 levels
**Mechanism.** Run 4 group swaps on a 1.25 base (e.g. all ssm_out K1 → K0.5; gate_up L48–63 K1 → K0.5).
- **Why:** #44: proxy pricing underestimates K1-heavy swaps by 20–36%. #114: the proxy is ~13% optimistic at K0.125.
- **Fit:** Knapsack input.
- **Test:** 4 evals on card 0, once sub-1 levels exist.
- **Gain/risk:** 1.0–1.25: 2–6% (sign unknown).
- **Conflicts:** #44.

#### G4 · Low-base context sensitivities
**Mechanism.** Re-measure 10–20 group sensitivities with the rest of the model at ~1.5 bpw rather than V2's 4-bpw base. Refit the κ compounding.
- **Source:** MixQuant 2607.23047 (context-aware scores: Llama-3.2-3B 1.25 GB AWQ PPL 10.70 vs 12.43); GEMQ.
- **Fit:** Knapsack input.
- **Test:** ~10 h on card 0.
- **Gain/risk:** 1.0–1.5: 1–4%. 2.5: 0–0.5%.
- **Conflicts:** #14 (single swaps too noisy). Use group swaps.

#### G5 · Intra-layer pair terms in the layer DP
**Mechanism.** Measure Δ(pair) − Δ(a) − Δ(b) for (gate, up, down) and (ssm_qkv, z, out) at K1 on a 1.5 base. knap5's DP runs layer by layer, so intra-layer pair terms can be added exactly.
- **Source:** TWLA ILA-AMP 2606.13054 (pairwise DP 12.83 vs unary 15.03 PPL), LoopQuant, 2609.09867 (composition error dominates).
- **Fit:** Knapsack input.
- **Test:** 10–20 evals on card 0.
- **Gain/risk:** 1.0–1.5: 1–3%. 2.5: 0–1%.
- **Conflicts:** #11 (symmetric depth compounding) is different.

#### G6 · KL-per-proxy factors for structured error types
**Mechanism.** At matched proxy error, measure KL for low-rank, neuron-drop and spike-only options against trellis (2–3 group swaps each).
- **Why:** These options cannot enter the knapsack honestly without this.
- **Fit:** Knapsack input.
- **Test:** ~6 evals on card 0 after diagnostics 1 and 4.
- **Gain/risk:** Gate for A1/B1/A10.
- **Conflicts:** None.

#### G7 · Price head and embedding from the calib part
**Mechanism.** Refit the head/embedding terms on calib-part rows. The head's share is ~5% on calib-style documents vs 0.3% on xcorp, so pooled two-part slices understate it.
- **Source:** LOG 2:55 PM entry (#34-02 consequence).
- **Fit:** Knapsack input (reweight existing measurements).
- **Test:** CPU refit plus knap5 re-solve.
- **Gain/risk:** 2.5: 0.2–1%. Low sizes: 0.5–2%.
- **Conflicts:** None.

#### G8 · Guard cells outside the measured range
**Mechanism.** Forbid options whose price is extrapolated beyond the proxy→KL fit range. Keep them only after G3/G6 anchors exist.
- **Source:** HQMQ 2605.27646 App. R (greedy allocation into an unmeasured rung: +0.471 PPL vs +0.175 uniform).
- **Fit:** Knapsack input.
- **Test:** knap5.
- **Gain/risk:** Risk control at 1.0–1.25.
- **Conflicts:** None.

#### G9 · Uncertainty-aware solve for extrapolated options
**Mechanism.** Penalise options by the bootstrap spread of their fitted price, as a soft version of G8.
- **Source:** BACKLOG B11.
- **Fit:** Knapsack input.
- **Test:** knap5 under bootstrap refits.
- **Gain/risk:** 1.0: 0–2%.
- **Conflicts:** None.

#### G10 · Family factors per K band with partial pooling
**Mechanism.** Fit separate family factors for sub-1, K1–1.875 and K≥2.
- **Source:** #44 (K1 factor 1.35 vs 1.09 for K2–3). BACKLOG "partial pooling".
- **Fit:** Knapsack input.
- **Test:** kfit2 CV.
- **Gain/risk:** 1.0–1.5: 0.5–2%.
- **Conflicts:** None.

#### G11 · Category reweighting at low sizes
**Mechanism.** Weight per-row scores by category to the private corpus' mix. Agentic text gains most and may dominate at low sizes.
- **Source:** Review 11-07 item 9.
- **Fit:** Knapsack input (free reweight).
- **Test:** CPU.
- **Gain/risk:** 0–1%.
- **Conflicts:** None.

#### G12 · Position check at low bpw
**Mechanism.** One per-position-bin readout of a 1.0–1.25 mix. #52's flat KL-vs-position was measured at 3–4.5 bpw; GDN state error may grow with position at 1.0. If it does, weight long-document rows.
- **Fit:** Knapsack input.
- **Test:** `posbins.py` on existing rows of one eval.
- **Gain/risk:** 0–2% at 1.0.
- **Conflicts:** #52.

### H. Library outputs that are cheap only before the freeze

#### H1 · Emit per-tensor error maps from every ladder job
**Mechanism.** Record per-output-row, per-128-input-block and per-head weighted error (one- and two-sided), plus g_scale and drift.
- **Why:** Near-free; enables E6, A8, B2 and D3 without rerunning 55 levels.
- **Fit:** Pipeline (pre-freeze, since the ladder runs once).
- **Test:** A dry-run job.
- **Gain/risk:** Enabling.
- **Conflicts:** None.

#### H2 · Save the top-256 H-weighted singular vectors per tensor during Hessian prep
**Mechanism.** Store U, Σ and V (fp16) per tensor while the Hessian is prepared.
- **Why:** A1, A4, A9 and C3 can then be built later without reloading.
- **Fit:** Pipeline.
- **Test:** Time on one layer.
- **Gain/risk:** Enabling, about +5% job time.
- **Conflicts:** None.

#### H6 · Make the scale refit re-runnable after the freeze
**Mechanism.** Keep each job's pre-refit su/sv (or ensure codes, W and H are pinned). Then D1, D2, D8 and F4, all post-code closed-form refits, can be redone on CPU later without Viterbi. This turns four pre-freeze questions into post-freeze knobs.
- **Fit:** Pipeline.
- **Test:** Re-run the refit on one stored job; it must reproduce bit for bit.
- **Gain/risk:** Large option value; cost ~0.
- **Conflicts:** None.

#### H3 · Build pruned-MLP variants only where diagnostic 4 predicts a win
**Mechanism.** Limit B1's variant levels to low K and to layers (likely late) where the drop census predicts a gain.
- **Fit:** Level menu.
- **Gain/risk:** Cost control.
- **Conflicts:** None.

---

## 3. Ruled out or not repeated (with reason)

- **i8c revival** (11-07 top-5 #5). Its premise is gone: H_out is now on at every level (`apply_k_min 0.0`), and #78-04 found i8c adds nothing on top of H_out.
- **Closed by the user on 11-08:** per-head K and row-slab K (#96), the 512-point Hadamard, per-tile salt, tile CD on attn_k. A8 and E6 are only conditional re-openers with stated pass bars.
- **#113** (2²⁰ beam, exact V18, random table).
- **Damping:** #114, #114-03, #121 (damping per level/family and its normalisation).
- **Per-level asym and GPTAQ-type quantized-prefix targets** (#105, #111, #88-02).
- **Bias side-car** (#120-03): post-freeze, needs no field.
- **Writers' row-3994 restore** (#120-02).
- **Learned rotations (HARP 2605.29843):** +0.02–0.08 bpw is 5–25% KL at 1.0, for a 0.7% gain at QTIP 3-bit.
- **Explicit KLT / data-aware bases:** the basis costs more than it saves (M2). SpectrumQuant shows ~nothing at 70B.
- **Low-rank or KLT for the embedding:** H-flat (M4); see #119.
- **Head vocabulary trimming** (#32).
- **Best-of-N seeds** (#41). **Per-group scales** (#40). **A4** (#46-03). **Hash codebook** (#100). **Clipped i8** (#109). **A1 down-predictor** (#81-04). **R1** (#47).
- **New Σdy² / H_out collections** (user, 11-05).
- **Trellis K0.0625:** low-rank dominates there (M1).
- **2:4 sparsity:** no RDNA4 path, and moesq loses to dense NVFP4.
- **Out of scope:** QAD, healing, per-mix passes and kernels.

---

## 4. Top 20 by expected gain per unit cost

Cost: S = ≤2 CPU-h or code; M = ≤1 card-day or a small format change; L = format plus kernel, or several card-days.

| # | idea | cost | 2.5 bpw | 1.0–1.25 bpw | why ranked here |
|---|---|---|---|---|---|
| 1 | D1 two-sided suh refit | S | 0.1–0.7% | 0.3–2% | free fix for a mismatch with an adopted setting |
| 2 | H6 re-runnable refit | S | enabling | enabling | moves D1/D2/D8/F4 off the freeze's critical path |
| 3 | G1 #118 offsets | S | 0.2–1% | 0.5–2% | already measured; one re-solve |
| 4 | D2 undamped refit | S | 0–0.3% | 0.5–3% | free; explains #114's worse-than-zero cell |
| 5 | C1 GDN a/b quantized | M | ≈1.5% | ≈3% | 0.0105 bpw freed; this model's data says insensitive |
| 6 | G7 calib-part head/emb pricing | S | 0.2–1% | 0.5–2% | reweight existing rows |
| 7 | A1 low-rank sub-1 options | S→L | 0 | 3–10% | measured wins on 13/17 tensors |
| 8 | B1 neuron-pruned MLP options | M–L | 0 | 2–8% | measured; late layers clearly favourable |
| 9 | C3 low-rank head core | S (headsim) | 0.2–1% | 1–4% | extends the existing mean-row field |
| 10 | C2 head K1/1.5 donors and low floors | M | 0 | 2–6% | head price today extrapolated below K2 |
| 11 | A11+A12 sub-1 tail-biting / NTRY | S test | 0 | 1–4% | wrap overhead dominates 32-bit tiles |
| 12 | F1 split fused qkv/qg | M | 0.3–1.5% | 1–3% | knapsack granularity; no per-row field |
| 13 | G5 intra-layer pair terms | M | 0–1% | 1–3% | exact inside the layer DP |
| 14 | D3 sub-1 drift table | S | 0 | 0–2% | the 1.08 clamp is a plain bug-class issue |
| 15 | G2 saturation fit | S–M | 0 | 1–5% | allocation at KL ≈ 1 is extrapolated |
| 16 | D5 tile CD at K≤1.875 | S test | 0 | 0.3–1% | ladder time only on cheap levels |
| 17 | A4/E3 low-rank + trellis hybrid | S test | 0–0.3% | 1–3% | M2 headroom in GDN readers |
| 18 | G4 low-base sensitivities | M | 0–0.5% | 1–4% | V2 sensitivities came from a 4-bpw base |
| 19 | D4 weighted g_scale search | S | 0–0.3% | 0.3–1.5% | shrink matters at low K |
| 20 | C6 metadata diet | S–M | ≈0.3% | ≈0.5% | 0.002 bpw for a dtype field |

---

## 5. The five cheapest diagnostics that would most change this ranking

1. **Full low-rank / spike / trellis census** (colo, ~3–4 CPU-h; extends `lr90.py`).
   - Scope: all 64 layers × all families at K0.0625–1.
   - Metrics: one- and two-sided (A14); int4 and trellis factor error (A2); separate vs shared-V siblings (A9); top singular direction vs single input channel (A10/F2).
   - It decides A1, A2, A9, A10, A16 and whether to reopen the exl3i side-car field.
2. **Refit-metric census** (colo settle64 or decoded levels, w87 held-out halves, 1–2 CPU-h).
   - Refit variants: one-sided damped (today) vs two-sided (D1) vs undamped (D2), at K0.125/0.5/1/2.5; plus the rel ≤ 1 guard (A6).
   - Two of the top-4 ideas depend on it.
3. **Sub-1 tail-biting census** (colo `viterbi.c`, ~1 CPU-h): free-start vs tail-biting vs chained vs NTRY16 at K0.125/0.25/0.5. It decides A11 (a format change) and A12 (free).
4. **Neuron-drop census with OBS compensation** (colo, ~2 CPU-h; extends `chimp2.py`): all 64 layers, drop 12.5/25/50%, compared with trellis on gate/up/down at equal MLP bits. It decides B1 and H3, and whether pruning reaches 1.5 bpw.
5. **knap5 what-if sweep at 1.0 / 1.25 / 1.5 / 2.5** (CPU minutes each). Inputs:
   - (a) #118 offsets;
   - (b) a/b priced at K3/K4 from proxy curves;
   - (c) low-rank, K0 and pruned options priced from diagnostics 1 and 4;
   - (d) K1/K1.5 head floors (extrapolated);
   - (e) calib-part head/embedding weights.

   It shows which ideas move predicted KL by ≥1% before any card time is spent.
