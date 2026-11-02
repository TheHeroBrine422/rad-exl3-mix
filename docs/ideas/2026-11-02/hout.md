I have what I need, so I'm writing the report now. The two findings that shape it most:

- **The quantizer:** the "diagonal" H_out arm already does cross-output feedback inside 128-column blocks. The current two-sided LDLQ loop updates every output column densely even though a diagonal H_out decouples the 128-column blocks, which matters for the 27B requant cost.
- **The collector:** it sums the loss over every position, but the score only counts generated rows.

# Research ideas: lens **hout** (output-side information)

## 0. What the code says (it shapes most ideas below)

I checked these in `exllamav3-frac/.../quantize.py`, `quantize/ourq.py`, `v3/yaqa/collect.py`, `proxy2b/yaqaB.py` and `proxy2b/evalq.py`:

1. **The "diagonal" H_out is not diagonal where the quantizer works.**
   - `prepare_H_out` multiplies diag(d) by sv² and applies the blockwise 128-point output Hadamard.
   - Each 128-output block of H_out_r therefore becomes a dense 128×128 matrix with eigenvalues v_j = d_j·sv_j². So the `bdiag` arm already does two-sided error feedback across outputs, but only inside 128-blocks.
   - Spread of v *within* a block is what two-sided LDLQ exploits. Variation *across* blocks does nothing for rounding; only scales or per-slab K could use it.
   - If v were flat, H_out_r = I and the arm would be identical to one-sided.
2. **`ldlq_2hess` does dense work it doesn't need.**
   - Each anti-diagonal step updates F over all output columns: `F[:k_hi,:n_hi] += Lkᵀ·(dE·Ln)`.
   - For a diagonal H_out, L_out is exactly block-diagonal (128). The dense path costs roughly k²n²/8 FLOPs, about 1e15 for a 27B gate/up (5120×17408), i.e. hours per tensor in fp32. A block path costs about 16k²n ≈ 7e12.
3. **`refit_scales` ignores H_out.**
   - The svh (column) refit is unaffected for a diagonal H_out, because each column is separable.
   - The suh (row) refit should be weighted by H_out.
4. **The collector and the metric see different positions.**
   - The collector puts sampled-label CE on **every position** (yaqaB: "Every position counts").
   - The score averages KL over **gen rows only** (evalq: `kl[gen].mean()`; about 55% of g2b-search1's positions are gen rows).
5. **Paper reference points.**
   - YAQA's gain is flat in K: Qwen3-8B KL −28% / −30% / −29% at K2/K3/K4 (ref 2505.22988, Table 8).
   - With Hadamard, H_Z-PTQ's token-level output factor E[G_ZᵀG_Z] equals YAQA's G_WᵀG_W in quality: 14.25 vs 14.26 PPL (2610.40345, Table 5).

Gains below are **increments over the planned diagonal-H_out requant**, as KL % at 2.5 bpw, unless noted. The planned diagonal itself: if the 0.8B result transfers, about −5…−8% at 2.5 bpw; transfer to 27B width is unproven.

---

## A. Before the 27B diagonal run: cheap add-ons to `collect.py YAQA_DIAG` / `hdiag27.sh`

These cost about one day of code and almost no extra GPU time, and they hedge the 8.5 h run. Do them first.

**A1. Split-half accumulators.**
- *Change:* also accumulate the sums for even-parity documents (as YAQA_FULL's `hout-even` already does), so total − even gives two independent halves.
- *Source:* our #33-09 / D2 practice.
- *Why:* every held-out tuning below (D1, C1, C2, C6) needs it. The 27B noise level at 1K sequences is unknown; 0.8B diagonal halves had cosine ≥0.997 at 4K sequences per half.
- *Fit:* collector.
- *Test:* identity check that the halves sum to the total within 1e-6.
- *Gain/risk:* enabling; no risk.
- *Conflicts:* none.

**A2. Free diagonal variants from the same backward.**
- *Change:* in the existing hook (x_t and dy_t are present), also accumulate:
  - (i) K-FAC Σ_t dy_tj²;
  - (ii) Sketch-A half-step Σ_t ‖x_t‖² dy_tj²;
  - (iii) leverage-weighted Σ_t (x_tᵀH_in⁻¹x_t) dy_tj², using the published damped calib H per input group and one triangular solve per token.
- *Source:* 2610.40345 (E[G_ZᵀG_Z]); 2609.38521 (KL-Shampoo bounded leverage weights); YAQA Sketch A.
- *Why:* hedges B2/B3 without a second 8.5 h run. Cost is far below the backward.
- *Fit:* collector.
- *Test:* the 0.8B B2/B3 runs choose which variant is used; the 27B run saves all of them.
- *Gain/risk:* enabling. *Conflicts:* none.

**A3. Block-128 accumulators.**
- *Change:* for every linear, accumulate the 128×128 diagonal blocks of Σ_b G_bG_bᵀ and of Σ_t dy dyᵀ, aligned to the output Hadamard blocks.
- *Cost:* about 2 GB fp32 total and roughly 0.3 s per sequence (m·128·n FLOPs per linear).
- *Why:* within-block structure is exactly what two-sided LDLQ can use beyond the diagonal. For GDN q/k one block is one head (head_dim 128).
- *Fit:* collector.
- *Test:* C5 on the 0.8B decides whether it is used.
- *Gain/risk:* enabling. *Conflicts:* #33-12 sized only diagonal or full (25–206 GB); blocks sit in between.

**A4. Full H_out for attention k_proj/v_proj (1024 outputs) and optionally q_proj.**
- *Cost:* 4 MB each for k/v (16 layers); q is 600 MB per layer.
- *Why:* on the 0.8B, k_proj H_out was precise (halves cos 0.997), very low-rank (PR 0.04) and mostly off-diagonal (diagonal share 0.13), so a diagonal misses most of its structure.
- *Fit:* collector.
- *Test:* C9 per-family arms on the 0.8B.
- *Gain/risk:* enabling (0 to −1% if used). *Conflicts:* none.

**A5. Residual-stream Fisher at each residual-add point.**
- *Change:* token-level Σ_t g gᵀ at the 128 residual-add points. Keep the diagonal and 128-blocks everywhere, and the full 5120² for the last 8 layers (1.7 GB).
- *Why:* it is the shared output factor of every residual writer at that point (o_proj, GDN out_proj, down). Token-level sums over T·N tokens are far less noisy than per-linear Sketch B (o_proj halves cos 0.656). It enables B11, B13, B18 and M2.
- *Fit:* collector. *Gain/risk:* enabling. *Conflicts:* none.

**A6. Embedding-output gradient covariance.**
- *Change:* one 5120² matrix, E[g gᵀ] at the input of layer 0 (the embedding output). Needed for I6.
- *Fit:* collector. *Gain/risk:* enabling.

**A7. Position-bucketed diagonals.**
- *Change:* separate accumulators for positions [0,64), [64,1024) and [1024,2048).
- *Why:* shows whether attention-sink/early positions dominate, and how much the diagonal depends on context length (B20).
- *Cost:* free. *Gain:* diagnostic.

**A8. Per-token scalar logs.**
- *Change:* sample 1 in 64 tokens; log ‖dy_t‖ at a few layers, ‖x_t‖, the gen flag and p(label).
- *Why:* feeds the effective-sample-size and heavy-tail diagnostics (E2) at 27B scale. A few MB. *Gain:* diagnostic.

**A9. Earlier snapshots plus online split-half cosine.**
- *Change:* snapshots at 64/128/256/384/512, and log each linear's half-vs-half diagonal cosine at each snapshot.
- *Why:* gives the 27B dose curve for free and lets the run stop early (B10).
- *Fit:* collector. *Gain:* time.

**A10. Gen-masked variants.**
- *Change:* if B1 is positive, use a gen-only loss. If still undecided, run two backwards per sequence: loss on gen positions and loss on non-gen positions. The gradient is linear, so this gives gen, non-gen and all-position variants (G_all = G_gen + G_non) at about 1.6× cost.
- *Fit:* collector. *Conflicts:* none.

---

## B. Better and cheaper H_out estimates

**B1. Gen-masked loss (match the metric's scored positions).**
- *Mechanism:* restrict the sampled-label loss to gen rows. Gradients still flow through prompt positions via attention and GDN state, so prompt-side activations still enter.
- *Source:* new (score rows via `gen.u8`); cf. turboderp's SC calibration on model-written text (tracker 10-31).
- *Why:* Sketch B also prices how well the model predicts user turns and tool outputs, which the score ignores. Those positions have higher entropy, hence larger gradients, so they are over-weighted in both H_out and Sketch B's H_in. This may explain part of the +20% H_in loss (#33-10).
- *Fit:* per-level (collector setting).
- *Test (0.8B, 4070):* collect 1K windows from fit/qad-pool calib documents with golden gen masks (collection on calib is allowed), gen vs all positions. Arms `bdiag` K3/K2 vs `base`, about 1.5 h. Pass: gen beats all with z ≤ −2 at one K and is not worse at the other.
- *Gain/risk:* 0 to −4% (median about −1%). Risk: only half the loss positions → noisier at 1K, so check the dose.
- *Conflicts:* none; new axis.

**B2. K-FAC output factor Σ_t dy_t dy_tᵀ instead of Σ_b G_bG_bᵀ.**
- *Mechanism:* Sketch B weights tokens through the token Gram XXᵀ (cross terms x_t·x_s, dominated by massive-activation tokens). Since the calib H_in is already ∝ Σ xxᵀ, Sketch B counts x-scale twice; K-FAC counts x once and dy once. It needs no weight gradients.
- *Source:* 2610.40345 (same quality as YAQA's factor under Hadamard); K-FAC.
- *Why:* lower variance. Per sequence, G_bG_bᵀ is often near rank 1 (the mean-x direction), while dYᵀdY has effective rank about T. That could rescue the full gate/up factor, which is noisy now (halves cos 0.80/0.85; hurt at K2), and it simplifies the collector (B8).
- *Fit:* per-level.
- *Test (0.8B):* one collection storing both diagonals plus full K-FAC for gate/up (0.8B-sized). Arms `bdiag-kfac` and `bfull-kfac` vs `bdiag` / `bhout`, K3/K2, about 1.5 h. Pass: K-FAC diagonal not worse (z < 2), which enables B8.
- *Gain/risk:* 0–2% for the diagonal; 0–3% if full K-FAC gate/up becomes usable. Low risk.
- *Conflicts:* #33-09 (Sketch B factor noise) — this is a different estimator.

**B3. Leverage-weighted output factor (one step of the matrix-normal / KL-Shampoo fit).**
- *Mechanism:* B_ii = Σ_t (x_tᵀA⁻¹x_t) dy_ti², with A = calib H_in. The weights are bounded and suppress massive-activation tokens.
- *Source:* 2609.38521 (1.7B, ≈1 bpw: KL-Shampoo PPL 17.04 vs untempered NKP 19.21, i.e. no gain for NKP; 16.72 with tempering).
- *Why:* a principled Kronecker fit that does not double-count either side.
- *Fit:* per-level. *Test:* same collection as B2; arm `bdiag-lev`.
- *Gain/risk:* 0–2%; low risk. *Conflicts:* none.

**B4. Metric-matched coarse Fisher target.**
- *Mechanism:* sample labels from the 513-cell distribution the score uses (bf16 top-512 plus one tail cell); spread the tail gradient ∝ q.
- *Source:* new (`common.score_rows`).
- *Why:* the score lumps the tail into one cell, but rare tail labels produce large, metric-irrelevant gradients.
- *Fit:* per-level. *Test:* one 0.8B arm (needs its own backward).
- *Gain/risk:* 0–1%; low priority.

**B5. Rademacher √Fisher target.**
- *Mechanism:* back-propagate δz = √p⊙ε − p(√p·ε) with ε Rademacher. Its covariance is exactly diag(p) − ppᵀ (the same Fisher), but there are no rare-label spikes.
- *Source:* new; cf. 2610.18139 (Rademacher logit probes).
- *Why:* lower fourth moments → fewer sequences for the same diagonal precision.
- *Fit:* per-level.
- *Test:* 0.8B dose at 256/512/1K with both targets; compare split-half cosine and KL.
- *Gain/risk:* time only (perhaps 1.5–2× fewer sequences); no risk.

**B6. Chunked Sketch B.**
- *Mechanism:* G_c = Σ_{t∈chunk} dy_t x_tᵀ over 256-position chunks; H_out += Σ_c G_cG_cᵀ. Same backward.
- *Source:* new.
- *Why:* drops the long-distance cross terms, which have about zero expectation but add variance. Only matters for full or block factors.
- *Fit:* per-level. *Test:* 0.8B collector variant. *Gain/risk:* 0–1%.

**B7. Random-projection diagonal.**
- *Mechanism:* diag(G_bG_bᵀ) ≈ (1/p)·rowsumsq(dYᵀ(XΩ)) with Ω ∈ {±1}^{n×p}, p = 64. Cost T·p·(m+n) instead of T·m·n.
- *Source:* new (randomized sketching).
- *Why:* if the Sketch B form is kept, this avoids forming G_b, the weight-gradient half of the backward → 1.3–1.5× faster collector.
- *Fit:* collector.
- *Test:* unit check: relative error of the summed diagonal over 64 sequences ≤1e-2, below the diagonal's own sampling noise.
- *Gain:* time.

**B8. Skip weight gradients entirely (if B2 passes).**
- *Mechanism:* set `requires_grad=False` on trunk weights and hook only output gradients.
- *Why:* about 1/3 fewer backward FLOPs and less memory. Combined with B14 or x2, a 27B 1K run might take ~4 h.
- *Fit:* collector. *Test:* 27B gate, time per sequence. *Gain:* time.

**B9. Shorter windows at equal tokens.**
- *Mechanism:* 4×512 instead of 1×2048 tokens.
- *Why:* more independent G_b samples per token → lower variance. Less activation memory → room to pipeline two windows across the two cards.
- *Risk:* loses long-range dy structure.
- *Fit:* collector.
- *Test:* 0.8B, compare diagonal and KL for 512 windows × 2048 vs 2048 windows × 512 at equal tokens.
- *Gain:* time.

**B10. Dose below 1K sequences.**
- *Mechanism:* rerun yaqaB to 512 sequences (bit-exact order, ~13 min) with snapshots at 128/256/512; diagonal arms at K3/K2.
- *Source:* #33-13 (1K kept 85% / 99% of the 8K gain).
- *Why:* if 256 sequences keep ≥80% at both K, the 27B run drops to about 2 h, and gen-masked or multi-variant runs become affordable.
- *Fit:* collector.
- *Gain/risk:* time. Risk: the wider 27B may need more; A9 checks this.
- *Conflicts:* #33-13 (extends it below 1K).

**B11. Pool residual-writer diagonals.**
- *Mechanism:* o_proj/out_proj and down write the same residual dimensions. Shrink each writer's diagonal toward the A5 residual-Fisher diagonal of its layer, and optionally toward adjacent layers.
- *Why:* precise per-dimension weights for the writers, which were noisiest as full factors.
- *Fit:* per-level.
- *Gain/risk:* small, because writers have flat diagonals (CV ~0.2). F1 decides whether it is worth it.

**B12. Analytic local H_out (no backward).**
- *Mechanism:*
  - gate_r: E[(SiLU′(g_r)u_r)²]·‖W_d[:,r]‖²_S
  - up_r: E[SiLU(g_r)²]·‖W_d[:,r]‖²_S
  - GDN z_r: E[(norm(o)_r·SiLU′(z_r))²]·‖W_out[:,r]‖²_S
  - attention v: attention-weighted ‖W_o[:,r]‖²
  - S = I, or the A5 residual diagonal. Only forward statistics are needed.
- *Source:* 2610.41408 (SACQ SwiGLU formulas), 2608.06291 (local MLP Hessian), BoA.
- *Why:* if this matches the Sketch B diagonal (cosine ≥0.95), the largest tensors (gate/up, about 45% of trunk FLOPs) need no backward, and it gives a noiseless prior to shrink toward.
- *Fit:* per-level.
- *Test (0.8B):* forward over 256 sequences; cosine vs the 8K Sketch B diagonal per family; arm `bdiag-local` (local for gate/up/z/v, Sketch B elsewhere). About 1.5 h.
- *Gain/risk:* time. Risk: misses downstream variation through the residual; S from A5 fixes that.

**B13. Hybrid: residual Fisher plus local propagation.**
- *Mechanism:* estimate only the residual-Fisher diagonals (5120 per point, token-level, shared) from very few sequences, then propagate to every reader and writer with B12.
- *Why:* still needs a backward, but perhaps only ~128 sequences → about 1 h at 27B instead of 8.5 h.
- *Fit:* per-level.
- *Test:* 0.8B, 128-sequence residual diagonal + local formulas vs the 1K Sketch B diagonal, K3/K2 arms.
- *Gain:* time.

**B14. Depth-split precision.**
- *Mechanism:* bf16 (or x2) backward above layer X, fp32 below.
- *Evidence:* bf16 H_out diagonal cosine vs fp32 was 0.88 at layers 48–63 and 0.36 at layers 0–15 (#33-03); errors grow downward.
- *Fit:* collector.
- *Test:* on the 27B gate run, per-layer diagonal of 2 sequences vs all-fp32 (as in #33-03); a KL test on the 0.8B as in #33-15 if needed.
- *Gain:* time 1.3–2×.
- *Conflicts:* #33-14/#33-15 (precision) — this splits by depth instead.

**B15. Adaptive truncated backward.**
- *Mechanism:* once the lower layers' split-half cosine has converged (A9), truncate the backward above them, as gate 2c did.
- *Fit:* collector. *Gain:* time 20–40%.

**B16. H_out at the operating point.**
- *Mechanism:* collect the diagonal on a fake-quantized model (0.8B uniform K2 or K3 substituted) and use it for that K band.
- *Source:* 2605.23078 (importance from a 2-bit model: 9.66 → 9.33 PPL), 2607.23047.
- *Why:* at K2 the perturbation is outside the quadratic regime; the full H_out hurt at K2.
- *Fit:* per-level (one collection per K band).
- *Test:* 0.8B, collect 1K sequences through dequantized K2 weights; K2 arm. About 1 h.
- *Gain/risk:* 0–2% at K2. At 27B it needs EXL3 → bf16 dequant streaming plus another long run, so only worth it if large.

**B17. Progressive re-collection.**
- *Mechanism:* quantize with H_out, recollect on that model, requantize (as in GEMQ).
- *Test:* 0.8B, two rounds. *Gain:* 0–1%; low priority.

**B18. Exact last-layer H_out from the head Fisher.**
- *Mechanism:* for the last layer's residual writers, H_out = E_t[J_tᵀW_hᵀF_tW_hJ_t], with F_t the coarse top-512 Fisher from the golden and J_t the final-RMSNorm Jacobian. This is an exact expectation over labels and a full 5120² matrix with no sampling noise. Propagate to L63 gate/up/z with B12.
- *Source:* new. Motivated by the L63 ffn_gate_up spike (+0.0065 at K2, ~80% of the base's KLD; INDEX #21) and REAL-Q's true-KL last block.
- *Fit:* per-level.
- *Test:* 0.8B, quantize only the last layer at K2 (D2 probe) with exact vs Sketch B vs no H_out; paired. About 1 h; needs pre-norm residual dumps (E1b hook).
- *Gain/risk:* 0 to −2%, concentrated on the most sensitive tensors.

**B19. Data mix of the 27B windows.**
- *Mechanism:* the qad-pool 2048-token windows over-represent long documents. Weight windows to the corpus category/token mix.
- *Test:* 0.8B per-category diagonal cosine.
- *Gain:* 0–0.5%; low priority.

**B20. Long-context positions.**
- *Mechanism:* compare the diagonal from positions 16k–32k of long traces with positions <2k on a few 0.8B documents. No action if cosine ≥0.99.
- *Gain:* diagnostic; low priority.

---

## C. How the quantizer uses H_out

**C1. Tempering exponent.**
- *Mechanism:* v_j = d_j^p with p ∈ {0.25, 0.5, 0.75, 1, 1.5}, chosen per K.
- *Source:* 2609.38521 (p=½ beat 1 and ¼: 19.21 → 17.46 at 1.7B); 2610.20353 (NatScale whole-model tempering β≈0.2); 2609.08135 (KBBQ: partial flattening λ<1 best).
- *Why:* at K2 the relative error is ~0.3, where the quadratic model over-trusts dominant curvature. It may also explain why the diagonal beat the full factor at K2.
- *Fit:* per-level knob.
- *Test (0.8B):* existing 8K and 1K diagonals, 10 arms, about 45 min. Pass: the best p beats p=1 with z ≤ −2.
- *Gain/risk:* 0 to −2% (mostly K2); no risk. *Conflicts:* none.

**C2. Output damping per tensor.**
- *Mechanism:* choose sigma_reg_out from {0.001, 0.003, 0.01, 0.03, 0.1} per tensor with D1's held-out predictor; first run a global sweep on the 0.8B (10 arms).
- *Source:* #38 / damp3 (per-tensor input-damping winners agreed only 53–70%).
- *Why:* the rotated diagonal has CV up to 4 on input projections, so a fixed 0.01 floor treats tensors very differently.
- *Fit:* per-level.
- *Gain/risk:* 0–1.5%. *Conflicts:* #38 (input side); this is the output side.

**C3. Per-block normalisation.**
- *Mechanism:* normalise each 128-block of H_out_r by its own mean before damping.
- *Why:* scale across blocks doesn't affect rounding, but it sets each block's effective damping; low-sensitivity blocks now get relatively huge damping (no feedback).
- *Fit:* per-level. *Test:* 2 arms on the 0.8B. *Gain:* 0–1%.

**C4. Sensitivity floor instead of additive damping.**
- *Mechanism:* clamp v_j ≥ f·median (like exllamav3's `out_sensitivity_floor` 0.02 for scales).
- *Why:* keeps near-insensitive channels from absorbing unbounded error.
- *Test:* 0.8B arms. *Gain:* 0–0.5%.

**C5. Block-diagonal (128) H_out.**
- *Mechanism:* keep full 128×128 blocks aligned with the output Hadamard; drop cross-block terms.
- *Source:* new, matching `prepare_H_out`'s rotation. Caution from 2610.03027 (TaSQ): a dense metric overfit; the diagonal generalised (GSM8K error 0.00933 vs 0.00234).
- *Why:* uses within-block structure (one block = one GDN q/k head) and drops cross-block noise. Expected to land between diagonal (K3 −8.0%) and mixed (K3 −9.9%).
- *Fit:* per-level; about 2 GB at 27B (A3).
- *Test:* 0.8B offline from the 8K full snapshot, K3/K2 arms, about 20 min.
- *Gain/risk:* 0 to −2%. *Conflicts:* #33-12 (only diagonal/mixed/full were tested).

**C6. Shrinkage from full toward diagonal.**
- *Mechanism:* H(γ) = (1−γ)H + γ·diag(H), with γ per tensor maximising D1's held-out score.
- *Source:* Ledoit–Wolf-type shrinkage; 2609.38521 used γ=0.2.
- *Why:* the full gate/up factor hurts at K2 (noise) but the mixed arm helps at K3; shrinkage interpolates.
- *Fit:* per-level. At 27B it needs blocks (C5) or low rank (C7); full gate/up is 155 GB.
- *Test:* 0.8B offline, 3 γ × 2 K.
- *Gain:* 0 to −2%.

**C7. Low-rank plus diagonal.**
- *Mechanism:* H ≈ D + UUᵀ with r = 16–64, from a streaming Nyström sketch (Y = Σ_b G_b(G_bᵀΩ) via activation products, or dYᵀ(dYΩ) in K-FAC form); no G_b is formed.
- *Storage:* m·r per linear (gate: 4.5 MB at r=64).
- *Source:* new (randomized linear algebra).
- *Why:* k_proj PR 0.04 and GDN qkv PR 0.05 — a few directions dominate.
- *Fit:* per-level.
- *Test:* 0.8B offline from the full factor, r ∈ {16, 64}.
- *Gain:* 0 to −2%.

**C8. Cross-fitted eigenvalue shrinkage.**
- *Mechanism:* eigenvectors from half A, eigenvalues re-estimated on half B. Removes the upward bias of noisy top eigenvalues that the LDL feedback chases.
- *Fit:* per-level. *Test:* 0.8B gate/up full factor at K2. *Gain:* 0–1% (only if any full factor is kept).

**C9. Choose the structure per family.**
- *Mechanism:* generalise `bmix`: pick one of {none, diagonal, block, low-rank+diagonal, full} per family with D1, then confirm with arms. Plausible outcome: full for attention k/v (cheap), block for GDN q/k, diagonal for gate/up/z, none for residual writers.
- *Fit:* per-level. *Test:* about 6 arms on the 0.8B.
- *Gain:* up to −2% at K3 (bmix was −1.9% vs diagonal).
- *Conflicts:* #33-12.

**C10. Choose the structure per K.**
- *Mechanism:* full/block at K ≥ 3, diagonal at K ≤ 2.5 (K2: bhout −5.6% vs bdiag −8.0%; K3: −9.7% vs −8.0%). The library is per-level, so the choice per K is free.
- *Test:* existing K2/K3 numbers plus K2.5/K4 arms. *Gain:* about 1–2% at K3+ levels.

**C11. Skip H_out where the predicted gain is <1%.**
- *Mechanism:* use D1 to decide per tensor; residual writers (CV 0.2) likely skip.
- *Why:* saves two-sided time and avoids noise harm. *Fit:* per-level. *Gain:* time; KL ±0.

---

## D. Tools that turn GPU arms into CPU arithmetic

**D1. Held-out high-rate predictor (the key tool).**
- *Mechanism:*
  - Per tensor and 128-block (or full), take the LDL factor L_A of the regularised rotated H_out from half A and the held-out estimate B_B.
  - Predicted error ratio vs one-sided: R = tr((I+L_A)⁻¹B_B(I+L_A)⁻ᵀ) / tr(B_B). This follows from high-rate LDLQ: E = η(I+L)⁻¹ with white tile innovations, and the input factor cancels under the Kronecker product.
  - Weight each tensor's R by its KL share (kfit2 sensitivities) to get a predicted whole-model gain for any recipe: p, damping, structure, dose, estimator variant, permutation.
- *Source:* LDLQ theory (2307.13304); new use.
- *Why:* dozens of 4-minute GPU arms become CPU seconds, and it works at 27B right after collection (with A1).
- *Fit:* tooling.
- *Test:* compute R for every existing 0.8B arm (ldlq, bhout, bdiag, bmix, 1K/2K/4K dose). Check Spearman ≥0.8 against measured KL, and whether it predicts "full worse than diagonal at K2". About half a day of CPU.
- *Gain/risk:* enabling, high leverage. Risk: the high-rate assumption fails at K2; the test shows how badly.

**D2. Single-family / single-layer K2 probes.**
- *Mechanism:* quantize only family X (or one layer band) at K2, with and without H_out, rest bf16; paired KL increment. About 5× larger effect per arm than whole-model ablations.
- *Source:* our 47 single-layer probes (#21).
- *Test:* 0.8B, 6 families × 2 arms, about 50 min. *Gain:* enabling.

**D3. Does the two-sided proxy track KL?**
- *Mechanism:* for the existing 0.8B quants, compute the per-linear two-sided proxy tr(EᵀH_inEH_out,B) / tr(WᵀH_inWH_out,B) with calib H_in and held-out H_out, weight the sum, and compare with measured KL ranks.
- *Context:* objcheck (#33-08) found Sketch B's two-sided objective did not track KL, but that used Sketch B's H_in.
- *Why:* if it tracks, H1/H3/H4 pricing and per-tensor decisions can use it.
- *Test:* about 2 h CPU.

**D4. Within- vs across-block decomposition.**
- *Mechanism:* per tensor, split the variance of log v_j (v = d·σ²) into within-128-block and across-block parts. Within-block is what two-sided LDLQ uses; across-block is usable only by K or scales (G1, H4, H6).
- *Test:* minutes of CPU on the 0.8B diagonal; repeat at 27B. *Gain:* enabling.

**D5. Predicted 27B gain right after collection.**
- *Mechanism:* run D1 and D4 on the 27B diagonal and compare per-family distributions with the 0.8B. If 27B within-block dispersion is ≥ the 0.8B's, expect ~−8%; if much flatter, lower expectations before spending requant nights.
- *Cost:* minutes. *Fit:* validation.

---

## E. Why gradient-weighted H_in hurts, and the input side

Note: in the one-sided `hin` arm (H_out = I), each token is already weighted only once by ‖dy‖², yet it still lost 20%. Double counting therefore does not explain that arm; something about the weights does.

**E1. H_in factorial (find the cause).**
- *Mechanism:* one 0.8B collection storing:
  - (a) Sketch B Σ G_bᵀG_b (already have);
  - (b) token-level Σ_t ‖dy_t‖² x xᵀ (no cross-token terms);
  - (c) gen-masked (b);
  - (d) tempered Σ‖dy‖^{2α} x xᵀ, α = ½;
  - (e) (b) on calib fit-pool windows instead of xcorp;
  - (f) the error-aware marginal of E3.
  
  One-sided arms at K3 (plus with bdiag) vs calib H_in. Each variant tests one hypothesis: cross-token Gram concentration (b), unscored positions (c), heavy tails / non-quadratic regime (d), distribution shift (e), wrong marginal (f).
- *Fit:* understanding.
- *Test:* about 3 h on the 4070.
- *Gain/risk:* product gain only if some variant beats calib xxᵀ (0–5%, low probability).
- *Conflicts:* #33-07 (gate 2e), #33-10 — this explains them instead of re-running them.

**E2. Effective sample size of the token weights.**
- *Mechanism:* ESS = (Σw)²/Σw² for w_t = ‖dy_t‖² per layer, the share of mass in the top 1% of tokens, and gen vs non-gen.
- *Why:* if ESS/T ≪ 1%, the weighted H_in is effectively built from a few thousand tokens vs 10M for calib xxᵀ. Upstream noted "H_out dominated by relatively few high-gradient tokens".
- *Test:* CPU on A8 logs (a quick 0.8B run). *Gain:* diagnostic.

**E3. Error-aware one-sided marginal.**
- *Mechanism:* the quantizer's output error is isotropic only after rotation; in the original basis its covariance is ∝ diag(sv²). The correct one-sided marginal is therefore Σ_t (Σ_j sv_j² dy_tj²) x xᵀ, not Σ‖dy‖² x xᵀ.
- *Source:* new derivation.
- *Why:* output scales are on and σ_j varies; Sketch B ignores where the error will land.
- *Fit:* per-level. *Test:* inside E1. *Gain:* unknown, low probability.

**E4. Rate dependence.**
- *Mechanism:* one-sided gradient-weighted H_in at K5/K6 on the 0.8B. If it helps at high K but hurts at K2/K3, the cause is large perturbations, and tempering is the fix. If it hurts everywhere, the cause is the estimator.
- *Note:* high K is also where turboderp's tensors beat ours (tracker 10-31).
- *Test:* 2–4 arms, about 15 min. *Gain:* diagnostic.

**E5. Gen-weighted H_in.**
- *Mechanism:* H_in(β) = H_gen + β·H_nongen from our own calib collection using gen-mask weights, β ∈ {0, 0.3, 1}. The score counts gen positions only; activations at prompt positions matter only through later attention/GDN reads.
- *Source:* new; turboderp's SC Hessians from self-generated text (tracker 10-31).
- *Why:* the input factor dominates the quantizer, so even a few % there is large.
- *Fit:* per-level. At 27B it needs our own forward-only H_gen collection (about half the calib tokens), blended with the published H.
- *Test (0.8B):* recollect calib H_in with weights (~15 min), 3 arms × 2 K, plus with bdiag. About 1 h. Pass: z ≤ −2 at both K.
- *Gain/risk:* 0 to −4%. Risk: low β starves prompt-side directions (attention keys); β > 0 hedges.
- *Conflicts:* none (the published H is untouched).

**E6. Massive-token-robust H_in.**
- *Mechanism:* cap each token's leverage at x_tᵀH⁻¹x_t ≤ c, or drop each document's first position (the attention sink).
- *Source:* 2609.38521 (bounded weights); tracker note that 0.025 × mean diag ≈ 22% of median curvature because mean ≈ 9× median.
- *Why:* our damping fight may be a symptom of a few tokens dominating H_in.
- *Fit:* per-level. *Test:* 0.8B recollection, 2 arms × 2 K.
- *Gain:* 0–1.5%. *Conflicts:* #38 damping (complementary).

**E7. One KL-Shampoo alternation for H_in.**
- *Mechanism:* A′ = Σ_t (dy_tᵀB⁻¹dy_t) x xᵀ with B = the current H_out diagonal; normalise; then B′ from A′.
- *Source:* 2609.38521.
- *Why:* a joint Kronecker fit with bounded weights, unlike Sketch B's gradient weighting on both sides.
- *Test:* 0.8B arms. *Gain/risk:* 0–2%; risk of repeating Sketch B's H_in loss (E1 explains first).

**E8. Tempered blend.**
- *Mechanism:* H_in = calib + α·(normalised token-level gradient-weighted H_in), α ∈ {0.1, 0.3}.
- *Context:* the planned gate-2e blend was skipped (#33-07); on the 0.8B with ample data it is 2 arms.
- *Gain:* 0–1%; low priority.

**E9. Noise-aware H_in combined with H_out.**
- *Mechanism:* G = H + λ²N + λ(C+Cᵀ), pretransform W·F; C and N come from one noisy forward pass.
- *Source:* 2610.07522 (closes 18–31% of the parallel-vs-sequential gap). Orthogonal to H_out.
- *Fit:* per-level (fixed noise level). *Test:* 0.8B.
- *Gain:* 0–3%. *Conflicts:* the ROADMAP's sequential Hessians (V3.5).

**E10. Quantized-prefix H_in combined with H_out.**
- *Mechanism:* GPTAQ-style inputs from a uniform-K prefix (the per-level version).
- *Gain:* 0–3% at a higher cost. *Conflicts:* ROADMAP V3.5 item.

**E11. Output-cluster-specific H_in.**
- *Mechanism:* cluster output channels into 4–8 groups by gradient pattern. Each group gets H_in^c = Σ_t(Σ_{j∈c} dy_tj²) x xᵀ — a sum-of-Kroneckers metric that LDLQ handles column-group-wise (one LDL per cluster).
- *Source:* new; 2610.18139 (Jacobian-aware row metric lowered logit MSE in 24/24 comparisons, but a fixed α was unsafe).
- *Fit:* per-level, 4–8× the LDL cost.
- *Gain/risk:* 0–3%; same weighting risk as gradient-weighted H_in, so do E1 first.

---

## F. Where the gain lives (attribution)

**F1. Family attribution.**
- *Mechanism:* D2 probes per family with bdiag.
- *Prediction from the 0.8B diagonal CV* (qkv 1.69, z 3.96, q 2.3, k 1.5, v 0.5, o 0.2, gate/up 1.0, down 0.17): gain in qkv/z/q/k/gate/up, about zero in o/down/out.
- *Why it matters:* decides collector scope (skip writers), C11, H4/H5.
- *Test:* about 1 h.

**F2. Depth attribution.**
- *Mechanism:* bdiag on layers 0–5 / 6–11 / 12–17 / 18–23 only.
- *Why:* if the last quarter carries most of the gain, the 27B collection can truncate the backward at ~layer 40 and run bf16 above it (cost ÷3), and the late-layer pilot (K1) suffices.
- *Test:* about 30 min.

**F3. Rate dependence.**
- *Mechanism:* bdiag vs base at K2.5 (fractional; this also exercises qtf2 inside `ldlq_2hess`), K4 and K5.
- *Reference:* YAQA's gain was flat in K (−28/−30/−29% at K2/K3/K4).
- *Why:* if ours falls to ~0 at K4+, the 3.5–4.5 bpw sizes gain little and requant priority moves to K2–K3.5.
- *Test:* about 30 min. Critical for expected value.

**F4. Crude mixed-K check.**
- *Mechanism:* on the 0.8B, e.g. gate/up/down at K2 and the rest at K3, base vs bdiag.
- *Why:* checks the gain survives when sensitive tensors already get more bits.
- *Test:* about 20 min.

**F5. 2B width check.**
- *Mechanism:* Sketch B diagonal at 512–1K sequences on the 2B (width 2048; 1024-token windows if memory requires); bdiag vs base at K3/K2.
- *Why:* reduces risk before the 27B (width 5120).
- *Test:* 3–5 h on the 4070, or an R9700 slot.

**F6. Ladder-exact settings.**
- *Mechanism:* one bdiag arm with the ladder's per-tensor input damping sweep (`--sigma-in-sweep`) and i8 codebook, instead of bhout's fixed damping.
- *Why:* the setting actually tested must be the one that ships.
- *Test:* about 10 min.

---

## G. Scales and refits

**G1. Retest sensitivity output scales with a clean diagonal.**
- *Mechanism:* exllamav3 `regularize` with `out_hessian_diag` (t ∝ (σ²/s)^{1/4}), using the 0.8B Sketch B diagonal. Arms {scales, scales + bdiag} vs {base, bdiag} at K3/K2.
- *Source:* 2607.07964 (KronQ: output-side diagonal pre-scaling S_G = (h/a)^{1/4}; removing it moved W2 PPL 11.92 → 12.94).
- *Why:* scales can use the across-block variation that two-sided LDLQ cannot (D4).
- *Fit:* per-level. *Test:* about 30 min.
- *Gain/risk:* 0 to −3%. Risk: scales and feedback may substitute rather than add.
- *Conflicts:* #33-06 (gate 2d failed on the 27B with 200-sequence diagonals from the old collectors) — this is the first test with a precise diagonal.

**G2. sv exponent grid with LDLQ active.**
- *Mechanism:* t = σ^a·d^{−b}, a ∈ {0.5, 1}, b ∈ {0, 0.25, 0.5}, with bdiag on.
- *Test:* 6 arms on the 0.8B. *Gain:* 0–1%.

**G3. H_out-aware su refit.**
- *Mechanism:* the row solve in `refit_scales` becomes ((QDQᵀ)∘H)r = rowsum(Q∘(HWD)), with D = diag(H_out) in the original basis. The column refit is unchanged for a diagonal H_out.
- *Fit:* free code change. *Test:* 2 arms. *Gain:* 0–0.5%.

**G4. Two-sided g_scale search.**
- *Mechanism:* weight the sample-tile error in `g_scale_search` by H_out_r.
- *Gain:* 0–0.5%; low priority.

**G5. Two-sided proxy in the input damping sweep.**
- *Mechanism:* `--sigma-in-sweep` currently picks per-tensor input damping by a one-sided proxy against the undamped H. With H_out, use the two-sided proxy; the winners may shift.
- *Test:* 0.8B arms. *Gain:* 0–0.5%.

**G6. Joint sv refit for block or full H_out.**
- *Mechanism:* with a non-diagonal H_out, the per-column scales couple; solve a 128×128 system per block.
- *When:* only if C5/C9 adopt non-diagonal structure. Low priority.

**G7. Output scales on/off from the clean diagonal.**
- *Mechanism:* recompute `out_scales_ratio` with the Sketch B diagonal.
- *Context:* #48 found the auto rule wrong (+2.5%), but with a different sensitivity source. Covered by the G1 arms. Low priority.

---

## H. Allocation (knapsack, granularity, activations)

**H1. Two-sided hproxy, then refit kfit2.**
- *Mechanism:* recompute the ~8,100 V2 hproxy rows (`mix/hproxy.cpp`, plugin decoder) with output row weights from the 27B diagonal: err2 = Σ_j d_j ΔW_j H ΔW_jᵀ (normalised). Refit kfit2 on V2's ~1,250 measurements. Check grouped-CV log-RMSE (now 0.118), pair ranking (79.9%), and whether per-family factors move toward one.
- *Source:* #67/#68. 2609.33923: on Qwen3.8-27B, allocating by the oracle per-layer objective gave 6.087 PPL vs 6.041 for a block-output signal. 2610.39931: gradient-squared-weighted pricing far beat local-distortion pricing.
- *Fit:* knapsack model.
- *Test:* about a day on a CPU box after the 27B collection. Pass: CV gain above the bootstrap noise.
- *Gain:* 0 to −3% at 2.5 bpw.
- *Conflicts:* #20 (the residual re-ranker learned nothing) — this is a better physical feature, not a learned correction.

**H2. Ours vs turboderp at high K.**
- *Mechanism:* compare err2/err1 ratios for his and our K5–K8 tensors (V2 takes his raised tensors: 37 of 49 at 6.0).
- *Why:* if ours put relatively more error into high-d channels, the two-sided proxy explains the 1.25–1.58× family factor.
- *Test:* CPU. *Gain:* ~0 at 2.5 bpw; matters at ≥5 bpw.

**H3. Priors for unmeasured options.**
- *Mechanism:* use the two-sided proxy as the prior in partial pooling (backlog B11) for fractional levels, sub-2 K and two-sided options.
- *Why:* K1 is now under-predicted by 20–36%; fewer stream evals are needed for new options.
- *Gain:* 0–1%.

**H4. MLP channel partition ("hot / cold sub-MLPs").**
- *Mechanism:* an MLP is exactly the sum of sub-MLPs over any partition of its intermediate channels. Give the channels with the highest sensitivity their own K, and the rest another. Sensitivity per channel r = gate/up output diagonal × σ², plus down's input-side H_in diagonal for r.
- *Format:* no new tensor format — two ordinary EXL3 MLPs whose outputs are summed. It needs a graph change in `qwen35_exl3` and one extra launch per GEMM; A8 prep is shared.
- *Quantization:* permute the channels first (J5; free for intermediate dims) so the hot set is 128-aligned, then requantize once per K. Row subsets are nearly reusable, but the scale refits are joint.
- *Source:* new; head slabs (#34); E3c (#29-02, input side only: 0.05 bit for gate_up + down); 2606.15652 (error hotspots).
- *Why:* gate/up diagonal CV ~1.0 on the 0.8B, and the across-block part is invisible to two-sided LDLQ.
- *Fit:* format/plugin change plus knapsack items.
- *Test:* D4 ceiling first (½·log₂ AM/GM and the 2-way split gain at equal bits; CPU). Then a 0.8B fake-quant arm: last 6 layers' MLPs split 25/75 at K3/K2 vs uniform K2.25 at equal bits.
- *Gain/risk:* 0 to −4% (high uncertainty).
- *Conflicts:* v3.md dropped trunk row-slab K because output weighting "hurt in every form" (gates 2–2e). The 0.8B diagonal result overturns that premise.

**H5. Section split for GDN in_proj_qkv (q | k | v) and attention q (query | gate).**
- *Mechanism:* a separate K per section (split the weight at conversion; the plugin runs the sections as linears on the shared input).
- *Source:* new; 2609.26173 (component type predicts sensitivity: V dominates); 2609.04098 (Qwen3.8-27B GDN per-projection errors add in quadrature).
- *Fit:* small plugin/format change.
- *Test:* per-section ceiling from the diagonal (CPU), then one 0.8B arm.
- *Gain:* 0 to −2%.

**H6. Trunk row-slab K revisited.**
- *Mechanism:* K per 128-output slab for large tensors, reusing the head's slab machinery; only if D4 shows across-block dispersion ≫ E3c's estimate (0.12 bit over all tensors, 0.05 for gate_up + down).
- *Fit:* format change. *Gain:* 0–3%.
- *Conflicts:* #29-02; v3.md §6b.

**H7. Requant priority.**
- *Mechanism:* order the two-sided requant as K2–K3.5 input projections first (where F1/F3 put the gain); residual writers keep one-sided quants.
- *Fit:* enabler (ladder time). *Conflicts:* #58.

**H8. Two-sided quants as a new kfit2 family.**
- *Mechanism:* their KLD per unit of one-sided proxy differs; add them with partial pooling toward "ours" and calibrate with early swaps.
- *Fit:* enabler.

**H9. A8 activation cost with H_out.**
- *Mechanism:* E1c's rel_out = Σ‖We‖²/Σ‖Wx‖² becomes Σ_j d_j(We)_j² / Σ_j d_j(Wx)_j². Refit aqcal's family factors (log-RMSE 0.26; late layers under-predicted 1.7×).
- *Test:* CPU on the existing dumps after the 27B diagonal.
- *Gain:* ≤0.5% at 2.5 bpw (A8 costs about +0.6% at 3.0); better speed/quality picks.
- *Conflicts:* #27.

**H10. Activation clip/group per tensor by output-weighted error.**
- *Mechanism:* choose γ_x by the Fisher-weighted output error, as H_Z-PTQ does.
- *Gain:* <0.3%; low priority.

**H11. GDN head prior.**
- *Mechanism:* compare per-head mean H_out diagonal with head decay half-life (A_log) on the 0.8B.
- *Source:* 2609.38169 (Qwen3.8-27B: half-life vs state error ρ ≈ 0.80).
- *Why:* if it correlates, it is a data-free prior for per-head scales or K.
- *Test:* CPU. Low priority.

---

## I. Head and embedding

**I1. Head two-sided with the analytic Fisher diagonal.**
- *Mechanism:* the head's output-side Fisher is closed-form, E_t[diag(p_t) − p_tp_tᵀ]. Its diagonal Σp(1−p) is exactly headsens's Fisher, already built from the golden's top-512. Quantize head donors with `--hout` = that diagonal (no backward) and score with headsim.
- *Source:* 2608.12026 (SoftWater: class-aware allocation under KL gave 6.5–8.3× lower head KL vs WaterSIC at 2 bits, with entropy coding).
- *Why:* rows are sorted by Fisher, but sensitivity still varies strongly inside the top slabs (400 rows hold 50% of the mass).
- *Fit:* per-level (donors per K).
- *Test:* one donor K (K4 or K5), two-sided vs one-sided; re-price the slab plans with headsim. About 2 card-hours.
- *Gain:* −0.3…−1% total (the head is ~5–7% of KLD). *Conflicts:* #34.

**I2. Exact block head H_out including −E[ppᵀ].**
- *Mechanism:* per 128-row slab, the full exact block from the goldens (no sampling noise, so the "full is noisy" objection does not apply). Couples co-predicted tokens such as casing and spacing variants.
- *Test:* as I1. *Gain:* +0–0.5% over I1.

**I3. Slab-specific head H_in.**
- *Mechanism:* H_slab = Σ_t(Σ_{v∈slab} p_tv(1−p_tv)) x_t x_tᵀ for the top ~16 slabs (2048 rows), using exact expectations from the 32k held-out head inputs.
- *Why:* a row's error only matters where its token is plausible. This is the gradient-weighted H_in idea with exact (not sampled) weights.
- *Fit:* per-level. *Test:* headsim. *Gain:* −0.2…−1%.

**I4. Head su refit, two-sided (G3 applied to the head).** Low priority.

**I5. Rare-token guard pricing with the Fisher-weighted error.**
- *Mechanism:* predict the guard KL (×1.2–1.5 at K5–6) inside slab-plan pricing. Low priority.

**I6. Embedding hidden-dim metric.**
- *Mechanism:* rows are quantized with H = I along the 5120 hidden dims. Use G = E[g gᵀ] at layer 0's input (A6) — shared by all rows, so frequency-neutral — as the LDLQ Hessian along each row.
- *Source:* new; v3.md §1b.6 chose H = I to avoid frequency tiering (#32).
- *Why:* residual channels differ strongly in sensitivity (the massive channels are shrunk by norm gains).
- *Fit:* per-level.
- *Test:* the 0.8B is tied, so untie for the test (substitute only the input embedding in evalq; head stays bf16). Trellis embedding at K1.5/K2/K3 with H = I vs H = G. About 2 h.
- *Gain:* −0.5…−1.5% at 2.5 bpw (the embedding sits near its K1.5 floor at ≤2 bpw; the trellis embedding was −5.4% at 2.5).
- *Conflicts:* #32 is avoided (no per-row frequency weighting); #35.

**I7. Embedding per-row scale refit under G.** Small; low priority.

**I8. 9B (untied) confirmation of I6.** Needs an R9700 slot; low priority.

---

## J. Trellis, Hadamard, fractional K, engineering

**J1. Block-diagonal fast path in `ldlq_2hess`.**
- *Mechanism:* with a diagonal H_out, L_out is exactly block-diagonal (128). Reshape into n/128 independent k×128 problems (batched): ~16k²n FLOPs instead of ~k²n²/8. For a 27B gate/up that is minutes instead of hours.
- *Source:* new for the block case; 2608.06291 (FastKron) for dense.
- *Fit:* quantizer engineering.
- *Test:* output-identical check on 0.8B tensors; time one 27B gate tensor first (diagnostic 5).
- *Gain:* requant time; possibly the difference between feasible and not.

**J2. Batch the blocks.**
- *Mechanism:* the anti-diagonals of all blocks go in one launch, giving bigger tile batches. Part of J1.

**J3. Verify fractional K / qtf2 / i8 through the two-sided path.**
- *Mechanism:* byte-identity at K2.5 of `ldlq_2hess` with L_out = 0 vs one-sided, as amendment 2 checked for integer K.
- *Why:* prevents a silent fallback to the old kernel. About 15 min.

**J4. Weighted Viterbi inside tiles.**
- *Mechanism:* use the D_in ⊗ D_out diagonals as per-weight metric in the trellis search.
- *Expectation:* inert after RHT (2610.09212: Γ_b fell 0.038 → 0.00013 bit). Diagnostic only: compute Γ_b on the 0.8B. Low priority.

**J5. Free output permutations before the Hadamard.**
- *Mechanism:* these can be permuted at zero inference cost — intermediate MLP channels (gate/up rows with down columns), GDN value-head internal dims (with the gated-norm gain and out_proj columns), attention v dims (with o_proj input). Either interleave d-sorted channels so each 128-block carries the tensor's full spread (maximises what two-sided LDLQ uses), or sort them into homogeneous blocks for H4/H6.
- *Source:* new; 2608.04074 (equal-volume grouping: random 74.4 vs sorted 37.2 on RULER).
- *Fit:* per-level (baked into the quant).
- *Test:* D1 predicts on CPU; then a 0.8B gate/up arm, interleaved vs natural, K2/K3.
- *Gain:* 0 to −2%.

**J6. Output Hadamard size (64 vs 128).**
- *Mechanism:* a smaller block keeps more structure for H_out but less incoherence. Kernel change; estimate with D1 first. Low priority.

**J7. FastKron recursion for any tensors that keep a full factor at 27B (C9).** Low priority.

**J8. Fractional-K extra-bit placement.**
- *Mechanism:* align the mask positions that get the extra bit with high-D_out rotated columns. Pivot spread after the Hadamard is small. Low priority.

---

## K. Validation at 27B

**K1. Late-layer pilot.**
- *Mechanism:* diagonal for layers 48–63 only (backward truncated at 48; bf16 is near-accurate there, check), 256 sequences, about 1 h on both cards. Requant layers 48–63 at K3, splice into the ladder K3 container, paired KL on a search slice.
- *Why:* go/no-go before the 8.5 h run.

**K2. Split halves plus D5 prediction on CPU (see A1 and D5).**

**K3. Band swap.**
- *Mechanism:* 8 middle layers at K3, two-sided vs one-sided, in a fixed base; the gate-2 harness; pass rule written first. 2–3 h.

**K4. Uniform-trunk swap (the decider).**
- *Mechanism:* whole trunk at K3 and K2.5 with `--hout` and the ladder's settings (with J1), vs the ladder's K3/K2.5 containers. Paired on a ~300k-position search slice; ver3 only as monitor. About 4 h.

**K5. Per-tensor proxy-improvement table, 27B vs 0.8B pattern (CPU, from K4's quants).**

**K6. Precision checks.**
- *Mechanism:* the planned single-linear fp32 vs x2 check, plus per-layer bf16 diagonal cosine for B14.

**K7. Pass-rule template.**
- *Rule (write it before the run):* K3 two-sided must be ≥3% better with z ≤ −2 on the search slice and have the same sign at K2.5; otherwise stop. Never select on public.

---

## L. Training and healing

**L1. H_out-weighted block-healing loss.**
- *Mechanism:* for block-wise scale healing or a QAD warm-up, minimise Σ_j d_j‖Δy_j‖² (RMSE-normalised per block, per 2610.00983) before the end-to-end KL stage.
- *Source:* 2609.00049 (REAL-Q block Fisher); 2610.00983.
- *Fit:* training (V5). *Test:* 0.8B scale-only healing, both ways. *Gain:* 0–2% after healing.

**L2. Two-sided closed-form scale polish as healing step 0 (G3 plus a joint sv refit).** Free.

**L3. H_out-guided trainable subset.**
- *Mechanism:* train only the scales of high d·error tensors/channels to fit 27B healing in memory. Low priority.

---

## M. Longer shots

**M1. First-order two-sided compensation (G²PTQ, 2609.31009).**
- *Mechanism:* KL gradient at the partially quantized model, with a trust region.
- *Fit:* per-mix → final pass only.

**M2. REAL-Q with A5's block Fisher.**
- *Mechanism:* REAL-Q's per-128-column Adam corrections.
- *Fit:* per-mix, ~40–50 GPU-h per size (ROADMAP item). A5 makes the Fisher free.

**M3. Two-sided block-CD repair.**
- *Mechanism:* after LDLQ, re-run Viterbi on single tiles given the residual under the two-sided metric; accept on the held-out proxy (DRQ's gate).
- *Source:* 2610.32127 (LDLQ+CD: 15.01 → 12.76 PPL, Llama-3-8B, E8P12, 2 bits); 2610.11226.
- *Fit:* per-level. *Gain:* 0–3% at K2.

**M4. Robust H_out (worst case over halves A/B).** Low priority.

**M5. Output-side learned rotation (HARP, 2605.29843).**
- *Context:* only +0.7% PPL on QTIP at 3 bits, with a kernel change. Low priority.

---

## Ranked top 20 (expected gain per unit cost)

| # | Idea | Cost | Expected effect |
|---|---|---|---|
| 1 | **D1** held-out high-rate predictor (validated on existing 0.8B arms) | ~0.5 day CPU | Enabler: most tuning moves to CPU and works on the 27B diagonal |
| 2 | **A1–A9** collector add-ons before the 27B run | ~1 day code, ~0 GPU | Hedges the 8.5 h run (split halves, K-FAC/leverage variants, blocks, k/v full, residual/embedding Fisher) |
| 3 | **J1** block-diagonal fast path (+ timing) | ~0.5 day | 27B two-sided requant: hours/tensor → minutes |
| 4 | **B1** gen-masked H_out | 1.5 h 4070 | 0 to −4% |
| 5 | **C1** tempering exponent | 45 min 4070 | 0 to −2% (K2) |
| 6 | **F3** K2.5/K4/K5 arms | 30 min | Calibrates the value at 3–4.5 bpw; checks the fractional path |
| 7 | **F1+F2** family/depth probes | 1.5 h | Collector scope; may cut 27B collection 2–3× |
| 8 | **B2/B3** K-FAC and leverage diagonals (same run as B1) | +0 | 0–2% and a faster collector (B8) |
| 9 | **C5/C6/C7/C9** structure from existing full snapshots | 1.5 h | 0 to −2% at K3 |
| 10 | **B10** dose 128/256/512 | 40 min | 27B run possibly ~2 h |
| 11 | **E5** gen-weighted H_in | 1 h | 0 to −4% (input side) |
| 12 | **H1** two-sided hproxy → kfit2 | ~1 day CPU (after 27B) | 0 to −3% via allocation |
| 13 | **G1** sensitivity scales retest with clean diagonal | 30 min | 0 to −3% |
| 14 | **D4 → H4** MLP channel partition (ceiling, then prototype) | CPU minutes; then plugin work | 0 to −4% |
| 15 | **K1** late-layer 27B pilot | ~2 h both cards | Early go/no-go |
| 16 | **B12/B13** analytic / hybrid local H_out | 1.5 h | Could cut the 27B collection to ~1 h |
| 17 | **I1/I3** head analytic two-sided + slab H_in | ~2 card-h | −0.3…−1.5% |
| 18 | **K4** uniform-trunk 27B swap | ~4 h | The V3 decision |
| 19 | **I6** embedding hidden-dim metric | ~2 h | −0.5…−1.5% |
| 20 | **E1** H_in factorial | 3 h | Explains #33-07/#33-10; may unlock E-ideas |

## The five cheapest diagnostics that would most change this ranking

1. **Gen-masked vs all-position diagonal on the 0.8B (B1, same run as B2/B3).** If gen-masking wins clearly, E5 and the 27B collector setting move to the top, and #33-07/#33-10 get a partial explanation.
2. **D1 predictor vs the existing 0.8B arms (CPU only).** If it reproduces the measured KL ordering (including "full worse than diagonal at K2"), most of C, J5, H4 and the dose questions become CPU arithmetic.
3. **Single-family and depth K2 probes with bdiag (F1/F2).** These decide whether residual writers matter (B11, A5's value), whether gate/up partition (H4) is worth it, and whether the 27B backward can be truncated or run in bf16.
4. **bdiag at K4/K5 (+K2.5) on the 0.8B (F3).** If the gain vanishes above K3, the 3.5–4.5 bpw sizes get nothing, requant priority shrinks to the low levels, and the expected value of the whole line roughly halves.
5. **`ldlq_2hess` wall time on one 27B gate_up tensor with a diagonal H_out, plus D4 on the 0.8B diagonal.** The timing sets the real requant cost (J1 urgency). D4 decides whether across-block structure (G1, H4, H6) or within-block structure (C5, J5) is where the remaining gain sits.

**Before launching `hdiag27.sh 1024`:** run diagnostic 1 (~1.5 h) and add A1/A2/A3/A4/A9 to the collector. Collecting split halves and multiple diagonal variants in one pass costs almost nothing, and a second 8.5 h run costs a night.
