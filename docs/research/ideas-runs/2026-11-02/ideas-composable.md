# Research-idea run, lens **composable**: 94 ideas for the EXL3-V3 pipeline

## Code checks behind several ideas

I read `exllamav3-frac/.../exl3_lib/quantize.py`, `quantize/ourq.py`, `v3/yaqa/collect.py`, `hdiag27.sh` and the exl3i importer. Six facts shape the ideas below.

1. **The one-sided path ignores the output scales.** `ldlq()` rounds in the rotated basis with an implicit identity output metric. `prepare_H_out` builds the rotated output metric as `H · S_v H_out S_v · H` per 128-row block. So every `--hout` arm on the 0.8B also brought in the per-row weighting **sv²**, which one-sided LDLQ never sees. Two-sided LDLQ with **H_out = I** is the output-side twin of the "scaled Hessian" (−12% proxy on the input side). Nobody has run it (idea C1).
2. **The scale refit after two-sided LDLQ uses H_in only.** In `refit_scales` the su step solves `((Q Qᵀ)∘H) r = …` with equal output weights. The `proxy_err` logged for `--hout` quants is that one-sided refit error.
3. **The global scale search ignores both Hessians.** `g_scale_search_batch` scores sampled raw tiles by plain rotated-domain MSE, with no H_in, no H_out and no LDLQ feedback.
4. **The diagonal collector does not match what the board scores.** It uses mean CE over *all* positions of 2048-token qad windows. The score counts only `gen.u8` rows, about 55% of rows on the 2B golden (166k of ~300k).
5. **The collector's backward hook already has what extra accumulators need.** It holds `dy` [T, m] and `x` [T, n] for every linear, so extra diagonals cost almost nothing.
6. **Stacked parts must share one rate.** The importer refuses "stacked parts at two rates", so gate and up share one K. A trunk "row-slab" therefore has to be split into separate stacked weights (D6).

Gains below are KL % at 2.5 bpw, where V2 is at 0.0666, and are guesses with wide error bars. "0.8B arm" means one ourq + evalq run on the 4070 (~4 min); a 0.8B diagonal collection at 1K sequences takes ~25 min.

---

## A. Add before the 27B diagonal run starts (≈8.5 h on both cards; extras cost near zero)

**A1. Collect several diagonal estimators in one pass.** In `fwd_hook_diag`, add these sums next to Sketch B's `rowsum(G_b∘G_b)`:
- (i) per-token Σ_t dy_t² (the H_Z factor);
- (ii) Σ_t ‖x_t‖² dy_t² (the weighting gate 2b used);
- (iii) Σ_t (x_tᵀH_in x_t) dy_t² (upstream Sketch-A weights);
- (iv) Σ_t (x_tᵀH_in⁻¹x_t) dy_t² (KL-Shampoo leverage weights), for linears whose H_in⁻¹ fits on the card.

- **Src:** 2610.40345 (H_Z-PTQ), 2609.38521 (ShamAN-Q), upstream yaqa_hessians.py. New: all four from one backward, chosen afterwards by the 0.8B factorial (B1).
- **Why:** On Llama-3.2-1B, YAQA's output factor was unstable (PPL 19.12) while H_Z was best (14.26 vs GPTQ 14.50). Inverse-quadratic token weights cut the factor's condition number from hundreds to ~40. The 0.8B showed Sketch B works, but not that it is the best diagonal.
- **Fit:** per-level (collection only).
- **Test:** 0.8B gate first: estimators (i)–(iv) vs Sketch B at K3 and K2 (B1). On the 27B, add the sums to the gate run and check time per sequence. Pass if the slowdown is under 3% and memory under 40 MB.
- **Gain/risk:** 0 to −3% if a better estimator exists. Risk: none for the run itself.
- **Conflicts:** #33-12/13 measured Sketch B only.

**A2. Odd/even split-half sums plus a stopping rule set in advance.** Keep two diagonal sums (odd and even sequences); YAQA_HALVES exists only for full mode. Stop when the per-family cosine between halves is ≥ the value at which 1K sequences sufficed on the 0.8B (≥0.997). Snapshots at 256/512 then become real decision points.
- **Src:** new; uses #33-09's split-half statistic.
- **Why:** Gives a noise estimate per tensor on the 27B itself. It also enables cross-validated damping and tempering (B2, B3, B5) and may cut the 8.5 h run.
- **Fit:** per-level.
- **Test:** compute the halves' cosine at every snapshot. Write the stopping threshold before the run.
- **Gain/risk:** 0% KL directly; may save 2–6 h of both cards. Risk: none.
- **Conflicts:** none.

**A3. One diagonal per document category.** Keep separate sums for code, chat, agentic, math and multilingual windows (5 × 10 MB).
- **Src:** 2606.17118 (MODE: per-modality normalized sensitivity). New for output-side Hessians.
- **Why:** If one category dominates gradient mass, the pooled diagonal under-weights the others. The private corpus has a known category mix, so the pools can be reweighted to it. This also shows whether multilingual text needs other rows (the reason #32 was dropped).
- **Fit:** per-level.
- **Test:** cosine between category diagonals. If any pair is below 0.9, run a reweighted-mix arm on the 0.8B with xcorp categories.
- **Gain/risk:** 0 to −1.5%. Risk: low.
- **Conflicts:** #32 (vocab tiering) dealt with rows, not this.

**A4. Log the per-sequence trace of each tensor's contribution.** Store tr(G_bG_bᵀ) per sequence per tensor (496 × 1K floats).
- **Src:** upstream note that H_out is dominated by a few high-gradient tokens; 2610.14934 (tail-KL selection).
- **Why:** A cheap census of heavy tails, and the input for B6 winsorization and J4 data selection.
- **Fit:** diagnostic.
- **Test:** effective sample size (Σw)²/Σw² per tensor. Below 100 out of 1K means few sequences drive the factor.
- **Gain/risk:** diagnostic only. Risk: none.
- **Conflicts:** none.

**A5. Gradient covariance at the embedding output.** Hook the embedding output and accumulate G_emb = Σ_t g_t g_tᵀ (5120², 100 MB). Optionally do the same at the final-norm output, to check the head input.
- **Src:** new. The design quantizes the embedding with H = I across its 5120 columns. Token frequency (the dropped row axis) is a separate question.
- **Why:** Enables G6. Massive-activation residual dimensions make an isotropic column metric unlikely.
- **Fit:** per-level.
- **Test:** compute the participation ratio of G_emb; if it is below 0.3, run G6.
- **Gain/risk:** enabler. Risk: none.
- **Conflicts:** #35 tested frequency weighting of rows, not a column metric.

**A6. Residual-stream gradient second moments per layer.** Collect the diagonal for all 64 layers, and the full 5120² for ~8 sampled layers (≈0.8 GB).
- **Src:** 2610.41408 (SACQ output-side metrics), 2609.11716.
- **Why:** In one layer, o_proj, down and ssm_out share the same output metric. This is the input for F11 (analytic gate/up H_out) and H3 (depth prior). It also checks the 0.8B finding that residual writers have nearly isotropic H_out (participation ratio 0.4–0.6).
- **Fit:** per-level or knapsack prior.
- **Test:** compare with the writers' own diagonals (cosine).
- **Gain/risk:** enabler. Risk: host RAM (30 GB); keep the full matrices on disk.
- **Conflicts:** none.

**A7. Nyström low-rank sketch of H_out per linear.** Keep Y = H_out Ω (Ω an m×r Gaussian, r = 32–64; 4.4 MB for a 17408-row tensor). That is Y += G_b(G_bᵀΩ), with one extra pass over G_b's row blocks per sequence. Reconstruct H ≈ Y(ΩᵀY)⁺Yᵀ and add the off-subspace diagonal back.
- **Src:** 2610.04204 (Clean: Nyström plus reintegration of off-subspace curvature), 2610.16792 (frequent directions). New use.
- **Why:** The full H_out added extra at K3 (−9.7% vs −8.0% for the diagonal), and the mixed arm was best (−9.9%). The full matrix is 206 GB and noisy for gate/up. A low-rank-plus-diagonal factor is the middle option (B4 decides).
- **Fit:** per-level.
- **Test:** first check B4 offline on the 0.8B. Add to the 27B run only if rank-r plus diagonal beats the diagonal at K3 with z ≤ −2. Allow ≤15% time overhead.
- **Gain/risk:** 0 to −2%. Risk: G_b recomputation time.
- **Conflicts:** #33-12 (full and mixed vs diagonal); a sketch was not tested.

**A8. Mask the backward loss to scored positions.** Apply the CE only at `gen.u8` positions of each window.
- **Src:** new (the score definition: aggscore.py counts gen rows only).
- **Why:** The board's Fisher is the Fisher of KL at gen rows. Today non-scored rows (prompts, tool schemas) also add gradient. Their hidden states still matter through context, and the masked loss keeps exactly that dependence.
- **Fit:** per-level.
- **Test:** 0.8B first (B8). It changes the backward, so it cannot be an extra accumulator.
- **Gain/risk:** 0 to −3%. Risk: the qad windows need gen masks (build them the way the golden does).
- **Conflicts:** none.

**A9. Gaussian Fisher probe in place of a sampled label.** Instead of a one-hot sampled-label CE, backpropagate d logits = r with r = √p⊙ξ − p·(√pᵀξ), ξ ~ N(0, I) over the full vocabulary. Its covariance is exactly diag(p) − ppᵀ, the true Fisher, but without one-hot kurtosis.
- **Src:** new (cf. the Rademacher logit probe of 2610.18139, which is not Fisher-weighted).
- **Why:** For the same number of backwards, a lower-variance second-moment estimate. If 256 probe sequences match 1K sampled ones, the 27B run takes ~2 h.
- **Fit:** per-level.
- **Test:** B9 dose curve on the 0.8B.
- **Gain/risk:** cost only (up to 4× cheaper). Risk: the full-vocab r per position costs memory; chunk it like `chunk_loss`.
- **Conflicts:** #33-13 dose was measured with sampled labels.

---

## B. Better H_out from data we already have (0.8B; mostly offline transforms of the 1K–8K snapshots)

**B1. Diagonal-estimator factorial.** Compare Sketch B (8K) against A1's (i)–(iv) at K3 and K2 on base's settings, and pick the one to deploy.
- **Src:** see A1.
- **Fit:** per-level.
- **Test:** one 0.8B collection with all accumulators (~30 min), then 8 arms (~35 min). Pass if an arm beats `bdiag` with z ≤ −2 at both K.
- **Gain/risk:** 0 to −3%. Risk: low.
- **Conflicts:** #33-12.

**B2. Tempering exponent on the diagonal.** Use D^p, rescaled to the same trace, for p ∈ {0.25, 0.5, 0.75, 1, 1.25}.
- **Src:** 2609.38521. Spectral tempering at p = ½ was best; untempered dense NKP gave no gain (19.21), tempered 17.46. The rationale is that the errors are outside the quadratic regime.
- **Why:** The diagonal CV is 1.5–4 for input projections; quantization error at K2 is large, so the local curvature over-trusts extreme rows.
- **Fit:** per-level (and possibly per K).
- **Test:** offline transform of `hdiag8192`, 10 arms on the 0.8B (~40 min).
- **Gain/risk:** 0 to −3%. Risk: low.
- **Conflicts:** none; #33-11 used only p = 1.

**B3. Output damping sweep for the diagonal, per K.** Try sigma_reg_out ∈ {0.001, 0.003, 0.01, 0.03, 0.1} at K2, K3 and K4.
- **Src:** gates 2b/2c (more regularization helped the noisy full factor), trust-region argument.
- **Why:** Only 0.01 was used with the diagonal. The best value may fall as K rises, which is library-compatible.
- **Fit:** per-level.
- **Test:** 15 arms (~1 h). If the best value differs by K, use a per-level table. A per-tensor choice on the split halves is possible with A2.
- **Gain/risk:** 0 to −2%. Risk: low.
- **Conflicts:** #38 (input damping), not output.

**B4. Diagonal plus low rank, offline.** From the stored full 8K snapshots take the top-r eigenvectors, r ∈ {4, 16, 64}, and add back the residual diagonal.
- **Src:** 2610.04204, 2609.34884 (SubRot: concentrated activation-gradient Fisher spectra), #33-09 (participation ratio 0.04–0.6).
- **Why:** It could keep the full factor's K3 extra (−1.9 points over the diagonal) without the gate/up noise that hurt at K2.
- **Fit:** per-level.
- **Test:** 6 arms (~25 min), with the full H_out already on the 4070. Pass if K3 beats `bdiag` at z ≤ −2 and K2 is not worse; this decides A7.
- **Gain/risk:** 0 to −2%. Risk: low.
- **Conflicts:** #33-12 (bmix best at K3).

**B5. Shrink the full factor toward its diagonal.** H = (1−λ)·full + λ·diag(full), with λ chosen per tensor by the two-sided proxy on the held-out 4K half.
- **Src:** Ledoit–Wolf; 2610.03027 (the dense metric overfit while the diagonal generalized: 0.00933 vs 0.00234).
- **Why:** Suppresses noise in a principled way. It differs from damping, which adds identity and loses the diagonal's structure.
- **Fit:** per-level.
- **Test:** offline λ fit, then 2 arms.
- **Gain/risk:** 0 to −1.5%. Risk: low. It only matters if full or low-rank factors are collected for the 27B.
- **Conflicts:** #33-12.

**B6. Winsorize token contributions before accumulating.** Cap each token's dy_t² contribution at the per-tensor 99.5th percentile, or the per-sequence share (with A4).
- **Src:** upstream note (a few high-gradient tokens dominate); robust statistics.
- **Why:** If the dominant tokens are sinks or BOS-like positions, they are not where KL is lost (see E2).
- **Fit:** per-level.
- **Test:** 0.8B collection variant (~25 min) plus 2 arms.
- **Gain/risk:** 0 to −2%. Risk: removes real signal; E2 tells which.
- **Conflicts:** none.

**B7. H_out at the operating point.** Collect the diagonal on the 0.8B *quantized* to base K3 (fake-quant), with labels still sampled from bf16, then requant K2 and K3 with it.
- **Src:** 2605.23078 (GEMQ: importance from the 2-bit model beat FP, 9.66 → 9.01 PPL), 2609.38521 (sequential refresh).
- **Why:** The quadratic model is taken around bf16, but quants live far from it.
- **Fit:** per-level (one quantized model per size band; reused by all levels).
- **Test:** 0.8B (~30 min collection plus 4 arms).
- **Gain/risk:** 0 to −2%. Risk: a second 27B collection (8.5 h) if adopted.
- **Conflicts:** none.

**B8. Gen-masked loss test (A8's gate).** 0.8B collection with the loss masked to gen rows (masks from the xcorp render), vs unmasked, at K3 and K2. Pass if masked is better by z ≤ −2 at either K and not worse at the other.
- **Src:** new.
- **Fit:** per-level.
- **Test:** ~1 h.
- **Gain/risk:** 0 to −3%. Risk: low.
- **Conflicts:** none.

**B9. Dose curve for the Gaussian probe (A9's gate).** Diagonal from 128/256/512/1K probe sequences vs sampled labels, with the same windows.
- **Src:** new.
- **Fit:** per-level.
- **Test:** ~40 min. Pass if 256 probe sequences keep ≥85% of the 8K diagonal's gain.
- **Gain/risk:** saves 27B time. Risk: low.
- **Conflicts:** #33-13.

**B10. Window length vs sequence count at equal tokens.** Compare 2K × 1024 tokens with 1K × 2048 tokens.
- **Src:** new. Sketch B is sequence-level, so cross-position terms grow with window length.
- **Why:** If shorter windows are as good per token, the 27B run fits more category diversity.
- **Fit:** per-level.
- **Test:** 0.8B (~50 min).
- **Gain/risk:** 0 to −1%. Risk: low.
- **Conflicts:** none.

**B11. Category-balanced pooling.** Reweight A3's per-category diagonals to the private corpus's category mix, vs plain pooling.
- **Src:** 2606.17118.
- **Fit:** per-level.
- **Test:** 0.8B with xcorp categories, 2 arms.
- **Gain/risk:** 0 to −1%. Risk: low.
- **Conflicts:** none.

---

## C. Output-side information inside the quantizer (per-level library)

**C1. Two-sided LDLQ with H_out = I: account for the output scales sv².** Pass an all-ones `hout_diag`. `prepare_H_out` then makes the rotated output metric H·S_v²·H, and LDLQ feeds errors back across rotated rows so that error moves to low-sv channels. Also run a diagonal-only-without-S arm (D with S = 1).
- **Src:** new. It is the output-side twin of the input-side scaled Hessian (held-out proxy 0.88 at K2.5).
- **Why:**
  - It needs no backward, so it can run on the **27B today**.
  - It splits the 0.8B −8% into an S part and a D part, which tells whether the 8.5 h run carries information that S alone does not.
  - Its value depends on how much sv varies inside each 128-row block (diagnostic 1 at the end).
- **Fit:** per-level, no format change.
- **Test:** CPU census of svh dispersion within 128-row blocks from the existing ladder quants (minutes). Then 0.8B arms: S-only, D-only, S+D (= bdiag) at K3/K2 (~15 min). If S-only reaches ≥50% of bdiag's gain, do a 27B swap test of K2.5/K3 levels with S-only on card 0 (~3 h plus evals).
- **Gain/risk:** 0 to −5% standalone; part of it may already be inside the diagonal gain. Risk: low.
- **Conflicts:** #48 and #33-06 changed sv itself, not the rounding metric. This leaves sv unchanged.

**C2. Two-sided scale refit.** After `ldlq_2hess`, refit with the output metric. With diagonal D, the su step becomes ((Q D Qᵀ)∘H) r = rowsum((Q D)∘(HW)). With a full or low-rank H_out, the sv step couples columns too. Log the two-sided proxy.
- **Src:** new (fixes check 2 above); 2609.38599 (JARQ: refit against the true objective after rounding).
- **Why:** Today's refit polishes a two-sided quant toward the one-sided optimum.
- **Fit:** per-level.
- **Test:** code change of ~30 lines; 0.8B arms at K3/K2 with diagonal H_out (~10 min). Pass if z ≤ −2 at either K.
- **Gain/risk:** 0 to −1.5%. Risk: none.
- **Conflicts:** none.

**C3. Use the two-sided proxy for every internal choice.** Pick the per-tensor sigma_in sweep (--sigma-in-sweep) and the proxy_err written to proxy.jsonl by the two-sided proxy, for `--hout` quants.
- **Src:** new.
- **Why:** The sweep currently judges damping by the one-sided proxy against undamped H_in. With H_out on, the best sigma_in can differ.
- **Fit:** per-level.
- **Test:** count the tensors whose winner changes on 2 layers; if over 20%, run a 0.8B arm.
- **Gain/risk:** 0 to −0.5%. Risk: none.
- **Conflicts:** #38.

**C4. Choose the global scale under LDLQ, not raw tile MSE.** Score g_scale candidates by the LDLQ proxy (one- or two-sided) of a fast quant of sampled 16-row groups with real feedback, instead of plain MSE of raw rotated tiles (check 3 above).
- **Src:** 2609.36654 (Schur Replay: score a candidate scale in the state it creates; a Euclidean score cost −90 of 12,047 questions), 2609.37416.
- **Why:** LDLQ quantizes feedback-adjusted targets, whose spread differs from the raw weights. #39 found the best factor drifts with K, which is consistent with a systematic mismatch.
- **Fit:** per-level (cost: a few extra fast-quant passes per tensor).
- **Test:** CPU on 3 layers at K2/K3/K5 comparing the proxy at the MSE-chosen and proxy-chosen g_scale; then a 0.8B level if the gap is over 1%.
- **Gain/risk:** 0 to −2%. Risk: low.
- **Conflicts:** #39 (blocked; this is a different criterion, not a fixed factor).

**C5. Block act-order inside LDLQ, on both axes.** Permute 16-wide blocks by descending rotated-H diagonal energy (and rotated-H_out energy in two-sided mode) before the LDL; unpermute the result. Tile storage is unchanged, so inference is free.
- **Src:** 2610.00432 (energy ordering 8.06 → 8.00 PPL), 2610.40345 (Fisher-trace block order).
- **Why:** The Hadamard flattens energy inside 128-blocks but not between them.
- **Fit:** per-level.
- **Test:** 0.8B arms (~10 min).
- **Gain/risk:** 0 to −1%. Risk: none.
- **Conflicts:** none.

**C6. H_out gain across K before requanting the ladder.** 0.8B arms at K1.5, K2.5, K4, K5 and K6 (plus one quarter step) with the diagonal.
- **Src:** new. Only K2/K3 have been measured.
- **Why:** Decides whether all ~16 ladder runs go two-sided. High K is where turboderp's tensors beat ours on KL despite worse input-side proxy, so an output-side signal may matter more there.
- **Fit:** diagnostic for the library.
- **Test:** 10 arms (~40 min).
- **Gain/risk:** decides scope. Risk: none.
- **Conflicts:** #33-11/12.

**C7. Second pass with refitted scales.** Rebuild the rotated output metric from the *refitted* sv and the scaled H_in from the refitted su, requantize, and repeat once.
- **Src:** 2609.38599 (JARQ alternation; small gains on strong hosts), 2610.37336 (CoS-VQ: scale coupling).
- **Fit:** per-level (2× quant time).
- **Test:** 0.8B arms at K2/K3.
- **Gain/risk:** 0 to −1%. Risk: doubles ladder time.
- **Conflicts:** none.

**C8. Turn H_out on per family.** Skip two-sided for tensors whose S²D is nearly flat inside 128-blocks (on the 0.8B, residual writers have diagonal CV ~0.2). Keep it where S²D varies.
- **Src:** #33-09.
- **Why:** Saves ldlq_2hess time and avoids spending noise where there is no signal.
- **Fit:** per-level.
- **Test:** the per-family arms already exist (bmix); add a "residual writers one-sided" arm.
- **Gain/risk:** 0 to −0.5%, plus time. Risk: none.
- **Conflicts:** #33-12.

**C9. List-Viterbi re-ranking under the exact tile metric.** Keep the top-L paths (L = 4–8) per tile from the trellis search. Pick the one minimizing the exact block-LDL tile objective, which has a dense 16×16 D_in block on the input side (and on the output side in two-sided mode), rather than the Euclidean error.
- **Src:** coding theory (list Viterbi); 2610.09212 and 2610.00432 (the weighted metric is what helps); 2610.32127 (coupled decisions).
- **Why:** Within a tile, the Viterbi is Euclidean while the LDL block metric is not identity.
- **Fit:** per-level (quantizer-only change; decode unchanged).
- **Test:** prototype in `viterbi.c` on CPU sampled rows (K2/K3): proxy vs best-path. Pass if ≥1% proxy.
- **Gain/risk:** 0 to −1.5%. Risk: kernel work in qtf2.
- **Conflicts:** #57 is about tail-biting, not this.

---

## D. Free symmetries: permutations and splits the kernels already absorb

**D1. Reorder MLP intermediate channels to equalize down's input Hadamard blocks.** Permuting gate/up rows and down columns jointly by P leaves the model exactly invariant. Choose P so each 128-block of down's input gets similar total energy (zigzag on diag H_in(down)), or by the down proxy.
- **Src:** 2605.28803 (HoloQ-VLA zigzag), 2610.07121 (SoloQ: block energy imbalance 67.9% → 22.1%), 2605.24754 (permutation symmetries).
- **Why:** SwiGLU outputs are heavy-tailed, and a 128-block that holds several large channels dominates its rotated block. The permutation is folded into the weights.
- **Fit:** per-level (offline weight permutation before quantizing).
- **Test:** CPU proxy on 27B Hessians with the trellis tool, for natural vs zigzag vs 4 random permutations on 4 layers. Then a 0.8B level.
- **Gain/risk:** 0 to −2%. Risk: low.
- **Conflicts:** #41 (random sign seeds: 0.5% spread). Structured permutation changes which channels share a block, which a random sign seed does not.

**D2. Choose the gate/up row order to exploit two-sided LDLQ.** With S²D in the rotated output metric, compare natural, sorted-by-D and interleaved (equal-volume) orders within 128-row blocks. It is the same joint permutation as D1, so a joint choice is needed.
- **Src:** 2608.04074 (NOVA-KV: sorted grouping 37.2 vs random 74.4 vs equal-volume 75.4); new for LDLQ.
- **Why:** The diagonal gain comes through mixing inside blocks; the order sets how much structure the rotated metric has.
- **Fit:** per-level.
- **Test:** 0.8B (6 arms).
- **Gain/risk:** 0 to −2%. Risk: may conflict with D1's best order.
- **Conflicts:** none.

**D3. Free permutations inside GDN layers.**
- Value channels within a head: v rows, z rows, conv1d channels, gated-norm weight and ssm_out columns.
- Key channels jointly for q and k: the per-head L2 norm, the depthwise conv and the delta rule with a scalar decay are all invariant.
- Head order.

Use them as in D1/D2: in_proj_z has diagonal CV 3.96 and diagonal share 0.81 on the 0.8B.
- **Src:** 2610.39418 (LinQuant's invariance analysis for GDN), new for weights.
- **Fit:** per-level.
- **Test:** check invariance exactly on the 0.8B (bit-for-bit logits within fp32 noise), then proxy and KL arms.
- **Gain/risk:** 0 to −1.5%. Risk: easy to break an invariance (the conv layout, the [q|k|v|z] spans).
- **Conflicts:** none.

**D4. Permute v/o channels within attention heads** (16 layers; q/k are blocked by RoPE pairing).
- **Src:** as D3.
- **Gain/risk:** 0 to −0.3%. Risk: low.
- **Conflicts:** none.

**D5. Row-slab ceiling with output-side weights.** Redo E3c's ½·log₂(AM/GM) with 128-row slab importance from the H_out diagonal, after sorting rows via the free permutation, for gate/up, GDN z/qkv and attention q.
- **Src:** #29-02 (E3c: input-side ceiling 0.05 bit); 2605.26660 (finer allocation units helped).
- **Why:** With diagonal CV ≈ 1 (gate/up), a lognormal estimate gives ~0.25 bit per row before slabbing. This decides D6.
- **Fit:** diagnostic.
- **Test:** CPU on the 0.8B diagonal now, on the 27B diagonal later. A ceiling of ≥0.1 bit on gate/up justifies D6.
- **Gain/risk:** decides D6. Risk: none.
- **Conflicts:** #29-02 (the input side could not tell rows apart; the output side can).

**D6. Split the MLP into two K classes.** Sort intermediate channels by a joint score (gate/up H_out rows plus down's H_in columns). Build two stacked weights, [gate_a|up_a] at K_a and [gate_b|up_b] at K_b, plus down_a and down_b (two GEMMs accumulated). Each part is its own knapsack item.
- **Src:** new. It is a row-slab K that needs only an arch-plugin change, because each part is an ordinary exl3i tensor and stacked parts must share a rate (check 6).
- **Why:** Spends bits on the 25% of channels that matter most in the largest tensors (gate_up ≈45% of FLOPs).
- **Fit:** format change at the arch-plugin level (kernels unchanged); per-level quants of each part.
- **Test:** if D5 passes, fake-quant on the 0.8B (evalq) with a hand mix (top 25% at K+1, rest at K−⅓, equal bits) vs uniform. Then the 27B time table.
- **Gain/risk:** −1 to −5% if the ceiling is real. Risk: extra launches (+5–10% MLP time); split-down accumulation.
- **Conflicts:** #29-02 (row-slab K dropped on input-side evidence).

**D7. Group-wise K for GDN heads.** Split in_proj and ssm_out by head groups (head permutation is free), ranked by H_out mass per head and the decay half-life prior (L1).
- **Src:** 2609.38169 (STEPQuant: half-life vs accumulated error ρ ≈ 0.80 per head on Qwen3.8-27B).
- **Fit:** format change (arch plugin).
- **Test:** as D5 per head on the 0.8B, then a fake-quant mix.
- **Gain/risk:** 0 to −2%. Risk: medium.
- **Conflicts:** none.

**D8. Choose permutations by proxy from structured candidates.** For each MLP, evaluate {natural, zigzag, sorted, 2 random} by the fast two-sided proxy and keep the best.
- **Src:** 2610.05265 (Loopy: discrete transform choice by end loss).
- **Why:** Unlike #41's sign seeds, these candidates differ in block content.
- **Fit:** per-level.
- **Test:** CPU proxy spread across the candidates; worthwhile only if the spread is over 2%.
- **Gain/risk:** 0 to −1%. Risk: overfitting a noisy proxy; check on split halves.
- **Conflicts:** #41.

---

## E. Why gradient-weighted H_in hurts (diagnostics first)

**E1. Effective sample size of the token and sequence weights.** From the 0.8B Sketch B data and gate-2e-style weights ‖dy_t‖², compute (Σw)²/Σw² per linear.
- **Src:** new.
- **Why:** If the ESS is ≪ N, gradient weighting turns 10M tokens into a few thousand, which would explain +20…+44% without any theory.
- **Test:** CPU, an hour.
- **Gain/risk:** diagnostic. Risk: none.
- **Conflicts:** #33-07, #33-10.

**E2. Where KL is lost vs where the Fisher is.** For a K2 0.8B quant, compare per-token KL (evalq rows) with the per-token Fisher trace at bf16 (one backward). Report the share of KL from the lowest-Fisher quartile.
- **Src:** 2609.11716 (errors concentrate on high-probability tokens), 2606.15682 (low-entropy tokens fail through tail-mass inflation).
- **Why:**
  - Confident tokens have near-zero Fisher, but quantization pushes them out of the quadratic regime.
  - If they carry a large share of KL, uniform token weights (calib H_in) are a robust prior and gradient weighting is wrong by construction.
  - It also predicts whether gen masking (A8) and winsorization (B6) help.
- **Test:** 0.8B, ~1 h.
- **Gain/risk:** diagnostic. Risk: none.
- **Conflicts:** none.

**E3. Tempered gradient weights for H_in.** H_in = Σ w_t^p x xᵀ with w_t = ‖dy_t‖²/mean, p ∈ {0.25, 0.5}, plus a winsorized variant.
- **Src:** 2609.38521 (tempering), 2610.18139 (Jacobian-weighted input Hessian blended per tensor; a fixed α is unsafe).
- **Why it might work now:** gate 2e used p = 1 on 100 sequences. Tempering limits ESS collapse (E1).
- **Fit:** per-level.
- **Test:** 0.8B collection (~30 min) plus 4 arms. Pass only if it beats base by z ≤ −2. Run only if E1 shows ESS collapse.
- **Gain/risk:** 0 to −2%. Risk: high; three negative results so far.
- **Conflicts:** #33-07, #33-10.

**E4. Matrix geometric mean of the calib and gradient Hessians.** H = H_calib # H_grad (or the t = ¼ geodesic), instead of an arithmetic blend.
- **Src:** 2610.40345 (# mean for transforms).
- **Why it might work now:** the gate-2e blend was never run, and the geodesic keeps H_calib's spectrum shape.
- **Fit:** per-level.
- **Test:** offline on the 0.8B (Sketch B H_in is on disk), 2 arms.
- **Gain/risk:** 0 to −1%. Risk: high.
- **Conflicts:** #33-07.

**E5. Deterministic reweighting toward scored positions.** H_in = Σ (1 + λ·gen_t) x xᵀ with λ ∈ {0.5, 1}. The weights are bounded and noise-free, unlike gradient weights.
- **Src:** new (score definition).
- **Fit:** per-level (a 27B Hessian recollection over ≥2M tokens if adopted).
- **Test:** 0.8B input Hessians with gen masks over calib (~15 min), 2 arms.
- **Gain/risk:** 0 to −1%. Risk: prompt positions matter through context.
- **Conflicts:** none.

**E6. Hessian from bf16-generated continuations of calib prompts (turboderp's "SC" idea).** Compare H_in from bf16-model continuations of calib prompts vs calib H, and a 50/50 blend.
- **Src:** tracker note on turboderp's SC quants (model-written text).
- **Data rule:** allowed as preference (1) in the ROADMAP.
- **Test:** 0.8B, generate 1M tokens on the 4070 (~1–2 h), collect, 2 arms.
- **Gain/risk:** 0 to −1.5%. Risk: distribution shift toward the model's own style.
- **Conflicts:** none.

---

## F. LDLQ and trellis quantizer (per-level)

**F1. Coordinate-descent sweeps over tiles after LDLQ.** Re-quantize each tile given *all* other tiles' current errors (earlier and later), target w_b − H_bb⁻¹H_b,¬b e_¬b. Re-Viterbi and accept if the exact proxy drops; do 2–3 sweeps. In one-sided mode only tiles in the same output column block interact.
- **Src:** 2610.32127 (MPQ: LDLQ + 3 CD sweeps took Llama-3-8B E8P 2-bit PPL from 15.01 to 12.76; the gain was small at 4 bit).
- **Fit:** per-level (≈2–3× quant time; apply at K ≤ 3).
- **Test:** CPU prototype on sampled full tensors with viterbi.c at K2/K3; pass if ≥3% proxy. Then qtf2 GPU on the 0.8B and a 27B level.
- **Gain/risk:** −1 to −4% (mostly at K ≤ 3). Risk: over-fitting the Hessian sample (F2).
- **Conflicts:** none.

**F2. Held-out-Hessian gate for every refinement.** Accept a CD or list-Viterbi change only if it also lowers the proxy under an independent Hessian half (H_A/H_B from ≥1M tokens per half).
- **Src:** 2610.11226 (DRQ: refining to lower reconstruction loss *raised* PPL in 47–74% of layers), D2 (9% in-sample proxy edge at 1.2M tokens).
- **Fit:** per-level.
- **Test:** split-half Hessians for 6 layers (collection exists for 3), then F1 with and without the gate on the 0.8B.
- **Gain/risk:** protects F1. Risk: needs half-Hessians for all layers.
- **Conflicts:** none.

**F3. Activation-denoising LDLQ target with level-matched noise.** A noisy forward pass (Gaussian at γ times the per-channel range, propagated) gives C = E[xηᵀ] and N = E[ηηᵀ]. Quantize W·F in the metric G = H + αI + λ²N + λ(C + Cᵀ). Set γ per ladder level to match that level's real error scale, which keeps it library-compatible.
- **Src:** 2610.07522. It closed 18–31% of the parallel→sequential gap at W3 GPTQ; at W2 with QuIP# it beat sequential on 8B/13B; the noise pass costs 1.3× a clean one. New here: noise scale per level.
- **Why:** Our quants come from clean-input Hessians, so error compounding is not modelled at all.
- **Fit:** per-level (collection like a Hessian: forward-only, 1–2M tokens, shrunk toward the published H).
- **Test:** 0.8B full pipeline (~2 h on the 4070) at K2/K3. Pass if z ≤ −2 vs base. 27B cost ≈ a 2M-token forward with hooks on both cards (several hours).
- **Gain/risk:** −1 to −4%. Risk: four knobs; C and N are as big as H.
- **Conflicts:** ROADMAP "sequential Hessians" (this is the per-level, parallel version).

**F4. Per-level quantized-prefix Hessians.** Each ladder level is a uniform whole-trunk quant, so collect H_in^(K) from that level's own activations, which is library-compatible. Shrink it toward the published H.
- **Src:** ROADMAP item; 2609.11716 (most error counteraction comes from the block's response to its input error).
- **Fit:** per-level (expensive: one collection per level).
- **Test:** 0.8B at K2 (H from a K2 prefix vs calib H), 2 arms.
- **Gain/risk:** −1 to −5%. Risk: noise at small token counts; mixes differ from uniform levels.
- **Conflicts:** ROADMAP (V3.5–V5), now framed per level.

**F5. Within-tile anisotropy diagnostic, then a weighted Viterbi if needed.** Compute Γ_b (within-tile anisotropy of the LDL D after the 128-point Hadamard) per family. Only if Γ_b > 0.01 bit somewhere, weight the Viterbi's per-position cost by D_ii.
- **Src:** 2610.09212 (Γ_b drops from 0.038 to 0.00013 bit after a full RHT; ours is blockwise), 2610.00432.
- **Fit:** per-level.
- **Test:** CPU from existing Hessians (an hour).
- **Gain/risk:** 0 to −1%. Risk: likely inert.
- **Conflicts:** none.

**F6. Census of Hessian conditioning.** Per tensor, compute the AM/GM of the Cholesky pivots and the condition number after damping and the Hadamard.
- **Src:** 2608.25188 (LDLQ slack ∝ AM vs GM of pivots), 2609.11687 (down_proj blow-up under ill-conditioned H).
- **Why:** Flags tensors where damping or ordering choices matter, and gives kfit2 a covariate.
- **Test:** CPU.
- **Gain/risk:** diagnostic. Risk: none.
- **Conflicts:** #38.

**F7. Check the scale-shrinkage bias against KL.** LS-refit scales shrink reconstructions toward zero by roughly 1/(1 + NSR), about 5–9% at K2. Sweep a global multiplier on svh per level and family, ε ∈ {−2, 0, +2, +4, +6}%, and measure KL directly.
- **Src:** 2610.07121 (unbiased-projection scale), 2606.20474 (an MSE-optimal shrink beats raw absmax), 2609.11716 (the length error is small, 3.7%).
- **Why:** Whether MSE-optimal shrinkage is also KL-optimal through 64 RMSNorm layers and the head is unknown.
- **Fit:** per-level (a constant folded into svh).
- **Test:** 0.8B (5 arms at K2, ~20 min).
- **Gain/risk:** 0 to −2%. Risk: low.
- **Conflicts:** #39 (CPU tool, proxy-based).

**F8. Attention-weighted H_in for v_proj.** In the 16 full-attention layers, quantize v_proj with A = E Σ_s ω_s x_s x_sᵀ, ω_s = Σ_t P²_t,s (how much attention a position's value receives), shrunk toward the plain H.
- **Src:** 2610.41408 (SACQ's structure-aware metric for v), 2609.26173 (v is the dominant sensitivity in 7 of 9 models).
- **Fit:** per-level (collect A over 1M tokens on the 16 layers).
- **Test:** 0.8B (6 attention layers): proxy and KL arm.
- **Gain/risk:** 0 to −0.5%. Risk: low.
- **Conflicts:** none.

**F9. SwiGLU-aware per-level refit of gate/up scales.** With clean inputs and only this MLP quantized, fit per-channel multipliers a_j (gate, inside SiLU) and b_j (up) that minimize ‖SiLU(a⊙g_q)⊙(b⊙u_q) − SiLU(g)⊙u‖² over ~100k tokens. Fold them into svh.
- **Src:** 2609.34680 (QuantForge: three ridge-fitted MLP scalars per layer halved Fit-3 damage at W4A4), 2610.41408.
- **Why:** refit_scales fits gate and up separately in their own output metric, which ignores the product nonlinearity.
- **Fit:** per-level (parallel, library-compatible; zero format change).
- **Test:** 0.8B (one forward per layer plus Adam on 2×3584 scalars, ~15 min), K2/K3 arms.
- **Gain/risk:** 0 to −2%. Risk: low.
- **Conflicts:** none.

**F10. Analytic H_out diagonal for gate/up through the SwiGLU chain (backward-free).**
- d_gate,r = (W_dᵀ Ĝ W_d)_rr · E[(SiLU′(g_r) u_r)²]
- d_up,r = (W_dᵀ Ĝ W_d)_rr · E[SiLU(g_r)²]
- Ĝ is the residual gradient metric (A6) or the identity.
- **Src:** 2610.41408 (SACQ formula).
- **Why:** If it matches Sketch B (cosine ≥ 0.95), the largest tensors' factor is nearly free and can be checked on the 27B before collection. Its mismatch also measures how much cross-token structure Sketch B uses.
- **Fit:** per-level.
- **Test:** 0.8B (one forward plus the existing diagonal), then K3/K2 arms with the analytic diagonal.
- **Gain/risk:** cost only; possibly 0 to −1% if less noisy. Risk: the per-token correlation is dropped.
- **Conflicts:** none.

**F11. Embedding trellis with a separable row×column metric.** Covered by G6; listed here for completeness of the quantizer list.

---

## G. Head and embedding (both count toward bpw)

**G1. Restrict head statistics to scored rows.** The head's only effect on the score is through gen rows. Recompute the head's H_in, the Fisher row ranking for slabs, and headsim's evaluation over gen-masked head inputs only, via the E1b dump hook on gen rows.
- **Src:** new (score definition).
- **Why:** Prompt-position hidden states (system prompts, tool schemas) currently shape the head quant even though their logits are never scored.
- **Fit:** per-level (head donors).
- **Test:** headsim on gen-masked inputs, masked-H vs unmasked-H head at K4/K5.
- **Gain/risk:** −2 to −8% head KL, i.e. about −0.1 to −0.5% total. Risk: none.
- **Conflicts:** #34.

**G2. Two-sided head quant with the analytic softmax Fisher.** Output-side factor = diag(E[p(1 − p)]) per vocab row, which is already computed for slab ranking. Optionally add the top-r of E[ppᵀ].
- **Src:** 2608.12026 (SoftWater: class-weighted head metric; 6.5–8.3× lower head KL than WaterSIC, though partly from variable rate), and the 0.8B diagonal win.
- **Fit:** per-level (head donors).
- **Test:** headsim (exact head-only KL) on uniform and slab heads at K4–K6.
- **Gain/risk:** −5 to −15% head KL, about −0.3 to −1% total. Risk: within a slab the rows are already similar, so the metric is flatter.
- **Conflicts:** #34 (mean-row correction already adopted).

**G3. Search the softmax shift beyond the mean row.** W_t = W − t·1μᵀ for t ∈ {−2, −1, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8}, chosen by headsim.
- **Src:** 2609.31291. Phi's GPTQ head went from 0.158 to 0.059 KL at t* = 4; the Qwen3.5 GPTQ head gained ~0. Ours uses t = 1, which already gave −30%.
- **Fit:** per-level (head donors).
- **Test:** 11 head quants on card 0 (one at a time: ~22 GB host RAM each), headsim.
- **Gain/risk:** 0 to −5% head KL. Risk: low.
- **Conflicts:** #34.

**G4. Fisher-weighted shift vector.** Use a = Σ_r f_r w_r / Σ_r f_r (rows weighted by E[p(1 − p)]) instead of the plain mean row; still a single shared vector, so softmax invariance is exact. Then apply G3's t grid along a.
- **Src:** new; the paper offers per-group coefficients, which break invariance unless shared.
- **Fit:** per-level.
- **Test:** 3 headsim arms.
- **Gain/risk:** 0 to −3% head KL. Risk: low.
- **Conflicts:** #34.

**G5. Low-rank residual correction for the head.** After quantizing, fit (W − W_q)·C^p ≈ UΣVᵀ at rank 6–8 (p = 0.75) and store the factors in int8 (≈0.001 bpw). It adds one tiny bf16 GEMM to logits; the head runs once per token, so the cost is decode-only and negligible.
- **Src:** 2608.02703 (ARCHead: rel-PPL 1.0104 → 1.0058 vs a GPTQ head at 4.1 bit on an 8B).
- **Fit:** format change (plugin side branch) on shipped mixes.
- **Test:** headsim with the correction at K4 and slab budgets.
- **Gain/risk:** −5 to −15% head KL on top of slabs, about −0.3 to −1% total. Risk: plugin work.
- **Conflicts:** #34.

**G6. Embedding with a Fisher column metric.** Quantize rows with the shared metric G_emb (A5, in the embedding's within-row basis) instead of H = I. Rows remain independent (no frequency tiering).
- **Src:** new; analogous to using H_in for linears.
- **Why:** The design chose H = I to avoid frequency tiering. The column metric is a separate axis, likely anisotropic because of massive residual dimensions. The embedding sits at its t1.5 floor at ≤2 bpw, and the trellis embedding was worth −5.4% at 2.5 bpw.
- **Fit:** per-level (embedding menu).
- **Test:** needs A5; CPU embedding sweep scored by G_emb-weighted error, then real swaps on golden-ml and a search slice.
- **Gain/risk:** −0.3 to −2%. Risk: multilingual guard (score golden-ml as in #35).
- **Conflicts:** #32/#35 (row-axis weighting, not this).

---

## H. Sensitivity measurement and the knapsack model

**H1. kfit3: refit the sensitivity model on a two-sided proxy.** Recompute hproxy for all ~8,100 (tensor, option) rows as err₂ = Σ_n d_n e_nᵀ H_in e_n (and the sv-aware form), using the plugin's host decoder and the 27B diagonal. Refit kfit2 on err₂ and compare cross-validated log-RMSE (0.118) and pair ranking.
- **Src:** 2609.28270 (price ½Tr(HΣ): ρ = 0.925–0.947 with measured KL), 2609.33923 (on Qwen3.8-27B the layer objective mis-ranks: Spearman −0.15 isolated vs 0.83 propagated).
- **Why:** The fit prices our K7–K8 at 1.58× KLD per unit of input-side proxy and turboderp's at ~1.0×. That points to error moving into output directions the input-side proxy cannot see.
- **Fit:** per-level (knapsack; CPU only).
- **Test:** CPU day after the 27B diagonal exists. Pass if CV improves by more than its bootstrap SE.
- **Gain/risk:** −1 to −4% through better allocation and pricing of new options. Risk: low.
- **Conflicts:** #20 (re-ranker); this improves the input feature, not the model.

**H2. Analytic prior for unmeasured options.** Use S = tr(D_out)·tr(H_in) (and err₂) as the prior mean in the planned partial pooling for new fractional levels, sub-2 levels and two-sided variants.
- **Src:** 2607.07964 (KronQ: picked down_proj first, 7.21 vs 7.40 PPL), 2610.34886 (ThinkBit).
- **Fit:** knapsack.
- **Test:** cross-validate on V2 data, holding out whole families.
- **Gain/risk:** 0 to −1.5%. Risk: low.
- **Conflicts:** B11 partial pooling (complementary).

**H3. Depth and compounding prior from residual gradient norms.** Use the per-layer E‖g_res‖² (A6) as a covariate for kfit2's κ_l depth knots.
- **Src:** 2609.32060 (predictive amplification decomposes over sites), #21 (the sensitivity spike is at the end, L63).
- **Fit:** knapsack.
- **Test:** CPU refit, cross-validation.
- **Gain/risk:** 0 to −1%. Risk: low.
- **Conflicts:** #19 (edge offsets gave nothing); this is a measured covariate, not free parameters.

**H4. Measured interactions inside a layer.** On ~8 layers, measure joint demotions (gate_up and down; qkv and o_proj; GDN in and ssm_out) against the sum of single demotions. If the interaction exceeds 2 SE, add a pair term to the layer-wise DP; the DP is already per layer, so it absorbs it exactly.
- **Src:** 2606.13054 (TWLA chain-DP pair costs: PPL 15.03 → 12.83), 2610.16368 (interaction-aware 12.47 vs local 13.05), 2609.04098 (GDN projection errors add in quadrature, i.e. no interaction there).
- **Fit:** knapsack.
- **Test:** ~24 group-swap evals on a search slice.
- **Gain/risk:** 0 to −1.5%. Risk: noise (#14), so use group swaps.
- **Conflicts:** #11 (that was a global quadratic, not intra-layer pairs).

**H5. Cap the number of changes per accepted local-search candidate, and re-price after.** At most ~8 tensor-groups changed vs the incumbent; re-measure after acceptance.
- **Src:** 2609.09867 (composition error 26–42× the pricing error; 134 additive changes raised NLL; a cap of 2 avoided the harm).
- **Fit:** stream.
- **Test:** compare on V2's logged stream which accepted moves later reversed, as a function of move size.
- **Gain/risk:** 0 to −1% (fewer wasted evals). Risk: slower moves.
- **Conflicts:** #70.

**H6. Offer one-sided and two-sided variants as separate knapsack options.** At the levels the 2.5 bpw mixes use (≈K1.75–K3.5), quantize both variants and let measured swaps choose per family or tensor.
- **Src:** new; the gate/up full factor hurt at K2 while others helped.
- **Fit:** per-level (≈+40% ladder time on those levels).
- **Test:** only if C6/C8 show family-dependent signs.
- **Gain/risk:** 0 to −1% over all-two-sided. Risk: library cost.
- **Conflicts:** none.

**H7. Category-specific proxy check.** With A3, compute per-category err₂ for the chosen 2.5 bpw mix and look for tensors whose rank differs strongly by category (e.g. multilingual).
- **Src:** 2606.17118, 2607.00908 (mixed calibration beats single-domain).
- **Fit:** diagnostic.
- **Test:** CPU.
- **Gain/risk:** guard. Risk: none.
- **Conflicts:** #32.

**H8. Tempering knob on sensitivities.** Shrink fitted per-tensor sensitivities toward the family mean, s → s̄^(1−β)·s^β, with β tuned on the search pool by re-solving only.
- **Src:** 2610.20353 (NatScale: β ≈ 0.2 best; removing the error power raised KLD 0.194 → 0.292).
- **Fit:** knapsack.
- **Test:** re-solve at 3 values of β, measure 3 mixes per value at 2.5 bpw.
- **Gain/risk:** 0 to −1%. Risk: overlaps planned partial pooling.
- **Conflicts:** partial pooling (B11 in the tracker).

**H9. Lazy-heap measured descent as an exploration kind in the stream.** From a mix ~0.1 bpw above target, demote the tensor-group whose one-level demotion raises *measured* KL least. Re-evaluate lazily with stale-marginal refresh (~2.5 evals per move).
- **Src:** 2608.30564 (Q-Strata: JSD 749 vs one-shot ILP 823; the lazy schedule needed ~2.5 evals per move instead of re-pricing every candidate).
- **Fit:** stream.
- **Test:** run as 10% of the stream's kinds at 2.5 bpw; judge on the kind-specific win rate.
- **Gain/risk:** 0 to −1%. Risk: eval budget.
- **Conflicts:** #70.

**H10. Validity map of the proxy by family.** Per family and K, correlate the one-sided proxy, err₂ and measured single-group swap KL (V2 data plus new swaps).
- **Src:** 2609.26173 (component type predicts sensitivity better than reconstruction error).
- **Fit:** diagnostic.
- **Test:** CPU.
- **Gain/risk:** decides where H1 matters. Risk: none.
- **Conflicts:** none.

**H11. Ours vs turboderp under the two-sided proxy.** Compute err₂ for his K5–K8 tensors against ours. If his win under err₂, the high-K misprice is explained, and our high-K levels should go two-sided (C6).
- **Src:** design §1d.
- **Fit:** diagnostic.
- **Test:** CPU.
- **Gain/risk:** settles §1d. Risk: none.
- **Conflicts:** design §1d (the ours-vs-his swap test).

---

## I. Search, stream and eval statistics

**I1. Group-sequential early stopping per candidate.** Score documents in a fixed random order. At 25/50/75% compare to the incumbent with a paired z and O'Brien–Fleming bounds set in advance; abort clear losers.
- **Src:** clinical-trial statistics (new here).
- **Why:** Most candidates lose, so roughly 30–40% of eval time could be saved.
- **Fit:** stream (runner change only).
- **Test:** replay V2's per-document scores offline. Pass if the abort decisions agree with full evals ≥99% at ≥25% time saved.
- **Gain/risk:** indirect: about 1.4× more evals per day. Risk: needs the runner to emit per-document rows.
- **Conflicts:** none.

**I2. Successive halving over candidate batches.** Screen N queued candidates on a quarter of the slice, keep the best half, extend, and reuse the scores.
- **Src:** 2610.05265 (4.2× fewer evals at similar quality).
- **Fit:** stream.
- **Test:** offline replay on V2 data.
- **Gain/risk:** indirect. Risk: misses late-surging candidates.
- **Conflicts:** none.

**I3. Choose search-slice documents by discriminative power.** From V2's ~1,450 evals, estimate each document's between-candidate variance relative to its noise. Build search slices from the most discriminative documents within each category stratum (search slices only; ver3/ver4 untouched).
- **Src:** new (Neyman allocation).
- **Fit:** stream.
- **Test:** compare SE per GPU-second vs a random 2k-capped slice.
- **Gain/risk:** indirect, perhaps ~20% less noise. Risk: selection bias toward document types; categories are fixed by stratification.
- **Conflicts:** capstudy (#55-adjacent).

**I4. Regression-adjusted paired estimator.** Regress per-document differences on covariates (base KL, length, category) and use the adjusted mean.
- **Src:** CUPED / control-variate methods (new here).
- **Fit:** stream.
- **Test:** offline on V2 pairs: SE reduction.
- **Gain/risk:** indirect, 5–15% SE. Risk: low.
- **Conflicts:** none.

**I5. Tail guard.** Log p95/p99 per-document KL for finalists and reject candidates whose mean gain comes with a ≥20% p99 regression.
- **Src:** 2610.20826 (mean KL is tail-dominated), 2610.32243.
- **Fit:** final selection.
- **Gain/risk:** guards robustness. Risk: none.
- **Conflicts:** none.

**I6. GDN state precision in our eval path.** Check whether radiance carries GDN state at chunk boundaries in fp32 or bf16. Measure the KL of a 5–6 bpw mix on a 250k trace with both.
- **Src:** 2609.38166 (bf16 per-step state dropped AIME 87.9 → 72.1), 2610.39418 (fp32 carry needed), 2609.38169.
- **Why:** At 5.0 bpw (0.0026) a state-precision floor could be a visible share of the score.
- **Fit:** eval and engine.
- **Test:** 2 evals.
- **Gain/risk:** 0% at 2.5 bpw, up to a few % at ≥5 bpw if state is bf16. Risk: none.
- **Conflicts:** #60-01 (2B bf16 vs fp32 inference, 2k docs).

---

## J. Final pass on 2–3 shipped mixes: scale and norm healing

**J1. End-to-end KL healing of scales only.** Train su, sv and RMSNorm gains (≈12M parameters) with the trellis codes frozen.
- A torch 27B with decoded bf16 weights, layer-split over both cards (the collector's TP2 streaming), with gradient checkpointing.
- Teacher: cached top-512 logits (radiance recorder on qad-pool windows).
- Parameterization s = s₀·e^r with an r² trust-region penalty; keep the best checkpoint by held-out KL on the qad-verify split.
- **Src:** 2610.34886 (ThinkBit: scales plus norms took KL from 0.641 to 0.282; scales did most of it; the refit scale barely correlates with the weight-MSE optimum), 2609.35465 (Tetra: +1.7–3.3 MMLU, 1–2 GPU-h), 2610.21808 (RoleQ), 2610.31931 (scale tuning beats code moves: 0.325 vs 0.371).
- **Fit:** final pass on shipped mixes (zero format change; scales are already stored).
- **Test:** 0.8B pilot first (J10). On the 27B, bf16 backward (~5 s per sequence) is acceptable for SGD even though it was unusable for second moments (#33-03); about 500–2,000 sequences ≈ 1–3 h per size.
- **Gain/risk:** −3 to −10% at 2.5 bpw. Risk: overfitting (needs held-out stopping); bf16 gradient noise in early layers.
- **Conflicts:** ROADMAP V5; this is the "scale-only healing" item made concrete.

**J2. Block-wise warm start with an RMSE loss.** Per block on one card, use quantized-prefix inputs and cached bf16 block outputs as targets, training only that block's su, sv and norm gains.
- **Src:** 2610.00983 (RMSE vs MSE: Qwen3.5-27B W3 PPL 8.42 → 8.26; it fixes early-block starvation), 2609.19213 (cached features: 4.2 GB vs 20.1 GB), 2609.33601 (JustQuant: local before global; full QAD collapsed for A4 without it).
- **Fit:** final pass.
- **Test:** 0.8B pilot; on the 27B ~10 min per block.
- **Gain/risk:** −1 to −4% alone; mainly makes J1 cheaper and more stable. Risk: block loss ≠ KL.
- **Conflicts:** none.

**J3. Heal RMSNorm gains only.** Train only the norm weights by KL (≈0.6M parameters, folded into norm weights).
- **Src:** 2610.30779 (ProQuant RoCo: 64.77 → 67.30).
- **Fit:** final pass (zero format change).
- **Test:** 0.8B (~30 min), then the 27B (≈1 h).
- **Gain/risk:** −1 to −3%. Risk: low.
- **Conflicts:** none.

**J4. Choose healing data by tail KL.** Rank qad-pool windows by the mean of their top-25% token KL under the shipped mix, and train on the top k plus a random share.
- **Src:** 2610.14934 (FidelityQ: Qwen3-8B 17.05 → 16.69 vs random selection; the worst decile of tokens carries 45% of KL).
- **Fit:** final pass.
- **Test:** 0.8B A/B.
- **Gain/risk:** 0 to −1% on top of J1. Risk: overfitting hard documents.
- **Conflicts:** none.

**J5. Stage the healing by depth.** Unfreeze scales in layers 48–63 first, then the full depth.
- **Src:** ROADMAP field report (~22% from the last 16 layers vs 6–13% for all).
- **Fit:** final pass.
- **Test:** within J1.
- **Gain/risk:** improves J1's ratio. Risk: low.
- **Conflicts:** ROADMAP V5.

**J6. Heal with int8 activations switched on.** For mixes shipped with A8, fake-quantize A8-g128 activations during healing so the scales absorb activation error too.
- **Src:** 2610.09202 (CanonQ: the A4/A8 gap shrinks when trained with the activation quantizer on).
- **Fit:** final pass.
- **Test:** compare KL under A8 after A8-aware vs plain healing on the 0.8B.
- **Gain/risk:** 0 to −1% of the A8 cost. Risk: low.
- **Conflicts:** #26.

**J7. Start healing from the two-sided-refit scales (C2).** This shrinks the distance J1 must travel.
- **Src:** new.
- **Fit:** final pass.
- **Gain/risk:** convergence speed. Risk: none.
- **Conflicts:** none.

**J8. Set the healing stopping rule in advance.** Choose the checkpoint on a qad-verify split, never on ver3/ver4; log the document ledger entries.
- **Src:** 2610.34886 (best-held-out-KL checkpointing); project data rules.
- **Fit:** process.
- **Gain/risk:** protects held-out integrity. Risk: none.
- **Conflicts:** none.

**J9. Discrete code moves after healing (low priority).** Moves of one trellis step guided by gradient plus Kronecker energy, with KL accept/reject.
- **Src:** 2610.31931 (KL 0.491 → 0.371, but scales alone reached 0.325), 2610.41822 (TVQ: STE-trained trellis indices, ≤0.2 PPL over QTIP at 8× cost).
- **Fit:** final pass.
- **Gain/risk:** 0 to −1%. Risk: expensive Viterbi per move.
- **Conflicts:** none.

**J10. 0.8B healing pilot before any 27B run.** Run J1/J2/J3 on the 0.8B with evalq.
- **Src:** proxy methodology.
- **Test:** 4070, a few hours. Pass if J1 gives ≥5% at K2.5-equivalent on g08-search1 with held-out stopping.
- **Gain/risk:** decides J. Risk: proxy-to-27B transfer.
- **Conflicts:** none.

---

## K. Int8 activations

**K1. LDLQ target that accounts for A8 activations.** For tensors that will run A8-g128, quantize against ‖Wx − ŴQ(x)‖²: metric E[Q(x)Q(x)ᵀ] plus the cross term, so the weights absorb the part of the activation error that lies in their span.
- **Src:** 2610.35955 (ResCom error decomposition), 2504.02692 (GPTAQ).
- **Fit:** per-level (a second variant per level; only where A8 is planned).
- **Test:** 0.8B with the E1b-style hook at A8-g128, K3.
- **Gain/risk:** recovers 0–30% of A8's ~+0.00005 cost; ≤0.3% total at 2.5 bpw. Risk: doubles library levels.
- **Conflicts:** #26/#27.

---

## L. GDN-specific

**L1. Decay half-life as a sensitivity covariate.** Compute per-layer mean gate retention (from A_log/dt_bias and the gate statistics of one forward). Test it as a kfit2 covariate for GDN families.
- **Src:** 2609.38169 (ρ ≈ 0.80 with accumulated error on Qwen3.8-27B heads), 2609.30950 (observability Gramian).
- **Fit:** knapsack.
- **Test:** CPU correlation with fitted GDN sensitivities (V2 data).
- **Gain/risk:** 0 to −0.5%. Risk: low.
- **Conflicts:** #19.

**L2. Measure healing gain separately for GDN and attention layers.** Unfreeze only one type at a time in J1.
- **Src:** 2606.18114 (post-hoc corrections failed on SSM recurrences).
- **Fit:** final pass.
- **Gain/risk:** informs J. Risk: none.
- **Conflicts:** none.

**L3. Separate rates for GDN q/k/v/z parts.** The exl3i spans already support parts of different *width* in the stacked [q|k|v|z]. Allow different *rates* per part so z (diagonal CV 3.96) and qkv get different K.
- **Src:** importer check (stacked parts must agree), #33-09 (z vs qkv structure).
- **Fit:** format change (importer plus the kernel's per-part plan).
- **Test:** a fake-quant 0.8B mix (z at K+½, rest at K−⅙) at equal bits.
- **Gain/risk:** 0 to −1%. Risk: plugin work; the time table changes.
- **Conflicts:** design §1d (K ties for fused GEMMs).

---

## M. Proxy methodology

**M1. Add C1's arms to the 27B swap-test pass rule now.** S-only vs S+D vs one-sided, at K3 and K2.5, so the 27B swap test separates the output-scale effect from the H_out information.
- **Src:** new.
- **Fit:** process.
- **Gain/risk:** clarity. Risk: none.
- **Conflicts:** #33-13.

**M2. 2B check of how the diagonal gain scales with width.** If the backward fits the 4070 (halves.py style), collect 1K diagonals on the 2B (width 2048) and measure the K3 gain.
- **Src:** ROADMAP proxy plan.
- **Fit:** process.
- **Test:** maybe 2–3 h; skip if it doesn't fit.
- **Gain/risk:** de-risks transfer. Risk: memory.
- **Conflicts:** #60-04 (paused calibration).

**M3. Record a proxy→27B transfer ratio per idea family.** For each idea confirmed on the 27B, keep (0.8B gain, 27B gain) in a table, to calibrate future screens.
- **Src:** 2609.00049 (REAL-Q's gains shrink with size: −49% at 1.7B vs −5…−9% at 8–70B).
- **Fit:** process.
- **Gain/risk:** better triage. Risk: none.
- **Conflicts:** none.

---

## Top 20 by expected gain per unit cost

| # | Idea | Cost | Expected KL @2.5 | Why it ranks here |
|---|---|---|---|---|
| 1 | **C1** sv-aware two-sided LDLQ (H_out = I) + census | minutes (0.8B), 3 h (27B) | 0 to −5% | No backward; splits the −8% into S and D parts; can run on the 27B today |
| 2 | **A1+A2+A3+A5** extra accumulators in the 27B run | ~0 GPU-h (code + gate run) | enabler | 8.5 h of both cards are spent anyway |
| 3 | **C2** two-sided scale refit | 1 h dev, 10 min test | 0 to −1.5% | Fixes a refit that works against the two-sided result |
| 4 | **B2+B3** tempering and damping of the diagonal | 1 h (offline + arms) | 0 to −3% | Only p = 1, σ = 0.01 tested |
| 5 | **C6** H_out gain across K (incl. K4–K6) | 40 min | scope decision | Decides the two-sided requant of the whole ladder |
| 6 | **B4** diagonal + low rank, offline | 30 min | 0 to −2% | Could keep the full factor's K3 extra without K2 noise; decides A7 |
| 7 | **B8/A8** loss masked to scored rows | ~1 h (0.8B) | 0 to −3% | The collector's loss does not match the score |
| 8 | **C4** H-aware global scale search | 0.5 day | 0 to −2% | g_scale is chosen by raw tile MSE |
| 9 | **H1** kfit3 on the two-sided proxy | 1 CPU day after the 27B diagonal | −1 to −4% | Improves pricing of every option, incl. turboderp's |
| 10 | **G1** head statistics on scored rows | half a day of headsim | −0.1 to −0.5% | Free correctness fix |
| 11 | **C5** block act-order in LDLQ | 10 min test | 0 to −1% | No inference cost |
| 12 | **G3+G4** shift search beyond the mean row | ~6 h on card 0 | 0 to −0.4% | Mean-row already gave −30% head KL; cheap to extend |
| 13 | **F7** KL-optimal scale multiplier | 20 min | 0 to −2% | Possible systematic LS shrink bias |
| 14 | **F9** SwiGLU-aware per-level refit | 0.5 day | 0 to −2% | Library-compatible, zero format |
| 15 | **J3→J1** norm-only, then scale-only KL healing | 0.8B pilot + 3–6 h per size | −3 to −10% | Largest single gain; final pass only |
| 16 | **F1(+F2)** tile CD sweeps after LDLQ | 1–2 days dev, 2–3× quant time | −1 to −4% | MPQ's large gain at 2 bit |
| 17 | **D1/D2** MLP channel permutations | CPU proxy + 0.8B | 0 to −2% | Exact invariance, absorbed by the weights |
| 18 | **I1** group-sequential early stopping | 1 day runner work | ~1.4× evals per day | Compounds through the stream |
| 19 | **G2** head with analytic Fisher H_out | headsim runs | −0.3 to −1% | Analytic factor already exists |
| 20 | **D5→D6** row-slab ceiling, then split MLP | CPU, then plugin work | −1 to −5% if the ceiling is ≥0.1 bit | High ceiling possible; costs plugin work |

Just outside: F3 activation denoising (−1 to −4%, but a 27B noisy-forward collection), A9/B9 Gaussian probe (cost only), G6 embedding column metric (needs A5).

## The 5 cheapest diagnostics that would most change this ranking

1. **sv dispersion inside 128-row blocks plus the 0.8B "H_out = I" arm** (CPU minutes on existing ladder svh; 3 arms on the 4070).
   - If S-only gives most of the diagonal gain, C1 moves to the top. The 27B diagonal run then matters mainly for the D part.
   - If S-only gives nothing, the diagonal's information is real and A1–A9 rise.
2. **Offline transforms of the existing 0.8B 8K data**: tempering p, damping, rank-r plus diagonal, shrinkage, scored by the two-sided proxy and then KL (~1 h). This decides B2–B5 and whether A7's Nyström sketch goes into the 27B run.
3. **KL-vs-Fisher token decomposition with the gen-mask share** (E2, ~1 h on the 0.8B).
   - If low-Fisher, confident tokens carry most of the KL, gradient-weighted H_in is wrong by construction (close E3–E4), and winsorizing or masking (B6, A8) becomes likely to help.
   - If not, gen masking rises.
4. **H_out gain at K1.5, K4, K5 and K6 on the 0.8B** (C6, ~40 min). This decides whether the whole ladder is requantized two-sided, and whether H11 explains turboderp's high-K wins.
5. **Row-slab and permutation ceiling from the 0.8B diagonal and the 27B Hessians** (D5 plus D1's CPU proxy, an afternoon).
   - Gate/up ceiling ≥0.1 bit: the split-MLP format (D6) jumps into the top 10.
   - Below 0.03 bit: drop D6 and D7, keep only the free permutations.
