# Lens **open**: research ideas for V3 and after (2026-11-07)

I made 84 proposals plus 8 diagnostics, grouped by theme; the top 20 and the 5 cheapest diagnostics are at the end. The most useful new finding is the massive channel at residual dim 3994 (§0). The most promising cheap ideas are closed-form gain/bias corrections on frozen codes (§A), built on a mechanism in §1 that says where they can help and roughly how much.

## 0. Measured this run

I ran four checks on colo (CPU only, about 1.5 core-hours). Scripts and outputs are in `/data/work/ideas-open/` (`convtaps.py`, `massive.py`, `superw.py`, `hspec.py`, `hspec2.py`; `out.*.json`, `o2.*.json`). Nothing was written in the repo.

**1. Residual dim 3994 is a massive channel, at every depth.**
- **Writers:** row 3994 is the largest row of every writer checked (L0–7, 30, 62, 63; down / out_proj / o_proj). It is 2–21× the median row norm: L0 out_proj 20.7×, L3 o_proj 17.8×, L0 down 11.5×, L63 o_proj 3.3×.
- **Norm gains:**
  - `post_attention_layernorm` gain (1+w) at 3994 is **0.0039** in every layer checked (L0–10).
  - `input_layernorm` reads it at 0.30–0.44 (L2–10) and 0.96 (L62). So the attention/GDN readers see it, and the MLP readers are told to ignore it.
- **Reader Hessians:** 3994 is the top diagonal of every reader input Hessian from L3 to L62, and the top eigenvector (cos 0.98–0.99) holds 64–94% of the trace in layers 3–30.
  - L3: 133,852× the median diagonal. L7: 126,337×. L11: 84,894×. L10: 17,756×. L30: 13,025×. L62: 1,466×.
  - MLP input at L30: 3,816×, despite the 0.0039 gain.
- **Secondary dims:** 310, 3842, 253, 1089.
- **No "super weights":** the largest element of row 3994 holds ≤3% of the row's energy. Any protection has to cover whole rows.

**2. Bypassing the massive input channel barely helps weight rounding.**
- Setup: LDLQ pivot sum after a random-sign 128-point Hadamard, with the damping held at its absolute value. Removing dim 3994 from the reader Hessian lowers the pivot sum by:
  - 1.4% at L3 and L7 (attention);
  - 0.9% at L11;
  - 0.4% at L10 and L30 (GDN);
  - 0.2% at L62 and on the L30 MLP input.
- An earlier run showed −47% at L3, but it re-derived the damping from the reduced matrix, so that number was the damping.
- At σ = 0.005 × mean diagonal, the damping is **14% of the median channel's curvature at L3/L7 but 0.9% at L62**, because the massive channel sets the mean.

**3. Low-rank side paths don't pay on the trunk.**
- Removing the top 16 / 64 eigendirections cuts the pivot sum by ≤5% / ≤11% on MLP and GDN inputs. That is an upper bound, inflated by the damping effect above.
- On gate (17,408 rows), a rank-64 fp16 path costs ~0.2 bpw and buys ~0.08 bit of error.
- This rules out GPTQ-intrinsic LoRA columns (2606.01412) and EoRA-style side paths for the trunk (shortlist #10).

**4. The GDN conv is far from neutral per channel.**
- Per channel, the 4-tap conv's (DC gain)² / tap energy is: p10 0.18–0.41, median 0.74–1.18, p90 1.18–2.9 per layer.
- Rows of in_proj_qkv therefore differ about 10× in how much low-frequency input error survives the conv.
- The shortlist's drop gate for conv-aware statistics (#9: "drop if ≥90% of channels lie in [0.8, 1.25]") is not met, so that idea stays alive.

**Caveats:** the pivot sum is a proxy in one fixed column order, not exllamav3's quantizer. The Hessian checks covered 15 tensors.

## 1. Why the coherent fixes grow, and where they can't help

**LS refit leaves every channel short by exactly its proxy error.** exllamav3's `refit_scales` does a least-squares fit per output channel in the (damped) Hessian metric: c = qᵀHw / qᵀHq. A least-squares fit leaves the residual orthogonal to the fit, so after an output-side step:

  ‖ŵ_n‖²_H = ‖w_n‖²_H − ‖e_n‖²_H

This holds exactly after that step, and approximately after the alternation with the input-side step. Every channel is therefore short by exactly its own proxy error, which the quantizer already computes.

With the trellis's Gaussian MSE (≈0.073 at 2 bits per LiftQuant/ApexQuant, ≈×¼ per extra bit), the amplitude deficit 1−√(1−D) is:

| K | D (Gaussian trellis) | amplitude deficit |
|---|---|---|
| 1.5 | 0.146 | 7.6% |
| 2 | 0.073 | 3.7% |
| 2.5 | 0.036 | 1.8% |
| 3 | 0.018 | 0.9% |
| 4 | 0.0046 | 0.23% |

This is consistent with:
- swj's fitted c_g of 1.065–1.080 at K1.5 on the 0.8B;
- swj at 27B: −6.5% at K2.5 but only −1% at K4. Small per-layer deficits that compound over 64 layers fit the "coherent errors grow with depth" hypothesis.

**A second shrink at every RMSNorm.** A norm whose input carries quantization noise with energy share ε divides by the noisy RMS. The signal part of its output shrinks by (1+ε)^(−½). That is coherent and grows with depth.
- This predicts that a writer behind a norm of a noisy signal wants gains beyond norm restore.
- That is what #112 found: gdn_out α = 3 at the grid edge (≈4–4.75 per the brief), behind GDN's gated RMSNorm.

**Where coherent per-head gain errors are invisible (gauge):**
- full-attention q/k (per-head q_norm/k_norm);
- GDN q/k (per-head L2 norm);
- GDN v: its state is linear in v, and the per-head gated RMSNorm on o removes any uniform gain.

Gains matter for attention v, GDN z (SiLU gate), gate/up (SiLU), the writers, the embedding, the head and the norms. swj and #112 cover gate/up and the writers. The rest of this coherent surface is mostly untested.

Expected gains below are KL % at 2.5 bpw:
- noise-reshaping ideas: ⅓ of their 0.8B effect (the brief's default);
- coherent fixes: 1–2× their 0.8B effect.

Tests fit on the qad pool, decide on search, and confirm on verify.

---

## A. Coherent errors on frozen codes (closed form, zero or near-zero runtime)

**A1. Per-channel gain from the LS identity.**
- **What:** apply c_n ← c_n·(1−ε_n)^(−α_f/2), with ε_n = ‖e_n‖²_H/‖w_n‖²_H taken from the quantizer's own per-channel error. Apply it to every family whose gain is visible (§1); α_f is predicted from A2's inflation term instead of searched.
- **Source:** exllamav3 `refit_scales` (fits on damped H); #85 C9; #112; swj fit logs. New: per channel rather than per family, needs no data, and the gauge list says where it cannot help.
- **Why:** the deficit is exact (§1). The ~1.8% amplitude deficit at K2.5 compounds over 64 layers.
- **Fit:** library (per level). **Runtime:** none (svh).
- **Test:** 0.8B fixed codes, 4070 ~1 h. Per-channel α ∈ {1, 2, 3} vs #112's family α, applied to writers, attention v and GDN z. Three seeds at K2/K3; pass ≤0.99 vs the #112 stack. Then a 27B fixed-code pair (~1 card-h).
- **Gain / risk:** −0.3…−1.5% beyond swj + #112, P 0.3. Risk: overlap with #112.
- **Conflicts:** #85, #112 (running at 27B), #108 (GDN/attention refits vanish in the full stack). The gauge list predicts #108's attention nulls, so this targets only visible gains.

**A2. Noise-aware RMSNorm gains (RMS-inflation correction).**
- **What:** multiply each norm gain by γ_ℓ = √(1+ε_ℓ). Here ε_ℓ is the noise energy share of that norm's input, measured as RMS(h_q)² vs ⟨h_q, h_bf16⟩ alignment.
- **Scope:** all input / post-attention / final norms (129), the 48 GDN gated norms, and the q/k norms (whose per-head gain acts as a temperature).
- **Source:** new. The length/angle split in 2609.11716 bounds it: length was 3.7% of the error at NVFP4 4-bit.
- **Why:** explains #112's large gdn_out α. If it works it replaces a grid-edge fit with a measured quantity, and it reaches the attention and GDN paths that #108's refits could not.
- **Fit:** per mix (final pass), or library via A3. **Runtime:** none (norm weights).
- **Test:** existing 0.8B K2/K3 quants, fixed codes, 4070 ~1.5 h. One forward with hooks over 64 qad docs gives ε_ℓ; set γ_ℓ and eval the search golden, 3 seeds. Pass ≤0.985 at K2. Side check: does γ on the GDN norms reproduce #112's gdn_out gain?
- **Gain / risk:** −0.5…−2%, P 0.3, coherent class so it may grow at 27B. Risk: the angular error dominates and this term is small.
- **Conflicts:** none in INDEX. In #89-02 the norm gains were ≤0.2% of the healing gain, but there the healing also trained scales.

**A3. Library form of A2: a γ table predicted from the mix.**
- **What:** kfit2 already predicts the cumulative upstream error P_<ℓ for any mix. Fit γ_ℓ = f(P_<ℓ) once from 3–4 calibration mixes per size, and apply it at pack time with no per-mix forward.
- **Source:** new. **Fit:** packer (library-like). **Runtime:** none.
- **Test:** after A2 passes, predict γ for a fifth mix; pass if within 20% of measured.
- **Gain / risk:** keeps ~80% of A2's gain, P 0.6 conditional on A2.
- **Conflicts:** none.

**A4. Logit temperature per mix.**
- **What:** one scalar τ on the final norm gain, chosen by a 1-D search of mean KL over the mix's recorded top-512 logits on qad positions. It captures the net of LS shrink, final-norm inflation and noise-induced over- or under-confidence. The sign is empirical.
- **Source:** IQR-V4's head temperature ("0–2% or none"), here per mix on the whole model.
- **Fit:** final pass (or one constant per size). **Runtime:** none.
- **Test:** one record eval of an existing 27B 2.5 mix (~1 card-h) plus a CPU fit. Pass: a fitted τ ≠ 1 cuts search-slice KL ≥0.5% at z ≤ −3, same sign on verify.
- **Gain / risk:** −0.2…−1%, P 0.4; negligible risk.
- **Conflicts:** ranked E3 (head healing incl. τ, never run).

**A5. SwiGLU Jensen-gap bias, written as a down_proj output bias.**
- **What:** random gate/up noise through the curved silu(g)·u produces a coherent per-channel offset δ̄ = E[silu(ĝ)û] − E[silu(g)u] (≈ ½E[silu″(g)u]σ_g² + E[silu′(g)n_g n_u]). down_proj writes it into the residual stream as a fixed vector W_d·δ̄.
- **Library-safe:** it depends on the noise *statistics* of the gate/up level, not on its exact rounding. A bias per (layer, K_gu) should therefore transfer across seeds and across down's K.
- **Source:** bias correction (DFQ, Nagel 2019). Shortlist #6 (writers' mean shift) uses bf16 inputs and misses this term.
- **Fit:** library per (layer, K_gu). **Format:** bias field (G1). **Runtime:** ~0 (+0.0003 bpw).
- **Test:** M5 first. Then 0.8B K2/K3: bias fitted on seed 0, applied to seeds 1–2 (transfer is the point); 4070 ~2 h. Pass ≤0.985 at K2, and the transferred bias keeps ≥70%.
- **Gain / risk:** −0.3…−1.5%, P 0.25. Risk: swj's c_g shift may already absorb it.
- **Conflicts:** shortlist #6; #88-02 (all-tensor asym 1.53× worse; that used exact quantized upstream, this uses statistics).

**A6. Jensen-gap through GDN's conv→SiLU path and z gate.**
- **What:** the same offset for in_proj_qkv rows (conv then SiLU) and z (SiLU gate), corrected with a bias on out_proj's output per (layer, K_qkv, K_z).
- **Source:** extends A5. **Fit:** library. **Format:** G1 bias field. **Runtime:** ~0.
- **Test:** 0.8B, after A5, ~1 h.
- **Gain / risk:** −0…−0.5%, P 0.15. GDN's gated norm renormalizes o per head, which shrinks the effect.
- **Conflicts:** #108.

**A7. Reader bias from the massive input mean.**
- **What:** the normalized residual has a large nonzero mean on dim 3994 (and 310, 3842), concentrated at sink positions. ΔW[:,3994]·E[x_3994] is then a coherent per-output offset. Correct it with b = ΔW·μ on readers, with μ collected once on bf16 (G3).
- **Source:** DFQ bias correction; §0.1.
- **Fit:** library. **Format:** bias field. **Runtime:** ~0.
- **Test:** CPU, colo: share ‖ΔWμ‖² / tr(ΔW H ΔWᵀ) on decoded 27B levels. Go to the 0.8B only if ≥5%.
- **Gain / risk:** −0…−0.8%, P 0.15.
- **Conflicts:** ranked H11; shortlist #6.

**A8. Per-mix sequential bias "nudging".**
- **What:** run one staged forward of a shipped mix with bf16 in lockstep. At each writer, sequentially, set the bias to the mean of (bf16 increment − quantized increment) over qad tokens. This is data assimilation applied to the residual stream.
- **Source:** "nudging" in weather data assimilation; ranked H3 (scales, not biases).
- **Fit:** per mix (final pass). **Format:** bias field. **Runtime:** ~0 (64×5120 f16).
- **Test:** 0.8B K2/K3, 3 seeds, forward-only, minutes. Arms with and without sink positions in the fit. Pass ≤0.98 at K2. At 27B ~2 card-h per mix, staged on one card.
- **Gain / risk:** −0.3…−1.5%, P 0.25. Risk: overfitting position-specific (sink) offsets.
- **Conflicts:** #89 (healing trains scales); ranked H3.

**A9. Depth-scheduled gain.**
- **What:** fit α(ℓ) with 3 knots (early / mid / late) instead of one α per family. kfit2's amplification (κ≈38 at layer 0) says early coherent errors cost more.
- **Fit:** library. **Runtime:** none.
- **Test:** 0.8B fixed codes, ~1 h; pass ≤0.995 vs the flat α.
- **Gain / risk:** −0.2…−0.8%, P 0.2.
- **Conflicts:** #112.

**A10. Inflated trellis target.**
- **What:** quantize W·(1+δ_K) with δ_K from A1's law, so the codes themselves are chosen for the norm-restored target, then refit. This may beat a post-hoc svh gain at K ≤ 2.
- **Fit:** library (requant). **Runtime:** none.
- **Test:** 0.8B K2 requant, 3 seeds, vs A1 at equal δ; 4070 ~2 h. Pass ≤0.99 vs A1.
- **Gain / risk:** −0…−0.5% beyond A1, P 0.15.
- **Conflicts:** #39 judged the gain flat by proxy MSE, which by construction cannot see a coherent benefit. Judge by KL.

**A11. Refit scales in the undamped metric.**
- **What:** `refit_scales` fits on the damped H. At L3/L7 attention inputs the damping is 14% of the median channel curvature (§0.2), so the fit trades output error for weight-space error. Refit on H_err, which the function already builds for its error report.
- **Fit:** library. **Runtime:** none.
- **Test:** refit only, on the existing 27B K2.5 attention/GDN inputs (GPU minutes or CPU); undamped proxy, then 0.8B KL. ~1 h.
- **Gain / risk:** −0…−0.5%, P 0.25.
- **Conflicts:** ranked B1 (sub-item, unrun); #38.

**A12. swj with a pre-activation offset (pre-distortion).**
- **What:** extend swj from (c_g, c_u) to (c_g, b_g, c_u). A per-channel offset in the gate pre-activation cancels the Jensen offset (A5) inside the nonlinearity, as RF amplifier pre-distortion does. Same grid machinery plus a 1-D offset grid.
- **Fit:** library. **Format:** gate bias (G1; added in the epilogue before SiLU). **Runtime:** ~0.
- **Test:** 0.8B fixed codes on top of swj, 3 seeds, ~1.5 h; pass ≤0.99 vs swj.
- **Gain / risk:** −0.2…−1%, P 0.2.
- **Conflicts:** extends #104.

**A13. Position-dependent temperature.**
- **What:** KL rises with position for every quant alike (#52). A τ(t) = 1 + a·log t schedule on the final norm targets any coherent drift with position.
- **Fit:** per size. **Runtime:** ~0 (a position-aware scalar; engine change).
- **Test:** fit a on recorded long-trace logits (CPU).
- **Gain / risk:** −0…−0.3%, P 0.1.
- **Conflicts:** #52.

## B. The massive channel (dim 3994)

**B1. Exact row 3994 in every writer.**
- **What:** store row 3994 of all 128 writers in fp16 outside the trellis: ≈3 MB, **≈0.0009 bpw**, optionally plus rows 310/3842/253/1089.
- **Runtime:** none in the two-pass prefill, since the decode pass simply writes those rows into the f16 or int8 plane. A GEMV row patch at decode.
- **Source:** §0.1. Different from #96 and #29-02: one known row, nearly free.
- **Why:** with LDLQ's equal relative error per row, the value written to the massive dim is off by roughly √ε_row. At sinks, the RMS-dominated normalization turns that into a global coherent gain on every other dim.
- **Fit:** format (row patch, G1). **Runtime:** ~0.
- **Test:** M6 first (oracle patch on the 0.8B). Pass ≤0.985 at K2, then a 27B decoded-plane patch.
- **Gain / risk:** −0…−2%, P 0.2. Risk: at sinks the normalized direction is dominated by 3994 itself, which would make the value error second order.
- **Conflicts:** #96 (row-slab K, negative in general).

**B2. Gain-only fix on row 3994 (free in today's format).**
- **What:** fit svh[3994] of each writer by LS on the output at high-|h_3994| tokens (bf16 inputs). This fixes only the coherent part of B1.
- **Fit:** library. **Runtime:** none.
- **Test:** as an arm of M6.
- **Gain / risk:** −0…−0.5%, P 0.2.
- **Conflicts:** none.

**B3. Massive-channel bypass for A8 (fast variant only).**
- **What:** on residual readers, zero x_3994 (and 310, 3842) before the Hadamard and int8 quantization, and add W[:,3994]·x_3994 through a 1–3-column f16 side product in the epilogue. This takes the outlier out of its 128-block's A8 scale, which was the #47 failure mode. The weight side gains ≤1.5% (§0.2), so this is for activation error only.
- **Fit:** format/kernel (fast variant). **Runtime:** prefill, +1–3 columns per reader GEMM (≪0.1% of FLOPs).
- **Test:** aqproxy on the E1c dumps for readers L3–L30: A8-g128 rel_out with and without the bypass, CPU ~1 h. Pass ≥30% lower A8 error on early readers, then reprice in knap4.
- **Gain / risk:** a fraction of the fast variant's A8 penalty. It could let knap4 use A8 on L3–L11 readers it now keeps f16. P 0.3.
- **Conflicts:** #47 (R1 spreads the outlier; this removes it); ranked D8.

**B4. Median-referenced damping.**
- **What:** use λ = σ·median(diag) instead of σ·mean. This removes the ~15× layer-to-layer swing in relative damping that dim 3994 causes (§0.2).
- **Fit:** library (requant). **Runtime:** none.
- **Test:** settle64 (order-fixed) on 10 reader tensors × 3 σ values, ~4 CPU-h. Pass ≥1% proxy at each scheme's best σ.
- **Gain / risk:** −0…−0.5%, P 0.15.
- **Conflicts:** #38 (the per-tensor choice of σ already spans 25×).

**B5. Secondary massive dims as priced options.**
- **What:** treat rows and columns for 310, 3842, 253 and 1089 as byte-priced patch options in knap4 (G4) rather than a fixed set.
- **Fit:** format + knapsack.
- **Test:** after M6, rank dims by their oracle-patch effect.
- **Gain / risk:** small, P 0.15.

**B6. Precomputed sink prefix (prompt-cache trick).**
- **What:** documents start with the same template token(s). Their bf16 KV (16 attention layers, ~64 KB per token) and GDN state are constants that a container could ship, making attention to the sink exact.
- **Source:** prompt caching (serving).
- **Fit:** engine/container. **Runtime:** none (a cache).
- **Test:** M8 first (do sink-adjacent positions carry outsized KL?). **Check the challenge rules first**: this may count as gaming.
- **Gain / risk:** unknown, P 0.1.
- **Conflicts:** none.

## C. Library vs mix (writers' asym, sibling dependence)

**C1. Writer variants indexed by sibling level.**
- **What:** asym writers fail across sibling K (#111: ×1.08–1.24) because the correction is fitted to the exact sibling rounding. Library siblings are fixed artifacts, so an asym writer fitted to the library sibling at K_s is exact whenever a mix uses that K_s.
- **How:** build writer variants for |K_w − K_s| ≤ ½ (3–5 per writer level), with gate/up tied as one sibling for down. knap4's layer DP enforces the (sibling, writer variant) consistency.
- **Fit:** library (per level × sibling level). **Runtime:** none.
- **Cost:** writer quant ×3–5 on 31% of trunk weights. Sibling moments per (layer, K_s) on bf16 inputs (sib27, ~5–7 card-h per level at 1M tokens). Donor-computable.
- **Test:** 0.8B, #111's design: writers fitted on K2.5 vs K3 siblings, shipped in matching and mismatching mixes, 3 seeds, 4070 ~3 h. Pass: matched variants ≤0.90 in mixes (keeps #111's per-mix gain).
- **Gain / risk:** if #105 passes, the per-mix asym gain survives in shipped mixes. Guess −3…−6% at 27B 2.5, P 0.35 conditional.
- **Conflicts:** #87-07, #111 (the failure this fixes), #105 (prerequisite).

**C2. Asym on ensemble sibling moments.**
- **What:** sum sibling moments over several sibling quantizations (seeds and K_s ∈ {K−½, K, K+½}). The fitted F then targets the transferable part, while the seed-specific part (F−I cosine 0.02–0.10 across seeds) averages out.
- **Fit:** library. **Runtime:** none.
- **Test:** 0.8B, F from 3 sibling variants applied to a 4th. Pass: never >1.0, and ≥40% of the matched gain.
- **Gain / risk:** −1…−3% at 27B if asym passes, P 0.25.
- **Conflicts:** #87-07 (seed transfer 1.049 at K3; that used one other seed and no averaging).

**C3. Asym-lite: plain writer codes + scale refit to the ensemble target.**
- **What:** no requant. Refit svh/suh of plain writers against C2's ensemble asym target.
- **Fit:** library. **Runtime:** none.
- **Test:** 0.8B fixed codes, ~1 h.
- **Gain / risk:** −0.5…−2%, P 0.25.
- **Conflicts:** review-11-05 idea 4 (per mix; this is per level).

**C4. Domain-randomized per-level fits.**
- **What:** any per-level fit that will face quantized upstream in a mix (writer α, A5/A6 biases, C2/C3) is fitted with upstream K sampled from the knapsack's empirical distribution at the target size (the w58/w63 repricing outputs), not bf16 and not one mix.
- **Source:** robotics domain randomization.
- **Fit:** library. **Runtime:** none.
- **Test:** 0.8B; compare bf16-upstream, randomized and per-mix fits on 3 mixes. Pass: randomized ≥80% of per-mix.
- **Gain / risk:** per-mix quality at library cost for every fit it touches, P 0.3.
- **Conflicts:** swj (bf16 inputs by design; keep that if C4 shows no difference).

**C5. Writer variant fitted to the modal sibling K.**
- **What:** the cheapest form of C1: one asym variant per writer level, fitted to the sibling level that w58/w63 most often pairs with it.
- **Fit:** library. **Runtime:** none.
- **Test:** an arm of C1's test.
- **Gain / risk:** ~½ of C1, P 0.3.

**C6. Pair penalty for writer–sibling K gaps in knap4.**
- **What:** if per-level asym ships anyway, price #111's measured mismatch costs (×1.08–1.24) as pair terms, so the knapsack mismatches only where it pays.
- **Fit:** knapsack. **Test:** CPU.
- **Gain / risk:** protects 1–3%, P 0.5 conditional.
- **Conflicts:** #111's open decision (user).

## D. Quantizer internals and statistics (noise-shaping class, expect ⅓)

**D1. Conv-aware metric for GDN in_proj_qkv.**
- **What:** (a) row weights w_r = Σ_kl c_k c_l ρ(k−l), using the measured lag-0–3 autocorrelation of the normalized input, for knap4 per-head pricing and a two-sided diagonal; (b) lag-covariance Hessians for rows clustered into ~4 groups by tap profile.
- **Why:** taps vary ~10× per channel (§0.4).
- **Fit:** library (qkv requant). **Runtime:** none.
- **Test:** 0.8B lag hook (4070, 30 min) plus qkv requant, 3 seeds at K2/K3 (~2 h). Pass ≤0.985 at K2. 27B lag statistics by a mkhin27-style forward (1–2 card-h; G3).
- **Gain / risk:** −0.2…−0.8%, P 0.25.
- **Conflicts:** shortlist #9 (its drop gate fails, so the idea stays).

**D2. Analytic output weights for q/k rows from the normalization Jacobian.**
- **What:** GDN q/k pass a per-head L2 norm; attention q/k pass per-head q_norm/k_norm. Per-head uniform gains are free, and row errors weigh ≈ g²/E[‖k_pre‖²]. Use these as diagonal output weights in `ldlq_2hess` for q/k rows only. Forward-only, so it respects "no new Σdy²".
- **Fit:** library (requant of q/k rows). **Runtime:** none.
- **Test:** 0.8B, 3 seeds; pass ≤0.985.
- **Gain / risk:** −0.1…−0.5%, P 0.2.
- **Conflicts:** #33 (H_out not adopted ≤K3.5; this is a narrower substitute); #96.

**D3. Token-weighted Hessians per row group for gate/up.**
- **What:** up row j's error reaches down multiplied by silu(g_j(x)), so its exact metric is H_j = E[silu(g_j)² x xᵀ]; gate row j weighs (silu′(g_j)u_j)². Cluster rows into 16–32 groups by gating pattern, then run LDLQ per group.
- **Fit:** library (requant). **Runtime:** none.
- **Test:** 0.8B, 16 groups, 3 seeds at K2/K3; pass ≤0.98. 27B statistics: one GPU forward with group accumulators, ~2–4 card-h.
- **Gain / risk:** −0.3…−1.5%, P 0.2 (mlp_in carried 44% of the 0.8B H_out gain, #82).
- **Conflicts:** #84 (A9 analytic diagonal failed its cosine gate; this is a per-row input metric, not a diagonal output factor); #33.

**D4. Gate→up code-level feedback within the MLP.**
- **What:** quantize gate first, then up with gate's quantized output known (weights silu(ĝ_j)², plus the cross-moment with silu(g_j)). This is asym for a sibling that shares the same bf16 input. When knap4 ties gate/up K (as the fused GEMM requires), it is library-safe.
- **Fit:** library. **Runtime:** none.
- **Test:** 0.8B on top of swj, 3 seeds; pass ≤0.98.
- **Gain / risk:** −0.3…−1%, P 0.2.
- **Conflicts:** #88-02 (quantized *upstream*; here it is a same-input sibling); swj (overlap).

**D5. BCJR soft-decision feedback in LDLQ.**
- **What:** propagate each tile's BCJR posterior mean (temperature T) through LDLQ feedback in early passes, then hard-decode and run tile CD.
- **Source:** MPQ (2610.32127), BCJR-QAT (2605.10655).
- **Fit:** library (quant ×3–5). **Runtime:** none.
- **Test:** settle64 (order-fixed), 16 tensors at K2/K2.5; pass ≥1.5% proxy, then 0.8B.
- **Gain / risk:** −0.2…−0.8%, P 0.15.
- **Conflicts:** #77 (GPU tile CD 0.5–1.2%); #113.

**D6. Cross-tile M-best lookahead.**
- **What:** keep the M best Viterbi paths per tile and pick jointly with the next tile's feedback cost.
- **Fit:** library (quant ×M). **Runtime:** none.
- **Test:** settle64.
- **Gain / risk:** ≤−0.5%, P 0.1.
- **Conflicts:** #113, #77.

**D7. Robust, sink-capped input Hessians.**
- **What:** recollect H_in with each token's contribution Huber-clipped, or with sink positions down-weighted.
- **Fit:** library. **Runtime:** none.
- **Test:** 0.8B, 3 seeds.
- **Gain / risk:** −0…−0.5%, P 0.1. §0.2 shows the massive direction is already cheap for LDLQ.
- **Conflicts:** #91, #101 (statistics changes dropped).

**D8. Entropy-weighted H_in.**
- **What:** weight tokens by the logit Fisher trace 1−Σp² (bounded, forward-only).
- **Fit:** library. **Runtime:** none.
- **Test:** 0.8B, 3 seeds.
- **Gain / risk:** 0…−0.5%, P 0.1.
- **Conflicts:** #86 (inconclusive), #33-07 (gradient weights: +31–44%).

**D9. Massive-block placement in LDLQ order.**
- **What:** dim 3994 sits in input block 31 (3994/128 = 31.2), whose rotated Hessian block dominates. Test placing it first vs last in exllamav3's last-to-first order.
- **Fit:** library. **Runtime:** none (or an input permutation).
- **Test:** CPU minutes: pivot sums under the two placements. Run only if Δ ≥1%.
- **Gain / risk:** small, P 0.1.
- **Conflicts:** #95-01, #103, #106.

**D10. Low-rank feedback columns / EoRA side paths (measured, negative).**
- **What:** GPTQ-intrinsic LoRA columns and EoRA-style rank-r side paths on the trunk.
- **Why not:** §0.3 — rank 64 costs ~0.2 bpw for ~0.08 bit. Recorded so it isn't re-proposed.
- **Conflicts:** shortlist #10; ranked E6.

## E. Head and embedding

**E1. Affine KL refit of the head: per-vocab bias + row gains + final-norm gain, on bf16 head inputs.**
- **What:** with the head fixed, the KL gradient in the bias is E[q − p], so the bias part is marginal matching (convex, a few Newton steps). The quantized rows' offsets ⟨Δh_v, μ⟩ on a nonzero-mean final hidden state are exactly such biases. Bias plane: 248,320 × f16 = 0.5 MB (0.00015 bpw).
- **Source:** #34 (mean-row cut head KL 30%, the common part of this bias); shortlist #7 (gain only).
- **Fit:** library (per head donor). **Format:** head bias plane (G1). **Runtime:** ~0.
- **Test:** headsim on dumped head inputs (exact head-only KL), CPU 1–2 h. Pass ≥5% head-KL cut on held-out dumps, then one whole-model swap.
- **Gain / risk:** −0.3…−1% (head ≈5–10% of KL at 2.5), P 0.35.
- **Conflicts:** shortlist #7; ranked E5 (shift).

**E2. Dead-vocab rows.**
- **What:** rows of tokens with negligible bf16 probability everywhere (reserved/unused specials that never appear in any top-512 of the calib golden) store nothing and emit a fixed large-negative logit. At K3, each 1% of the vocab saves ≈0.0014 bpw.
- **Fit:** format (head row mask). **Runtime:** saves.
- **Test:** CPU count from the head Fisher and the golden; guard on golden-ml.
- **Gain / risk:** −0…−0.7%, P 0.2. Risk: #32 — keep the set to non-text specials.
- **Conflicts:** #32 (tiering dropped; this removes only unreachable rows).

**E3. Norm-preserving embedding row scales.**
- **What:** choose each row's scale to preserve ‖e‖ rather than by LS. The un-normalized embedding rides the residual stream into every later norm.
- **Fit:** library (CPU). **Runtime:** none.
- **Test:** one whole-model swap at 2.0/2.5.
- **Gain / risk:** −0.1…−0.5%, P 0.2.
- **Conflicts:** #35 (scale-search null under MSE; this uses a different criterion).

**E4. Exact embedding rows for chat-structure tokens.**
- **What:** im_start/im_end, newline, think tags and role names are 5–10% of agentic/chat positions. ~50 exact rows ≈ 0.5 MB.
- **Fit:** library. **Runtime:** none.
- **Test:** swap at 2.0.
- **Gain / risk:** −0…−0.3%, P 0.15.
- **Conflicts:** shortlist #14; #32 (a handful of rows, not tiers).

## F. Per-mix final passes

**F1. One-pass closed-form corrector.**
- **What:** one staged forward of a shipped mix with bf16 in lockstep computes A2's γ, A8's biases, A4's τ and A1's α checks together.
- **Cost:** ~2 card-h per mix at 27B (one card, staged).
- **Fit:** final pass. **Runtime:** none / ~0.
- **Test:** the 0.8B tests of A2/A4/A8 run as one tool.
- **Gain / risk:** the sum of what passes, P 0.3.

**F2. Per-mix swj on quantized inputs.**
- **What:** refit (c_g, c_u) with the mix's quantized upstream. 2609.11716 finds blocks cancel ~50% of the error they inherit; a refit on quantized inputs can exploit that.
- **Fit:** per mix. **Runtime:** none.
- **Test:** 0.8B, 3 seeds.
- **Gain / risk:** −0.3…−1%, P 0.2.
- **Conflicts:** #104 (library form); #108.

**F3. Per-mix head refit on quantized-trunk inputs.**
- **What:** E1 with the head inputs the mix actually produces, so the head also compensates trunk error.
- **Fit:** per mix. **Runtime:** ~0.
- **Test:** one dump eval per mix plus CPU.
- **Gain / risk:** −0.3…−1%, P 0.3.
- **Conflicts:** ranked E3.

**F4. One Gauss–Newton step on all continuous parameters.**
- **What:** svh, suh, norms, biases; diagonal Fisher from one backward pass of the mix. A "one-pass healing" step.
- **Fit:** per mix. **Runtime:** none.
- **Hardware:** rented 2×H100, or staged on gpubox (slow).
- **Test:** 0.8B first.
- **Gain / risk:** −1…−4%, P 0.3.
- **Conflicts:** #89 (deferred healing; this is a one-pass fit, which the brief allows).

## G. Format and process (before the freeze)

**G1. Reserve correction fields in exl3i now.**
- **What:** optional per-tensor output bias (f16), gate pre-activation bias, sparse exact rows (index + f16 row), sparse exact input columns (for B3), and a head bias plane. Zero-length by default, so V3 containers carry no bytes. Later passes (A5–A8, A12, B1, E1, E2) then need no encoding bump.
- **Fit:** format. **Runtime:** none when empty.
- **Test:** selftest refusals plus byte identity with empty fields.
- **Gain:** keeps ~1–4% of later gains possible without a format change, P 0.9.
- **Conflicts:** none.

**G2. Log per-channel error statistics in every ladder level.**
- **What:** ‖w_n‖²_H, ‖e_n‖²_H and ⟨e_n, w_n⟩_H after refit, in both damped and undamped metrics. ~100 KB per tensor.
- **Why:** makes A1, A9, A11 and M1 free later, instead of re-decoding 71 levels.
- **Fit:** process. **Runtime:** none. Must precede the ladder.

**G3. Collect input means and lag-1–3 autocovariance diagonals for every reader input.**
- **What:** one bf16 forward (staged, 1–2 card-h, or piggybacked on any pending forward).
- **Why:** enables A7, D1, D2.
- **Fit:** process.

**G4. Byte-priced correction options in knap4.**
- **What:** biases, exact rows/columns and the head bias plane as DP items, priced by measured or headsim ΔKL.
- **Fit:** knapsack. **Test:** CPU.

**G5. Subtractive-dither trellis (from universal quantization).**
- **What:** encode W + d and decode Q − d, with d a seeded pseudo-random dither regenerated in the decode pass. The error becomes independent of W, so there is no shrink and no coherent component, at the cost of a larger error variance.
- **Fit:** format. **Runtime:** ~0 in the two-pass prefill; +ALU on the GEMV.
- **Test:** settle64 at K2/K2.5 (proxy and per-row ⟨e, w⟩), then 0.8B KL vs plain + A1.
- **Gain / risk:** probably loses to a gain fix, P 0.1.
- **Conflicts:** #100 (decode ALU, dropped).

**G6. Store LS and norm-restored svh side by side.**
- **What:** two svh vectors per level; knap4 or the search picks one per tensor.
- **Fit:** library. **Runtime:** none.
- **Gain / risk:** small, P 0.2.

## H. Allocation and search

**H1. Reprice after every adopted coherent fix.**
- **What:** the deficits scale with D, so low K gains most. Reprice from fixed-code pairs at 2–3 K points per family, as w63 did for swj (×0.91 K2, ×0.935 K2.5, ×0.989 K4).
- **Fit:** process (CPU).
- **Gain:** protects 0.5–2%, P 0.6.

**H2. Depth-tilt check against RAM's oracle on this model.**
- **What:** at IQ3_M bytes, RAM's oracle allocation on Qwen3.8-27B used 2.7–2.8 bits on L0–31 and 4.3–4.5 on L32–63 (2609.33923). If knap4's 2.5/3.0 depth profile is much flatter, add a depth-tilt kind to the stream.
- **Test:** CPU minutes.
- **Gain / risk:** 0…−2%, P 0.2.
- **Conflicts:** kfit2's depth curve.

**H3. Gate/up × down pair term in kfit2.**
- **What:** the SiLU Jensen term makes gate/up noise raise down's effective error, a superadditive pair.
- **Test:** 0.8B 2×2 factorial of swaps (12 evals).
- **Gain / risk:** 0…−1%, P 0.2.
- **Conflicts:** ranked F5.

**H4. Coherent / incoherent split in kfit2.**
- **What:** two features per tensor (H-norm deficit; orthogonal error) with separate depth amplification. Fitting this on V2's ~1,250 measurements tests the "coherent errors compound" hypothesis directly.
- **Test:** CPU.
- **Gain / risk:** fit quality, P 0.3.

**H5. Correction-aware local search.**
- **What:** after each accepted move, re-apply the cheap closed-form corrections (A3's γ, A4's τ) before scoring, so the stream optimizes corrected mixes.
- **Runtime:** ~0.
- **Gain / risk:** 0…−1%, P 0.3.

## I. Activations (fast variant)

**I1.** = B3 (massive-channel bypass for A8).

**I2. A8-aware input scales on frozen codes.**
- **What:** refit suh for fast-variant tensors to minimize tr(EHEᵀ) plus the aqproxy A8 noise, stored as a second suh per level.
- **Fit:** library. **Runtime:** prefill (none extra).
- **Gain / risk:** −10…−30% of the A8 penalty, P 0.2.
- **Conflicts:** #47.

**I3. Residual compensation for A4 tensors.**
- **What:** a rank-32 branch fed by the A4 rounding residual (ResCom, 2610.35955; Llama-3-8B W4A4 PPL 7.708 → 7.147). Only if #46-03 ships A4.
- **Runtime:** prefill, ~0.4% MAC.
- **Gain / risk:** ~30% less A4 error on those tensors, P 0.25 conditional.

## J. High-bpw floor

**J1. f32 residual accumulation and f32 linear-output stores at ≥6 bpw.**
- **What:** bf16 output stores cost ~1.1e-3 relative L2 per GEMM (v3.md §1c.9), on the order of A8's cost at K6. The 7.0/8.0 sizes sit near a non-weight floor (0.00028–0.00034).
- **Fit:** engine. **Runtime:** small activation bandwidth.
- **Test:** the #99 floor split plus one f32-store arm (3–4 evals).
- **Gain / risk:** −5…−20% of KL at 7.0, 0 at 2.5, P 0.3.

## K. Training-heavy / rented compute (post-freeze, V4)

**K1. Noise-robust preconditioning before quantization.**
- **What:** LoRA ΔW on the bf16 27B minimizing E_n KL(p_bf16 ‖ p_{W+ΔW+n}), with n Gaussian noise shaped like a target level's error (H-whitened, scaled by D_K). Then quantize W + ΔW. The reference stays the original bf16.
- **Source:** 2608.20988 (Jacobian noise injection); 2606.09012 (QAT theory).
- **Hardware:** rented 8×H100, 1–2 days. Proxy: full 0.8B run on the 4070 (~1 day).
- **Test:** pass if the 0.8B at K2 is ≤0.90 vs base after quantization.
- **Gain / risk:** −3…−10%, P 0.2. It changes the base, so it is a V4 ladder.
- **Conflicts:** deferred healing (#89).

**K2. Trellis-STE QAD with periodic re-encode.**
- **What:** TVQ / E8-QAT recipe on EXL3 codes: latents in the rotated domain, Viterbi re-encode every 16 steps.
- **Evidence:** 2-bit Llama3-70B 4.97 → 4.48 PPL; ~0 on Llama2-70B.
- **Hardware:** rented 8×H100.
- **Gain / risk:** −2…−6%, P 0.25.

**K3. Domain-randomized library healing.**
- **What:** C4 in training form: heal each level's scales against random sibling K, so healing becomes a library item.
- **Test:** ranked H4's level → mix transfer test on the 4070.
- **Gain / risk:** the library form of healing's −3…−10%, P 0.3.

**K4. Bias + gain-only healing as a ceiling.**
- **What:** train only biases and norm gains end to end on the 0.8B (4070, 3 h). If this gains <2%, drop A5–A8.
- **Fit:** diagnostic.

**K5. Rented seed pairs for borderline 27B decisions.**
- **What:** two to four more pairs (~$200–400 rented) for any 27B rule that misses by less than its noise.
- **Fit:** process.

## M. Diagnostics (cheap; listed as proposals because they reorder everything)

**M1. Coherent-error audit.** Decode 10–20 27B tensors per family at K2/K2.5/K4 from existing levels (colo, 4–8 CPU-h). Per channel: ε_n, the H-norm deficit and ⟨e, w⟩_H. Confirms the §1 identity and maps coherent headroom by family and K.

**M2. Residual-stream trace on a quantized 0.8B (and one 27B 2.5 mix, staged).** Per layer: RMS(h_q)/RMS(h), cosine, h_3994 ratio, ε at every norm, and GDN o's noise share. 4070, ~20 min.

**M3. Logit-temperature headroom.** Fit τ on recorded top-512 logits of an existing mix. Seconds once the logits exist.

**M4. Head-bias headroom.** Optimal per-vocab bias on headsim for the current head donors. CPU, ~1 h.

**M5. Size of the Jensen gap.** On 0.8B fixed codes: per channel E[silu(ĝ)û − silu(g)u] / RMS, and ‖W_d δ̄‖ vs down's own error. 4070, ~15 min.

**M6. Oracle massive-row patch.** Find the 0.8B's massive dim (writer row norms plus the post-attention norm gain). Replace that row in every writer with bf16 at K2/K3, 3 seeds; arms: exact row, gain-only. 4070, ~30 min.

**M7. Scaling curve for coherent fixes.** swj fixed-code gain on 0.8B, 2B and 9B at K2.5 (staged on the 4070, ~4 h). Sets the 27B prior for every idea in §A.

**M8. KL by position class.** First token / after delimiters / the rest, from existing per-row scores. CPU.

---

## Top 20 by expected gain per unit cost

| rank | idea | expected @2.5 | P | cost | precedes freeze? |
|---|---|---|---|---|---|
| 1 | G1 reserve correction fields in exl3i | enables 1–4% | 0.9 | days of format work, 0 bytes | **yes** |
| 2 | G2 per-channel error logging in the ladder | enables A1/A9/A11/M1 | 0.9 | ~0 | **yes** |
| 3 | A4 logit temperature per mix | −0.2…−1% | 0.4 | 1 card-h + seconds CPU | no |
| 4 | A2 noise-aware RMSNorm gains (+A3 table) | −0.5…−2% | 0.3 | 4070 1.5 h | no (post-quant) |
| 5 | A1 per-channel gain law + gauge map | −0.3…−1.5% | 0.3 | 4070 1 h, 27B pair 1 card-h | no (svh) |
| 6 | E1 head affine KL refit (bias + gain) | −0.3…−1% | 0.35 | headsim CPU 2 h | bias field: yes (G1) |
| 7 | C1 writer variants by sibling level | −3…−6% if #105 passes | 0.35 cond. | 4070 3 h, then writer requants ×3–5 | **yes** (ladder writer plan) |
| 8 | H1 reprice after each coherent fix | protects 0.5–2% | 0.6 | CPU | no |
| 9 | A5 SwiGLU Jensen bias (down bias) | −0.3…−1.5% | 0.25 | 4070 2 h | field: yes (G1) |
| 10 | A8 sequential bias nudging per mix | −0.3…−1.5% | 0.25 | minutes 0.8B, 2 card-h/mix | field: yes |
| 11 | C4 domain-randomized per-level fits | makes per-mix gains library | 0.3 | 4070 2 h | no |
| 12 | B1 exact row 3994 in writers (after M6) | −0…−2% | 0.2 | 4070 30 min; 0.0009 bpw | field: yes |
| 13 | A9 depth-scheduled α | −0.2…−0.8% | 0.2 | 4070 1 h | no |
| 14 | H2 depth tilt vs RAM oracle | 0…−2% | 0.2 | CPU minutes | no |
| 15 | D1 conv-aware GDN qkv metric | −0.2…−0.8% | 0.25 | 4070 2.5 h, 27B 1–2 card-h | **yes** (qkv codes) |
| 16 | E2 dead-vocab head rows | −0…−0.7% | 0.2 | CPU | head mask: yes |
| 17 | A12 swj + pre-activation offset | −0.2…−1% | 0.2 | 4070 1.5 h | gate bias field: yes |
| 18 | E3 norm-preserving embedding rows | −0.1…−0.5% | 0.2 | CPU + 1 swap | no |
| 19 | B3 A8 massive-channel bypass (fast variant) | part of the A8 penalty | 0.3 | CPU 1 h | kernel field: yes |
| 20 | F4 one Gauss–Newton step per mix | −1…−4% | 0.3 | rented 2×H100 | no |

## Five cheapest diagnostics that would most change this ranking

1. **M2 residual-stream trace** (4070, 20 min). If the RMS inflation at the norms (ε_ℓ) is ≥2% by mid-depth, A2/A3/A4 move to the top. If it's <0.5%, they drop, and the gdn_out α in #112 needs another explanation.
2. **M1 coherent-error audit on existing 27B levels** (colo, 4–8 CPU-h). It confirms or refutes the LS identity and shows which families still carry a gain deficit after swj and #112, which sets A1/A9/A11.
3. **M6 oracle massive-row patch** (4070, 30 min). It decides B1/B2 and tells whether the massive channel matters for KL at all beyond A8.
4. **M5 Jensen-gap size** (4070, 15 min). It decides A5, A6, A12 and H3 together.
5. **M3 + M4 temperature and head-bias headroom** (CPU, once one record eval exists). Two hours to settle the cheapest whole-model and head fixes.

One more for the scale question: **M7** (swj at 0.8B / 2B / 9B) would turn "coherent fixes grow with size" from a hypothesis into a prior for every §A idea.
