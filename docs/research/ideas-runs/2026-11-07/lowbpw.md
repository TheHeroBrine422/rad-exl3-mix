# Lens lowbpw: research ideas for 2.0–2.5 bpw (2026-11-07)

Five of the top ideas look like real gains at 2.5 bpw, and three of them are cheap to test:

1. **Rank-r deflation of the LM head (A1, A2).**
2. **Trellis-quantizing the GDN a/b projections, which are still stored bf16 (A8).**
3. **Reviving the #108 GDN junction refits.** Their own precondition for a 27B port is now met at ≤ K3.5 (B1).
4. **Splitting the fused ssm_qkv and attn_qg tensors into separate knapsack items (E1).**
5. **Reviving i8c at ≤ K3.5 (D1).** It was dropped only because it added nothing on top of H_out, and H_out is not used there.

The massive-channel ideas came out smaller than expected for readers. Today's proxy run shows it only matters in the early full-attention layers.

Conventions:
- "G" = expected change in mean KL at 2.5 bpw on the 27B.
- Noise-shaping ideas get the brief's default ⅓ shrink from 0.8B results. Coherent-error fixes are not shrunk; I say so each time.
- **Conversion used throughout:** the w63 re-pricing gives 0.0787 → 0.0534 → 0.0364 at 2.25 / 2.5 / 2.75. That is ln(1.467)/0.25 = 1.53 per bpw, so **0.01 bpw ≈ 1.5% KL at 2.5** (≈ 1.7% at 2.0). Ideas that free bytes are priced with this.

---

## 0. What I measured today (new data; colo `/data/work/ideas-lowbpw/`)

**Where the 2.5 budget goes** (knap5 `main-acc-2.5`, swj2 run):
- **Head:** uniform `ours-K4`. That is 5.0% of params and **0.189 bpw, 7.6% of the budget**.
- **Embedding:** `t2`, 0.095 bpw.
- **GDN a/b:** bf16 and outside the knapsack. config: `linear_num_value_heads` 48, so a+b ≈ 23.6M params ≈ **0.014 bpw**.
- **Trunk:** mean K ≈ 2.4.
  - 7.5% of params sit at K1.375–1.875, and 27% at K2–2.25.
  - Early layers are lowest. Mean K by 16-layer band: gate_up 2.07 / 2.17 / 2.40 / 2.84; attn_v 2.16 → 3.62; ssm_qkv 2.28 → 2.67.
  - Per-tensor relative output error at K2.5 on the order-fixed proxy is 0.6–1.4%.

**Channel 3994 is a persistent massive residual dimension.**
- **Share of the residual-reader Hessian trace:**

  | layer | 1 | 2 | 3 | 4 | 7 | 10 | 11 | 20 | 30 | 43 | 50 | 62 | 63 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | share | 0.82 | 0.80 | 0.93 | 0.64 | 0.91 | 0.67 | 0.88 | 0.64 | 0.63 | 0.55 | 0.39 | 0.16 | 0.11 |

  Max/median of H diag reaches 1.3e5. Channels 310 and 3842 are second and third.
- **It is present at a typical token, not just at sinks.** RMSNorm is zero-centred, and at L11 γ₃₉₉₄ = 1.026 ≈ the median γ (1.035). An 88% share is then only possible if the channel holds about 88% of the residual energy at a typical token. So it is also the RMSNorm denominator for every reader.
- **MLP input:** the post-attention norm suppresses it (γ ≈ 0.004 in L0–L3), except mid and late layers (L30 share 0.39).
- **Head input:** share 1.7% (final-norm γ₃₉₉₄ = 0.715 vs median 1.965).
- **Writers:** GDN out_proj and attention o_proj write it. Row 3994 has 8–21× the median row norm in L0–L11, which is up to **6.9% of L0 out_proj's weight energy in one row**. down_proj rows are 1.6–11.5×.
- **Readers:** column 3994 carries a large part of the reader's output energy tr(WHWᵀ):

  | tensor | L11 attn_qg | L30 ssm_z | L30 gate_up | L43 attn_qg | L10 ssm_qkv | L3 attn_qg |
  |---|---|---|---|---|---|---|
  | share | 86% | 73% | 35% | 35% | 4.4% | 2.5% |

  So for some readers most of the "signal" is a near-constant bias.

**Pivot structure after the 128-Hadamard + LDL.**
- **Column-block K: dead.** AM/GM of per-128-block mean pivots is 0.005–0.026 bit for readers (o_proj 0.09–0.10, consistent with #106's −3%).
- **Pivot-weighted Viterbi (CurveTQ): dead.** The within-16-column-tile Γ is ≤ 0.002 bit.

**Proxy test: column-3994 split** (settle64, order-fixed, K2.5, one 128-row sample; true output error vs baseline):

| tensor | L3 attn_qg | L11 attn_qg | L11 attn_qg, top-3 channels | L30 ssm_z | L10 ssm_qkv | L30 gate_up | L43 attn_qg |
|---|---|---|---|---|---|---|---|
| ratio | **0.921** | **0.950** | **0.898** | 0.985 | 0.994 | 0.997 | 0.998 |

LDLQ feedback already handles the massive direction everywhere except the early full-attention readers.

**Head and embedding PCA** (H-metric for the head, Euclidean for the embedding):
- **Head:** the mean row is **49.8%** of the head's H-weighted energy, which is why mean-row gave −30%. After centring, the **top-1 principal component holds 28% and the top-8 hold 41%** (top-64 54%).
- **Embedding:** near-isotropic (mean row 1.0%, top-8 PCs 1.8%).

---

## A. Budget outside the trunk: head, embedding, fixed bf16 tensors

**A1 · Rank-r H-metric deflation of the LM head.**
- **Mechanism:** Generalize the adopted mean-row trick. Factor W H^{1/2} and take the top-r components: W ≈ W_res + C·B, with C ∈ ℝ^{248320×r}, B ∈ ℝ^{r×5120}, both bf16. Quantize W_res with the usual trellis. Logits = Ŵ_res h + C(Bh).
- *Source:* new. It extends #34 mean-row. ARCHead (2608.02703) fits a low-rank term *after* quantization; this removes it *before*.
- *Why:* The top-8 components hold 41% of the centred H-weighted energy (top-1 28%). At fixed K the trellis error scales with the quantized matrix's energy. The head is about 5–7% of total KL.
- *Fit:* per level (head donors). Needs a low-rank adder in the plugin.
  - Decode adds about 2M MACs per token next to 1.27G for the head (+0.16%). Prefill runs the head on the last token only.
  - Storage for r=8: 0.0012 bpw (0.0006 with int8 coefficients); r=1: 0.00015 bpw.
- *Test:* CPU headsim (exact head-only KL on the 32,768 held-out head inputs), r ∈ {1,2,4,8} × K ∈ {3, 3.5, 4}, with the rare-token guard. Colo, about 1–2 h. Counts if r ≤ 8 cuts head-only KL by ≥ 20% at K4. Then one 27B pack pair (about 1 card-h).
- *G:* −1.5…−3% at 2.5 (direct head-KL cut, or re-spent via A2). Risk: the head-only to whole-model transfer is unverified (#99 pending), and the gain may be less than energy-proportional.
- *Conflicts:* #34 (mean-row adopted). #32 is unrelated (no vocabulary tiering here).

**A2 · Deflated heads as cheaper knapsack options.**
- **Mechanism:** Price deflated heads at K3–4.25 in quarter steps in knap4 (headsim prices) so the solver can trade head bits for trunk bits.
- *Source:* new, on top of A1.
- *Why:* K4 → K3.5 frees 0.024 bpw ≈ −3.6% in the trunk. A half step costs about ×2 head error, about ×1.2–1.5 after deflation.
- *Fit:* per level, then the knapsack.
- *Test:* CPU knap re-solve with A1's prices, about 15 min.
- *G:* part of A1's range. It is mostly the 2.0 product that gains (head K3 there).
- *Conflicts:* none.

**A3 · Low-budget slab heads (budgets 3.0–4.5) with floor search in the menu.**
- **Mechanism:** knap4 offers slab heads only at K5/5.5/6. Add plans at 3.0–4.5 from the floor search, which priced the curve 43–54% below uniform at every budget 3.5–7.5.
- *Source:* #34 data. SoftWater (2608.12026) gives the same direction (frequent tokens get finer grids).
- *Why:* A slab-3.5 head ≈ the uniform-K4 head's KL at 0.024 bpw less.
- *Fit:* per level.
- *Test:* add the `fcurve`/`lcurve` plans to KN_HEADSLAB with the guard-adjusted price, CPU. Then the whole-model check (#99 / #34-02 `headwm.sh`).
- *G:* −1…−3%. Risk: rare-token KL was ×2.5 at K4 budgets, so multilingual text pays.
- *Conflicts:* #34 ("K4 not offered: n.s."). The floor-search results came later and change that.

**A4 · ARCHead residual correction after (deflated) head quantization.**
- **Mechanism:** Rank 4–8 SVD of (W − Ŵ)C^{p} (p = 0.75, H-metric), stored in int8, added to the logits.
- *Source:* 2608.02703. Their gain vs a GPTQ head was relative PPL 1.0104 → 1.0058.
- *Why:* It catches the error structure left after A1.
- *Fit:* per level, same adder as A1.
- *Test:* headsim, CPU.
- *G:* −0.3…−1%. Risk: overlaps A1.
- *Conflicts:* none.

**A5 · Vocabulary logit-bias vector.**
- **Mechanism:** Add b ∈ ℝ^{248320} to the logits. Minimizing KL(p‖softmax(z+b)) is convex in b.
- *Source:* new. Motivation: LS refit shrinks noisier head rows more, a coherent per-token logit bias.
- *Why:* Systematic per-token bias does not average out.
- *Fit:* per level (expected bias from per-row shrink factors, closed form) or per mix (fit on stored logits). Storage 0.00015 bpw, free at runtime.
- *Test:* 0.8B fixed codes; 27B headsim variant.
- *G:* −0.3…−1%.
- *Conflicts:* shortlisted "head KL-side gain refit". This one is a bias, not a gain.

**A6 · Unbiased per-row head scales.**
- **Mechanism:** Replace the LS row scale with c = ‖w‖²_H / ⟨ŵ,w⟩_H.
- *Source:* SoloQ (2610.07121) bias correction.
- *Why:* It removes the noise-dependent shrink per token row.
- *Fit:* per level, zero cost.
- *Test:* headsim, CPU.
- *G:* −0.2…−0.8%.
- *Conflicts:* the shortlisted head gain refit (this is the parameter-free version).

**A7 · Head rare-token guard computed after deflation.**
- **Mechanism:** Re-run the slab floor search pricing guard KL on the deflated residual.
- *Source:* new.
- *Why:* Deflation lowers every row's error, which may make low-budget slabs safe for rare tokens.
- *Fit:* per level.
- *Test:* headsim plus guard slice.
- *G:* enables A3 (no direct gain).
- *Conflicts:* #34.

**A8 · Trellis-quantize the GDN in_proj_a / in_proj_b as knapsack items.**
- **Mechanism:** Quantize a/b at K3–6 (i8 codebook) under the layer-input H (already in the Hessian files) and add them to the 337-item knapsack. They are currently fixed bf16.
- *Source:* 2609.04098, the same model at NVFP4: single-projection GDN output error a 2.1%, b 2.6% vs qkv 10.4%; the log-parameterized gates absorb error.
- *Why:* about 0.014 bpw at bf16 → 0.0035 at K4. That saves 0.0105 bpw ≈ −1.6%; at K3 it is −1.9%. If the design doc's "N = 48 and 128" is right rather than 48+48, double that.
- *Fit:* per level. **Format/kernel choice, so it should be decided before the freeze.** a/b are separate `gemm_nt` ops; N=48 works with 16-wide tiles.
- *Test:* quantize a/b of all 48 layers at K3/K4 on a GPU box (minutes), swap into the K2.5 base, one 27B pair (~1 card-h). Counts if the KL cost is ≤ 0.3%.
- *G:* −1…−1.8% (up to −2.9%). Risk: a decode-time decay error could compound over very long contexts. Check with one 32k trace.
- *Conflicts:* shortlisted "GDN a/b int8". Int8 saves only 8 bits per weight; trellis K3–4 saves 12–13.

**A9 · Container byte census.**
- **Mechanism:** List every container byte that is not in the 337 knapsack items: a/b, conv1d (≈ 2M params, 0.0012 bpw), norms, A_log, dt_bias, and any padding or reserved head rows.
- *Source:* new.
- *Why:* Every 0.01 bpw is 1.5%. The A8 number rests on the config, not the container.
- *Fit:* n/a (diagnostic).
- *Test:* a CPU script over the V2 2.50 container, minutes.
- *G:* sets A8/A12's prize.
- *Conflicts:* #53 (board bytes = payload).

**A10 · Embedding per-row unbiased norm.**
- **Mechanism:** Rescale each dequantized embedding row so that ⟨ê,e⟩ = ‖e‖², or its norm matches. The row scale is already stored.
- *Source:* SoloQ.
- *Why:* An embedding error repeats on every occurrence of a token (systematic per token). LS shrink lowers the norm of noisier rows.
- *Fit:* per level (embedding levels), zero cost.
- *Test:* 27B embedding swap pair at the 2.5 base (CPU quant, 2 evals, ~1 card-h).
- *G:* −0.2…−0.8%.
- *Conflicts:* #35 (menu), #32 (no tiering here).

**A11 · Embedding fitted under the layer-0 readers' metric.**
- **Mechanism:** Quantize the embedding with M = H of layer 0's input, normalized, instead of H = I.
- *Source:* OPTIC (2610.31309): fitting in the output-path metric beat Euclidean by about 2 retention points.
- *Why:* Error directions that layer 0's readers do not use are free.
- *Fit:* per level.
- *Test:* 27B embedding swap pair, ~1 card-h. The 0.8B is useless here (tied embedding).
- *G:* −0.3…−1%. Risk: later layers read the residual too; the frequency-weighting precedent (#32) hurt multilingual text.
- *Conflicts:* #35 run6 (null on scale search), #32. This weights directions, not tokens.

**A12 · Reserved / unused head rows.**
- **Mechanism:** Census rows that are padding or never-valid outputs (near-zero norms or duplicates). Store them at sub-1 K or as a fixed floor.
- *Source:* new.
- *Fit:* per level.
- *Test:* CPU census, minutes.
- *G:* 0…−0.3%.
- *Conflicts:* #32. This is not frequency tiering: rows the tokenizer never emits only.

**A13 · Embedding stored as a head-row predictor plus residual.**
- **Mechanism:** If untied embedding and head rows correlate (cos(E_t, W_t)), code E_t − αW̃_t.
- *Source:* new.
- *Fit:* format change.
- *Test:* CPU cosine census first, minutes.
- *G:* 0…−0.5%. Likely nothing: the embedding is isotropic.
- *Conflicts:* none.

---

## B. Coherent-error fixes on frozen codes (closed form or one pass)

**B1 · Revive #108's GDN junction refits for levels ≤ K3.5.**
- **Mechanism:** The jfit z-gate / qkv+z / down-weighted refits (svh edits on in_proj_qkv / in_proj_z, fitted with bf16 inputs).
- *Source:* #108.
- *What changed:* #108 failed only "in the full stack (H_out dy + swj + writers asym)". Against trunk + swj it screened at **K2 0.954 / 0.945, K3 0.983**. V3 at ≤ K3.5 has swj but **no H_out** (#33-28) and, given #111, likely plain writers. The log's own condition ("not ported unless H_out and asym both fail there") is met at 2.5.
- *Why:* coherent per-channel error at the z / gated-norm junction, the same class as swj, which grew at 27B.
- *Fit:* per level, library-safe (bf16 inputs). Zero runtime.
- *Test:* 0.8B 3 seeds against the actual V3 ≤ K3.5 stack (swj + plain or C9 writers, cd3), then a 27B fixed-code pair at K2.5 (~1–2 card-h).
- *G:* −1…−3% (not shrunk: coherent class). Risk: overlap with C9 writers (#112).
- *Conflicts:* #108. The difference is the stack it is measured against.

**B2 · C9 H-norm restoration for reader families after swj.**
- **Mechanism:** One α per family (GDN qkv, z, attn_v), α grid extended past 2.
- *Source:* #85 (C9 alone K3 0.973 / K2 0.911). #85-02's "C9 on swj" was a broken run (wrong order) and was never redone for readers. #112 covers writers only.
- *Fit:* per level, fixed codes.
- *Test:* 0.8B 3 seeds, ~1 h; 27B fixed-code pair.
- *G:* −0.5…−2%.
- *Conflicts:* #85-02, #108 (KL-fitted joint refits ≠ an analytic H-norm restore).

**B3 · Parameter-free unbiased refit replacing LS refit_scales.**
- **Mechanism:** Per output channel, s = ‖w‖²_H / ⟨ŵ,w⟩_H (inner-product-preserving) instead of LS ⟨w,ŵ⟩/‖ŵ‖². Generalize as s = s_LS·(1+ε)^β with one β per family.
- *Source:* SoloQ (2610.07121, c_bc). This is the brief's "interaction with LS refit_scales".
- *Why:* LS shrinks exactly the coherent component that swj and C9 found harmful. It matters most for the 7.5% of params at K < 2, where ε is largest.
- *Fit:* per level. It changes the quantizer's final step, but it can be tested on frozen codes.
- *Test:* 0.8B fixed codes K2/K2.5/K3, β ∈ {0, 0.5, 1} (zero rounding noise within a pair).
- *G:* −0.5…−1.5%.
- *Conflicts:* #33-06 / #48 (output-scale variants). Those changed which scales exist; this changes the fit rule.

**B4 · Attention temperature restoration.**
- **Mechanism:** Scale q_norm × k_norm gains per layer by √((1+ε_q)(1+ε_k)). ε comes from per-level tables, so the correction composes per mix at pack time (see B12).
- *Source:* new. #108's q/k refits were absorbed by q_norm/k_norm; the norm gain itself (= temperature) was never tested.
- *Why:* Noise inflates the per-head RMS, so logits shrink. Measured ε at K2.5 is 0.6–1.4%, so the effect is likely small.
- *Fit:* per level (tables) plus pack-time assembly.
- *Test:* 0.8B fixed codes, 1-D sweep per layer group.
- *G:* −0.2…−0.8% (only 16 layers).
- *Conflicts:* #108 (attention refits ≥ 1.00, but a different knob).

**B5 · Logit temperature probe.**
- **Mechanism:** Fit one scalar on the final-norm gain per mix by 1-D KL search. Hidden-state rotation α shrinks logits by cos α (2609.11716: α ≈ 12.5° at NVFP4 32B).
- *Source:* 2609.11716.
- *Fit:* per mix, zero cost.
- *Test:* 0.8B logits, minutes; 27B, 3 evals.
- *G:* −0.2…−1%. Also a cheap diagnostic of the coherent-shrink hypothesis.
- *Conflicts:* shortlisted head gain refit (overlap); do only if that item doesn't cover it.

**B6 · Final-norm per-channel KL fit.**
- **Mechanism:** Fit 5120 gains on final-norm outputs of the quantized mix through the frozen head (head-only backward), one pass over about 1M qad tokens.
- *Source:* RoCo (2610.30779); ThinkBit norms-only recal (2610.34886, KL 0.641 → 0.418).
- *Fit:* per mix (~1 card-h: forward to collect final states + CPU/GPU fit).
- *Test:* 0.8B first.
- *G:* −0.5…−1.5%.
- *Conflicts:* healing is deferred, but this is a one-pass, 5120-parameter fit.

**B7 · Noise-aware swj.**
- **Mechanism:** Fit (c_g, c_u) with Gaussian noise added to the gate/up inputs at the level-typical upstream error of a 2.5 mix (σ grid table).
- *Source:* new. E[silu(g+n)] = silu(g) + ½silu''(g)σ² is a per-channel bias that swj on bf16 inputs cannot see.
- *Fit:* per level with a small σ table (library-safe).
- *Test:* 0.8B fixed codes, compare to swj.
- *G:* −0.5…−1.5%.
- *Conflicts:* #88 (denoising target for LDLQ was negative); this is a scale fit, not a rounding target.

**B8 · SwiGLU second-order bias correction.**
- **Mechanism:** b = −W_down·E[δ(silu(g)u)] from the noise model, closed form. Needs a bias vector per writer.
- *Source:* new; same mechanism as B7.
- *Fit:* per level, plus format support for a bias (shared with the shortlisted writers' mean-shift bias).
- *Test:* 0.8B.
- *G:* −0.3…−1%.
- *Conflicts:* shortlist "writers' mean-shift bias" (adjacent; this one is analytic).

**B9 · Per-channel moment matching at norm outputs.**
- **Mechanism:** Match the mean (via writer biases) and std (via norm gains) of each channel at every norm output to bf16, sequentially through the depth.
- *Source:* data-free bias correction (Nagel 2019) applied to the residual stream.
- *Fit:* per mix, one pass (~2 card-h).
- *Test:* 0.8B.
- *G:* −0.5…−2%.
- *Conflicts:* shortlist mean-shift bias (overlaps the mean part only).

**B10 · Σdy²-weighted output-scale refit at ≤ K3.5.**
- **Mechanism:** Use the existing 27B block-normalized diagonal as row weights in the refit only, not in two-sided LDLQ. No new collection, no 5× quant cost.
- *Source:* #85 B1 (0.8B screen on bdiag K3 0.993 / K2 0.975, never finished).
- *Fit:* per level, fixed codes.
- *Test:* 27B fixed-code pair, ~1 card-h.
- *G:* −0.5…−1.5%.
- *Conflicts:* #33-28 (two-sided H_out at ≤ K3.5 not adopted). This uses the same data at about 0 quant cost.

**B11 · 36-knob end-to-end gain grid.**
- **Mechanism:** One scalar per family × depth band, 9 × 4, fitted by coordinate search on search-pool KL.
- *Source:* generalizes C9.
- *Fit:* per mix, ~100 evals ≈ 3–4 card-h. Only for 2.5.
- *G:* −0.5…−2%. Risk: overfitting the search slice; verify on ver.
- *Conflicts:* #108.

**B12 · Pack-time composition of per-level corrections (enabler).**
- **Mechanism:** Store per-level tables (relative error ε, LS shrink factor, h₃₉₉₄ write shrink). Corrections that depend on the mix (B4, C4, the per-mix parts of B9) are then assembled at pack time with no GPU.
- *Source:* new.
- *Fit:* library plus pack time.
- *Test:* —
- *G:* makes several ideas library-compatible.
- *Conflicts:* addresses the brief's library-vs-mix theme.

**B13 · Exact first-token (sink) states.**
- **Mechanism:** Position 0 is context-free. Precompute bf16 attention K/V at position 0 for the canonical first token (16 layers × 4 KV heads × 256 × 2 ≈ 64 KB). Also precompute the GDN first write (k, v, β per head) and conv state.
- *Source:* 2609.09085 (sinks and massive activations at the initial position whatever the token).
- *Fit:* format and runtime feature. **Check the challenge rules first.**
- *Test:* 0.8B hook replacing position-0 states.
- *G:* 0…−2% (unknown).
- *Conflicts:* none in the index. Needs the corpus first-token census (G6).

**B14 · GDN state-drift diagnostic and per-head fix.**
- **Mechanism:** Compare per-head state Frobenius norm and effective erase β(k·k̂)² between bf16 and the 2.5 mix over 32k docs. If the drift is systematic, use a per-head scalar on dt_bias / b.
- *Source:* LeapQuant / STEPQuant (2609.38166 / 2609.38169).
- *Fit:* per level.
- *Test:* 0.8B hooks.
- *G:* 0…−1%.
- *Conflicts:* none.

**B15 · swj fitted through down_proj.**
- **Mechanism:** Fit (c_g, c_u) jointly to minimize ‖W_down(silu(ĝc_g)⊙ûc_u − silu(g)u)‖² (bf16 down) instead of matching each channel's product.
- *Source:* SACQ (2610.41408) G_gate / G_up weights.
- *Fit:* per level.
- *Test:* 0.8B fixed codes.
- *G:* −0.2…−0.8%.
- *Conflicts:* #85 / #104 (swj).

**B16 · Bias-excluded LS refit for readers.**
- **Mechanism:** Compute refit_scales against H with channel 3994 removed, i.e. fit the scales to the informative part.
- *Source:* new, from today's diagnostic: column 3994 is 86% of L11 attn_qg's tr(WHWᵀ) and 73% of L30 ssm_z's. The current LS row fit is dominated by reproducing a near-constant bias.
- *Fit:* per level, fixed codes.
- *Test:* 0.8B (if it has the channel, C9) or 27B fixed-code pair.
- *G:* −0.3…−1%.
- *Conflicts:* none.

**B17 · GDN conv1d tap refit.**
- **Mechanism:** Refit the 4 conv taps per channel (bf16, 2M params) to reproduce the bf16 conv output given quantized ssm_qkv projections on bf16 inputs. Use an unbiased (not LS) target.
- *Source:* new junction, not in #108's list.
- *Why:* The taps can partly denoise projection noise across 4 tokens.
- *Fit:* per level (depends on ssm_qkv's K only). Zero runtime.
- *Test:* 0.8B fixed codes.
- *G:* −0.2…−1%.
- *Conflicts:* shortlisted "conv-aware GDN qkv H_in" (a different stage: statistics vs post-fit).

---

## C. The massive channel 3994 (structural)

**C1 · Exact column(s) 3994 (+310, 3842) for early full-attention readers only (L3, L7, L11, L15).**
- **Mechanism:** Store the column bf16 and quantize the rest with that channel zeroed in H. Runtime is one rank-1 update.
- *Source:* shortlist ("massive-channel handling"). Today's numbers sharpen it: **L3 attn_qg 0.921, L11 0.950, top-3 0.898**, but ≤ 1.5% gain elsewhere.
- *Fit:* format (outlier-column list per tensor). Storage < 0.0002 bpw.
- *Test:* settle64 on all 16 attn_qg/k/v at K2/K2.5 (CPU, ~1 h), then one 27B KL pair.
- *G:* −0.2…−0.6% (only a few small tensors), unless the coherent bias error matters more in KL than in proxy error.
- *Conflicts:* the shortlisted item. Restrict it to early attention layers.

**C2 · Writers' row-3994 exactness.**
- **Mechanism:** Extract row 3994 of ssm_out / o_proj / down in L0–L15 (bf16), quantize the rest, add the row's dot product at the store.
- *Source:* new. The writers put up to 6.9% of their weight energy in that row. Under the output Hadamard, LS shrink of block 31's rows shrinks the massive channel's write coherently.
- *Fit:* format. Storage ≈ 0.0009 bpw for all layers, 0.0002 for L0–15.
- *Test:* 0.8B (if C9 holds); 27B requant of 16 ssm_out/o_proj (~30 min) + pair.
- *G:* 0…−2% (uncertain). Note: #21 measured L0 ffn_down at K2 as nearly free (+0.00006), evidence against.
- *Conflicts:* none.

**C3 · Is #112's gdn_out α ≈ 3–4.75 a massive-channel artefact?**
- **Mechanism:** Fixed-code ablation: apply the C9 gain to block-31 rows only, vs all rows except block 31, vs all rows.
- *Source:* new. Same family, same layers that write channel 3994. A family α at the grid edge is suspicious.
- *Fit:* per level.
- *Test:* 0.8B fixed codes, 30 min.
- *G:* indirect. If true, a targeted fix replaces a blunt family α and may beat it.
- *Conflicts:* #112 (running; this informs its rollout).

**C4 · RMS-denominator drift compensation.**
- **Mechanism:** Channel 3994 holds 55–93% of residual energy in L1–L43, so its write shrink scales every other normalized reader input coherently. Per layer, multiply non-3994 norm gains by the measured rms ratio.
- *Source:* new.
- *Fit:* per mix via B12 tables.
- *Test:* G3 diagnostic first.
- *G:* 0…−1.5%.
- *Conflicts:* none.

**C5 · Mean-removed LDLQ + exact bias.**
- **Mechanism:** x = μ + x̃. Quantize against the covariance C = H − Nμμᵀ and add b = (W−Ŵ)μ. Needs first moments, a forward-only collection, not Σdy²; the user's freeze applies to Σdy² only.
- *Source:* classic bias correction.
- *Why:* It generalizes C1 to every mean direction.
- *Fit:* per level. Format: bias.
- *Test:* 0.8B first-moment pass (minutes); CPU proxy.
- *G:* 0…−1% (today's data says LDLQ already handles the dominant direction).
- *Conflicts:* #88 (unrelated mechanism).

**C6 · Rest-normalized damping.**
- **Mechanism:** Damping = σ·trimmed-mean(diag H), excluding the top-3 channels, so effective relative damping is consistent with depth. Today it ranges from ≈ 4% of typical curvature at L11 to ≈ 0.5% at L62, because mean diag is dominated by channel 3994.
- *Source:* IQR-V3 note on over-damping; today's shares.
- *Fit:* per level (requant).
- *Test:* settle64 damp mode with alternative normalization, CPU.
- *G:* −0.2…−0.8%.
- *Conflicts:* #38 (per-tensor sweep over 3 values partly compensates).

**C7 · Larger Hadamard (512/1024) for early attention readers.**
- **Mechanism:** An alternative to C1: spread channel 3994 beyond block 31.
- *Source:* faraway M5 (L11 attn_qg at 1.74× an ideal coder).
- *Fit:* format and kernel (per-tensor rotation size).
- *Test:* settle64 with a larger rotation, CPU.
- *G:* similar to C1, with more kernel work.
- *Conflicts:* #47 (R1 residual rotation, a different thing).

**C8 · Bias-excluded proxy normalization.**
- **Mechanism:** Proxy "rel" for readers is deflated by up to 86% bias energy. Any rule or price built on cross-tensor proxy rel (option pricing from curves, kfit features, the #94-style ratio checks) should use tr(W H_rest Wᵀ).
- *Source:* today.
- *Fit:* methodology.
- *Test:* recompute existing tables, CPU.
- *G:* indirect.
- *Conflicts:* none.

**C9 · 0.8B validity check.**
- **Mechanism:** Does Qwen3.5-0.8B have a persistent massive channel with the same writer pattern (H diag shares, norm γ, writer row norms)? Decides whether B16 / C2–C4 can be screened there.
- *Source:* —
- *Fit:* diagnostic.
- *Test:* minutes of CPU.
- *G:* —
- *Conflicts:* none.

**C10 · Massive-share covariate in the kfit2 prior.**
- **Mechanism:** The reader share curve (0.9 → 0.1 with depth) as a feature for unmeasured options.
- *Source:* new.
- *Fit:* allocation.
- *Test:* CV log-RMSE.
- *G:* ±0.5%.
- *Conflicts:* #19 / #20 (edge offsets and re-ranker were null).

---

## D. Codebook, trellis and LDLQ at K1.4–3

**D1 · Revive i8c for levels ≤ K3.5.**
- **Mechanism:** Use the cubic 0.02 int8 codebook at the levels where H_out is not adopted.
- *Source:* #78. Plain quants: 0.8B K3 −1.7%, K2 −2.7%, K4 −2.3%; 27B CPU proxy −1.2…−1.5%.
- *What changed:* It was dropped as "#78-04 null on top of H_out". H_out is not used at ≤ K3.5 (#33-28).
- *Fit:* per level. **Codebook = format: decide before the freeze.** Decode +2 FMAs per weight (exl3cb has it).
- *Test:* 0.8B i8c vs i8 on the V3 stack (swj, no H_out), 3 seeds K2.5/K3 (~2 h), then a 27B K2.5 pair (requant 2 levels, ~2.5 card-h).
- *G:* −0.8…−2%. Risk: overlap with swj; decode cost.
- *Conflicts:* #78-03 / #78-04 (null only on H_out), #100 (a different codebook).

**D2 · i8c as a per-tensor option at ≤ K3.5.**
- **Mechanism:** Offer both codebooks per tensor, as BACKLOG suggests.
- *Source:* BACKLOG "27B confirmation queue".
- *Fit:* per level (~20 extra levels ≈ 12 card-h).
- *Test:* after D1.
- *G:* an extra −0.3% over D1.
- *Conflicts:* same as D1.

**D3 · K-dependent warp strength at K1.375–1.875.**
- **Mechanism:** Sweep the cubic coefficient at the low levels (7.5% of params at 2.5).
- *Source:* #78 sweep (cubic 0.05 worse at K2.5).
- *Fit:* per level.
- *Test:* settle64, CPU, 1 h.
- *G:* −0.1…−0.4%.
- *Conflicts:* #78.

**D4 · Tile CD on attn_k and cd5 on early attn_qg.**
- **Mechanism:** Extend the adopted CD to attn_k (GPU proxy 0.991 / 0.994, ~2 s per tensor) and run more sweeps on L3–L15 attn_qg, where the massive channel hurts most.
- *Source:* #77.
- *Fit:* per level.
- *Test:* GPU quant + 27B no-harm check.
- *G:* −0.05…−0.2%.
- *Conflicts:* #77 (adopted on attn_qg only).

**D5 · Tile CD at K ≤ 2 for all families.**
- **Mechanism:** CD sweeps on every family at the lowest levels only.
- *Source:* #77 (proxy cd3 −1.0% at K2; GPU ~⅓ of that).
- *Fit:* per level.
- *Test:* CPU settle64.
- *G:* −0.1…−0.3%.
- *Conflicts:* #77.

**D6 · JARQ alternation at K ≤ 2.**
- **Mechanism:** After refit_scales, re-run Viterbi on the refit grid; 2 rounds.
- *Source:* JARQ (2609.38599).
- *Fit:* per level (quant time ×2 at low K).
- *Test:* settle64.
- *G:* −0.1…−0.5%.
- *Conflicts:* none.

**D7 · HARP learned butterfly for the ~20 most sensitive tensors.**
- **Mechanism:** Learn a per-tensor butterfly rotation in place of the fixed Hadamard for a few tensors.
- *Source:* 2605.29843. QTIP 2-bit 7B 6.87 → 6.63 at +0.05 bpw; 3-bit 5.41 → 5.37.
- *Fit:* format and kernel change.
- *Test:* settle64 with learned rotation.
- *G:* −0.3…−1%. High cost.
- *Conflicts:* #41 (best-of-N seeds: no gain).

**D8 · Soft trellis marginals (BCJR) + CD.**
- **Mechanism:** Use forward–backward path marginals in place of hard Viterbi, then CD (MPQ-style).
- *Source:* MPQ (2610.32127): LDLQ+CD 15.01 → 12.76 at 2-bit E8P12.
- *Fit:* quantizer research.
- *Test:* CPU prototype.
- *G:* −0.3…−1% at K ≤ 2. Long.
- *Conflicts:* #113.

**D9 · More tail-biting wrap candidates at K ≤ 2.**
- **Mechanism:** NTRY 16 recovered 12% / 8% of the tail-biting gap at K1 / K2 (0.8% / 0.4% error).
- *Source:* #57.
- *Fit:* per level (quant time only).
- *Test:* GPU quant.
- *G:* −0.05…−0.15%.
- *Conflicts:* #57, #102.

**D10 · Damped LDLQ feedback at K ≤ 2.**
- **Mechanism:** Scale the feedback by γ_i (QTEA's error decay) at the lowest levels.
- *Source:* 2609.00224 (+0.46 points on Qwen3-14B).
- *Fit:* per level.
- *Test:* settle64 λ sweep.
- *G:* 0…−0.3%.
- *Conflicts:* none.

**D11 · Family-specific codebook warp from LDLQ target kurtosis.**
- **Mechanism:** Measure kurtosis of feedback-adjusted targets per family at K2. If heavy-tailed, pick a per-family warp via the per-tensor `cb=` field.
- *Source:* new.
- *Fit:* per level.
- *Test:* CPU diagnostic.
- *G:* 0…−0.5%.
- *Conflicts:* #78.

**D12 · 1/16-step levels at K2.25–2.625.**
- **Mechanism:** Add finer levels where 58% of 2.5's params sit; price them first by interpolation in knap.
- *Source:* reprice (1/8 steps at K5–7: −0.1…−0.2%).
- *Fit:* per level.
- *Test:* CPU knap.
- *G:* 0…−0.4%.
- *Conflicts:* none.

---

## E. Allocation at low budgets

**E1 · Split fused ssm_qkv into q|k|v and attn_qg into q|gate as separate knapsack items.**
- **Mechanism:** Quantize the 128-aligned row ranges as separate tensors with their own su and K. Run them as adjacent sub-GEMMs.
- *Source:* #96 found the spread between GDN heads: ssm_qkv 0.33 bit, attn_qg 0.27 bit, ≈ 0.04 bpw ceiling for per-head K. Component-level splitting gets part of that with no new format.
- *Fit:* per level. **Ladder structure: decide before the freeze.**
- *Test:* 27B band swaps of q-only / v-only / gate-only (8 evals, ~1 card-h); knap re-solve.
- *G:* −0.5…−2%.
- *Conflicts:* #96, #29-02. This is not row-slab K inside a tensor; it uses sub-tensors.

**E2 · Untie gate and up K.**
- **Mechanism:** gate and up are quantized apart already; give them separate knapsack items. swj needs fits for each (K_g, K_u) pair used, |ΔK| ≤ 0.5.
- *Source:* design §1d (fused-GEMM K ties are an open decision).
- *Fit:* per level plus swj pair table.
- *Test:* band swaps at 2.5.
- *G:* −0.5…−1.5%.
- *Conflicts:* swj (#104).

**E3 · In-context re-measurement around the 2.5 mix.**
- **Mechanism:** Re-measure the most uncertain group swaps with upstream at the 2.5 mix rather than the 4 bpw base.
- *Source:* MixQuant (2607.23047), GEMQ (2605.23078).
- *Fit:* search.
- *Test:* 20–40 evals.
- *G:* −0.5…−1.5%.
- *Conflicts:* #14 (single-tensor swaps are too noisy, so use bands). MixFrag found re-measuring *hurt* — a risk.

**E4 · Band-level 2×2 interaction audit at 2.5.**
- **Mechanism:** gate_up × down and qkv × out, 2×2 factorial per band. Add a kfit2 pair term if non-additive.
- *Source:* TWLA (2606.13054), LoopQuant (2610.16368).
- *Fit:* allocation.
- *Test:* ~16 evals.
- *G:* −0.3…−1%.
- *Conflicts:* #11.

**E5 · Capped-move local search with re-pricing.**
- **Mechanism:** Limit the number of simultaneous changes per accepted step and re-measure after each.
- *Source:* 2609.09867 (composition error 26–42× pricing error).
- *Fit:* stream policy.
- *Test:* —
- *G:* protects the gains.
- *Conflicts:* #70.

**E6 · Lazy greedy measured-KL descent from 2.75 to 2.5.**
- **Mechanism:** Demote band × family groups one at a time by measured KL.
- *Source:* Q-Strata (2608.30564), about 2.5 evals per move.
- *Fit:* search.
- *Test:* ~150 evals.
- *G:* −0.5…−2%.
- *Conflicts:* #70.

**E7 · Minimum-K floors at L60–63 and on the massive writers (L0–L11 ssm_out / o_proj).**
- **Mechanism:** Force floors in the knapsack and measure.
- *Source:* GEMQ minimum-high-bit rule.
- *Fit:* allocation.
- *Test:* repack from the library + 1 pair.
- *G:* ±0.5%.
- *Conflicts:* #21.

**E8 · 27B swj at K ≤ 2.**
- **Mechanism:** One pair to confirm the r = 0.91 the 2.0 prediction rests on.
- *Source:* w63 note.
- *Fit:* diagnostic for the 2.0 product.
- *Test:* 1 fixed-code pair.
- *G:* —
- *Conflicts:* #104.

**E9 · Category-weighted pricing for head and embedding.**
- **Mechanism:** Multilingual guard text in the price.
- *Source:* #32, #34 guard.
- *Fit:* allocation.
- *Test:* —
- *G:* risk control.
- *Conflicts:* #32.

**E10 · Tempering exponent on predicted sensitivities.**
- **Mechanism:** Tune NatScale's τ on search slices.
- *Source:* 2610.20353.
- *Fit:* allocation.
- *Test:* CPU knap + stream.
- *G:* 0…−0.5%.
- *Conflicts:* kfit2 already fits an exponent.

---

## F. Per-mix passes for the key product (2.5 only)

**F1 · If #105 passes: run the per-mix writer pass for 2.5 only.**
- **Mechanism:** #111 found per-level asym fails next to mixed-K siblings, but a per-mix pass keeps 0.81–0.90 on the 0.8B.
- *Fit:* per mix, 4–5 card-h for the single most important size.
- *Test:* —
- *G:* −2…−6% (if #105 holds at 27B).
- *Conflicts:* #105, #111. This is a rollout recommendation, not a new method.

**F2 · Per-mix gdn_out-only writer pass.**
- **Mechanism:** The cheapest third of F1. It also targets the massive-channel writer (see C3).
- *Fit:* per mix, ~1.5 card-h.
- *G:* −1…−3%.
- *Conflicts:* same.

**F3 · Per-mix sequential norm-gain LS repair.**
- **Mechanism:** For each norm, closed-form fit of its gain vector over the stacked sibling readers: Gram (WᵀW) ⊙ (Σ x̂x̂ᵀ). Use quantized-prefix inputs and bf16 outputs as targets. Use an unbiased rather than LS-shrinking solve.
- *Source:* RoCo (2610.30779), ThinkBit norms-only.
- *Fit:* per mix, one pass (~3 card-h).
- *Test:* 0.8B first.
- *G:* −1…−4% (wide).
- *Conflicts:* #89-02 (norms ≤ 0.2% on top of healing; standalone not measured). Caution: plain LS gains on quantized inputs shrink *more*, which the coherent-error evidence says is wrong.

**F4 · F3 on the last 16 layers only.**
- **Mechanism:** Same fit, restricted to layers 48–63.
- *Source:* field note: QAD on the last 16 layers gained most.
- *Fit:* per mix (~1 card-h).
- *G:* −0.5…−2%.
- *Conflicts:* same as F3.

**F5 · Per-mix final-norm + logit-bias fit.**
- **Mechanism:** B6 and A5 fitted together on the shipped mix.
- *Fit:* per mix, ~1 card-h.
- *G:* −0.5…−1.5%.
- *Conflicts:* —

**F6 · Per-mix head requant against quantized-trunk final states.**
- **Mechanism:** "Head asym": quantize the head so that Ŵ ĥ ≈ W h.
- *Fit:* per mix (~30 min head quant).
- *G:* 0…−1.5%. Risk: #88-02 showed asym fails beyond the writers.
- *Conflicts:* #88-02.

---

## G. Diagnostics

- **G1 · Per-band × family KL attribution at the 2.5 mix.** Swap each band/family group up to K4 (~10 evals). Shows where 2.5's error lives.
- **G2 · Token-tail and category split of 2.5 KL** from stored per-row scores (CPU). Decides head/embedding vs trunk emphasis.
- **G3 · Per-layer hidden-state angle, norm ratio and h₃₉₉₄ drift**, bf16 vs a K2.5 quant (0.8B hooks; 27B dump). Gates B4/B5/C4.
- **G4 · Attention entropy per head**, bf16 vs 2.5. Gates B4.
- **G5 · KL-matched proxy regime.** 27B at 2.5 (0.053) corresponds to about 0.8B K3–3.5, not K2 (KL 0.5). Screen low-bpw ideas there.
- **G6 · Corpus first-token census** (for B13).
- **G7 · LS-shrink census.** s = ⟨w,ŵ⟩_H/‖ŵ‖²_H per family and K from library shards + published H (CPU). Gates B2/B3/B16.
- **G8 · Head-only KL of the actual 2.5 head** (headsim). Confirms its 5–7% share.
- **G9 · Embedding-only swap** (t2 → bf16) at the 2.5 mix, 2 evals. Measures the embedding's share.

## Ruled out by today's diagnostics

- Pivot-weighted Viterbi (Γ ≤ 0.002 bit).
- Column-block K (≤ 0.026 bit).
- Reader 128-block act order (pivot p90/p10 1.4–1.7 vs o_proj's 3.3–4.1; #103 already failed at a similar spread).
- Embedding mean/PC deflation (1.0% / 1.8%).
- Massive-column split outside the early attention layers (≤ 1.5%).

---

## Top 20 by expected gain per unit cost

| # | idea | G at 2.5 | cost |
|---|---|---|---|
| 1 | A1+A2 head rank-r deflation (+ re-spend) | −1.5…−3% | CPU headsim + small adder |
| 2 | A8 (+A9) trellis GDN a/b | −1…−1.8% (maybe −2.9%) | minutes of quant + 1 pair; pre-freeze |
| 3 | B1 #108 GDN refit revival at ≤ K3.5 | −1…−3% | fixed-code pair |
| 4 | F2 / F1 per-mix writer pass at 2.5 only (if #105 passes) | −1…−6% | 1.5–5 card-h, one size |
| 5 | A3 (+A7) low-budget slab heads | −1…−3% | CPU + #99 check |
| 6 | E1 fused-tensor split (q/k/v, q/gate) | −0.5…−2% | band swaps; pre-freeze |
| 7 | D1 i8c revival at ≤ K3.5 | −0.8…−2% | 0.8B + 27B pair; pre-freeze |
| 8 | B3 unbiased refit | −0.5…−1.5% | fixed codes |
| 9 | C3 gdn_out α massive ablation | informs #112 | 30 min 0.8B |
| 10 | B16 bias-excluded LS refit | −0.3…−1% | fixed codes |
| 11 | B10 Σdy²-weighted refit (existing data) | −0.5…−1.5% | fixed-code pair |
| 12 | E2 untie gate/up K | −0.5…−1.5% | swaps + swj pair table |
| 13 | B6 / F5 final-norm + logit bias | −0.5…−1.5% | ~1 card-h |
| 14 | A5 logit bias | −0.3…−1% | CPU / 0.8B |
| 15 | B7 noise-aware swj | −0.5…−1.5% | 0.8B fixed codes |
| 16 | F3 / F4 per-mix norm LS repair | −0.5…−4% | 1–3 card-h per mix |
| 17 | A10 embedding unbiased norm | −0.2…−0.8% | 1 pair |
| 18 | C1 early-attention massive column | −0.2…−0.6% | CPU + 1 pair |
| 19 | B9 moment matching | −0.5…−2% | 2 card-h |
| 20 | D9 more wrap candidates at K ≤ 2 | −0.05…−0.15% | quant time only |

## The 5 cheapest diagnostics that would most change this ranking

1. **headsim for a deflated head** (r ∈ {1,2,4,8} × K ∈ {3, 3.5, 4}, with guard; colo, ~1–2 h). Decides #1 and #5.
2. **Container byte census + a/b K3/K4 swap at the K2.5 base** (minutes + ~1 card-h). Decides #2 and its true size (N = 48+48 or 48+128).
3. **LS-shrink census (G7) from library shards** (CPU, no GPU). Shows whether B2/B3/B16 have room at K ≤ 2.5.
4. **0.8B massive-channel check (C9) + block-31-only C9-α ablation (C3)** (minutes + 30 min on the 4070). Decides whether the gdn_out α is the massive channel, and whether C2/C4 can be screened on the proxy.
5. **0.8B screen of #108 GDN refit and i8c on the actual V3 ≤ K3.5 stack** (swj, no H_out, plain writers; 3 seeds at K2.5/K3, ~3 h on the 4070). Decides #3 and #7 before any 27B pair is spent.

Scratch on colo: `/data/work/ideas-lowbpw/` holds `{a,b,c,d}.jsonl` (Hessian/pivot stats), `out/*.jsonl` + `.norm` (column-split proxy), `pca.out`, and the scripts. Nothing was written in the repo.
