# Research: RCO (Helcig & Alistarh 2026, arXiv 2605.00649) for our mix allocation, plus SLQ (2605.02404)

> **Status (2026-11-02).** Research snapshot from 2026-10-30, kept as written. Its recommendation (option C,
> model-based local search) was **adopted** (#70); RCO itself was **dropped** (#17, numbers =
> `../experiments/INDEX.md`). The RCO paper's digest is in the 2026-11 paper corpus (`2605.00649`).

*Date: 2026-10-30.*

**Sources read:**
- RCO paper, v3 HTML and PDF tables.
- RCO code: `IST-DASLab/RCO`, cloned to `/tmp/pi-github-repos/.../5440b0f…` (`src/search/quant.py`, `src/manifold.py`, `rco_search_quant.py`, `scripts/run_search_quant.sh`).
- SLQ paper, HTML.
- `docs/research/mix-allocation.md`, `kfit2.py`, `auto.sh`.

**Labels:**
- **[Lit]**: the source says it.
- **[Interp]**: my reading of the source.
- **[Own]**: my idea or derivation, not in any source.

---

## 0. Executive summary

1. **Adopting RCO as a whole needs things we don't have.** [Interp] It optimizes the *real* calibration KL by backpropagating through the full model. Its search variables are logits α_ik per (group, option). It makes the forward pass differentiable by blending all K pre-quantized weight versions: W = W_ref + Σ_k p_ik Δ_ik.
   - **What that requires for us:**
     - a differentiable torch forward and backward of Qwen3.8-27B (GDN + attention) on ROCm;
     - bf16 reference log-probs;
     - every option's dequantized weights touched at every step;
     - 800–3,200 forward+backward passes of 2,048 tokens per budget.
   - **Why it doesn't fit:**
     - Radiance has no backward pass.
     - The bf16 model (54 GB) does not fit one card.
     - Our option pool (~10–25 options per tensor) multiplies the weight traffic by K.
   - [Own] **Estimate:** 1–2 weeks of engineering with high ROCm risk, then roughly 15–30 GPU-hours per target size. That is too much for a gain that is probably small next to our approach.
2. **Its measured edge is mainly at ≤2.5 bpw, and over weak surrogates.**
   - [Lit] On Qwen3-8B (FineWeb-Edu ppl):
     - At 2.25 bpw RCO beats HIGGS-DP by 36% (20.47 vs 32.07) and EvoPress by 4% (vs 21.40).
     - At 3.5–4.0 bpw all methods are within about 2% of each other (RCO 11.41 / EvoPress 11.50 / HIGGS 11.65 at 3.5).
     - Seed standard deviation at 2.5 bpw is 0.18 ppl (about 1.2%), so the 2.5-bpw lead over EvoPress (15.43 vs 15.64) is about one SD.
   - [Interp] Our surrogate is fitted to measured mixes. It is not a noise-injection linear model like their HIGGS baseline, so the relevant comparison is "RCO vs measured-loss search". There RCO's lead is 0–4%.
3. **The manifold itself is irrelevant to us.** [Lit, App. E.3] With the DP in the forward pass, a tuned quadratic penalty comes within 1.3% of RCO's KL (0.1374 vs 0.1356 at 3.0 bpw). The paper concludes that for a single budget "the gain from the manifold is therefore exact feasibility without tuning, not solution quality". We already have an exact byte-bucket DP and never relax the problem, so there is nothing to gain from the geometry.
4. **The idea worth taking: one backward pass gives in-context sensitivities for all 337 tensors at once.** [Own derivation, §2.3]
   - RCO's straight-through gradient is essentially ⟨∂KL/∂W_i, W_i^(k) − W_i^(z_i)⟩ evaluated *at the sampled mix*. In the quadratic regime this mostly measures the current option's own in-context contribution, ≈ 2·s_i(z_i). Depth amplification is included automatically, because the Jacobians are taken at the quantized mix.
   - That is exactly the local tangent our surrogate needs. One forward+backward pass yields 337 numbers, where one eval yields 1.
   - A path-integrated version, Aumann–Shapley/integrated gradients with 3–4 passes, decomposes the measured KLD of a mix *exactly* into per-tensor shares. It is a continuous analogue of SLQ's Shapley games at a tiny fraction of their cost.
   - **Recommendation:** treat this as a scoped, optional sub-project:
     - first a 1-day feasibility spike (single-layer torch backward on ROCm, via exllamav3 dequant);
     - then a validation on existing fit-pool band-swap data before it touches the loop.
   - **Do not delay the planned model-based local search for it.**
5. **SLQ:** skip its Shapley + ILP; it costs about 15 h per 8B model [Lit]. Two cheap things are worth taking:
   - **EAR** = mean top-k Σ min(p, q) = 1 − TV. Use it as a *secondary diagnostic* column in the eval output, not as an objective (the challenge scores KLD).
   - **The single-point ratio calibration** is a trivial version of what kfit2 already does through c0 and the link. Nothing to adopt there.

---

## 1. What RCO optimizes, and with what inputs

**Problem** [Lit]:
- min_α L(α) subject to C(α) = Σ_i w_i ⟨softmax(α_i), c⟩ = B.
- L is the calibration KL vs the full-precision model, using cached reference log-probs (Eq. 6).
- Groups are linear layers (252 on Qwen3-8B). The options are GPTQ bitwidths 2–8, with c_k = bits and w_i = parameter fraction.

**One step (Algorithm 1; `optimize_projected_gumbel`)** [Lit, code]:
1. Draw Gumbel noise and form α̂ = (α + G)/τ.
2. **Exact MCKP DP** on α̂ under the budget gives a feasible hard assignment z.
3. Forward pass with the hard weights. In the backward pass, gradients flow through softmax(α̂) via the straight-through estimator (STE): `probs = hard − soft.detach() + soft`, and `W = W_ref + Σ_k probs_k·Δ_k`.
4. Average the gradient over g Gumbel samples.
5. Project the gradient onto the tangent plane of the budget surface. The normal n_ik = w_i p_ik (c_k − E_p[c]) is closed-form.
6. Take an Adam step.
7. Retract back onto the budget surface by binary search over a shift α + t·c. This is monotone, because dC/dt = Σ w_i Var_p[c] > 0.
8. Re-project Adam's first moment onto the new tangent plane.
9. At the end, solve the DP on log p once more to get the final assignment.

**Inputs** [Lit, code]:
- A per-layer database of dequantized fake-quant weights for every bitwidth (`<layer>/<bw>.pth`).
- The full HF model in torch, with gradient checkpointing.
- Calibration sequences (FineWeb-Edu, 256 × 2048).
- Reference log-probs for the KL: they report "~159 GB for 256 samples" of cached full-vocab log-probs.
- It needs **full backpropagation through the network to the α's** (gradient sanity check in `setup()`).
  - The `cpu_deltas` mode streams the deltas from CPU each forward.
  - A comment mentions a REINFORCE variant that does not need autograd. It is not in the paper's experiments.

**Default script** [code]: `--batch-size 1 --batches-per-step 1 --n-gumbel-samples 4 --n-steps 200`, i.e. **one 2,048-token sequence per step**, reused by all g samples. "Forward passes" = g·T forward+backward passes of 2k tokens, so 800 for g=4, T=200.

**Hyperparameter defaults** [Lit, Table 24]: g = 16–32, T = 50–200, τ_min = 0.01, lr = 0.1, no antithetic sampling (it hurt), no layer grouping (grouping cost 0.3–1.1 ppl).

## 2. Compute, and whether it can run for us

### 2.1 Reported compute [Lit]

**Search cost:**

| Configuration | Hardware | Time |
|---|---|---|
| Qwen3-8B, g=4, T=200 (800 fwd+bwd passes) | RTX 3090 | 62 min |
| Qwen3-8B, g=16, T=200 (3,200 passes) | RTX 3090 | 218 min |
| Qwen3-8B, g=32, T=50 (1,600 passes) | RTX 3090 | 110 min |
| EvoPress, 100 generations (matched protocol) | A6000 | 11–14 h |

**Per-pass cost:**
- Grouped Qwen3-8B on 4×A100: ~3 s of forward+backward per sample and ~14 s per step.
- The pure-Python DP takes 0.27–0.5 s per call at N=144.

**Model scale:**
- **No dense quantization experiment above 8B.**
- Larger models (Qwen3-30B-A3B, Qwen3-Coder-Next 80B-A3B) appear only in *expert pruning*, which has two options per expert and needs no deltas.
- Memory: the default per-GPU budget leaves room for "n_bw−1 delta copies plus ~2× activations". The deltas are bf16 copies the size of the full model for every non-reference option, unless `--cpu-deltas` is set.

### 2.2 Feasibility on 2× R9700 (32 GB), 30 GB host RAM [Own]

**Blockers as implemented:**
- **(a) No differentiable full-model forward.** Radiance is inference-only, and HF/torch GDN on gfx1201 is unproven. The `fla` Triton kernels on RDNA4 are uncertain; the pure-torch chunked delta rule is slow but differentiable.
- **(b) Memory.** The bf16 model is 54 GB; with deltas, (K−1) × 54 GB. With our K ≈ 10–25 that is 0.5–1.3 TB of bf16 deltas per mix, which can't be stored or streamed per step.
- **(c) Long context.** 2,048-token windows cannot see the long-trace positions (~35% of scored positions).

**What a workable redesign would look like** (none of this is in their code):
- Keep the *current mix* quantized on the GPU (≤17 GB at ≤5 bpw).
- Dequantize per layer on the fly for the forward and backward passes, using exllamav3 torch modules. This works for single layers on these cards (not measured here).
- Recompute layer by layer with checkpointing: hidden states are ~655 KB per token, about 5.4 GB per 8k tokens, so they fit on the card or in host RAM.
- In the backward pass, form the weight-gradient G_i = Σ_t g_t x_tᵀ per tensor and immediately contract it with the dequantized weights of each option: a_ik = ⟨G_i, W_i^(k)⟩. Then discard. This replaces K stored deltas with K transient dequantizations per tensor per pass.
- Option reads are still Σ_k size_k. A ±1-K trust region (~3 options) means ~40 GB read per pass; the full pool means several hundred GB.
- Reference log-probs: use the top-k (64–512) bf16 log-probs we already produce for evals. The KL gradient wrt the logits, q − p, is fine truncated.

**Cost:**
- Per pass at 2–4k tokens, I/O dominates. Estimate ~1–2 min.
- Full RCO at g=4, T=200 is therefore ~15–30 GPU-hours per *target*. We have ~20 targets. Feasible for 2–3 key tiers only.
- Engineering: ~1–2 weeks, with the ROCm GDN backward and the exllamav3 dequant path as the main risks.

**CPU-only:** a 27B backward on CPU is hours per pass with 30 GB RAM. Not viable.

### 2.3 What the STE gradient actually measures [Own derivation; check before relying on it]

**Derivation.** In `WeightInterpolation`, ∂L/∂soft_ik = ⟨∂L/∂W_i, Δ_ik⟩ =: a_ik, so ∂L/∂α_ik = (1/τ)·p̂_ik (a_ik − Σ_j p̂_ij a_ij).
- Take the second-order KL model, KL ≈ ½ δᵀFδ with δ = Σ_i J_i ε_i, and roughly independent rotated errors.
- Then a_i,z (the sampled option) ≈ δᵀF J_i ε_z ≈ 2·s_i(z) + cross terms.
- For an unsampled option k, a_ik ≈ cross terms only, ≈ 0 in expectation. Exceptions: options whose errors are correlated with ε_z (adjacent K of the same quantizer may be).
- So the signal mainly says "the sampled option costs ≈ its in-context contribution". It does *not* give s_i(k) for unsampled k. The true swap effect is s_i(k) − s_i(z); the first-order estimate is ≈ −2s_i(z). This is the STE bias the paper leaves "uncharacterized".

**Consequences:**
- RCO learns per-option costs by *sampling* every option over many steps. That is why g matters most in their ablation [Lit, F.1].
- For us, the robust and cheap part is the **in-context attribution of the current option**: A_i(z) = ⟨∂KL/∂W_i, W_i^(z) − W_i^bf16⟩ = Σ_t ⟨∂KL/∂y_t, (W_q − W_bf16) x_t⟩.
  - In the quadratic regime Σ_i A_i ≈ 2·KL.
  - **Integrated gradients** along W_bf16 + t·ε for t ∈ {¼, ½, ¾, 1} make the decomposition exact. KL(mix) = Σ_i ∫ A_i(t) dt is the Aumann–Shapley value, so compounding is shared out between tensors, not lost.
  - It costs 1–4 forward+backward passes per mix and needs only the current mix plus bf16 weights, *not* all options.

## 3. Discrete choices, budgets and option sets

- **Discreteness** [Lit]:
  - The softmax relaxation is used only for gradients.
  - Every forward pass uses a *feasible hard* assignment from the exact MCKP DP on the Gumbel-perturbed logits (perturb-and-MAP).
  - The final rounding is a DP on log p.
  - The budget is exact for the expected cost. Decoded models are feasible up to the integer cost grid (e.g. 2.251 vs 2.25 bits).
  - Inequality budgets use a slack variable s², and multiple budgets use a q×q projection plus a Newton retraction.
- **Option sets in the code** [code]:
  - α has shape (n_groups, n_bw), with one cost vector c shared by all groups, scaled by w_i. So every group has the same bitwidth ladder.
  - The reference is the highest bitwidth. Pruning is "bw 0".
  - Groups may tie several layers (as in fused k+v or gate+up).
- **Generalizing to our pool** [Interp/Own]:
  - The math generalizes directly to per-group cost vectors c_ik (bytes):
    - normal n_ik = p_ik (c_ik − E_{p_i}[c_i]);
    - the retraction shift α_ik + t·c_ik stays monotone, with derivative Σ_i Var_{p_i}[c_i] > 0;
    - variable K per tensor is handled by masking with −∞ logits.
  - Multiple quantizers at the same bytes, bf16, embedding and head are all fine as options. The embedding's gradient flows through its row lookups.
  - The code would need ~1 day of changes for this. It is not the hard part.

## 4. Reported gains [Lit]

**Qwen3-8B mixed precision** (252 groups; GPTQ 2–8 bit, same database for all methods; FineWeb-Edu / C4 ppl; FP16 = 10.96 / 17.20):

| bpw | RCO (best of 3 configs), FW | EvoPress, FW | IMPQ, FW | HIGGS-DP, FW |
|---|---|---|---|---|
| 2.25 | 20.47 | 21.40 | 24.18 | 32.07 |
| 2.5 | 15.43 | 15.64 | 18.79 | 20.08 |
| 3.5 | 11.41 | 11.50 | 12.31 | 11.65 |
| 4.0 | 11.14 | 11.16 | 11.42 | 11.28 |

- Wall time: RCO 62–215 min; EvoPress 11–14 h; IMPQ and HIGGS ~2 h.
- On C4 at 2.5 bpw, RCO's configurations land at 23.79–24.73 vs EvoPress 24.63, so g=4 is *worse* than EvoPress on C4.
- Seed standard deviation at 2.5 bpw is 0.18 FW ppl (range 0.41), and about 70 of 252 layers change assignment across seeds.

**Other results:**
- **Qwen1.5-MoE-A2.7B vs MxMoE** (per-layer ILP on layer-output sensitivities): Wikitext-2 ppl is 0.43 / 0.18 / 0.04 lower at 2.5 / 3.0 / 3.5 bpw.
- **Synthetic MCKP:** within 1% of the DP optimum, where a vanilla Lagrangian plateaus at a 37.6% gap (N=1000, K=32).
  - [Interp] This compares *gradient methods on a known additive objective*. It is irrelevant when you can run the DP directly, as we can.

**Interpretation** [Interp]:
- The gain over HIGGS and IMPQ is the "surrogate breaks at low bit" story: large at ≤2.5 bpw, ~2% at 3.5, ~1% at 4.0.
- The gain over measured-loss search (EvoPress) is 0–4%, mostly time savings.
- The evidence is all ppl on FineWeb/C4 with short 2k-token contexts, on one dense 8B model.

## 5. Fit to us, and recommendation

**Options compared (gains vs our current kfit2 loop are [Own] guesses):**

| Option | Engineering | Compute / evals | Expected gain vs our current kfit2 loop |
|---|---|---|---|
| **A. Full RCO** (torch fwd/bwd, all options, Gumbel+DP) | 1–2 weeks, high ROCm/GDN risk | ~15–30 GPU-h per tier, plus a few evals per tier to confirm | ≤3 bpw: 0–5%; ≥4 bpw: ~0–1%. Long-context blind |
| **B. RCO-style gradient attribution as surrogate data** (§2.3) | Spike ~1 day; full ~3–5 days (shares A's torch path minus the options) | 1–4 passes per mix (~minutes each); 0 extra evals to validate | Indirect: better local tangents/κ, so faster convergence of the model-based search; maybe +1–3% at ≤3 bpw |
| **C. Planned model-based local search** (Thompson + DP + antithetic pairs) | ~0 (already in `auto.sh`/`kfit2.py`) | ~100–250 evals | 2–5% at ≤3 bpw, ~1% at 4–5 (previous brief) |
| **D. RCO's Gumbel-DP on our surrogate** | trivial | 0 | **0**: the exact DP already optimizes the surrogate; Thompson sampling is the right randomization |
| **E. Zero-order RCO** (REINFORCE on measured KLD) | small | thousands of evals | not viable: variance with 337×K logits vs 20–30 evals/h |

**Recommendation** [Own]:
- **Do C now.** Do not adopt RCO as a whole (skip A unless B proves the torch path is cheap and the low-bpw tiers still lag).
- **B is the one idea worth testing.** Gate it in three stages:
  1. **Spike (1 day, no evals).**
     - On one card, compute A_i for one GDN layer, one attention layer and one MLP. Use exllamav3 dequant plus a torch forward/backward of that layer, with the output gradient taken from a finite-difference check.
     - If the full stack is unreachable on ROCm, stop.
  2. **Validation (fit pool only, no new evals).**
     - Compute integrated-gradient attributions on a fit-pool slice (golden-calA or fit2/3 docs, ≤8k caps) at 2–3 measured mixes: the sc4.00-e4 base, plus one 2.5-bpw and one 3.0-bpw knapsack mix.
     - Predict the ~100 already-measured band swaps around those bases, scaling attributions by (e_new/e_old)^q per family.
     - **Adopt** if it beats kfit2's leave-k-out *decision* metrics: pairwise ranking within 0.15 bpw and slope error.
  3. **Use.** Add the attributions to kfit2 as a prior mean, or as an extra likelihood term with its own variance, on log s_w at that operating point. Do not use them as a replacement. Then run C as planned.
     - Acceptance stays on the search pool via `paired.py`, with routine checks on verify slices and public only for the final 2–3 per size.
     - Attributions are "sensitivity measurements", so they live in the **fit pool**. Never compute them on verify or public.
- **If A is ever run:**
  - Optimize on the *fit* pool, using whole documents capped at 8k, not 2k chunks.
  - Treat RCO's output as just another candidate mix for C's search-pool acceptance.
  - It has ~337×K free parameters fitted to its calibration tokens, which is an overfitting risk exactly like QAD's.

## 6. SLQ (arXiv 2605.02404): anything beyond Shapley + ILP + EAR?

**What SLQ does** [Lit]:
- Multi-bitwidth Shapley runs one game per target bitwidth, with P permutations: O(P·M·|B|) forward passes, ~2.5 h per bitwidth or ~15 h total on an 8B with one A6000.
- Allocation is by ILP, with a binary search over the budget to meet an EAR or KL target.
- "Shapley-based allocation outperforms linear estimation at mid-range bitwidths (5.5–6.5 bits)."
- The full range {4..8} beats {4,8}: 4.98 vs 5.74 bpp at EAR ≥ 0.99 on Qwen3-32B.

**Items:**
- **EAR** [Lit]: mean over positions of Σ_{top-10} min(p, q) = 1 − TV, "robust to tail behavior".
  - [Own] Log it per row next to KLD; it is free in the eval kernel if we have top-k p and q.
  - Use it as a diagnostic: mixes with equal KLD but different EAR distinguish tail-driven KLD from token flips.
  - Do **not** optimize it. The private score is mean KLD.
- **Single-point calibration** [Lit]: ρ = KL_actual / KL_pred at one anchor, applied to all predictions, plus a 2× guardrail to re-anchor.
  - [Interp] This is a one-parameter version of kfit2's c0 and link. Our underprediction of 13–48% below 2.75 bpw is exactly the case their guardrail exists for: "3-bit [anchors] not recommended".
  - [Own] The useful takeaway is procedural: a *local* ratio correction from the nearest measured mix, as a kriging-style residual (long-tail item 6 of the previous brief), is cheap and reduces calibration error. It does not change rankings within a tier.
- **Recovery ≈ 1 − α·KL** in the near-lossless regime [Lit]. Irrelevant to KLD scoring.
- **The γ² symmetric vs asymmetric variance law** [Lit]. Irrelevant: it is a property of scalar quantizers, not of EXL3 trellis + Hadamard.
- **Qwen3.5-27B task-lossless at 3.30 bpp** [Lit]. Bits are averaged over a subset (MLP + linear-attention projections), with the rest in bf16, so it is not comparable to our whole-model bpw.
- **[Own] The better Shapley.** The integrated-gradient attribution of §2.3 gives an Aumann–Shapley decomposition of a mix's KLD from 3–4 forward+backward passes, vs SLQ's P·M·|B| forward passes. If B's infrastructure exists, that is the cheap way to get "in-context, interaction-aware" per-tensor costs.

## Contradictions

- **RCO's "matches or exceeds EvoPress"** holds for FW ppl at all four budgets. On C4 at 2.5 bpw the g=4 configuration is worse (24.73 vs 24.63), and the 2.5-bpw FW lead is about one seed SD.
- **The paper calls the surrogates "failing at high compression".** Yet HIGGS-DP is within 2% at 3.5 bpw and 1.2% at 4.0 bpw, and at 3.5 bpw HIGGS (11.65) beats IMPQ (12.31) even though IMPQ models interactions.
- **SLQ calls gradient-based search (RCO) "more accurate"** but uses Shapley + ILP for every result. Neither paper compares the two directly.

## Missing evidence

- No RCO result on a dense model above 8B, on hybrid GDN architectures, on long context, or with heterogeneous option pools (several quantizers, bf16, embedding, head).
- My analysis of the STE signal (§2.3) and the claim that integrated gradients are a usable Shapley substitute are unvalidated. In particular, I don't know:
  - how correlated EXL3 errors are across adjacent K;
  - how noisy per-tensor attributions are on ~50–100k positions.
- ROCm feasibility is unknown: the GDN backward kernels on gfx1201 and the exllamav3 dequant inside autograd.
- All compute numbers for us (§2.2) are estimates. No profiling was done here.

## Sources

**Kept:**
- RCO paper ([arXiv 2605.00649 v3](https://arxiv.org/abs/2605.00649), [HTML](https://arxiv.org/html/2605.00649v3), [PDF](https://arxiv.org/pdf/2605.00649v3)): method, Tables 2, 3, 4, 12, 14–24.
- RCO code ([github.com/IST-DASLab/RCO](https://github.com/IST-DASLab/RCO)): STE/interpolation mechanics, defaults, memory layout, shared cost vector.
- SLQ ([arXiv 2605.02404](https://arxiv.org/html/2605.02404)): EAR, Shapley cost, single-point calibration.
- Local: `research/mix-allocation.md`, `search/kfit2.py`, `search/auto/auto.sh`.

**Deprioritized:** EvoESAP and MxMoE details (pruning and MoE settings, not our problem).

## Next steps

1. Keep running C, the `auto.sh` Thompson+DP loop. Nothing in RCO argues for pausing it.
2. Optional: the one-day ROCm spike for single-layer forward/backward attribution (§5 B.1). Then validate against existing fit-pool band swaps before any loop integration.
3. Add an EAR column to the eval outputs if the eval kernel exposes top-k p and q cheaply.
