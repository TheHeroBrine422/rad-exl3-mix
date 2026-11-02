# Research: bit allocation / mix selection for the EXL3 entry (Qwen3.8-27B)

> **Status (2026-11-02).** Research snapshot from 2026-10-30, kept as written. What became of its top 5
> (numbers = `../experiments/INDEX.md`):
> - R2 exact byte-bucket DP solver: **adopted** (#69).
> - R4 model-based local search (Thompson + DP): **adopted** in V2's streaming search (#70).
> - R3 low-bpw anchor data: done as targeted K1/K2 swaps (#71).
> - R1 decision-focused CV and R5 category / position-aware objective: not tracked as experiments; KLD vs position
>   was checked (#52). The black-box surrogate and residual re-ranker ideas were **negative** (#18, #20).

*Date: 2026-10-30.*

**Scope.** This brief covers:
- the model that predicts the KLD of a mix of per-tensor options;
- the optimizer that picks one option per tensor under a size budget;
- the choice of which mixes to measure next.

It does not repeat the quantizer survey in `research/quant-sota.md`.

**Labels.**
- **[Lit]**: the source says it.
- **[Interp]**: my reading of a source.
- **[Own]**: my own reasoning or speculation, with no paper behind it.

---

## 0. Executive summary

1. **[Own] Our current "depth compounding" term is not a depth term.**
   - In `kfit.py`, Σ_w cost_w·(1 + c·U_w), with U_w = Σ_{v in earlier layers} cost_v, equals X + c·Σ_{v<w, different layers} cost_v·cost_w. That is ≈ X + (c/2)(X² − Σ_l X_l²).
   - The pair product is symmetric, so layer order never enters it. With 64 layers, Σ_l X_l² ≪ X², so the term is almost exactly a **global quadratic link** on X. It does the same job as `g ∈ {mult, pow}`.
   - Its re-linearised marginal cost is cost_w·(1 + c(X − X_layer(w))) ≈ cost_w·(1 + cX). That is a near-uniform multiplier, which a Lagrangian solver absorbs into λ.
   - **Consequence:**
     - The decisions of every model form we fit are essentially those of the additive model cost_w = s_w·e_b·(e/e_b)^q·famfactor.
     - The CV gain from 0.28 to 0.19 is mostly *calibration*: it gets the absolute KLD right.
     - It is not evidence that the *ranking* of mixes improved.
     - The 20–30% underprediction at 2.0–2.5 bpw is likewise mostly a link-function problem. It matters for decisions only if the *relative* sensitivities shift at low rate.
   - **Check first:** solve the knapsack with and without the depth term at 2.0, 3.0 and 4.0 bpw. I expect nearly identical mixes.
2. **Fix the predictor where decisions are made, not where log-RMSE is.**
   - Add a *directional*, regime-dependent amplification model: each layer's error is amplified by the error *downstream* of it.
   - Add low-bpw anchor measurements.
   - Evaluate models by **decision-focused CV**: pairwise ranking of measured mixes at similar bpw, and frontier regret. Do not evaluate them by global log-RMSE.
   - [Own] The product-form model still admits an **exact DP** (§4.2), so we lose nothing on the optimizer side.
3. **Change the measurement design from one-group swaps to budget-neutral perturbations around the frontier mixes.**
   - Use random multi-tensor ± K-step swaps, Thompson samples from the model posterior, and SPSA-style antithetic pairs.
   - Every local-search candidate then also becomes a regression row near the frontier. This merges EvoPress-style search and refitting into one loop: model-based local search with the exact knapsack as the acquisition optimizer.
   - [Lit] SPDY found the same pattern for pruning: searching over the *coefficients of a sensitivity model with the DP inside the loop* needed about 10× fewer evaluations than searching over configurations directly.
4. **Replace the Lagrangian sweep with an exact byte-bucket DP.**
   - It is cheap: about 340 items × about 20 options × about 10⁴ buckets.
   - It removes hull slack at tier edges and supports constraints such as min-K, trust regions and fused ties.
   - Keep the Lagrangian only for drawing the frontier.
5. **Use per-document scores to quantify uncertainty, not to inflate data.**
   - Per-document KLD gives parameter uncertainty via a document bootstrap, including slice-sampling error.
   - It also gives category- and position-specific sensitivities, so the slice can be reweighted toward the corpus mix (long traces are about 35% of scored positions).
   - It does not multiply the information about the population-mean objective much. The regressors (the mix) are identical across documents.

**Expected gain** [Own]:
- **3–8% KLD at ≤3 bpw**, mainly from removing the +3..+7% misses and from the low-bpw regime.
- **1–3% at 4–5.5 bpw.**
- About **0 at ≥6 bpw**, where the head and option pool dominate.

**Cost:** about 1 day of engineering plus about 150–250 evals spread over 3–4 days.

---

## 1. Literature survey (allocation only)

### 1.1 Additive sensitivity + knapsack/ILP (the mainstream)

| Work | Predictor | Solver | Evidence on additivity / numbers |
|---|---|---|---|
| **HAWQ-V2** ([1911.03852](https://arxiv.org/abs/1911.03852)) / **HAWQ-V3** ([2011.10680](https://arxiv.org/abs/2011.10680)) | Sensitivity = average Hessian trace × ‖Q(W)−W‖² (V2: "the right sensitivity metric is the average Hessian trace") [Lit, abstract] | V2: Pareto frontier. V3: ILP with memory/latency constraints [Lit] | CNNs; INT4/8 only. Origin of "trace × perturbation, then ILP" [Lit] |
| **FIT** ([2210.08502](https://arxiv.org/abs/2210.08502)) | Fisher information trace combined with a model of quantization noise [Lit, abstract] | Ranking | Argues Fisher (an output-loss quantity) is cheaper and as good as Hessian [Interp] |
| **HIGGS linearity theorem** ([2411.17525](https://arxiv.org/abs/2411.17525)) | E[PPL] ≈ PPL* + Σ_l α_l t_l², with t_l² the relative weight MSE. α_l comes from Gaussian noise injection, one layer at a time, **J = 15 noise levels**, 287k tokens [Lit] | "Knapsack-style DP", solved with **CP-SAT in seconds** [Lit] | "predicted perplexity is close to the real perplexity on relatively higher bitwidths (b>3.0) and diverges on lower bitwidths" [Lit]. Assumes the perplexity Hessian is near block-diagonal with a scalar per layer (checked only on OPT-125M) [Lit]. Data-free variant uses KL on random tokens [Lit] |
| **Q-Palette** ([2509.20214](https://arxiv.org/abs/2509.20214)) | Linearity theorem. a_l fitted from the **KL increase** under noise norms √i/16, i = 1..16, on 128K self-generated tokens (16·L evals). Optional "actual" variant: measured single-layer replacement loss on 256K tokens [Lit] | MCKP as ILP, SCIP with a 60 s limit; fusion-aware variables [Lit] | Llama-2-7B Wiki2: "linearity" coefficients 6.69 vs "actual" swap-measured 6.47 at 2 bit, 5.38 vs 5.34 at 3 bit [Lit]. **Measured swaps beat noise-fitted coefficients at low bit** [Interp]. MSQ vs uniform TCQ (Llama-3.1-8B): 6.78→6.28 at 3.00 b, 5.92→5.81 at 4.00 b [Lit] |
| **exllamav3 optimizer** ([optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md)) | kld(t,K) = S_t·rfn_t(K)^α. S_t comes from **LDLQ-shaped** noise injection into the bf16 model, suffix forward, KLD vs clean. α = 2 recommended (fits 1.7–1.9). α_low ≈ 2.35 for K2→K1 **on Qwen3.8-27B** [Lit] | Exact greedy (convex curves) plus an exchange-repair pass; fused ties k+v and gate+up [Lit] | Shaped noise: **Spearman 0.84** vs ground-truth swap attribution; unshaped 0.61. Isotropic noise "overestimates sensitivity by up to ~6x" on anisotropic tensors (v_proj) [Lit]. Sensitivities from self-sampled traces transfer better than wikitext [Lit]. About 35 min for 2 levels on a 27B [Lit] |
| **IQR-V2** (IQR-V2 technical report) | 24 units (8 bands × {gate_up, down, attn+GDN}). Unit KL share = band's measured share × weight fraction, since "projection types are about equally sensitive per weight". Shares measured from **whole-trunk bases at each k** [Lit] | Exact merge of the units' (bytes, KL) Pareto fronts [Lit] | Chosen mixes use 2–3 adjacent widths, widest at the last band [Lit]. IQR-V1's band-additive model predicted final mixes within 1–7% [Lit via our earlier survey]. ~50 candidates scored on public [Lit] |
| **LLM-MQ / MixLLM / SqueezeLLM / OWQ / SliM-LLM / CMPQ** (survey [2510.16805](https://arxiv.org/abs/2510.16805); [2412.14590](https://arxiv.org/abs/2412.14590); [2405.14917](https://arxiv.org/abs/2405.14917); [2410.13056](https://arxiv.org/abs/2410.13056)) | First-order Taylor (LLM-MQ); 2nd-order Fisher saliency sorted **globally over output channels** (MixLLM); diagonal Fisher (SqueezeLLM); Hessian-diagonal weak columns (OWQ); Hessian salience per 128-col group within a layer (SliM-LLM); activation norms (CMPQ) [Lit via survey] | ILP (LLM-MQ), global sort (MixLLM), double-pointer / top-k / thresholds [Lit] | The survey says the field "reveal[s] a clear convergence toward heuristic-based methods" because RL and metaheuristic search are too costly at LLM scale [Lit]. MixLLM: "10% more bits" cuts the Llama-3.1-70B ppl increase from ~0.5 to <0.2 [Lit, abstract]. These are intra-tensor granularities; EXL3 cannot express them except by splitting tensors [Interp] |

### 1.2 Non-additive models and measured-loss search

| Work | Idea | Numbers / caveats |
|---|---|---|
| **EvoPress** ([2410.14649](https://arxiv.org/abs/2410.14649)) | (1+λ) evolutionary search. Level-switch mutation preserves the budget, swapping only between equal-size projections. KL fitness. Multi-stage selection: each stage uses more tokens and keeps fewer survivors; the final stage uses ~20–50× the first stage's tokens; λ=64 offspring scored "on less than a single sample" in stage 1 [Lit] | "around 64K tokens… are sufficient". For quantization, "KL-Divergence yielded slightly better results" than ppl. Most of the gain comes in ~2 GPU-h; converged at ~8 GPU-h (8B). Levels 2..6 bit, mostly 2/3/4 used. Llama-3-8B 3 bit: +4.1 zero-shot avg over uniform [Lit]. Theorem: for *linear* fitness, O(k(n−k)/λ) generations (noise-free) [Lit]. The non-additivity evidence is mostly from pruning; for quantization the paper only asserts the "below 3 bits" breakdown [Lit/Interp] |
| **SPDY** ([2201.13096](https://arxiv.org/abs/2201.13096)) | Per-layer error model e_ℓ = c_ℓ·(level)², with **one learned scalar c_ℓ per layer**. A DP picks the profile for any c, so the budget is enforced inside. Neighborhood-shrinking random search over c (100 random starts, resample ⌈0.1L⌉ coordinates, shrink); fitness = calibration loss of the stitched model [Lit] | The reparametrized search "finds profiles with lower calibration loss… with much fewer evaluations (≈10× here)" than direct search; a GA with 2.5× more iterations was worse. Learned c "integrates global information, e.g. if two consecutive layers should not both be heavily pruned" [Lit]. Pruning, CNN/BERT: SPDY 71.17 vs best fixed metric 69.26 (RN50 2×) [Lit] |
| **AMQ** ([2509.12019](https://arxiv.org/abs/2509.12019), EMNLP'25) | Layer bits {2,3,4}. Search space pruned: layers with sensitivity >2× median fixed to 4 bit (0.45–2.1% of layers). Precomputed HQQ layers stitched together. **RBF surrogate** on JSD. NSGA-II. 50 real evals per iteration [Lit] | **250 initial + 200 iterations × 50 = 10,250 real evals** (7B, 5 GPU-h). RBF ≈ MLP; GP tried, "negligible gains" or slower. 7B 3-bit avg acc: AMQ 65.59 vs one-shot sensitivity 61.35 vs greedy 62.57 [Lit]. Outlier layers "destabilize predictor training" [Lit] |
| **IMPQ / CoopQ** ([2509.15455](https://arxiv.org/abs/2509.15455); ACL Findings 2026) | Shapley values from 100 random permutations of 4→2-bit drops. Pairwise interaction K = shrunk covariance of marginal contributions. Binary QP linearised to a MILP, solved with SCIP [Lit] | ~18 h on an A40 for 8B. Claims 20–80% lower ppl than **weak baselines** (first-order/LIM/Z-score). Interaction ablation: average ppl 2.83e3 (full) / **2.79e3** (α=0.5) / 2.81e3 (diagonal only), i.e. **interactions are worth ~1–1.5%**; the authors call it "relatively flat" [Lit] |
| **QEP** ([2504.09629](https://arxiv.org/abs/2504.09629), NeurIPS'25) | Error growth across layers is the main low-bit failure of layer-wise PTQ; propagate and compensate [Lit, abstract] | A quantizer-side fix, not allocation. Confirms compounding is a low-bit phenomenon [Interp] |
| **Our GDN source** ([2609.04098](https://arxiv.org/abs/2609.04098), via our earlier survey) | Qwen3.8-27B: projection errors **add in quadrature** within a GDN layer (19.4% predicted vs 19.2% measured) [Lit] | Supports additivity *within a layer*; non-additivity has to come from *depth* [Interp] |

### 1.3 Surrogate-model / BO / design literature (not LLM-specific)

- **BOCS** ([1806.08838](https://arxiv.org/abs/1806.08838)) [Lit]:
  - Surrogate: a 2nd-order polynomial in binary variables with a horseshoe prior, sampled by Gibbs.
  - Acquisition: Thompson sampling, optimized by SDP relaxation or simulated annealing.
  - Scale tested: d ≤ 25 binaries. The SDP "will scale easily to a few hundred dimensions".
  - An MLE point estimate instead of posterior samples "plateaus" because it is purely exploitative.
  - Categorical variables are one-hot encoded, with SA on the original domain.
  - [Interp] The structure is right for us: a parametric model, a sparse interaction prior and Thompson sampling. But our d (~340 tensors × ~20 options) is far larger, and our objective already has strong physical structure. Use the *recipe*, not the package.
- **CASMOPOLITAN** ([2102.07188](https://arxiv.org/abs/2102.07188)) [Lit, abstract]: trust-region local BO plus a categorical kernel, with a convergence guarantee, for high-dimensional categorical spaces. [Interp] The trust-region idea (only trust the model near measured points) is the transferable part.
- **BODi** ([2303.01774](https://arxiv.org/abs/2303.01774)) [Lit, abstract]: dictionary embeddings so that continuous-space GPs work on high-dimensional binary or categorical spaces.
- **Knowledge gradient with linear beliefs** ([Negoescu, Frazier & Powell 2011](https://optimallearning.princeton.edu/Papers/Negoescu%20Frazier%20Powell_drug_discovery_JOC_August2_2011.pdf)) [Lit]: sequential experiment selection under a **Free–Wilson** (additive substituent) model. It picks the measurement that most improves the *expected value of the best decision*, not the model's accuracy. [Interp] Our mix model is a Free–Wilson model: tensor = position, option = substituent. This is the closest classical answer to "how to choose measurements to improve the frontier".
- **Bayesian Bits / BOMP-NAS / TPE in MASE** [Lit, via survey/abstracts]: gradient or BO searches for small networks. They do not scale to our evaluation cost. Not used.

### 1.4 What the literature says about additivity: synthesis

**What the sources state** [Lit]:
- Additivity is good above ~3 bpw: HIGGS, the IQR band model at 1–7% error, and the Q-Palette proxy ablation.
- It degrades below ~3 bpw: HIGGS, EvoPress (assertion), IMPQ, QEP.
- Within a layer it holds very well on our model: GDN quadrature.

**The fixes in use:**
1. Measure in context: IQR's per-k bases, Q-Palette's "actual" swap, which is 3% better ppl at 2 bit than noise-fitted.
2. Measured-loss search: EvoPress, AMQ, SPDY.
3. Pairwise interactions: IMPQ. The measured effect is small, ~1–1.5%.

**[Own] Theory for why additivity breaks the way it does.**
- Second order: KL ≈ ½·E[δzᵀ F δz], with δz = Σ_w J_w ε_w.
- With random rotations, the per-tensor errors ε_w are close to independent and zero-mean. So the cross terms vanish *in expectation*, and the 2nd-order KL is additive. This explains the GDN quadrature result and the good additivity at ≥4 bpw.
- Non-additivity enters at the next order:
  - **(a) Jacobian drift.** The J_m of a downstream layer are evaluated on already-perturbed activations, so upstream error × downstream error gives an amplification term. It is *directional*: an earlier error gets amplified by later layers, not the reverse.
  - **(b) KL leaves the quadratic regime** on tail positions, where per-token KL is heavy-tailed and top-1 flips.
- Our 20–30% *under*prediction at 2.0–2.5 bpw says (a) dominates, i.e. the total is superadditive.
- A symmetric product (kfit's term) or a global link captures the *size* of (a). Only a directional or regime-dependent model can capture its *allocation consequence*: when downstream layers are cheap, upstream errors cost more.

---

## 2. Better regression / surrogate models (Q2)

**What the data can support** [Own]:
- 100–300 measurements and ~340 tensors.
- The parametric core must stay low-dimensional: a smooth layer profile per group, plus a handful of shape parameters.
- Anything per tensor must be a prior-shrunk deviation.

**Recommended model, "M2"** [Own]:

```
cost_w(o)  = s_w · e_w(o)^q_g · φ_fam(o)                      (proxy -> cost; q per group, family factor per option family)
log s_w    = f_g(layer) + δ_w,   f_g ~ GP(layer; Matérn-5/2, ℓ≈6–10 layers) + shared cross-group GP;  δ_w ~ N(0, τ²) small
X_l        = Σ_{w in layer l} cost_w                         (within-layer additive: GDN quadrature evidence)
KLD_pred   = g( c0 + emb + head + Σ_l X_l · Π_{m>l} (1 + κ(m) X_m) )      κ(m) smooth in m (few knots), ≥ 0
log y_obs  = log KLD_pred + slice_effect + ε,   ε ~ N(0, σ_model²)
```

**Model details:**
- **Directional product.** It is the minimal model in which upstream error is amplified by *downstream* degradation. It reduces to the additive model when κ = 0, and it admits an exact DP (§4.2).
- **Identification of κ(m).** κ(m) needs measurements whose depth profiles differ: low-bpw anchors, and mixes with cheap early but expensive late layers, or the reverse. The current band swaps around sc4.00 barely identify it.
- **Regime dependence (alternative, cheaper).** log s_w = f_g(layer) + β_g(layer)·log X_total. The sensitivity profile then tilts with total degradation, a HIGGS-style regime effect. Test it against κ by decision-focused CV.
- **GP-in-layer prior instead of knots every 8.**
  - It gives per-layer flexibility where the data supports it (first/last layers, the full-attention layers every 4th) without the knot-placement artifacts.
  - Fit it by MAP plus Laplace. With ~40–80 effective parameters this is fast.
  - Add an explicit **full-attention vs GDN** indicator. The attention/SSM groups are already separate, so this only matters for ffn at l%4==3.
- **Family factor per (family × group)**, not global. The plain/SC/ours difference likely depends on how each quantizer's error aligns with the true output sensitivity, which differs by tensor type [Own].
- **Slice random effect.** One log offset per golden slice lets us combine measurements from several fit-pool slices. It also detects slice overfitting.

**Target choice:**
- Fit in **log KLD** (multiplicative errors; the paired SE is relative, ~1.5%). This is the right choice for the *likelihood*.
- For the *decision*, the target is mean KLD. Since g is monotone and the budget is fixed, the scale is irrelevant to the argmin. It matters only for risk penalties and cross-size comparisons.
- **Better still: fit paired contrasts.** Use log(KLD_mix/KLD_base-on-same-slice) where possible. Common slice effects then cancel exactly [Own].

**Interactions** [Interp of IMPQ + BOCS; Own]:
- Prefer **structured** interactions over free pairwise terms. Free pairwise terms need O(d²) parameters, and IMPQ shows they are worth ~1%.
- Directional depth amplification is structured.
- **Adjacent-layer pairs** with a horseshoe prior keep the DP exact (the chain structure is preserved; §4.2).
- **Factorization machines** (low-rank pairwise, rank 1–2 over bands) are a fallback if residuals show band×band structure.

**Uncertainty, and how to use it** [Own]:
- **Posterior** from Laplace, a parametric bootstrap, or better a **per-document bootstrap**: resample documents, recompute every measurement's slice-KLD from the stored per-row scores, refit. This captures slice-sampling error, which is what makes a +3% "win" vanish on verify.
- **Thompson sampling.** Draw θ, solve the knapsack exactly, measure that mix. This is the BOCS recipe with the knapsack as the acquisition optimizer.
- **Risk-aware selection for shipping.** Minimize the mean plus κ_r·sd of the predicted log-KLD difference vs the incumbent (or its CVaR over posterior draws). This penalizes mixes far from any measured point. It is the cheap fix for the "+3..+7% worse than a reference container" failures.

**Per-document / per-category / per-position data** [Own]:
- **What it gives:**
  - (i) Parameter uncertainty that includes slice sampling.
  - (ii) Category-specific sensitivity profiles (code, chat, agentic, multilingual, math), so the objective can be reweighted to calib's category mix.
  - (iii) Position-bucket profiles, ≤8k only, because the slices cap documents at 8k.
  - (iv) Heteroscedastic weights.
- **What it doesn't give:** many independent replicates of the *mix effect*. Every document sees the same mix, so for population-mean parameters the effective N stays ≈ the number of distinct mixes. What it does reduce is the slice-level noise in each measurement, which is already small in-slice.
- **Important:**
  - An eval is **deterministic** on a fixed slice. So the in-sample residuals (0.07) and CV residuals (0.19) are *model misspecification*, not noise.
  - Measurement noise only enters when generalizing to other documents, at ~1.5%.
  - Model error is therefore 5–10× the noise. Improving the model is worth far more than measuring more precisely, until CV error falls to ~0.03 [Own].

**Model selection by decision-focused CV** [Own; cf. "predict-then-optimize" / SPO, [1710.08005](https://arxiv.org/abs/1710.08005)]:
- Replace leave-k-out log-RMSE with:
  1. **Pairwise ranking accuracy** of held-out measured mixes within ±0.15 bpw of each other, plus mixes vs the interpolated reference-container curve.
  2. **Local slope error.** d log KLD / d bpw along the frontier, predicted vs measured on knapsack-mix pairs.
  3. **Frontier regret.** Refit without the round-r measurements, solve, and compare the round-r measured KLD to the best measured mix of that size.
- These are what the knapsack consumes. Global log-RMSE is dominated by large K=2 band swaps far from any frontier.

---

## 3. Better proxies (Q3)

**1. [Lit] Input-side tr(dW H dW^T) is HAWQ/GPTQ-style. It misses output-side weighting:**
- YAQA's H_O ([2505.22988](https://arxiv.org/abs/2505.22988));
- GuidedQuant's end-loss-gradient-weighted output groups ([2505.07004](https://arxiv.org/abs/2505.07004));
- the Fisher of FIT and SqueezeLLM.

[Interp] For *allocation*, the per-tensor scalar s_w already absorbs the average output weighting. The output side matters only when **two options for the same tensor distribute their error differently across output rows or directions**. That is exactly our SC vs plain vs ours family factor (plain costs 1.23× SC per unit proxy). A per-(family × group) factor is the cheap patch. A real output-side proxy needs the O8 backward infrastructure (see `research/quant-sota.md`).

**2. [Lit] Noise injection is the literature's cheap calibration** (HIGGS J=15; Q-Palette 16 levels; exllamav3 2 shaped levels, ~35 min for a 27B).
- **Caveats:**
  - [Lit] Isotropic noise overestimates by up to ~6×. Shaped (LDLQ-distributed) noise reaches Spearman 0.84 vs real swaps. Q-Palette's measured "actual" swaps beat noise-linearity coefficients at 2 bit.
  - [Own] For us, noise injection needs a bf16 or near-bf16 forward with in-place weight edits. Radiance runs encoded EXL3 containers, and bf16 (54 GB) does not fit on one 32 GB card. So it is not cheaper than real swaps unless we build it on top of O8 or TP2.
  - [Own] We have something better than noise: **real swaps are reflinks (0.2 s)**. Our proxy calibration problem is a *design* problem (which swaps), not a cost problem.
- **[Own] Use exllamav3's `sc_measure` output only as an extra *feature*:** a prior mean for log s_w. Run it on a 6–8 bpw container instead of bf16 (shifted reference, small bias) if we can host it. It is not a replacement for measured mixes.

**3. [Own] Random-design screening instead of one-at-a-time swaps.**
- A single tensor's one-K-step effect is ~0.1–0.5% of total KLD, below the 1.5% generalization SE.
- Group swaps (our bands) raise the signal but confound tensors and induce large perturbations.
- **Random balanced designs** (Plackett–Burman-like, each eval flipping a random ~30–50% of tensors by ±1 K at a fixed budget) estimate *all* main effects at once with variance ∝ σ²/N. Combined with the GP-in-layer prior, ~40–60 evals pin down the profile around one operating point.
- Because these points sit at the operating point the knapsack cares about, the coefficients are the *local tangents*. This is what re-linearisation (O5 in the earlier brief) was trying to get.

**4. [Own] Antithetic pairs (SPSA).** Measure mix+Δ and mix−Δ, where Δ is a random budget-neutral swap set. The difference estimates the directional derivative along Δ for all tensors at once. The sum minus 2·mix estimates curvature, which is a direct test of local additivity. Two evals per direction; cheaper than the one-at-a-time route.

**5. [Own] Depth structure.**
- Fit the per-band κ(m) of §2 from anchors at two or three operating points.
- **Prediction to test:** at low bpw the optimal allocation shifts bits **toward early layers** if κ is large in the middle and late stack, or toward the last band if the head-adjacent layers amplify little.
- IQR's "widest at the last band" is an additive-model outcome at their bases.

**6. [Own] Head and embedding.**
- These are already measured directly as discrete items: keep that.
- The head's analytic output Fisher ≈ diag(E[p_v]) ⊗ E[hhᵀ] (earlier brief O1) is the only place where an output-side proxy is free.

---

## 4. Optimizer (Q4)

### 4.1 Is Lagrangian MCKP + local search right?

**What is right:**
- [Interp] The structure is right and matches HIGGS, Q-Palette, exllamav3 and IQR.
- With a *separable convex* model, the Lagrangian sweep is exact at hull points.

**Three issues** [Own]:
1. **Hull slack.** A target between hull points falls back to the hull point below and leaves bytes unused. With ~340 fine-grained items the slack is usually small (≪0.01 bpw), but not guaranteed. Exact DP removes it.
2. **Non-convex per-tensor curves.** Option sets mixing families (SC/plain/ours/bf16) are not convex in (bytes, cost). The Lagrangian silently skips non-hull options that a DP would use at specific budgets.
3. **Non-separable terms.** Re-linearising a symmetric quadratic does nothing (§0). A directional term needs a real solver: the DP below.

### 4.2 Exact DP for the directional model [Own]

**Inner step.**
- Per layer l, an MCKP DP over its 3–5 tensors gives m_l(b) = the minimal X_l for byte budget b.
- Fused tensors count as one item.
- Bucket at ~1–4 MB.

**Outer step.**
- The total is P_L, with P_l = P_{l−1}·(1 + κ_l·X_l) + X_l. This is increasing in both P_{l−1} and X_l, so optimal substructure holds exactly:
  - V_l(B) = min_{b ≤ B} [ V_{l−1}(B − b)·(1 + κ_l·m_l(b)) + m_l(b) ].
- Then add the head, embedding and fixed bytes as a last stage.

**Cost.** 64 × B_total × B_layer ≈ 64 × 10⁴ × 10³ ≈ 6×10⁸ elementary operations. That is seconds in numba or a vectorised numpy min-plus.

**Extensions:**
- **Adjacent-layer interactions:** carry the previous layer's choice class (e.g. its K bucket, 3–5 classes) in the state.
- **Trust region:** bound the Hamming distance in K-steps from the incumbent by adding a "steps used" dimension, or by restricting options to ±1–2 K per tensor.
- **Min-K floors.**

**Check:** for κ=0 the DP must reproduce the Lagrangian on hull targets. Use this as the unit test.

**Alternative:** an ILP or MIQP in HiGHS / CP-SAT / SCIP. HIGGS and Q-Palette solve in seconds to 60 s. Fine for separable plus sparse pairwise terms. The product model is awkward in an ILP; the DP is cleaner.

### 4.3 Local search after the solve

- [Lit] EvoPress-style (1+λ) search with budget-preserving level switches is the standard measured-fitness polish. Its convergence theorem assumes noise-free *linear* fitness.
- **[Own] Make it model-based.** Each round:
  1. Refit the posterior on all fit-pool data.
  2. Propose candidates for each target size:
     - (a) the posterior-mean DP solution;
     - (b) 2–4 Thompson-sample DP solutions;
     - (c) 1–2 antithetic random perturbations of the incumbent.
  3. Measure them on a fit-pool slice.
  4. Every candidate, accepted or not, becomes a regression row.
- **This differs from EvoPress, where rejected offspring are wasted.** Here they sharpen exactly the local coefficients that pick the next candidate.
- Acceptance into the *shipped set* happens only on the search pool (§5).

### 4.4 Choosing measurements for frontier improvement (not model accuracy)

**Policies, in order of cost** [Own, after KG/BOCS]:
1. **Thompson sampling** (above). It explores where the posterior disagrees about the *argmin*, and it is free given a sampler.
2. **Disagreement sampling.** Among the solutions of 20–50 posterior draws, measure the mixes most often optimal, plus the one with the largest predicted-KLD spread.
3. **Knowledge gradient (Free–Wilson KG).** For a candidate measurement x, the expected decrease in min predicted KLD at each target after a Bayesian update. With a Gaussian linear posterior (κ fixed) the update is closed-form and the DP gives the min. Score ~50 candidates per round this way. This matters more if evals get scarce.

**Allocation across sizes:** spend evals where (frontier slope × posterior sd) is largest. That is the ≤3 bpw tiers, where KLD changes fastest per byte and the model is least certain.

---

## 5. Top-5 recommendations (ranked)

All tuning on the fit pool, acceptance on the search pool, routine checks on the verify pool, public only for the final 2–3 per size.

### R1. Fix the model's decision structure and evaluate it by decision-focused CV

**Content:**
- Confirm §0: compare knapsack solutions with and without the depth term, and with lin vs pow.
- Implement M2 (§2): GP-in-layer, family × group factors, a directional κ(m) *or* regime-tilted s_w, and a slice random effect.
- Add a per-document bootstrap posterior, using the stored per-row scores.
- Replace CV log-RMSE with pairwise ranking, local slope error and frontier regret.

**Cost:** ~0.5–1 day of code, **0 evals**. It refits on the existing 106 measurements.

**Expected:** no KLD gain on its own. It tells us whether the model ranks mixes right, and it gives the uncertainty that R3 and R5 need.

**Validation:**
- Leave-one-round-out on the 13 knapsack mixes and 8 containers.
- Ranking accuracy target: ≥80% on pairs separated by >3%.

### R2. Exact byte-bucket DP solver with the directional term, trust region and floors (§4.2)

**Cost:** ~0.5 day, 0 evals.

**Expected [Own]:**
- 0–2% from removing hull slack and non-hull options.
- More importantly, it makes R1's directional model *usable*: the knapsack actually reacts to κ.
- The trust region (±1–2 K vs the nearest measured mix) prevents the +3..+7% blowups.

**Validation:** for κ=0 it reproduces the Lagrangian; then compare predicted gains vs measured gains in R4.

### R3. Low-bpw and multi-operating-point anchor data

**Measurement:**
- Measure the remaining whole containers (all 16, about 8 more evals).
- Run a **random balanced design** around two low-rate operating points, the current 2.0 and 2.5 knapsack mixes: ~20 evals each.
  - Budget-neutral random ±1 K flips on ~40% of tensors.
  - Include ~4 antithetic pairs.
- Add ~10 band swaps around a 2.2-bpw base, so κ(m) is identified (early-cheap/late-expensive and the reverse).

**Cost:** ~60 evals (~2.5 h on two cards at 2.5 min each).

**Expected [Own]:** this removes the 20–30% low-bpw bias. If κ or the tilt turns out significant, it also shifts allocation at ≤3 bpw: **2–6% KLD there**. IQR effectively does this by measuring band shares from per-k bases [Lit].

**Validation:** decision-focused CV on held-out low-rate mixes, then R4.

### R4. Model-based local search per tier (Thompson + DP + antithetic perturbations)

**Per tier** (focus on 2.0, 2.5, 2.99, 3.5, 3.99; later 4.5, 5.49), per round:
- 1 posterior-mean mix;
- 3 Thompson mixes;
- 2 antithetic perturbations of the incumbent.

That is 6 evals × 5 tiers = 30 evals/round on the fit pool (rotate among 2–3 fit-pool slices with the slice effect). Run 3–4 rounds, ~100–120 evals.

**Multi-fidelity option [Own]:**
- If eval time is mostly per-position (not load), screen on a 20–30k-position slice first: EvoPress's multi-stage idea.
- Profile one eval first (load vs prefill vs logits-to-host). The TODO already notes the ~1 GB/s logits copy as a bottleneck.

**Expected [Lit + Own]:**
- EvoPress / AMQ / SPDY show measured search beating one-shot sensitivity by large margins on *weak* starting points.
- From our much better start, expect **2–5% at ≤3 bpw and ~1% at 4–5 bpw**.

**Acceptance:**
- After the rounds, take the best 2–3 per tier by posterior mean, plus the current incumbent and the best reference container.
- Score them on a **search-pool slice of ≥250k positions**, which brings the paired SE down to ~1%.
- Accept on the paired per-document bootstrap (`paired.py`).
- Cost ~25 evals.

### R5. Category- and position-aware objective (guard against private mismatch)

**Content:**
- From the stored per-row scores, fit category-specific s-profiles (agentic, code, chat, math, multilingual) as hierarchical deviations from the M2 profile.
- Reweight the objective to the calib category × position mix.
- Measure the top finalists of each tier on one whole verify-pool long trace (rich or express, TP2 if >170k at high bpw) to check that the ≤8k-capped slices don't misrank them for the ~35% of positions in long traces.

**Cost:** ~0.5 day analysis, plus ~5–10 long-trace evals on finalists only.

**Expected [Own]:** 0–3%. Mainly variance and risk reduction on the private set. If attention layers matter more at long context, this is where it shows.

### Plan sized to the budget

| Day | Card 0 | Card 1 | Code |
|---|---|---|---|
| 1 | R3: remaining containers + random design around the 2.0 mix (~30) | R3: design around the 2.5 mix + band swaps around the 2.2 base (~30) | R1 + R2 |
| 2 | R4 round 1–2, tiers 2.0/2.5/2.99 (~36) | R4 round 1–2, tiers 3.5/3.99 (~24) + 4.5/5.49 (~24) | refit between rounds |
| 3 | R4 round 3–4 (~60 total) | search-pool acceptance (~25) | R5 analysis |
| 4 | verify-pool check of accepted mixes (~10) + long trace (~5) | spare / re-runs | write-up |

**Total:** ~250 evals. Public is untouched until the final 2–3 per size.

---

## 6. Long tail (one or two lines each)

1. **Measure all 16 containers on the fit slice** (8 left). They are free low-rate anchors and sentinels for the curve [Own].
2. **A sentinel in every batch** (base plus one container). This catches kernel or runtime drift between eval batches. The frozen install-card makes drift unlikely, but checking is cheap [Own].
3. **Fit contrasts, not levels.** Use log(mix/base) on the same slice, so slice effects cancel exactly [Own].
4. **SPDY-style decision-focused fitting.** Black-box-optimize ~10–20 model coefficients (band × group multipliers) directly against the measured KLD of the DP solutions they induce. Lit (SPDY: ≈10× fewer evals than direct search).
5. **Bayesian model averaging over link and depth forms** instead of picking one by CV. This matters only for calibration and cross-size choices [Own].
6. **A GP residual on top of the parametric model.** Use a kernel on per-band mean-K vectors, so predictions snap to measured values near measured mixes (kriging correction) [Own].
7. **Horseshoe-prior adjacent-layer interactions** (BOCS-style). The DP stays exact via a chain state. Lit (BOCS) + Own.
8. **Factorization-machine band × band interactions,** rank 1–2, only if residuals show structure. IMPQ says the total interaction value is ~1% [Lit/Own].
9. **Per-group exponent q_g.** Convexity of cost vs proxy may differ for GDN vs FFN. exllamav3 finds α steepening below the anchor (2.35 for K2→K1) [Lit].
10. **Monotone spline link g** instead of three parametric forms. Calibration only [Own].
11. **Heteroscedastic weights** from per-document bootstrap variances. Large-KLD mixes have larger absolute but similar relative variance [Own].
12. **Risk-aware objective** (posterior mean + κ_r·sd, or CVaR), used only for *shipping* decisions [Own].
13. **Trust-region constraint** in the DP: ≤N K-steps from the nearest measured mix (CASMOPOLITAN/TuRBO idea) [Lit/Own].
14. **Min-K floors per tier** (exllamav3 `--min_k`) as a cheap guard against extrapolating into unmeasured K=1/2 territory [Lit].
15. **AMQ-style pruning.** Freeze tensors whose single-swap sensitivity is >2× median at their current K. This keeps outliers from destabilising the fit [Lit].
16. **Knowledge-gradient scoring** of candidate measurements under a linear-Gaussian posterior (Free–Wilson KG). Worth it if evals get scarce [Lit/Own].
17. **Multi-fidelity slices** (20k screen → 100k fit → 250k+ accept), with a fidelity offset in the model. Lit (EvoPress multi-stage) + Own.
18. **Profile eval wall time.** If the logits-to-host copy dominates, top-512-on-GPU scoring (already in TODO) would multiply our eval budget. That changes which methods are affordable [Own].
19. **Shaped-noise sensitivities** (exllamav3 `sc_measure`, run on a 6–8 bpw base or TP2 bf16) as a *prior mean* for log s_w. Spearman 0.84 [Lit] is good for a prior, not for decisions.
20. **Proxy ensemble.** Compute err under H, diag(H) and a long-context-only H (if obtainable). Let the regression weight them per group. This captures error-shape differences between families [Own].
21. **Option-family factor per (family × group × K-range).** "ours" lowers the Hessian error but may shape it differently. Give it a separate factor and anchor it with 3–5 evals before trusting it [Own].
22. **Fractional / row-split K** adds finer options. Q-Palette attributes much of its 2-bit gain to finer allocation (bit-allocation gap 0.0129→0.0023 at 3.25 b, TCQ-2/3/4 → TCQ-ALL) [Lit]. The DP handles extra options at no extra cost.
23. **Untie gate and up** if the kernel allows. exllamav3 says the tie costs "tiny" KLD [Lit]. Measure once.
24. **Head × last-band interaction.** Logit-space errors from the head and from the final layers may not be independent (both act on the same logits). Test with one 2×2 factorial (head K × last-band K) [Own].
25. **Embedding × first-band interaction,** by the same factorial logic. Embedding error enters the residual stream before every layer, so the κ amplification applies to it too [Own].
26. **Shapley/IMPQ permutations:** skip. ~100 permutations × L evals is far beyond our budget for a ~1% effect [Lit/Interp].
27. **Full BO (GP over 340 categorical dims):** skip. AMQ needed ~10k evals with an RBF surrogate; GPs gave "negligible gains" [Lit].
28. **NSGA-II multi-objective search:** unnecessary. The DP gives the whole frontier exactly under the model [Interp].
29. **Ladder consistency constraint.** Require nested mixes across sizes (each tensor's K is monotone in the budget) to regularise; also convenient for "a container every 0.1 bpw". Costs a little optimality [Own].
30. **Use the frontier's local slope** (measured d log KLD / d bpw) to convert small bpw differences when comparing candidates of slightly different size. Never compare unadjusted KLD across 0.02 bpw [Own].
31. **Category-reweighted acceptance.** Compute search-pool paired differences per category and reweight to the calib mix before the bootstrap [Own].
32. **Re-quantize the final mix sequentially** (Hessians from the quantized upstream, GPTAQ/QEP style). This is a quantizer change that interacts with allocation. Lit (QEP) + Own; defer.
33. **Per-tier separate models** (local regression weighted by distance in bpw) as a nonparametric baseline for M2. If it wins decision-CV, the global model is misspecified [Own].
34. **Log every mix's per-tensor provenance and per-row scores.** These are the data for all of the above and for QAD later (already a TODO rule).

---

## 7. Caveats: where the literature's setting differs from ours

- **Metric.** Most results are WikiText or C4 *perplexity* on Llama/Gemma/Qwen3 small models. Ours is mean KLD vs bf16 on a mixed corpus with long agentic traces.
  - [Lit] EvoPress and Q-Palette use KL for fitting; EvoPress reports KL "slightly better" than ppl for quantization.
  - [Interp] KL is the better-behaved target, and our results should transfer *better* than ppl-based ones.
- **Step size.** The literature mostly chooses among {2,3,4}-bit *scalar or group* quantizers with large error steps. Ours are trellis options at K=1..8 plus family variants, i.e. much finer and lower-error steps.
  - [Interp] Additivity should hold *better* for us at the same bpw, so non-additivity gains from the literature are upper bounds.
- **Baselines.** IMPQ's 20–80% is against weak first-order heuristics; its own interaction ablation is ~1%. AMQ's gains are against one-shot or greedy rankings. EvoPress's are against uniform/DP-NMSE.
  - None start from a calibrated per-tensor knapsack like ours [Interp].
- **Uniform vs mixed containers.** Our "options" are themselves internally mixed exports, so a "swap" changes several things at once. The family factor partly covers this [Own].
- **Size accounting.** No paper counts the embedding and head (each 1.27B parameters, 1 bit = 0.047 bpw). Their allocations ignore the biggest single items in our budget [Lit/Interp].
- **Eval budget.** AMQ used ~10k evals and EvoPress ~8 GPU-h per target on an 8B model. We have ~100–300 evals in total. Hence the emphasis on design and model-based search, not black-box search [Interp].
- **Long context.** Allocation papers calibrate on ≤4k-token sequences. Our scored set has ~35% of positions in 256k traces, and our calib slices cap documents at 8k [Lit for public composition, via earlier brief].
- **Overfitting.** IQR selected among ~50 candidates on public and admits its margin is "within selection noise" [Lit]. Our pool discipline is stricter; keep it.

## Contradictions

- **Is additivity reliable at low rate?**
  - HIGGS and EvoPress: it breaks below ~3 bpw.
  - IQR-V1: 1–7% prediction error at band level for 2.5–7 bpw.
  - IMPQ: interactions worth only ~1%.
  - Our own: a +20–30% bias at 2.0–2.5 bpw, but mostly a link/scale effect.
  - **Resolution [Interp/Own]:** the *magnitude* is non-additive at low rate, but the *argmin* may be fairly robust. R1 tests this directly.
- **Noise injection vs real swaps.** exllamav3 reports Spearman 0.84 for shaped noise. Q-Palette finds swap-measured coefficients beat noise coefficients at 2 bit (6.47 vs 6.69 ppl) but not much at 3 bit (5.34 vs 5.38).
- **Predictor family.** AMQ found RBF ≈ MLP and GPs not worth it [Lit]. BOCS finds MLE point estimates plateau and posterior sampling is needed [Lit]. These are compatible: the *surrogate class* matters less than *having uncertainty for exploration*.

## Missing evidence

- I could not extract per-table numbers for:
  - HIGGS's predicted vs actual curves (figures only; the appendix has an unfinished "how small" placeholder);
  - EvoPress's quantization tables and token budgets (Table 8 not extracted);
  - MixLLM, SliM-LLM, CMPQ and BitStack beyond their abstracts.
  - BitStack and SliM-LLM are intra-tensor or decomposition methods. They are not directly relevant to per-tensor EXL3 allocation; included for completeness.
- No source tests directional depth amplification (§1.4, §2) for LLM quantization allocation. The model and the exact DP are my own and unvalidated. R1 and R3 are the test.
- I have not verified that kfit's depth term leaves the knapsack solutions unchanged. That rests on algebra plus the near-uniform multiplier argument (§0). It is a 5-minute check.
- The per-tensor single-swap effect sizes (~0.1–0.5% of total) are my estimate (1/337 of total × step factor), not measured.
- Eval wall-time breakdown is unknown. It decides whether multi-fidelity or EvoPress-scale search is affordable.

## Sources

**Kept:**
- HIGGS [2411.17525](https://arxiv.org/html/2411.17525v1): linearity theorem, noise calibration, b>3 validity.
- Q-Palette [2509.20214](https://arxiv.org/pdf/2509.20214): KL-based noise calibration, MCKP/ILP, measured vs noise coefficients.
- EvoPress [2410.14649](https://arxiv.org/html/2410.14649v2): measured-KL search, multi-stage selection, convergence theorem.
- SPDY [2201.13096](https://arxiv.org/pdf/2201.13096): DP-in-the-loop coefficient search; closest analogue to our refit loop.
- AMQ [2509.12019](https://arxiv.org/html/2509.12019v1): surrogate + NSGA-II, eval counts, search-space pruning.
- IMPQ / CoopQ [2509.15455](https://arxiv.org/html/2509.15455v1): pairwise interactions, magnitude of the interaction effect.
- exllamav3 [optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md): same format, same model, shaped noise, α values.
- IQR-V2 whitepaper (local): competitor's allocator on the same model and metric.
- BOCS [1806.08838](https://arxiv.org/html/1806.08838v2): sparse polynomial + Thompson + SDP.
- CASMOPOLITAN [2102.07188](https://arxiv.org/abs/2102.07188) and BODi [2303.01774](https://arxiv.org/abs/2303.01774): trust-region / categorical BO (abstract level).
- Negoescu–Frazier–Powell KG for Free–Wilson models: measurement selection for the best decision.
- HAWQ-V2 [1911.03852](https://arxiv.org/abs/1911.03852) / V3 [2011.10680](https://arxiv.org/abs/2011.10680) and FIT [2210.08502](https://arxiv.org/abs/2210.08502): origin of Hessian/Fisher trace + ILP (abstract level).
- Mixed-precision survey [2510.16805](https://arxiv.org/html/2510.16805v1): map of LLM allocation heuristics.
- QEP [2504.09629](https://arxiv.org/abs/2504.09629): error growth across layers (abstract level).
- GDN paper [2609.04098](https://arxiv.org/abs/2609.04098), via the earlier brief: within-layer quadrature on our model.

**Deprioritized:**
- BitStack, SliM-LLM, CMPQ, MixLLM, OWQ: intra-tensor granularity that EXL3 cannot express without tensor splitting; abstracts only.
- Bayesian Bits, BOMP-NAS: small CNNs, gradient/NAS-style.
- Any-precision LLM: a serving format, not allocation.

## Next steps

1. Run the 5-minute check: solve kfit's knapsack at 2.0/3.0/4.0 with and without `+depth` and `pow`, and diff the mixes. Then implement R1 + R2 (no evals).
2. Queue R3 (~60 evals) on both cards while coding.
3. Profile one eval's wall time to decide on multi-fidelity before R4.
