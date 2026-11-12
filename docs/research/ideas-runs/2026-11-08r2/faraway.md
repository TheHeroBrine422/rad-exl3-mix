# Faraway lens, round 2 (2026-11-09): replacing large parts of the approach

**Scope.** This is the exploration pass and is exempt from the pre-freeze scope.
- It does not repeat round 1's ideas (`runs/2026-11-02-faraway`, FA-*). It cites them where relevant.
- The other lenses this round cover low-rank/sub-1 codecs, slabs, pruning at ≤1.5 bpw, metadata and pricing. I only touch those topics from a different angle.

**Conventions.**
- Board slope at 2.5 bpw: 0.01 bpw ≈ −1.7% KL (board 2.0 → 2.5 → 3.0 is 0.164 → 0.0666 → 0.0306).
- Parameter shares of 26.9e9:
  - MLP 63.7% (1 bit on all MLP = 0.64 bpw);
  - GDN mixers 20.6%;
  - full attention 6.2%;
  - embedding + head 9.4% (1 bit on the head = 0.047 bpw);
  - one decoder layer 1.42% (0.036 bpw at 2.5).
- Effects measured on the 0.8B shrink to about ⅓ at 27B.
- P is my probability that the stated gain holds at 27B.

---

## 0. What I measured this round

### Setup

**Proxy.** Qwen3.5-9B in fp32 on colo CPU. It is the same family as the 27B (3 GDN : 1 full attention, untied 248k vocabulary). There is no 0.8B/2B on colo.

**Error model.** Quantization error is a fixed synthetic draw on all 248 trunk linears:
- iid Gaussian at relative Frobenius error REL = 0.045, which gives mean KL ≈ 0.05–0.10 (the 2.5 bpw range);
- or a variant where half the energy is a coherent shrink.

**Data.** 256-token windows from xcorp v1. Fit on docs A, score on held-out docs B.

**Caveats.**
- Synthetic iid noise is not LDLQ-shaped error.
- Only 8–16 docs per measurement.
- 9B, not 27B.
- Where the 27B checkpoint and its calib Hessians were enough, I measured the 27B directly.

### Results

| # | measurement | result | verdict |
|---|---|---|---|
| M0.1 | Zero-byte logit post-processing on 9B: global temperature, 438-token bias, entropy-binned temperature, tail-floor mixtures (bf16 unigram / flattened q), 3-parameter feature-conditioned temperature | **iid noise:** best temperature is s = 1.00 (no gain); bias 1.004–1.06; entropy bins 1.13. **Shrink noise, seed 1:** s = 0.95 gives 0.959. **Shrink noise, seed 3 (other B docs):** temperature fit on A (s = 0.90) gives **1.024 on B**; unigram mix 1.020; flat mix 0.997; feature temperature 0.866 on A → **1.042 on B**. Per-document oracle temperature: only 0.91–0.96. | **Dead.** Everything fit on A overfits. |
| M0.2 | KL anatomy (9B, 8 docs, REL 0.045) | **Top 1% of positions carry 51% of mean KL; top 10% carry 90%.** Per-doc KL: hermes-fc 0.29–0.31, glaive-fc 0.36, nemotron-swe (agent trace) **1.04**; prose/code/math/multilingual 0.006–0.064. | The score is a tail metric. |
| M0.3 | One hermes-fc doc at REL 0.005 / 0.015 / 0.045 | Doc KL 0.0052 / 0.060 / 0.299. Worst position (next token ` {"`, bf16 entropy 0.63 nats) goes 0.35 → 7.7 → 8.6 nats. Top 3 positions hold 46% / 67% / 35% of the doc's KL. Top positions are JSON structure (` "`, ` {"`, `description`); one has bf16 entropy 0.096. Clean vs clean with a different thread count: KL −4e−9. | **Confident structural tokens flip.** This is not numerics. |
| M0.4 | Which block type causes it (REL 0.045 on one block type only; 2 hermes-fc + 2 glaive-fc docs vs 4 other docs) | Attention (0.47B params): FC 0.080 / other 0.0020. GDN (1.62B): 0.244 / 0.0083. MLP (4.83B): 0.149 / 0.0082. Per billion params on FC docs: **attention 0.171, GDN 0.151, MLP 0.031**; on other docs: 0.0043 / 0.0051 / 0.0017. Positions with KL > 1 per FC doc (of 255): attention 2–9, GDN 10–19, MLP 7–18. | At equal relative error the **mixers are 5× more damaging per parameter on tool text** (2.5–3× elsewhere). On FC docs the groups are sub-additive (sum 0.47 vs about 0.3 jointly). |
| M0.5 | Residual-stream slicing: global PCA of the RMS-normalised residual at 65 norm inputs; project writers and embedding onto the top r | Dropping 3% of dims keeps 99.85% of the energy but costs **KL 0.088** unhealed. The saving is 0.075 bpw ≈ 0.008 KL at 2.5. | **Dead, 11× too costly.** |
| M0.6 | MLP neuron pruning by E[a²]·‖W_down col‖² | The bottom 5/10/25% of neurons hold 1.5/3.3/9.7% of the score, but pruning 5% costs **KL 0.131** and 15% costs 0.366. Mean compensation does nothing (0.368). 5% of the MLP is worth 0.08 bpw ≈ 0.009 KL. | **Dead, 15× too costly.** |
| M0.7 | Head absorbs trunk error: map noisy → clean final hidden state, fit on 40 docs (10k positions), folded into the head (zero bytes), 8 held-out docs | Base mean 0.271 (one nemotron-swe doc at 1.04; it was drawn twice). Diagonal map: 0.260 (−4%, all from the SWE doc). Full ridge: 0.078, but only because the SWE doc goes 1.04 → 0.034 while **every other doc gets 3–9× worse**. Rank-16 / rank-128 parts: 0.246 / 0.178, typical docs 1.5–5× worse. | A fixed head-side linear map does not generalise. The SWE drift is linear and category-specific (see J4). |
| M0.8 | 27B gate/up same-index rows | Median \|cos\| 0.06–0.12 (permuted 0.01); joint coding saves 0.006–0.014 bit. | **Dead.** |
| M0.9 | 27B calib input Hessians, eigen-spectrum | No eigen-direction below 0.01×mean (L10/L31/L50 MLP input, L10/L31 layer input). ½log2(AM/GM) is 0.29 / 0.59 / 0.69 bit for MLP inputs and 1.25 / 1.15 bit for layer inputs (the massive channel); LDLQ already uses this. | Null-space shrink (OSAQ) has nothing to work on. |
| M0.10 | 27B vocabulary | 248,320 rows vs 248,077 tokenizer entries: 243 padding rows (0.1%). | Vocabulary pruning ≤ 0.0003 bpw. **Dead.** |

### The 27B logs point the same way

#117 logged a pooled p99 KL of 3.9 at a pooled mean of 0.058 (K2.5). If that p99 is per position, the top 1% of positions carry at least 0.039 of the 0.058, i.e. **≥ 2/3 of the score**. The 9B gives 51%. Diagnostic 1 checks this.

### What this changes for the faraway lens

1. **The storage side is boxed in.**
   - Round 1 found no cross-layer redundancy, Gaussian weights, and a trellis 1.13–1.19× from an ideal coder.
   - Round 2 adds: no sibling redundancy, no Hessian null space, no slicing or pruning slack, no vocabulary slack, and no zero-byte output fix.
   - Replacing the codec, sharing weights or restructuring the model all lose at 2–3 bpw unless backed by large-scale training.
2. **The open frontier is the objective.**
   - The score is a mean over positions, and that mean is a tail: rare confident flips of structural tokens in tool-call and agent text.
   - Every statistic in the pipeline is second order:
     - Σxxᵀ weights all calib tokens equally;
     - Σdy² uses sampled-label Fisher.
   - Sampled-label Fisher weight at a position scales with 1 − p_top. At the 9B's worst flip (p_top ≈ 0.99), it is ≈ 0.02 while the flip costs 8–10 nats.
   - The qad windows behind Σdy² hold agent traces at 6% of scored rows, against about 25% in search/verify (LOG 11-04).
3. **Mixers matter more than their size suggests on agent text** (M0.4). The knapsack sees this only if its prices come from slices with the score's agent share.

Most living ideas below follow from points 2 and 3. Everything else is listed for breadth, with honest probabilities.

---

## 1. Ideas by theme

Each idea has these fields:
- **Mech** — mechanism.
- **Src** — source papers or "new", and what differs.
- **Why** — which error source it attacks, with numbers.
- **Fit** — where it sits in the pipeline.
- **Test** — cheapest test and pass rule.
- **Gain** — expected gain at 2.5 bpw.
- **P** — probability.
- **Conflicts** — similar INDEX experiments.
- **Reserve** — anything to decide or keep before the ladder.
- **2–3 bpw** — why it could beat the frontier there.
- **Falsifier** — first falsifying 0.8B/2B experiment.

**Status tags:** ALIVE · CRAZY-CHEAP · KILLED-today · KILLED-arith (killed by arithmetic) · KILLED-earlier.

### A. The score is a tail: change the objective and the statistics

**FB-A1 · Flip-risk output Hessian (margin kernel)** — ALIVE
- **Mech:** In the Σdy² collector, backprop the bf16 logit margin m = z_top1 − z_top2 at each scored position instead of a sampled-label CE. Weight it by w = φ(m/σ)/σ, where σ is the logit-difference noise a K2.5 level actually produces (read once from an existing level). Accumulate the block-normalised diagonal of Σ w·(∂m/∂y)² and use it as H_out, alone or blended 50/50 with bn.
- **Src:** new. Related: margin erosion in OnPTQ (2609.36828) is a calibration weight, not a Hessian. The whole Fisher family (#33, H_Z-PTQ 2610.40345, KronQ 2607.07964) is second order and vanishes as p_top → 1.
- **Why:**
  - Flips are threshold events: P(flip) ≈ Φ(−m/σ). The kernel puts weight where the crossing probability moves.
  - Sampled-label Fisher gives about 0.02 weight to a position that costs 10 nats when it flips (M0.3).
  - The top 1% of positions hold 51% (9B) and possibly ≥67% (27B) of the score.
- **Fit:** per-level statistic (needs a relevel), or per-mix if used only for writers and healing.
- **Test:** 0.8B on the 4070.
  - One margin-backward collection on the 1,024 qad windows (≈1 h).
  - Quantize K3/K2 × 3 seeds against edy.
  - Pass: pooled ≤ 0.985 on every seed, and the agent subset ≤ 0.97.
  - Total ≈ 3 h.
  - Go to the 27B only after a pass: 256 margin sequences (≈5 card-h) blended into bn.
- **Gain:** −1 to −4% (±3). Risk: a noisy margin gradient, and the choice of σ.
- **P:** 0.2.
- **Conflicts:** #33-18 and #33-29 found all Fisher-type estimators within 1% of each other; this is not Fisher-type. #83's gen mask chose positions; this weights them by flip risk.
- **Reserve:** keep the collector able to emit per-position bf16 top-2 ids and margins. No format change.
- **2–3 bpw:** σ grows at low bpw, so the kernel reaches more positions; flips are what separates 2.5 from 3.0.
- **Falsifier:** 0.8B K2 with margin H_out ≥ 1.00 vs edy, or no gain on the agent subset.

**FB-A2 · KL-reweighted statistics (closed loop, IRLS)** — ALIVE
- **Mech:**
  1. Quantize a level and score per-position KL on fit/qad windows.
  2. Recollect H_in, and optionally Σdy², with weights w_t = clip(KL_t / mean, ≤ 50)^γ, γ ∈ {0.5, 1}.
  3. Requantize, and do one more round.

  The statistics then describe the directions that are active where this quant actually fails.
- **Src:** FA-G5 (tail-position H_in, never run); DRQ 2610.11226 (worst-case covariance, scalar grids). New: the weights come from the measured KL of the actual quant.
- **Why:** the tail numbers above. The published H_in weights every calib token equally; agent text is about 19% of tokens but plausibly more than half of the K2.5 KL (diagnostic 1).
- **Fit:** per-level statistics, needs a relevel. At 27B, one H_in recollection is ≈12 card-h, single card, staged.
- **Test:** 0.8B K2/K3 × 3 seeds: base → per-position KL rows → reweighted H_in → requant. Pass: ≤ 0.97 on search and verify, every seed. ≈3 h on the 4070.
- **Gain:** −1 to −4%. Risk: overfitting the fit docs (DRQ saw lower layer loss raise PPL in 47–74% of cases); fewer effective samples.
- **P:** 0.15.
- **Conflicts:** #86 used a binary gen weight (K3 0.998, K2 0.976). #91 and #101 changed the corpus, not the weights.
- **Reserve:** keep per-position KL rows of every fit-pool scoring.
- **2–3 bpw:** more positions fail at low bpw, so the weights are less concentrated and the statistic is steadier.
- **Falsifier:** 0.8B K3 reweighted ≥ 0.99.

**FB-A3 · Category-matched statistics** — ALIVE (top pick)
- **Mech:** Reweight Σdy² (and H_in) per document category so each category gets its share of scored rows in the score corpus (agentic ≈ 25%), not its share of qad scored rows (6%).
  - **27B:** recombine the 128-sequence block snapshots of the 1,920-sequence collection with weights from each block's category mix (the window order is seeded and reproducible).
  - **H_in:** one forward collection with per-category sums.
- **Src:** new. LOG 11-04 noted the 6% vs 25% gap as an "extra knob"; it was never tested.
- **Why:**
  - 9B: tool and agent docs show 20–75× the KL of prose at the same weight noise.
  - LOG #93/d1110: H_out gained −19% and the full stack −35% on calib's agentic text, against about 0 on xcorp's tool text. The methods earn their gain on agent text, and the statistics under-sample it 4×.
- **Fit:** per-level statistic. At 27B this is nearly free if block categories are recoverable; otherwise ≈12 card-h for H_in.
- **Test:** 0.8B. One Σdy² collection keeping per-category sums; quantize K3/K2 with weights {natural, scored-row matched, 2× agent}. Pass: pooled ≤ 0.985, agent ≤ 0.97, no category > 1.01. ≈2 h.
- **Gain:** −0.5 to −2%. Risk: if the 128-sequence blocks mix categories, reweighting is diluted.
- **P:** 0.25.
- **Conflicts:** #33-19, #91 and #101 added off-distribution text; this reweights calib's own documents.
  - **Watch #33-27:** block normalisation equalised the heavy blocks 6 and 9, and that helped. If those blocks were agent-heavy, it is evidence *against* up-weighting agent text. Diagnostic 2 settles this.
- **Reserve:** record each qad window's category in collection order. This takes seconds and decides whether the 27B version is free.
- **2–3 bpw:** H_out gains are largest at low K.
- **Falsifier:** 0.8B with 2× agent weight ≥ 1.00 pooled.

**FB-A4 · Category-matched healing data** — ALIVE (per-mix healing, deferred stage)
- **Mech:** Draw the healing windows from calib's qad pool with agent traces at their scored-row share or 2×. Keep the plain KL loss.
- **Src:** new for calib-own data. #89-05 used xcorp plus the user's own traces.
- **Why:** same as FB-A3. Healing has the most capacity to fix systematic, document-specific drift, and M0.7 shows the agent drift is linearly correctable.
- **Fit:** per-mix healing.
- **Test:** 0.8B heal0 recipe, K2/K3 × 2 seeds, natural vs agent-matched data. Pass: pooled ≤ 0.99 vs heal0 and agent ≤ 0.97. 4070, ≈3 h.
- **Gain:** −0.5 to −2% on top of healing. **P:** 0.2.
- **Conflicts:** #89-05 (gray zone, 1.014), #91.
- **Reserve:** none.
- **2–3 bpw:** healing's own gain is largest at K2 (0.8B −38%).
- **Falsifier:** agent-matched heal ≥ heal0 pooled.

**FB-A5 · Margin-weighted head plan** — ALIVE
- **Mech:** The slab head ranks rows by the Fisher diagonal Σp(1−p) (#34), which vanishes at confident positions. Rank rows by E[w(m)·1(row ∈ {top1, top2})] with the FB-A1 kernel, or by a sum of both. Keep the K floors and the rare-token guard.
- **Src:** new. #34; SoftWater 2608.12026 (class-aware rate ∝ E[p(1−p)]).
- **Why:** the head is ≈5% of KL on the calib part (#34-02). JSON and code-syntax rows are top-1 at near-certain positions, so they get low Fisher and therefore low K today.
- **Fit:** per-level. Head donors come after the trunk, so the ladder is not affected.
- **Test:** headsim on the 32k held-out head inputs (CPU, minutes), margin plan vs Fisher plan at the K4/K5 budgets. Pass: ≥ 10% lower head KL with no worse guard KL. Then one 27B pair.
- **Gain:** −0.3 to −1%. **P:** 0.3.
- **Conflicts:** #32 dropped frequency tiering (this is not frequency); #34.
- **Reserve:** keep bf16 top-2 per scored position (already in the top-512 goldens).
- **2–3 bpw:** the head sits at K3–4 with low floors there, so head flips cost more.
- **Falsifier:** headsim margin plan ≥ Fisher plan.

**FB-A6 · Robust (worst-direction) acceptance in tile CD** — ALIVE, low
- **Mech:** Tile CD (#77) accepts moves on the H-loss. Add α·λmax(EᵀE, C) with C = (1−β)H + βI, as in DRQ, so a move must not push error into a direction a slightly shifted input distribution would expose.
- **Src:** DRQ 2610.11226 on scalar grids. New: applied to trellis tile moves.
- **Why:** the tail suggests that worst-case input directions matter more than average ones.
- **Fit:** per-level quantizer.
- **Test:** settle64 CPU proxy on 27 tensors, scored on held-out H_B. Pass: held-out H-loss ≤ 0.99. Then 0.8B KL. Colo, ≈4 CPU-h.
- **Gain:** 0 to −1%. **P:** 0.08.
- **Conflicts:** #77, #114.
- **Reserve:** none.
- **2–3 bpw:** larger errors are more likely to align with rare directions.
- **Falsifier:** proxy ratio > 0.995.

**FB-A7 · 27B self-continuations of calib agent prompts as statistics** — ALIVE, low
- **Mech:** Calib's agent traces were generated by the 27B itself. Have the bf16 27B continue more calib-style agent prompts (from the qad pool only) and add those windows to the Σdy²/H_in sums at the agent share. This raises the effective number of agent windows by 4–8× without leaving the score's distribution.
- **Src:** #91 (0.8B continued the user's sessions, a different prompt distribution: −5…−15% on own text, +1…+6% on calib). New: 27B continuations of calib-like agent prompts.
- **Why:** the agent rows are under-sampled (6%), and FB-A3's reweighting cannot create samples.
- **Fit:** per-level statistic, 27B generation ≈ 5–10 card-h.
- **Test:** 0.8B analogue: 0.8B continuations of qad agent prompts added at 25%. Pass: agent ≤ 0.97 and calib pooled ≤ 1.00.
- **Gain:** −0.5 to −1.5%. **P:** 0.08.
- **Conflicts:** #91, #33-19, #98.
- **Reserve:** none.
- **2–3 bpw:** as FB-A3.
- **Falsifier:** pooled > 1.005.

**FB-A8 · Tail-aware damping** — ALIVE, low
- **Mech:** Damping (σ_reg × mean diag H) treats every input direction alike. Damp less along the input directions active at the top-1% KL positions: the top eigenvectors of Σ_{t ∈ tail} x_t x_tᵀ, kept undamped through a low-rank correction to H + δI.
- **Src:** new. #114 and #121 studied the size and normalisation of δ, not its shape.
- **Why:** damping blurs exactly the rare directions that a few hundred tail positions rely on. Round 1 measured 0.05–0.3 bit of real per-tensor structure, and damping could erase part of it on tail directions.
- **Fit:** per-level quantizer setting.
- **Test:** w87's held-out two-sided proxy plus a tail-H_B (built from the top-KL positions of the fit pool). Pass: tail-H_B loss −3% with all-H_B ≤ +0.5%. CPU, ≈3 h.
- **Gain:** 0 to −1%. **P:** 0.07.
- **Conflicts:** #114, #114-03, #121 (all null for isotropic damping).
- **Reserve:** none.
- **2–3 bpw:** damping matters most at low K (#114: σ* rises toward K0.125).
- **Falsifier:** tail-H_B gain < 1%.

### B. Allocation and pricing under the tail view

**FB-B1 · Mixer-first allocation on agent text** — ALIVE (top pick)
- **Mech:** Re-price the knapsack's per-family sensitivities on calib-mix slices that carry the score's agent share, instead of 65%-xcorp two-part slices. Test one forced equal-bytes move against the knapsack mix: +0.5 bit on all attention and GDN mixer tensors, −0.18 bit on the MLP.
- **Src:** M0.4 (new); knap5 family factors (#118).
- **Why:**
  - At equal relative error the mixers do 5× (tool text) and 2.5–3× (prose) the per-parameter damage of the MLP (M0.4).
  - LOG 11-08 found pooled two-part slices understate head effects about 15× relative to calib; mixer priority is likely understated the same way.
  - #118 measured attention as *over*-priced (attn_k 0.73). That is a per-unit-proxy statement; this test asks a different question.
- **Fit:** knapsack input, per mix.
- **Test:**
  - 0.8B first: equal-bytes shift, KL by category (diagnostic 4).
  - 27B: one forced mix vs the knapsack mix, scored on calib verify (agent subset plus pooled), 2 evals ≈ 1 card-h.
  - Pass: pooled ≤ 0.99 with both parts < 1.
- **Gain:** 0 to −3%. **P:** 0.25 (a null result is a cheap confirmation).
- **Conflicts:** kfit2 factors, #118.
- **Reserve:** none.
- **2–3 bpw:** at low bpw the knapsack takes bits from the cheapest-priced tensors; if mixers are mispriced, that is where it hurts.
- **Falsifier:** the 0.8B shift does not lower agent KL.

**FB-B2 · Flip-attributed sensitivity prior** — ALIVE
- **Mech:**
  1. Find fit-pool positions where an existing mix flips (KL > 1).
  2. Backprop the bf16 margin there.
  3. Score each tensor by Σ_flips |⟨∇_W m, ΔW_actual⟩|.
  4. Use the scores as a prior in the knapsack fit where swaps are too noisy (#14: SE ≈ 1.5% vs 0.1–0.5% effects).
- **Src:** new. Gradient × perturbation attribution, as in Mix-QVLA 2606.19565.
- **Why:** the tail is a few hundred flips per slice; attribution reads them directly, while a swap measures their noisy sum.
- **Fit:** knapsack input.
- **Test:** 0.8B: compare the attribution with measured group-swap deltas. Pass: rank correlation ≥ 0.6, and a knapsack with the prior beats one without on verify. ≈2 h (CPU + 4070).
- **Gain:** 0 to −2%. **P:** 0.12.
- **Conflicts:** #14, #15, #18.
- **Reserve:** keep per-position KL and bf16 top-2 on fit-pool slices.
- **2–3 bpw:** more flips give more signal.
- **Falsifier:** rank correlation < 0.3.

**FB-B3 · Heal-aware knapsack** — ALIVE (V4)
- **Mech:**
  1. Run a short frozen-code scale heal on each library level (one fit pass, like swj).
  2. Measure sensitivities after healing.
  3. Run the knapsack on the healed prices.

  The ranking of K choices before healing is not the ranking after.
- **Src:** QUBO-MP 2609.22238 (route rankings flip after LSQ+ recovery); BACKLOG "healing-aware reallocation" (V4).
- **Why:** healing gains differ by K and family (0.8B: K2 −38%, K3 −17%), so pre-heal prices misallocate.
- **Fit:** per-level fixed-code fit plus re-pricing.
- **Test:** 0.8B: pre-heal knapsack mix vs heal-aware mix, both healed. Pass ≤ 0.97. 4070, ≈4 h.
- **Gain:** −1 to −3%. **P:** 0.15.
- **Conflicts:** #89, #111 (per-level vs per-mix), #85.
- **Reserve:** none (already planned for V4).
- **2–3 bpw:** healing changes most at low K.
- **Falsifier:** heal-aware mix ≥ 0.99.

**FB-B4 · Differentiable allocation check on the 0.8B** — ALIVE, low
- **Mech:** Relax per-tensor K choice to a Gumbel-softmax blend of precomputed dequantised options with an expected-bpw penalty, and train the logits on fit-pool KL (GAMMA/dMX/RCO style). Compare the resulting argmax mix with the knapsack + local-search mix.
- **Src:** RCO 2605.00649, GAMMA 2605.18475, dMX 2606.04115. New: used as an audit of our knapsack, not a replacement.
- **Why:** measures how much non-additivity the stream search leaves. Interactions are about 1% (#16), but CASA found a 2× residual gap at 2.25 bpw.
- **Fit:** diagnostic for the knapsack.
- **Test:** 0.8B, all options resident (fits in 12 GB for 0.8B). Pass: the relaxed mix is ≥ 2% better on verify at equal bytes. ≈4 h.
- **Gain:** 0 to −2% (only through what it reveals). **P:** 0.08.
- **Conflicts:** #17 (RCO dropped as 27B-infeasible; the 0.8B is feasible), #70.
- **Reserve:** none.
- **2–3 bpw:** interactions grow at low bpw (CASA).
- **Falsifier:** relaxed mix within 1%.

**FB-B5 · Pairwise-interaction knapsack for siblings and the last layers** — ALIVE, low
- **Mech:** Add measured pair terms Γ_ij for intra-block pairs (gate/up → down, q/k/v → o, GDN qkv → out) and for the last 4 layers to the DP, as in TWLA's chain DP and LoopQuant's Shapley interactions.
- **Src:** TWLA 2606.13054, LoopQuant 2610.16368. The measure lens proposes this broadly; my angle is restricting it to the pairs where M0.4's sub-additivity shows up (tool text).
- **Why:** on FC docs the groups are sub-additive (0.47 summed vs about 0.3 jointly), so additive pricing over-penalises combining low bits there.
- **Fit:** knapsack input.
- **Test:** 0.8B pair measurements on agent-heavy slices. Pass: CV pair ranking improves beyond bootstrap noise.
- **Gain:** 0 to −1%. **P:** 0.07.
- **Conflicts:** #16, #11.
- **Reserve:** none.
- **2–3 bpw:** saturation bites at high KL.
- **Falsifier:** |Γ| below the noise.

**FB-B6 · Price tensors by flip rate, not mean KL** — ALIVE, low
- **Mech:** For the knapsack's swap measurements, record the count of positions crossing KL > 1 alongside mean KL, and fit kfit2 on both with a two-component cost. Use this only to stabilise noisy fits, since the score itself is mean KL.
- **Src:** new.
- **Why:** flip counts are integer events with lower variance than their KL sum (a few 8-nat spikes dominate).
- **Fit:** knapsack fit.
- **Test:** refit kfit2 on existing V2/V3 rows with the flip-count feature. Pass: CV log-RMSE −10%. CPU, ≈1 h.
- **Gain:** 0 to −1%. **P:** 0.08.
- **Conflicts:** #19, #20 (re-ranker failed), #68.
- **Reserve:** store per-position rows (already done).
- **2–3 bpw:** more flips there.
- **Falsifier:** no CV improvement.

### C. Output-side corrections (zero or tiny bytes)

**FB-C1 · Global temperature / bias / tail floor** — KILLED-today (M0.1)
- Fit-on-A ratios on B are 1.00–1.04 across two seeds and doc sets. The per-doc oracle is only 0.91–0.96, and nothing fit on A transfers.
- Revive only if the 27B's real quant error is shrink-dominated (#120-03 found 70% coherent at K2), and then only with a per-mix fit on fit docs and a verify check.

**FB-C2 · Fixed head-side linear map absorbing trunk error** — KILLED-today (M0.7)
- The diagonal map gives −4%, all from one document; that part is final-norm healing, already in #89.
- The full or low-rank map overfits and makes typical documents 1.5–9× worse.

**FB-C3 · Rate-distortion-perception decoder (preserve output entropy, not MSE)** — KILLED-today
- RDP theory (2607.17232) says matching output statistics can lower a divergence at fixed rate. Our test is the temperature and entropy calibration of M0.1: iid error already leaves Hq ≈ Hp (1.620 vs 1.612), and shrink error made Hq < Hp, but the correction does not transfer.

**FB-C4 · KL-trained low-rank head adapter** — ALIVE, low (per mix)
- **Mech:** A rank-8 to rank-64 bf16 adapter on the final hidden state before the head (≤ 0.0004 bpw), trained per mix with KL to bf16 on qad windows, with early stopping on held-out windows. Unlike M0.7, it is trained on KL, not hidden-state MSE, with low rank and early stopping.
- **Src:** LQER/QERA family; BACKLOG V4 "runtime adapter".
- **Why:** M0.7 shows some drift is linearly correctable (SWE doc 1.04 → 0.034), and a KL loss weights exactly the positions that matter.
- **Fit:** per-mix final pass, plus an engine op.
- **Test:** 0.8B, cache final hidden states plus bf16 top-512 logits. Pass: verify ≤ 0.98 and no category > 1.01. ≈2 h.
- **Gain:** 0 to −2%. **P:** 0.1.
- **Conflicts:** #89 (healing), BACKLOG LoRA notes.
- **Reserve:** an optional head-adapter weight in qwen35_exl3 (post-freeze, like the #120 bias note).
- **2–3 bpw:** more drift to correct.
- **Falsifier:** verify > 0.99.

**FB-C5 · Logit soft-capping or flip insurance at small margins** — KILLED-today by proxy
- Flattening q's top-2 when its margin < τ bounds the cost of near-ties, but the costly flips have q confident on the wrong token.
- The M0.1 flat-mixture variant gave 0.997.

**FB-C6 · Per-position adaptive temperature from features (hidden-state norm, entropy, position)** — KILLED-today
- 0.866 on A → 1.042 on B.

**FB-C7 · Final RMSNorm / last-layer scale healing as a standalone step** — covered by #89. M0.7's diagonal map gives −4%.

### D. Training, healing and end-to-end optimisation

**FB-D1 · STE trellis QAD with periodic Viterbi re-encode** — ALIVE (V4)
- **Mech:**
  1. Keep latent weights in the rotated domain; the forward pass uses the trellis decode of the current codes, with an identity STE.
  2. Re-run our trellis encoder every ≈16 steps.
  3. Train in two stages: block-wise L1 with the quantised prefix, then end-to-end KL over continuous parameters.

  Indices move, not just scales.
- **Src:** TVQ 2610.41822 (QTIP: Llama3-70B 2-bit Wiki 4.97 → 4.48; **no gain** on Llama2-70B; 8× QTIP cost); E8-QAT 2610.27699 (cell-centre init, LR ∝ 1/scale, escape tracking).
- **Why:** it attacks the code-choice error that scale healing cannot reach. ThinkBit, however, shows scales do most of the work.
- **Fit:** per mix (V4). Needs 27B latents (fp32/bf16) plus Adam, i.e. rental.
- **Test:** 0.8B K2: heal0 vs STE-QAD of equal steps. Pass ≤ 0.95 vs heal0. 4070, ≈6 h.
- **Gain:** −1 to −4% on top of healing. **P:** 0.08.
- **Conflicts:** #89, V4 QAD notes.
- **Reserve:** none.
- **2–3 bpw:** code moves are worth more at low K.
- **Falsifier:** ≥ 0.98 vs heal0.

**FB-D2 · GDN decay healing (A_log, dt_bias)** — ALIVE (top pick)
- **Mech:** After quantization, train only the per-head A_log and dt_bias (96 numbers per GDN layer, bf16, already stored) with KL to bf16 on qad windows; codes and scales stay frozen. Quantized q/k/beta shift the effective state lifetime, and the decay gates can re-tune it.
- **Src:** new. #108 refit qkv/z scales (fail in the stack), not the recurrence time constants. STEPQuant (2609.38169) and DAMP (2608.27513) show GDN error persistence is set by the decay.
- **Why:** M0.4: GDN is 5× the MLP's per-parameter damage on tool text; long agent traces depend on state lifetime. Zero bytes.
- **Fit:** per-level fixed-code fit (one fit pass, like swj) or per mix.
- **Test:** 0.8B K2/K3 × 3 seeds, decay-only heal vs none. Pass ≤ 0.98, every seed. ≈1 h.
- **Gain:** −0.5 to −2%. **P:** 0.15.
- **Conflicts:** #108, #89 (healing trains suh/svh/norms; check whether A_log is included).
- **Reserve:** none (the fields exist).
- **2–3 bpw:** low-K q/k distort retention more.
- **Falsifier:** ≥ 0.99.

**FB-D3 · GDN short-conv healing** — ALIVE
- **Mech:** Train the 4-tap depthwise conv1d (10240×4 per GDN layer, bf16) and its bias with KL, codes frozen. The conv sits right after in_proj_qkv, so it can absorb per-channel scale and offset errors and a short temporal correction of the quantised projection.
- **Src:** new.
- **Why:** a per-channel, sequence-aware corrector for 52% of GDN parameters (in_proj_qkv), at zero bytes.
- **Fit:** per-level fit or per mix.
- **Test:** 0.8B as FB-D2, ≈1 h. Pass ≤ 0.98.
- **Gain:** −0.5 to −1.5%. **P:** 0.12.
- **Conflicts:** #108 (a qkv scale refit is a subset of this), #85.
- **Reserve:** none.
- **2–3 bpw:** larger per-channel errors at low K.
- **Falsifier:** ≥ 0.99, or a gain that vanishes once swj/C9 is applied.

**FB-D4 · Lifetime-weighted H_out rows for GDN projections** — ALIVE, low
- **Mech:** Multiply the H_out diagonal of in_proj_qkv rows by each head's state lifetime L_u = Σ_j e^{2jℓ_u}, where ℓ_u is the mean log retention computed from A_log, dt_bias and calib activations. Errors in long-memory heads persist across tokens; Σdy² sees them only through the window length.
- **Src:** STEPQuant (gate half-life vs accumulated error, ρ ≈ 0.80 over 2,304 heads, on Qwen3.8-27B); the codes lens 2.12 uses this as a per-head-K prior. Mine is a no-format H_out weight.
- **Why:** qad windows of 2k tokens under-weight errors that persist over 250k-token traces.
- **Fit:** per-level statistic tweak.
- **Test:** 0.8B. Pass: KL on long (≥8k) windows ≤ 0.98, short windows ≤ 1.00.
- **Gain:** 0 to −1%. **P:** 0.07.
- **Conflicts:** #52 (KL vs position flat for all quants).
- **Reserve:** none.
- **2–3 bpw:** larger state errors at low K.
- **Falsifier:** long-window gain < 1%.

**FB-D5 · Quantization-native conditioning fine-tune of the bf16 model** — ALIVE, very low (V4+)
- **Mech:** Briefly fine-tune the bf16 27B with weight-noise injection shaped like LDLQ error at K2.5 (Jacobian-guided), with a KL anchor to the original, then quantize.
- **Src:** FA-G4 (prior); Jacobian noise injection 2608.20988 (small gains); depth-law QAT paradox 2609.32060 (gains fade with depth).
- **Why:** flattens the model along LDLQ's error directions.
- **Fit:** training, rental.
- **Test:** 0.8B conditioning on 50M tokens, then K2. Pass ≤ 0.95 vs plain quant **and** KL of the conditioned bf16 vs original ≤ 0.002.
- **Gain:** −2 to −8% if it works. **P:** 0.04.
- **Conflicts:** none in the INDEX.
- **Reserve:** none.
- **2–3 bpw:** larger errors leave more room for flatness gains.
- **Falsifier:** the conditioning drift alone exceeds the gain.

**FB-D6 · Width prune + distil (Minitron-style), bits reinvested** — ALIVE, very low
- **Mech:** Prune 10% of MLP neurons and GDN heads, distil on several billion tokens, and spend the freed ≈ 0.16 bpw on bits.
- **Src:** Minitron; M0.6.
- **Why:** 0.16 bpw ≈ −25% KL if distillation recovers the loss. M0.6 says the unhealed loss is about 15× the value, so distillation must recover > 93%.
- **Fit:** training, rental.
- **Test:** 0.8B, prune 10%, distil on 200M tokens, KL vs bf16 0.8B. Pass: KL of the pruned bf16 ≤ 0.004.
- **Gain:** large if it works. **P:** 0.02.
- **Conflicts:** FA-E5, lowbpw lens B1.
- **Reserve:** none.
- **2–3 bpw:** bytes matter most there.
- **Falsifier:** pruned bf16 KL > 0.01.

**FB-D7 · Depth→precision: drop 4–8 layers, distil, reinvest** — ALIVE, very low (crazy)
- **Mech:** Drop the 4–8 most redundant GDN layers (by angular distance), distil, and spend 0.14–0.29 bpw elsewhere.
- **Src:** ShortGPT, 2608.25068; FA-E4 (P 0.08).
- **Why:** 0.29 bpw ≈ −40% KL if depth is recoverable.
- **Fit:** training.
- **Test:** 0.8B drop 2 of 24 layers, distil 200M tokens. Pass: bf16 KL ≤ 0.005.
- **Gain:** large. **P:** 0.02 (lowered from FA-E4 by M0.6's evidence that per-parameter redundancy is small).
- **Conflicts:** FA-E4, #21.
- **Reserve:** none.
- **2–3 bpw:** as FB-D6.
- **Falsifier:** pruned bf16 KL > 0.01.

**FB-D8 · Learned residual-stream corrector network** — ALIVE, low (per mix)
- **Mech:** A small bf16 side network (≤ 50M params, ≤ 0.03 bpw) that reads the residual at layers {16, 32, 48} and adds a correction, trained per mix with KL. #120-03 found 70% of K2 error coherent across seeds, i.e. deterministic and potentially learnable.
- **Src:** new as a byte trade. Related: ResCom's closed-form correction 2610.35955 (activation side), LQER.
- **Why:** it must beat 0.03 bpw spent on bits (≈ −5% KL). Healing already captures much of the coherent part.
- **Fit:** per mix, plus an engine op.
- **Test:** 0.8B K2 healed vs healed + corrector at equal bytes. Pass ≤ 0.95.
- **Gain:** 0 to −4%. **P:** 0.06.
- **Conflicts:** #89, #120-03.
- **Reserve:** an optional side-module field in qwen35_exl3 (post-freeze).
- **2–3 bpw:** coherent error is largest at K2 (70% vs 32% at K3).
- **Falsifier:** ≥ 0.99 vs equal-bytes bits.

**FB-D9 · Sequential (quantised-prefix) statistics with a trust region, per mix** — ALIVE, low
- **Mech:** For a finished mix, recollect H_in from the quantised prefix and requantize only the most sensitive tensors with G²PTQ's update: block-output target, first-order gradient term, trust region β.
- **Src:** G²PTQ 2609.31009 (LLaMA3-8B W3 KL 0.140 → 0.122 vs GuidedQuant; without a trust region it collapses to 25.3). It differs from #88-02, which used a full asym target for all tensors (1.53× worse) and had no trust region.
- **Why:** deployed activations come from the quantised prefix (#111 lesson).
- **Fit:** per mix, ≈ 12 card-h of H_in plus partial requant at 27B.
- **Test:** 0.8B mixed-K mix, top-30 tensors. Pass ≤ 0.97.
- **Gain:** −1 to −3%. **P:** 0.06.
- **Conflicts:** #87, #88-02, #111.
- **Reserve:** none.
- **2–3 bpw:** prefix drift is largest at low bpw.
- **Falsifier:** ≥ 0.99.

**FB-D10 · Distil from a cheaper quantised teacher (bootstrapped healing)** — ALIVE, low
- **Mech:** Heal the 2.5 mix toward the 4.5–5.0 bpw mix instead of bf16, so teacher and student fit one 32 GB card side by side and the teacher can run on-the-fly on any data. Finish with a short bf16-golden phase.
- **Src:** new for our setting.
- **Why:** removes the teacher-logit storage and TP2 bottleneck for healing on fresh data (calib goldens exist only for calib).
- **Fit:** per-mix healing infrastructure.
- **Test:** 0.8B: heal K2 toward the K4 mix vs toward bf16, equal steps. Pass: within 10% of the bf16-teacher gain.
- **Gain:** enables, roughly 0 directly. **P:** 0.3 that it matches.
- **Conflicts:** #89.
- **Reserve:** none.
- **2–3 bpw:** the teacher error (≈0.006 at 4.5) is small relative to 0.067.
- **Falsifier:** < 80% of the gain.

### E. Rounding as global inference (physics, ECC, message passing)

**FB-E1 · Soft trellis LDLQ (BCJR denoiser + VAMP) then block CD** — ALIVE
- **Mech:** Replace greedy commit-per-tile LDLQ with VAMP over the whole row. The blockwise denoiser is the trellis forward-backward (BCJR) at temperature T, giving per-weight posterior means and variances. Then take the MAP path per tile and run exact tile-CD sweeps on the same Hessian.
- **Src:** FA-H1 (prior proposal). New evidence:
  - MPQ 2610.32127: Llama-3-8B 2-bit E8P Wiki 12.76 (LDLQ + CD) → 11.63; 3-bit scalar 7.95 → 7.81.
  - BCJR-QAT 2605.10655: the BCJR relaxation of a QTIP trellis works.
- **Why:** greedy commitment, in the tail-biting and feedback order. Our tile CD measured 0.5–1.2% on attn_qg only (#77), so the headroom for local moves is small; VAMP changes the basin.
- **Fit:** per-level quantizer, no format change.
- **Test:** settle64 CPU proxy, 27 tensors × 2 samples, K2/K2.5. Pass: VAMP + CD ÷ (LDLQ + cd3) ≤ 0.98. Colo, ≈2 days of implementation plus 6 CPU-h.
- **Gain:** −1 to −3% at 2.5. **P:** 0.12.
- **Conflicts:** #77, #113 (search width), FA-H1.
- **Reserve:** none.
- **2–3 bpw:** MPQ's gains are largest at 2 bit (−9%) and small at 4 bit.
- **Falsifier:** proxy ratio > 0.99.

**FB-E2 · Discrete KL-guided code flips (Ising healing)** — ALIVE, very low
- **Mech:** After quantization, move individual trellis paths by one step using a KL gradient plus the Kronecker energy, accepting a step only if held-out KL falls (TWIG's schedule: move fraction ×1.5 on accept, ×0.3 on reject).
- **Src:** TWIG 2610.31931. Its own ablation: KL 0.491 → 0.371, while scale tuning reached 0.325.
- **Why:** reaches code choices scales cannot.
- **Fit:** per mix.
- **Test:** 0.8B K2 after heal0. Pass ≤ 0.98.
- **Gain:** 0 to −1%. **P:** 0.04.
- **Conflicts:** #89.
- **Reserve:** none.
- **2–3 bpw:** coarser codes.
- **Falsifier:** ≥ 0.99.

**FB-E3 · Partial-denoising statistics (parallel quant that anticipates upstream error)** — ALIVE, very low
- **Mech:** Inject Gaussian noise at every linear's input and propagate it through the network (one extra FP pass), collect H, C = E[xηᵀ] and N = E[ηηᵀ], then quantize WF in metric G with ρ < λ (partial denoising).
- **Src:** 2610.07522 (closes 18–31% of the parallel→sequential gap; with local-only noise 11%, propagated 28%). Differs from #88, which used level-matched local noise and failed (K3 1.031, K2 1.10).
- **Why:** EXL3 quantizes from clean-input Hessians, so compounding error is unmodelled.
- **Fit:** per-level statistic.
- **Test:** 0.8B, λ, ρ ∈ {(0.4, 0.3), (0.2, 0.1)}. Pass ≤ 0.98.
- **Gain:** 0 to −2%. **P:** 0.05.
- **Conflicts:** #88.
- **Reserve:** none.
- **2–3 bpw:** compounding is worst at low bpw.
- **Falsifier:** ≥ 1.00 at both settings.

**FB-E4 · Annealed / replica LDLQ ensembles** — KILLED-earlier
- Best-of-N seeds: spread 0.5%, not reproducible (#41). Tile CD shows ≤ 1.2% local headroom.

**FB-E5 · Unequal error protection by bit significance (ECC view)** — KILLED-arith
- Fractional K and the knapsack already set the rate per tensor. Within a tensor, slabs are closed (#96, codes lens D-1/D-2).

**FB-E6 · Wrap-around (tail-biting) decoding to recover the 4–6% loss (#57)** — KILLED-earlier
- The loss comes from the tail-biting *constraint*: the 16 − K start bits per tile cannot be stored. Search is already near-ML (#57; #102 chained negative; stored start state costs +0.22 bpw, per CurveTQ).

### F. Codecs and representations far from per-tensor trellis

Most of this theme is bounded by the audit: the trellis is 0.07 bit from ideal, plus 4–6% tail-biting, plus at most 3–4% from a better 20-bit table (#113).

**FB-F1 · Random-coding residual (SeedLM on the trellis residual)** — KILLED-arith
- A b-bit seed choice per 256-weight tile from 2^b Gaussian vectors removes ≈ 2b·ln2/256 of the residual energy. For b = 16 (1/16 bit per weight) that is 8.7%; spending the same 1/16 bit in the trellis rate gives 2^{−2/16} → 8.3%.
- Equal at best, with a 2^16 search per tile.

**FB-F2 · Entropy-coded trellis indices or variable-rate tiles** — KILLED-arith
- Bitshift-trellis indices are near-uniform (entropy ≈ K bits).
- Variable rate would need load-time expansion (the target is VRAM) or a new decode path.

**FB-F3 · Leech / E8 / spherical lattice VQ in place of the trellis** — KILLED-earlier
- Equal to QTIP or worse at 2 bpw (2609.02652, 2609.35465, LiftQuant 2606.04050). The lattice kernel reads 2.4× more bytes.

**FB-F4 · Learned 2^20-state table with near-exact search** — KILLED for V3 (#113: 3–4% bound with exact search, encode ×13–46). V4 note stands.

**FB-F5 · Nearest-Kronecker-product core plus trellis residual** — ALIVE, very low, CRAZY-CHEAP
- **Mech:** W ≈ A ⊗ B (Van Loan rearrangement SVD) at high precision, plus a trellis on the residual.
- **Src:** new for our weights; Kronecker structure appears in K-FAC, not in weights.
- **Why:** only worth it if real weights have Kronecker energy far above Gaussian (≈ 1/min-dim for iid). Round 1 found Gaussian weights.
- **Test:** CPU, minutes: top-1/top-8 NKP energy of 10 tensors vs Gaussian surrogates. Pass ≥ 5× surrogate.
- **Gain:** 0 likely. **P:** 0.02.
- **Conflicts:** prior M2 (MP artefact).
- **Reserve:** none.
- **Falsifier:** energy ≤ 2× surrogate.

**FB-F6 · Neural post-filter of decoded weights (codec in-loop filter)** — KILLED-arith
- For Gaussian sources with near-white trellis error, the optimal post-filter is the MMSE shrink, which the svh/suh refit already applies.

**FB-F7 · Subtractive dither (unbiased reconstruction)** — KILLED
- Turns the coherent shrink (handled by swj/C9) into noise, at no MSE gain. M0.1: shrink-type error was not worse than iid error (0.0508 vs 0.0539).

**FB-F8 · HashedNets / whole-model dictionary / INR hypernetwork** — KILLED-earlier (prior M1/M5, FA-K1; #119 for embeddings: ×1.03–1.09 of the Gaussian bound).

**FB-F9 · Vocabulary pruning (unused rows at 0 bits)** — KILLED-today (M0.10): 243 padding rows, 0.1%.

**FB-F10 · Byte-composed embeddings for rare tokens** — ALIVE, very low
- **Mech:** Generate the embedding rows of rare tokens from their UTF-8 bytes with a tiny shared bf16 network plus a low-bit residual.
- **Src:** n-gram embedding papers 2610.31646, 2610.35985.
- **Why:** #119 found no cross-row structure via the head or KLT, but byte-composition structure was not tested.
- **Test:** CPU: R² of a 2-layer byte-CNN predicting the rows of 50k rare tokens. Pass ≥ 0.3.
- **Gain:** ≤ −0.5% at ≤ 1.5 bpw, ≈0 at 2.5. **P:** 0.02.
- **Conflicts:** #119, #32.
- **Reserve:** none.
- **Falsifier:** R² < 0.1.

**FB-F11 · Joint sibling coding (gate/up, q/k, GDN qkv/z)** — KILLED-today (M0.8: 0.006–0.014 bit).

**FB-F12 · Wyner–Ziv down_proj (side information from decoded up/gate, without an extra GEMM)** — KILLED
- The A1 gain (0.09–0.23 bit) is value prediction, which needs the extra GEMM.
- Side information that only sets scales or seeds carries no information (scales are already cheap).

### G. Model restructuring

**FB-G1 · Residual-stream slicing** — KILLED-today (M0.5, 11×).

**FB-G2 · MLP neuron pruning, unhealed** — KILLED-today (M0.6, 15×). The healed version is FB-D6.

**FB-G3 · Layer drop, unhealed** — KILLED-today by inference: 5% of one block family already costs 0.13 KL; one whole layer is worth only 0.036 bpw. The healed version is FB-D7.

**FB-G4 · Cross-layer sharing, merging or looping** — KILLED-earlier (prior M1: adjacent-layer cosine at the random baseline).

**FB-G5 · MLP → MoE expert split** — KILLED: compute only, same bytes.

**FB-G6 · Full attention → GDN conversion** — KILLED: 372M vs 383M parameters per layer, no byte change.

**FB-G7 · Tie embedding and head (one table, 0.118 bpw)** — KILLED-earlier (#119: the head predicts the embedding poorly, least squares leaves 85%). Tying needs retraining.

**FB-G8 · Remove the attention output gate (q_proj's gate half, 1.9% of parameters)** — ALIVE, very low, CRAZY-CHEAP
- **Mech:** Replace the sigmoid output gate of the 16 full-attention layers with a per-head constant fit on calib, freeing 0.047 bpw.
- **Test:** 9B CPU (I can run this): KL of the gate-removed bf16 on 16 docs. Pass ≤ 0.004.
- **Gain:** −8% if free. **P:** 0.02 (the gate is input-dependent by design).
- **Conflicts:** codes lens 2.4 (split q/gate).
- **Reserve:** none.
- **Falsifier:** KL > 0.01.

**FB-G9 · GDN head pruning / KV head merge** — KILLED (FA-E6 P 0.07; k/v projections are 10.5M parameters per layer, too small to matter).

**FB-G10 · Slice only the embedding and head (vocabulary-side low rank)** — KILLED (#119: KLT on unit-RMS rows ≤ 0.093 bit).

### H. Radically different uses of the bit budget

**FB-H1 · Logit-average two half-size models** — KILLED-arith
- Averaging two independent errors at best halves variance-type KL. Two 1.25-bpw models would need KL(1.25)/2 < 0.0666, but even two 2.0-bpw models (4.0 total) give 0.164/2 = 0.082.

**FB-H2 · Mixture of category-routed correctors** — ALIVE, CRAZY
- **Mech:** K ≈ 5 tiny per-category adapters (agent / code / math / prose / multilingual), each rank-16 on the final state and a few writers, ≈ 0.001 bpw each. A router picks one per document from the first 256 tokens. Trained per mix with KL.
- **Src:** new. M0.7: the agent drift is linear and category-specific, and a single map that fixes it hurts the other categories.
- **Why:** the category KL ratio is 20–75×, and one correction cannot serve all categories (M0.7).
- **Fit:** per mix plus an engine router op.
- **Test:** 0.8B real K2 quant. Pass: verify ≤ 0.97 with every category ≤ 1.00. ≈4 h.
- **Gain:** −1 to −5% (very uncertain). **P:** 0.05.
- **Conflicts:** FA-G1 (per-document adaptivity rejected because bytes are fixed; here bytes are fixed and tiny).
- **Reserve:** an optional per-category adapter table in qwen35_exl3 (post-freeze).
- **2–3 bpw:** drift is largest at low bpw.
- **Falsifier:** the routed adapters don't beat one global adapter (FB-C4).

**FB-H3 · Bytes into exact copy paths: full-attention k/v at 8 bit everywhere** — ALIVE (variant of FB-B1)
- **Mech:** The 16 layers' k_proj and v_proj total 168M parameters; raising them from 2.5 to 8 bits costs 0.034 bpw (≈ +6% KL if taken from elsewhere). Test as a forced knapsack option against agent-subset KL.
- **Src:** M0.4; copy behaviour in agent traces.
- **Why:** long-range exact copying (tool names, JSON keys, file paths) runs through full attention.
- **Fit:** knapsack forced option.
- **Test:** 27B one eval pair, ≈1 card-h.
- **Gain:** −1 to +2%. **P:** 0.15.
- **Conflicts:** #118 (attn_k "overpriced").
- **Reserve:** none.
- **2–3 bpw:** k/v are cheap to protect.
- **Falsifier:** the 0.8B analogue does not lower agent KL.

**FB-H4 · Spend bytes on bf16 exceptions chosen by flip attribution** — ALIVE, very low
- **Mech:** Store about 10^6 exact weights with indices (0.002 bpw), chosen by FB-B2's flip attribution rather than magnitude.
- **Src:** SpQR (magnitude), new selection.
- **Why:** sensitivity is only useful here if it is concentrated. #120 showed energy concentration is not error concentration.
- **Test:** 0.8B oracle: restore the top-10^5 attributed weights to bf16. Pass ≤ 0.97.
- **Gain:** 0 to −1%. **P:** 0.03.
- **Conflicts:** #120-02 (restore oracle 1.0000).
- **Reserve:** none.
- **Falsifier:** oracle ≥ 0.99.

**FB-H5 · int8 activations as a free regulariser** — KILLED: speed only (A8-g128 ×1.0001, #46-03).

**FB-H6 · Store information in uncounted tensors (MTP / drafter / vision)** — REJECTED: against the rules' intent. FA-D1's rules question stands; I do not re-propose it.

**FB-H7 · Per-document dynamic precision** — KILLED-earlier (FA-G1): all stored bytes count.

**FB-H8 · "All bits to mixers and head, MLP at sub-1 with low rank"** — covered by the knapsack plus the lowbpw lens. M0.4 suggests the knapsack's MLP floors may be too high only if FB-B1 passes.

### I. Hybrid- and data-specific

**FB-I1 · Long-trace copy census → attention priority** — this is diagnostic 5 (§3); it drives FB-B1/FB-H3.

**FB-I2 · Sink-aware and copy-aware q/k statistics** — covered by the codes lens 3.4.

**FB-I3 · Mean/bias correction per linear** — covered by #120-03 (0.8B K3 −1.1%, post-freeze option).

**FB-I4 · Gen-only H_in** — KILLED-earlier (#86).

**FB-I5 · Scored-position-only head statistics** — covered by #83/#34.

### J. Crazy but worth one cheap test

**FB-J1 · Copy-continuation logit prior (zero bytes, runtime)** — CRAZY-CHEAP
- **Mech:** When the last n ≥ 4 tokens match an earlier span in context, add β to the logit of that span's continuation token. Fit β (one or two numbers) on fit docs to minimise KL vs bf16. The idea is that agent and SWE flips are often failed copies of earlier text.
- **Src:** prompt-lookup decoding, induction heads. New as a KL repair.
- **Why:** SWE-trace KL 1.04 and JSON key flips (M0.3).
- **Fit:** engine op (a small logit kernel).
- **Test:** 9B synthetic noise on colo, ≈1 CPU-h; then the 0.8B real quant. Pass: agent docs ≤ 0.95, others ≤ 1.00 on B.
- **P:** 0.05.
- **Conflicts:** none.
- **Reserve:** none.
- **Falsifier:** B ratio ≥ 0.99.

**FB-J2 · Context-keyed sparse logit bias (structural-token contexts)** — CRAZY-CHEAP
- **Mech:** A bias table keyed by (previous-token class, target ∈ about 100 JSON/code syntax tokens), fit on fit docs.
- **Why:** this is the context-conditioned version of M0.1's failed global bias.
- **Test:** 9B, ≈1 CPU-h. Pass B ≤ 0.97.
- **P:** 0.03.
- **Falsifier:** B ≥ 1.00.

**FB-J3 · Category-routed final-state corrector, closed form** — CRAZY-CHEAP
- **Mech:** Repeat M0.7 with a real quant: a ridge map fit only on agent docs, applied only when a doc classifier says "agent". This tests whether the SWE result (1.04 → 0.034) survives real quantization error and held-out documents.
- **Test:** 0.8B real K2, cached hidden states, ≈1 h. Pass: agent held-out ≤ 0.8 and others unchanged.
- **P:** 0.08 that the effect is real at 0.8B. This is a cheap precursor to FB-H2.
- **Falsifier:** agent held-out ≥ 0.95.

**FB-J4 · Seed lottery for the 16 attention layers scored on agent KL** — CRAZY-CHEAP
- Four sign seeds for attention tensors only, picked by agent-slice KL on the search pool.
- **P:** 0.03 (#41: 0.5% spread on proxy error).
- **Falsifier:** the best seed is not reproduced on verify.

**FB-J5 · Test-time adaptation of scales on the document prefix** — CRAZY (FA-G2, prior). No new angle; not re-proposed.

**FB-J6 · 9B real-quant replication of M0.2–M0.4** — CRAZY-CHEAP diagnostic
- Run the same tail and block-type anatomy with actual EXL3 levels instead of synthetic noise, to check that "mixers 5× per parameter" survives LDLQ shaping.
- ≈4 h on the 4070 with 0.8B levels.

**FB-J7 · Structural-token "format head"** — CRAZY
- **Mech:** A 20M-parameter bf16 predictor (0.012 bpw) that outputs logits only for about 200 structural tokens, mixed in with a learned gate.
- **P:** 0.02.
- **Falsifier:** an oracle mixture (true bf16 logits at structural positions) gains < 10% → kill.

**FB-J8 · Mode-covering under-confidence (soften q at runtime)** — KILLED-today (M0.1).

**FB-J9 · The oracle that bounds the whole theme** — CRAZY-CHEAP
- **Mech:** On the 27B golden rows of an existing K2.5 pair, replace the quant's distribution with bf16 at the top-1% KL positions only, and recompute the mean.
- **Why:** this is the ceiling for every flip-targeted idea. If it is < 50% of the score, theme A is overrated.
- **Test:** CPU, minutes.
- This is diagnostic 1.

**Count:** 90 entries (FB-A1–A8, B1–B6, C1–C7, D1–D10, E1–E6, F1–F12, G1–G10, H1–H8, I1–I5, J1–J9). Of these, 46 are alive or crazy-cheap. The killed ones carry today's numbers or the arithmetic that kills them.

---

## 2. Ranked top 20 (expected gain per unit cost)

| # | idea | gain at 2.5 (unc.) | cost | P | why this rank |
|---|---|---|---|---|---|
| 1 | FB-J9 / diagnostic 1: tail-share oracle on 27B rows | gates themes A/B | CPU, minutes | — | decides whether everything below matters |
| 2 | FB-A3 category-matched statistics | −0.5…−2% | 2 h 4070; 27B ≈ free if block categories are recoverable | 0.25 | direct fix of a measured 4× under-sampling |
| 3 | FB-A5 margin-weighted head plan | −0.3…−1% | CPU headsim, minutes; 1 pair | 0.30 | cheap; the head is 5% of calib-part KL |
| 4 | FB-B1 mixer-first allocation test | 0…−3% | 0.8B 1 h + 27B 1 card-h | 0.25 | M0.4 effect is 5×; a null still confirms the knapsack |
| 5 | FB-D2 GDN decay healing | −0.5…−2% | 0.8B 1 h | 0.15 | zero bytes; untested parameter group |
| 6 | FB-A1 flip-risk H_out | −1…−4% | 0.8B 3 h; 27B 5 card-h | 0.20 | attacks the score's dominant term |
| 7 | FB-D3 GDN conv healing | −0.5…−1.5% | 0.8B 1 h | 0.12 | zero bytes |
| 8 | FB-H3 attention k/v at 8 bit | −1…+2% | 1 card-h | 0.15 | cheap forced option |
| 9 | FB-A4 agent-matched healing data | −0.5…−2% | 0.8B 3 h | 0.20 | healing is planned anyway |
| 10 | FB-A2 KL-reweighted IRLS statistics | −1…−4% | 0.8B 3 h; 27B 12 card-h per round | 0.15 | closed-loop tail weighting |
| 11 | FB-J3 routed final-state corrector (closed form) | precursor | 0.8B 1 h | 0.08 | checks M0.7 on a real quant |
| 12 | FB-B3 heal-aware knapsack | −1…−3% | 0.8B 4 h | 0.15 | V4 but cheap to test now |
| 13 | FB-B2 flip-attributed sensitivity prior | 0…−2% | 0.8B 2 h | 0.12 | reduces knapsack noise |
| 14 | FB-J1 copy-continuation prior | 0…−3% | CPU 1 h | 0.05 | crazy-cheap |
| 15 | FB-C4 KL head adapter | 0…−2% | 0.8B 2 h | 0.10 | tiny bytes |
| 16 | FB-E1 soft trellis VAMP + CD | −1…−3% | days of development + CPU | 0.12 | the only codec-side path left |
| 17 | FB-B4 differentiable allocation audit | via audit | 0.8B 4 h | 0.08 | bounds the knapsack's gap |
| 18 | FB-A7 27B self-continuations of agent prompts | −0.5…−1.5% | 5–10 card-h | 0.08 | more agent samples, on-policy |
| 19 | FB-D9 per-mix sequential statistics with trust region | −1…−3% | 27B ≈ 12 card-h | 0.06 | prefix drift |
| 20 | FB-D1 STE trellis QAD | −1…−4% | rental | 0.08 | V4 |

---

## 3. The five cheapest diagnostics that would most change this ranking

1. **27B tail anatomy from stored per-position rows** (CPU, minutes; rows on gpubox from #117/#120/#34-02).
   - Measure:
     - share of mean KL in the top 0.1 / 1 / 10% of positions;
     - share per category (agent / code / math / chat / multilingual);
     - the token classes at the top positions (JSON or tool syntax, code syntax, think tags);
     - bf16 entropy and margin there;
     - for one adopted method pair (bn vs base), whether its gain is in the tail or the bulk.
   - Add FB-J9's oracle: substitute bf16 at the top 1%.
   - If the top 1% carries < 30% of the K2.5 score, theme A drops several ranks.
2. **Category composition of the 27B Σdy² 128-sequence blocks** (CPU, seconds: replay the seeded window order against pools-v3).
   - Do blocks 6 and 9 (#33-27's heavy blocks) consist of agent traces?
   - If yes, bn already down-weights agent text and helped, which argues against FB-A3/A4.
   - If no, FB-A3's 27B version is nearly free.
3. **Fisher mass vs margin mass at the top-1% KL positions of a 0.8B K2 quant** (4070, 30 min). What fraction of the sampled-label Σdy² mass sits at the positions that carry half the KL? If ≥ 30%, FB-A1/FB-A5 add little.
4. **Equal-bytes mixer↔MLP shift on the 0.8B with real levels, KL per category** (4070, ≈1 h). Does M0.4's synthetic-noise result (mixers 5× per parameter on tool text) survive LDLQ-shaped error? This decides FB-B1/FB-H3 and validates the synthetic-noise proxy behind this report.
5. **Copy-event census** (CPU, from 27B rows plus tokens). Among the top-1% KL positions, what fraction have a bf16 top-1 token that continues an n ≥ 4 match from earlier context? If > 40%, FB-J1 and FB-H3 move into the top 10; if < 10%, drop them.
