# Lens **measure**: ideas report (2026-11-07)

I ran two cheap checks for this report on colo's CPU (under 5 minutes in total) and read the relevant code. Everything below that isn't marked "measured here" is a proposal.

## Findings from the checks

1. **Σdy² as a sensitivity prior (measured here).** I compared the per-row mean of the 27B block-normalized Σdy² diagonal (`/data/work/hout27/hout-bn`, 400 tensors) with kfit2's fitted per-tensor sensitivity `s` (`data/v3/kfit2-model-final.json`, 336 tensors).
   - Spearman over all tensors is **0.82**.
   - A fit of log s against log g with one intercept per family has a **shared slope of 0.49**, not the 1.0 the Kronecker/Gauss–Newton price predicts. Its residual log-SD is 0.58, against a within-family log-SD of 1.09, so it explains about 70% of the variance.
   - The fit is good for the writers and the GDN inputs (slopes 0.30–0.65, residual 0.28–0.43).
   - It fails for the attention inputs (q/k/v). There kfit2's `s` is nearly flat with depth (range ×11) while Σdy² falls ×50–170 with depth (slope ≈ 0.03–0.06).
   - Conclusion: Σdy² is a usable prior for pricing levels and families that have no measurements, but only with a calibrated exponent and with attention checked separately (ideas B3, D1).
2. **Possible sub-additivity in V2 data (measured here).** Source: V2 group swaps in `data/v2/results.jsonl`, all on the same 90,802-position slice, base sc4.00, base KL 0.00806.
   - Attention inputs at plain-2-bit, window L00: the single-layer probes at L00 and L08 sum to +0.00122. The whole 16-layer window (4 attention layers, including those two) gives +0.00110.
   - Window L16: the probes at L16 and L24 sum to +0.00224; the whole window gives +0.00215.
   - If the layers added up and were similar, each window would be about twice its two singles. The observed ratio is about 1×, roughly 7 paired SE (about 0.00012 each) below that.
   - Caveats: one seed, attention only, and the single-layer rows lack the golden tag (the position count matches the window rows, so it is very likely the same slice).
   - kfit2's model compounds errors across depth (super-additive), so within-family saturation would be a structural misfit. Idea C1 tests it.
3. **knap5 code (read).**
   - It prices ffn_gate_up as one tensor (geometric-mean proxy), so gate and up are tied in K and the swj partner mismatch cannot arise today.
   - `KN_SWJ_PTS` still defaults to the withdrawn `4:0.7786` (`knap5.py:20,370`). Only `mkjobs2.py` overrides it.
   - The V3 integer levels are still priced from V2's hproxy × DAMPF (the file's own caveat). The planned replacement, "ladder hproxy via .rad import", would measure tensors after swj (idea B1).

Each idea lists: mechanism; **Source**; **Why**; **Fit/Freeze**; **Test** (hardware, card-hours, pass criterion); **Gain/Risk** (KL % at 2.5 bpw; "enabling" means it changes decisions or cost, not KL directly); **Conflicts**. Card-hour arithmetic assumes about 30 evals per hour per R9700 on a ~100k-position slice, about 37 minutes per level without H_out and 3.2 h with it.

---

## A. Decision statistics and the 27B noise floor

**A1. Check whether the seed spread replicates on independent documents.**
Score base, s1 and s2 (K2.5, swapped into V2 3.00) on golden-verify27. Most score files exist: base and s1 were scored on verify by the swj insert. On search, base→s1 was 0.975 (z −2.5), s1→s2 1.010 and base→s2 0.985.
- If base→s1 is about 0.975 again on verify, the "seed floor" is a real difference between models (a rounding lottery). It can then be exploited (A3).
- If it is about 1.00, the floor is mostly document noise. The paired bootstrap already counts that, and the H27-style rules count it twice.
- **Source:** new; the A/A calibration of 2609.18005 applied to quantization seeds.
- **Why:** every 27B rule leans on σ_seed (0.018 at K2.5, 0.0062 at K4).
- **Fit/Freeze:** method. Should run before #105/#112 are judged.
- **Test:** 27B, one eval for s2 on verify, the rest pcmp on existing files. 0.1 card-hour. "Real" if |log(base/s1)| on verify is within 1 SE of the search value.
- **Gain/Risk:** enabling. No risk.
- **Conflicts:** #33-28. Not a re-litigation of H_out, only the rule for future tests.

**A2. Layer-split "virtual seeds".**
Build mixes from two existing whole-model quants of one recipe (base and s1 at K2.5), taking layer groups (halves, quarters, alternating) from either. About 8 evals give σ_seed from two quants instead of three or more. Under additivity they also show *where* the seed effect lives.
- **Source:** BACKLOG review idea 2. New here: using it for localization and a σ_seed(K) table.
- **Why:** a 1.8% whole-model seed σ implies about 15% per-layer relative noise if spread evenly, or a few hot layers if not. Those need different remedies.
- **Fit/Freeze:** method.
- **Test:** reflink mixes on gpubox, 8 evals (0.3 card-hour) at K2.5. Repeat at K3/K4 with any existing seed pairs. Pass if the halves sum to the whole within 1 SE.
- **Gain/Risk:** enabling. Interactions blur the attribution.
- **Conflicts:** none run.

**A3. Best-of-N seeds, selected by KL, only on the hot layers.**
If A1 and A2 show a real effect concentrated in a few layers, requantize only those tensors with 3 extra rotation seeds per shipped level, select by KL on search and confirm on verify.
- **Source:** new. #41 (best-of-4 by proxy) was null because the proxy's seed spread is 0.5% while the KL spread is 2.5%; the proxy cannot see this, KL can.
- **Why:** half of a 2.5% spread is about 1%.
- **Fit/Freeze:** per level (the chosen seed is per tensor). Post-freeze.
- **Test:** only after A1/A2. About 2 card-hours of quants plus 10 evals.
- **Gain/Risk:** 0.5–1.5% if real. Winner's curse, so confirmation on verify is mandatory.
- **Conflicts:** #41.

**A4. A variance-components decision model.**
Fit log per-document KL with crossed random effects (document, seed, arm), pooling the search and verify documents and all arms. Decide on the posterior P(effect ≤ −x). This replaces "geo ≤ 0.98 and |log geo| ≥ 2σ/√n", whose σ (taken from three pairs) already contains document noise.
- **Source:** new (standard mixed model).
- **Why:** it separates about 1% document noise from about 1.5% seed noise and uses all arms together.
- **Fit/Freeze:** method, CPU only.
- **Test:** refit #33-28's arms offline and compare the verdicts with the rule's.
- **Gain/Risk:** enabling. The model must be written down before data.
- **Conflicts:** H27 amendments (future use only).

**A5. A power calculator before every 27B test.**
Compute the number of seed pairs needed from σ_seed(K) (A2), the slice's document SE and the class-shrunk 0.8B effect (J1). Refuse underpowered tests. Example: #106's o_proj −3% proxy error is probably below −0.5% KL at 27B, which would need more than 10 pairs at K2.5. Not worth 27B time.
- **Source:** new.
- **Fit/Freeze:** method; decides 27B queue slots now.
- **Test:** CPU, an hour.
- **Gain/Risk:** saves card-hours. No risk.
- **Conflicts:** #106 and #112 queue decisions.

**A6. Group-sequential 27B pairs with futility stops.**
Pre-register interim looks after each pair (O'Brien–Fleming spending). Stop when the remaining pairs cannot reach the bar.
- **Source:** new.
- **Why:** a 3-pair K2.5 test that is futile after pair 2 saves about 1.5 card-hours.
- **Fit/Freeze:** method.
- **Test:** none; written into the next rule.
- **Gain/Risk:** about 30% fewer pairs.
- **Conflicts:** none.

**A7. Split-half coverage check of pcmp's bootstrap.**
For many past comparisons, compute the paired difference on random document halves. Var(diff_A − diff_B) should match the bootstrap prediction. If it is larger, the per-document bootstrap under-covers (heavy tails, a few documents dominating) and should become BCa/studentized or use more documents.
- **Source:** new.
- **Fit/Freeze:** method.
- **Test:** CPU on existing `.f32` score files.
- **Gain/Risk:** enabling.
- **Conflicts:** none.

**A8. A category-stratified estimator and bootstrap.**
pcmp resamples documents without strata. Stratify by category (slices are category-matched) and weight strata by the board's category token shares.
- **Source:** new (small pcmp change).
- **Why:** lower SE, and the estimator then targets the board's mix.
- **Fit/Freeze:** method.
- **Test:** CPU re-run of 10 recent comparisons. Expect SE −5…−15% (a guess).
- **Gain/Risk:** enabling.
- **Conflicts:** none.

**A9. Influence diagnostics on every comparison.**
Report the largest drop-one-document change in the ratio and the share of the difference coming from the top 5 documents. Flag decisions carried by one document.
- **Source:** 2610.14934 (the worst token decile carries 45% of KL); 2606.03458.
- **Fit/Freeze:** method, minutes.
- **Gain/Risk:** enabling.
- **Conflicts:** none.

**A10. A/A null controls in the stream.**
About 1 eval in 50 is an identical mix (repacked) or an identical-trunk pair. z should follow N(0,1) and the engine should be deterministic.
- **Source:** 2609.18005.
- **Fit/Freeze:** method.
- **Test:** 2% of eval time.
- **Gain/Risk:** catches silent drift (stale `.so`, kernel changes).
- **Conflicts:** none.

**A11. A mix-identity manifest check.**
Before any paired score, diff the (name, encoding, bytes, sha) manifests of both containers and refuse if anything outside the declared changed set differs. A head swap alone moved K4 KL ×0.76 (11-06).
- **Source:** the LOG 11-06 TODO.
- **Fit/Freeze:** hygiene; pre-freeze, minutes.
- **Gain/Risk:** prevents errors the size of the 20% swj episode.
- **Conflicts:** none.

**A12. A stale-constant audit.**
Make `KN_SWJ_PTS` (default `4:0.7786`, withdrawn), DAMPF and the "×1.02 i8 K7" leftovers refuse to run without an explicit value. Grep configs and scripts for withdrawn numbers.
- **Source:** code reading.
- **Fit/Freeze:** hygiene; pre-freeze, minutes.
- **Gain/Risk:** avoids a silent K4+ repricing error.
- **Conflicts:** w58/w63.

---

## B. Recipe-consistent prices for the knapsack

**B1. Price swj levels by the proxy before the refit.**
hproxy (tr(EHEᵀ) per row) is exactly minimized by the least-squares scale c\* = ⟨W,Ŵ⟩_H/‖Ŵ‖²_H. swj moves the gate/up scales off c\* on purpose (0.8B logs: c ≈ 1.07–1.08 ± 0.10). The input-side proxy then rises by Σ_r‖Ŵ_r‖²_H(c_r − c\*_r)², which is +0.006…0.016 of ‖W‖²_H. Against a K2.5 relative error near 0.04 that is about +15–40%, while KL falls 6.5%. The planned "ladder hproxy via .rad import" (knap5's caveat) and rqah's score-based verification both see post-swj tensors. They would rate swj'd gate_up as worse, and the ×0.935 factor would then go on top. Keep a pre-refit proxy column (ourq's logged proxy) for pricing and use the post-refit numbers only for integrity checks.
- **Source:** new (algebra); brief §1.
- **Why:** gate_up is the largest family.
- **Fit/Freeze:** per-level pricing. Pre-freeze for the price-table schema; the quantizer is unaffected.
- **Test:** CPU. hproxy on one layer's gate/up before and after `swjapply` at K2.5 and K4. Minutes. Act if the ratio exceeds 1.05.
- **Gain/Risk:** avoids a misallocation of up to 1–3% at 2.5. No risk.
- **Conflicts:** none.

**B2. A recipe factor at the K4 H_out boundary.**
Two-sided LDLQ trades input-side error for output-side error. The input proxy of a bn4 level is higher than base4's while its KL is lower (0.964). Priced by the input proxy, K3.875 (one-sided) against K4 (two-sided) is mispriced in favour of staying below K4. Measure the bn4/base4 proxy ratio per family from the arms' `proxy.jsonl`; set f_fam = (KL ratio) / (proxy ratio).
- **Source:** new; #33-28.
- **Why:** sizes 3.5–4.5 straddle the boundary.
- **Fit/Freeze:** pricing; needed before knap runs on the ladder.
- **Test:** CPU, minutes.
- **Gain/Risk:** 0.5–2% at 3.5–4.5, about 0 at 2.5.
- **Conflicts:** none.

**B3. A two-sided price with a calibrated exponent.**
Price each (tensor, level) by Σ_r d_r^β e_r H e_rᵀ, with d = the 27B Σdy² diagonal (on colo) and β fitted (my check suggests about 0.5, not 1). Refit kfit2 on V2's measurements with this price; compare CV log-RMSE and pair ranking.
- **Source:** BACKLOG w08-hprice; 2609.28270 (price = ½tr(HΣ), ρ = 0.93–0.95 with KL); 2607.07964 (KronQ); the check above.
- **Why:** gives new recipes (H_out levels, i4) prices without V2 data.
- **Fit/Freeze:** pricing.
- **Test:** colo CPU, 2–4 h. Pass if CV log-RMSE drops ≥10% or pair ranking improves beyond bootstrap noise.
- **Gain/Risk:** 0–2%.
- **Conflicts:** #33-29 (estimators). This is about pricing, not quantizer weights.

**B4. A SwiGLU-junction proxy for gate_up.**
Price gate_up by E‖W_d(silu(ĝ)û − silu(g)u)‖² on 2k calib tokens per (layer, K). This sees swj's gain, so the global f(K) multiplier becomes a per-layer factor.
- **Source:** 2610.41408 (SACQ's SwiGLU row metrics); 2609.33923 (on Qwen3.8-27B, block-output cosine allocation gave 6.041 PPL against 6.087 for the per-layer objective oracle).
- **Fit/Freeze:** per level.
- **Test:** needs decoded swj levels from gpubox; CPU at about 1–2 s per (layer, K). Validate against the 27B swj ratios (0.935 at K2.5, 0.989 at K4).
- **Gain/Risk:** 0.5–1%. Moderate risk: the local metric may miss downstream compounding.
- **Conflicts:** none.

**B5. Pricing writers after C9 α (if #112 is adopted).**
With α ≈ 3–4.75 on gdn_out, the input proxy of the writers would blow up. Price them by the pre-α proxy times a measured factor from #112's 27B pair, and check that α is stable across K.
- **Source:** #112.
- **Fit/Freeze:** pricing; pre-freeze if #112 passes.
- **Test:** CPU.
- **Gain/Risk:** avoids gross mispricing of the writers.
- **Conflicts:** #112.

**B6. Uniform-level KL anchors for every ladder level.**
Score each ladder level once as a uniform trunk (fixed V2 3.00 head and embedding) on the search slice. 71 anchors fix the global K-curve with every recipe effect included (i8 saturation, swj, H_out, CD), at about 2 minutes each (about 2.5 card-hours over the ladder). Fit kfit2's global and recipe terms to them.
- **Source:** new (w63 used three points).
- **Fit/Freeze:** per level, as levels land.
- **Test:** an anchor's measured value against knap5's uniform prediction. Refit if off by more than 3%.
- **Gain/Risk:** 0.5–2% at sizes priced by interpolation.
- **Conflicts:** w58/w63 repricing.

**B7. A small V3 calibration set of family factors.**
kfit2's `phi` (ours_lo/mid/hi, sc, plain) describes V2 families. Twelve family-band group swaps, V3 level against V2 level at K2.5 and K4 (24 evals, about 1 card-hour), anchor V3 family factors. That covers i8 codebook, damping and CD together.
- **Source:** #13 (families differ by 1.15–1.23×); B6.
- **Fit/Freeze:** pricing.
- **Test:** 1 card-hour.
- **Gain/Risk:** 0.5–2%.
- **Conflicts:** none.

**B8. A fixed-code swj-in-mix pair.**
Build the w63 knapsack 2.5 plan with swj on and off (identical codes, so no seed noise). Compare the measured ratio with knap5's "gu" prediction. If they differ by more than 2%, swj's pricing in mixes is wrong.
- **Source:** new.
- **Fit/Freeze:** pricing check.
- **Test:** 2 evals (0.1 card-hour).
- **Gain/Risk:** enabling. Tiny cost.
- **Conflicts:** #104 (does not re-test adoption).

**B9. Check top-end i8 pricing in KL.**
#94's ×1.27 at K7 is a proxy number. One family-band swap of i8-K7 against mul1-K7 measures the KL ratio (4 evals). This decides whether a fast 7.0 can beat V2 7.00 (0.00045).
- **Source:** #94, #109.
- **Fit/Freeze:** pricing, 7.0 only.
- **Test:** 0.2 card-hour.
- **Gain/Risk:** none at 2.5.
- **Conflicts:** #94.

**B10. Clip and shrink negative measured sensitivities.**
At high K, isolated swaps often measure ΔKL < 0. That is noise, or quantization regularizing on the calibration set (2609.34765). Clip at 0 and shrink toward the family curve before the knapsack.
- **Source:** 2609.34765.
- **Fit/Freeze:** pricing hygiene.
- **Test:** CPU.
- **Gain/Risk:** small. Avoids "free" bits.
- **Conflicts:** none.

---

## C. Interactions and in-context sensitivities

**C1. A within-window additivity audit (follows check 2).**
At the 2.5 plan, demote by one K step: 4 single layers, 2 pairs and the full window, for 3 families (attn_in, gdn_in, gate_up) × 2 depth windows. If saturation within a family is confirmed, add a compressive term next to kfit2's cross-depth compounding.
- **Source:** the check above; 2610.09227 (composition is compressive, β 0.26–0.77); 2609.01587 (pairs ≈ sum within 0.91–1.10); 2607.12266.
- **Fit/Freeze:** pricing model.
- **Test:** about 40 evals (1.5 card-hours).
- **Gain/Risk:** if strong, the knapsack is spreading bits too evenly: 1–3%.
- **Conflicts:** #11 (that symmetric term was a global link; this one is local and within a family).

**C2. A family-level KKT audit at every finalist.**
Shift ±1/4 bit per family (6–9 families) around the plan. Effects are about 2–6% of KL, against a paired SE near 1%. If the measured marginal ΔKL/Δbyte differs by more than 1.5× between families, move bytes.
- **Source:** standard; 2608.30564 (in-context marginals).
- **Fit/Freeze:** per mix; finals.
- **Test:** 12–18 evals (about 0.6 card-hour) per size.
- **Gain/Risk:** 1–3% at 2.5 if the plan is off.
- **Conflicts:** none.

**C3. A factorial ±1/4-bit screening design around the plan.**
A Plackett–Burman design over 32 groups (family × depth quarter) with 64 evals. Effect SE is about 2σ/√64 ≈ 0.25% of KL, against group effects near 1%. This gives in-context marginal prices for a kfit2 refit at each size.
- **Source:** #14 (single swaps too noisy, use designs).
- **Fit/Freeze:** per size.
- **Test:** about 2.5 card-hours per size.
- **Gain/Risk:** 1–2%.
- **Conflicts:** none.

**C4. Lazy measured descent (Q-Strata).**
Start from the plan at size + 0.25 bpw. Repeatedly demote the group with the smallest measured ΔKL/Δbyte, using a lazy heap (about 2.5 evals per move).
- **Source:** 2608.30564 (JSD 749 against 823 for a one-shot ILP at 1.75 bits; 1,378 queries against AMQ's 10,474 for the same quality).
- **Fit/Freeze:** search.
- **Test:** 100–200 evals per size against the current local search at equal evals.
- **Gain/Risk:** 1–2% against a one-shot knapsack, less against V2-style local search.
- **Conflicts:** V2's local search (#70).

**C5. Budget-neutral swap moves screened with a pair-aware proxy.**
Local-search moves become one group up and one group down, ranked by kfit2 plus pair terms. Only the top-K are evaluated.
- **Source:** 2609.25916 (cross-term screening 22.41 PPL against 28.15 for self-term screening at 2.25 bpw).
- **Fit/Freeze:** search.
- **Test:** A/B with equal evals.
- **Gain/Risk:** faster convergence.
- **Conflicts:** none.

**C6. Cap the number of changes per accepted move.**
Composition error grows with the number of simultaneous moves: it was 25.9–41.7× the pricing error, and capping at 2 changes avoided harm (2609.09867). Limit proposals to ≤k changed groups and re-anchor kfit2 after each acceptance.
- **Source:** 2609.09867.
- **Fit/Freeze:** search rule.
- **Test:** none.
- **Gain/Risk:** fewer false accepts.
- **Conflicts:** none.

**C7. Within-layer pair terms (gate_up–down, q/k/v–o).**
Measure pair demotions against singles in 8 layers. If significant, add a pair term to the per-layer DP; the DP stays exact. Expect GDN to be near-additive: its projection errors add in quadrature (19.4% predicted against 19.2%, 2609.04098, on our model).
- **Source:** 2606.13054 (a pair-term DP gave 15.03→12.83 PPL); 2610.16368.
- **Fit/Freeze:** pricing.
- **Test:** about 24 evals (1 card-hour).
- **Gain/Risk:** 0–1%.
- **Conflicts:** none.

**C8. The head × trunk cross term.**
The head swap alone (V2 3.00 head to V2 4.00 head) gave ×0.76 at a K4 trunk. Measure the same swap at K2.5 and K3 trunks. If the head's KL share scales with trunk error, knap needs a per-size head price instead of headsim's bf16-trunk price.
- **Source:** 11-06 7:50 PM LOG; #34.
- **Fit/Freeze:** pricing.
- **Test:** 4–6 evals.
- **Gain/Risk:** 1–3% at 2.5 if mispriced. The head is 5–25% of total KL.
- **Conflicts:** #34.

**C9. The embedding × trunk cross term.**
kfit2's compounding κ is about 50 at layer 0 (model file). The embedding error enters before layer 0 but is priced as an additive offset. Measure the swap from t2.5 to t4 at K2.5 and K4 trunks.
- **Source:** kfit2 model; #35.
- **Fit/Freeze:** pricing.
- **Test:** 4 evals.
- **Gain/Risk:** 0.5–1%.
- **Conflicts:** none.

**C10. Isolated or in-context prices: score both on verify.**
Solve the 2.5 knapsack once with isolated prices and once with C3's in-context prices; score both on verify. The literature disagrees: MixQuant and GEMQ found the context helps, MixFrag found re-measuring hurt (31.91→29.73).
- **Source:** 2607.23047, 2605.23078, 2607.28589.
- **Fit/Freeze:** method.
- **Test:** 2 evals plus C3.
- **Gain/Risk:** decides the stream's pricing loop.
- **Conflicts:** none.

**C11. Mine V2's evals for interacting pairs.**
Regress kfit2's out-of-fold residuals over the ~1,450 V2 evals on indicators of which family × depth groups changed together. This finds interacting pairs for free.
- **Source:** new.
- **Fit/Freeze:** method.
- **Test:** colo CPU, 1–2 h (needs the per-mix picks).
- **Gain/Risk:** enabling.
- **Conflicts:** #20 (that re-ranker used summary features; this is a structured pair search).

**C12. Compounding κ at the operating point.**
κ was fitted around a 4.0 base. Measure the same early-layer group swap on K2.5 and K4 trunks to see whether κ depends on trunk error.
- **Source:** #68.
- **Fit/Freeze:** pricing.
- **Test:** 4 evals.
- **Gain/Risk:** 0–1%.
- **Conflicts:** none.

**C13. A8 prices at K2.5 (the key product's fast variant).**
E1b/E1c costs were measured on a K6 base. Run all-A8 against f16 on a K2.5 trunk (one pair) plus one per family (6 evals), and check additivity (sum of families against all).
- **Source:** BACKLOG "activation cost on a low-K base"; E1c's ±30%.
- **Fit/Freeze:** pricing; before the fast variants are built.
- **Test:** 8 evals.
- **Gain/Risk:** a correct fast-2.5 trade-off.
- **Conflicts:** #26/#27.

---

## D. Gradient- and Fisher-based cheap prices

**D1. Σdy² as the prior for kfit2's depth curves, with attention checked.**
Use g^0.49 (check 1) as a partial-pooling prior for families and levels without data. For attention q/k/v, run 3–4 depth-quarter group swaps at 2.5 to decide whether kfit2's flat curve or Σdy²'s strong fall is right.
- **Source:** check 1; 2607.07964 (KronQ) and 2610.34886 (ThinkBit), both scoring E‖g‖²·E‖x‖² per tensor.
- **Fit/Freeze:** pricing.
- **Test:** CPU plus 4 evals.
- **Gain/Risk:** 0–1%.
- **Conflicts:** none.

**D2. One backward pass prices every swap at the current mix.**
Compute g = ∇_W KL(bf16‖mix) on 64 sequences (fake-quant 27B through the fp32 collector). The predicted change for moving tensor i to level k is ⟨g_i, Ŵ_k − Ŵ_cur⟩ + ½Σ_r d_r ΔW_r H ΔW_rᵀ. All roughly 400×71 options are screened at once, and the top ones get real evals.
- **Source:** 2609.09867 (the current-state gradient gets the sign right in 85/98 moves, the midpoint in 97/98); 2610.31931.
- **Fit/Freeze:** search.
- **Test:** 0.8B first (minutes). At 27B about 0.5 h on both cards per round (#33-24 timings).
- **Gain/Risk:** 1–2% at equal evals. Gradients are noisy in early layers unless fp32 (#33-03).
- **Conflicts:** #33-07 (that used gradient-weighted H_in as a quantizer statistic; here it is only for screening).

**D3. Aumann–Shapley in-context attribution.**
A 4-node path integral of ∂KL/∂α_i along tΔW (t = 1/8 … 7/8) gives per-tensor shares that sum to the total KL. Combined with each tensor's rate-distortion slope this gives marginal prices.
- **Source:** 2607.12266 (KL 0.0264 against 0.0331 for ModelOpt at 5.0 bits on Qwen3-30B).
- **Fit/Freeze:** pricing.
- **Test:** 0.8B against measured group swaps. At 27B, 4 backward passes × 64 sequences (about 2 h on both cards).
- **Gain/Risk:** 1–2%, uncertain.
- **Conflicts:** none.

**D4. A block-output sensitivity probe.**
Per (tensor, level), forward one block on a few thousand real calib tokens and use the block-output relative error (through the nonlinearity) as kfit2's input instead of hproxy.
- **Source:** 2609.33923 (block cosine allocates better than the layer objective on Qwen3.8-27B).
- **Fit/Freeze:** per level.
- **Test:** single-layer forwards on a GPU, or donated compute. Validate by CV on V2.
- **Gain/Risk:** 0–2%.
- **Conflicts:** none.

**D5. A Jacobian-weighted H_in as a price only.**
Weight tokens by w_t = ‖J_t‖² from a Rademacher logit probe and price with Σ w_t x xᵀ.
- **Source:** 2610.18139 (objective-aware weighting lowered logit MSE in 24/24 comparisons).
- **Fit/Freeze:** pricing.
- **Test:** 0.8B CV against D1 and B3.
- **Gain/Risk:** 0–1%.
- **Conflicts:** #33-07 and #91 failed as *rounding statistics*; this is only for pricing.

**D6. A trace-ratio correction for Kronecker prices.**
K-FAC scales differ from the empirical Fisher by 1–3× across layers (2606.00573). If B3 is adopted, correct each layer by η = tr(F)/(tr G · tr H) measured on a few batches.
- **Source:** 2606.00573.
- **Fit/Freeze:** pricing.
- **Test:** 0.8B.
- **Gain/Risk:** small.
- **Conflicts:** none.

---

## E. Library ↔ mix compatibility

**E1. A junction-dependency assertion in the mix writer.**
List each per-level fit and its partners:
- swj: gate↔up, tied in knap5 (fine);
- C9 α: per family (fine);
- asym writers: siblings' codes (needs a per-mix pass);
- tile CD: none;
- H_out: bf16 Σdy² (fine).

Make the writer refuse untied gate/up K while swj is on, in case the speed table later unties them.
- **Source:** brief §2; code reading.
- **Fit/Freeze:** hygiene; pre-freeze.
- **Test:** minutes.
- **Gain/Risk:** insurance.
- **Conflicts:** none.

**E2. The writer–sibling mismatch penalty curve (0.8B).**
Measure the asym-writer penalty against |ΔK| = 0, ±1/8, ±1/4, ±1/2, ±1 between writer and sibling levels. This prices "tied K" against "free K plus a per-mix pass", or finds a tolerance band.
- **Source:** #111 (×1.08–1.24 at a 0.5–1 bit mismatch).
- **Fit/Freeze:** only if #105 passes.
- **Test:** 4070, 2–3 h.
- **Gain/Risk:** decides the rollout form.
- **Conflicts:** #111.

**E3. Price the tie constraint with knap5 on CPU.**
Run knap5 with writer K tied to sibling K at each size and compare the predicted cost with the free solution. A cheap tie combined with asym's per-mix gain (0.81–0.90 on 0.8B) favours a tied rollout.
- **Source:** new.
- **Fit/Freeze:** knapsack; pre-freeze, input to the user's #105 decision.
- **Test:** CPU, under 1 h.
- **Gain/Risk:** decision support.
- **Conflicts:** #105, #111.

**E4. The value of partner-aware refits (0.8B, fixed codes).**
For a knapsack mix, compare per-level refits (swj, C9) against refits recomputed with the actual neighbours (bf16 inputs) and against refits on quantized upstream. This decides whether any per-mix final pass pays.
- **Source:** new.
- **Fit/Freeze:** final pass.
- **Test:** 4070, 1 h, no seed noise.
- **Gain/Risk:** 0–1%.
- **Conflicts:** #108.

---

## F. Cheaper and less noisy evaluation

**F1. Capstudy at 512 and 1k caps, from existing rows.**
Truncate existing 8k score rows to 512/1k/2k positions. Compute paired SE per GPU-second with the fixed+linear cost model (about 80 s + 0.4 s per 1k positions) and check rank agreement on recipe pairs.
- **Source:** capstudy (2k beat 8k by 27% less noise per GPU-second).
- **Fit/Freeze:** method.
- **Test:** CPU.
- **Gain/Risk:** possibly another 20–30% eval time.
- **Conflicts:** none.

**F2. Two-stage evals with a futility stop.**
Score candidates on one third of the slice first. Stop if they are clearly worse than the incumbent (z > 1.5 worse, pre-registered); most search candidates are rejects.
- **Source:** 2610.05265 (successive halving: 4.2× less calibration time).
- **Fit/Freeze:** search.
- **Test:** replay V2's stream offline to estimate the savings.
- **Gain/Risk:** −30–50% eval time.
- **Conflicts:** none.

**F3. Successive halving across a round's candidates.**
Evaluate N candidates on slice 1, the top half on slice 2, and so on, reusing scores.
- **Source:** 2610.05265.
- **Fit/Freeze:** search.
- **Test:** replay.
- **Gain/Risk:** like F2.
- **Conflicts:** none.

**F4. Neyman allocation of documents to categories.**
Allocate documents per category in proportion to w_c·σ_c (per-document difference SD from past pairs) and reweight. Same SE with fewer positions.
- **Source:** new.
- **Fit/Freeze:** slice building.
- **Test:** CPU from existing rows.
- **Gain/Risk:** 10–30% eval time.
- **Conflicts:** none.

**F5. Exact in-context head evaluation without full evals.**
Dump the final hidden states of a trunk mix on the eval documents once (the dump hook exists). Compute logits for every head option against the golden: about 2.5 GFLOP per position, a few seconds per option on a GPU or minutes on CPU. This gives the exact head-swap KL in context, so the head is re-optimized per trunk mix for free.
- **Source:** headsim (#34); C8.
- **Fit/Freeze:** per mix.
- **Test:** 1 dump eval plus CPU.
- **Gain/Risk:** 0.5–2% at 2.5–4.
- **Conflicts:** #34.

**F6. Suffix-only re-evaluation from cached boundary states.**
The residual stream at layer L fully determines every later layer. Cache it for the incumbent (about 1.2 GB per boundary on a 123k-position slice). Moves in layers ≥ 48 then cost about 25%.
- **Source:** new. Engine plumbing (low priority per the brief).
- **Fit/Freeze:** search.
- **Test:** none until the runner work.
- **Gain/Risk:** −40% eval time for late-layer moves.
- **Conflicts:** the "resident base" proposal.

**F7. A prefix-state restart diagnostic.**
On the 0.8B in PyTorch, score a 32k document twice: with the quantized prefix, and with bf16 prefix states (KV plus GDN state) handed to the quantized model at 16k. If the two agree on the window, late positions can be scored from cached bf16 states at 1/16 the cost.
- **Source:** new; B4 (#52).
- **Fit/Freeze:** method.
- **Test:** about 1 h on the 4070 or CPU.
- **Gain/Risk:** enabling for long-context measurement.
- **Conflicts:** none.

**F8. Top-512 truncation bias at high bpw.**
On the 0.8B (full logits available), compare full-vocabulary KL with top-512+tail KL at KL 1e-3 to 5e-4. Rare-token rows matter (#32). Bias above 5% or rank flips would mean the 7.0 and 8.0 goldens need more top-k.
- **Source:** #32.
- **Fit/Freeze:** method.
- **Test:** minutes.
- **Gain/Risk:** 7.0/8.0 only.
- **Conflicts:** #99.

**F9. Engine floor by position on long traces.**
Measure KL of the bf16 model in our engine against the golden, per position bin, on the 32k chunks; subtract or report it.
- **Source:** BACKLOG (#416 lesson).
- **Fit/Freeze:** method, ≥6 bpw.
- **Test:** part of #99.
- **Gain/Risk:** accurate 7.0/8.0 decisions.
- **Conflicts:** #99.

**F10. Screen with the A8 path.**
Screen with the two-pass A8 path (about 25% faster) and confirm finalists in f16, after checking rank correlation on 20 mixes. For the fast variant, A8 *is* the target.
- **Source:** new.
- **Fit/Freeze:** search.
- **Test:** 20 paired evals.
- **Gain/Risk:** about −20% screening time.
- **Conflicts:** none.

**F11. Cluster bootstrap by source as the default.**
`paired.py --cluster` exists. Make clustering by upstream document the default for any chunked golden (xcorp 8k pieces, own sessions); otherwise the SE is optimistic.
- **Source:** eval/traces.
- **Fit/Freeze:** method, minutes.
- **Gain/Risk:** honest SE.
- **Conflicts:** none.

**F12. A winsorized statistic for screening only.**
Screen on mean KL with per-position KL winsorized at p99.9. Confirm finalists on the raw mean; tail guards stay.
- **Source:** 2610.14934.
- **Fit/Freeze:** search.
- **Test:** replay.
- **Gain/Risk:** lower variance; misses gains that live only in the tail.
- **Conflicts:** none.

**F13. Document-fingerprint control variates.**
Per-document response vectors from past evals predict a candidate's per-document difference from its group changes. A regression estimator then lowers variance.
- **Source:** new.
- **Fit/Freeze:** method.
- **Test:** CPU on V2/V3 rows.
- **Gain/Risk:** speculative, 0–20% variance.
- **Conflicts:** none.

**F14. Warm prefixes for 32k trace chunks.**
Chunk resets drop GDN state and KV and underestimate long-context error. Prepend 2–4k unscored warm tokens to each chunk and measure convergence to whole-trace KL (0.8B first, then one 27B trace).
- **Source:** new.
- **Fit/Freeze:** eval design.
- **Test:** 0.8B, 1 h.
- **Gain/Risk:** a faithful long-context score.
- **Conflicts:** eval/traces design.

---

## G. Search

**G1. Gradient-screened local search (D2 applied to the stream).**
- **Source:** 2609.09867.
- **Fit/Freeze:** search.
- **Test:** see D2.
- **Gain/Risk:** 1–2% at equal evals.
- **Conflicts:** none.

**G2. Seed choice as a search dimension.**
Only if A1–A3 show a real seed effect: mixes pick the seed per hot layer group.
- **Source:** A3.
- **Gain/Risk:** 0.5–1%.
- **Conflicts:** #41.

**G3. Cross-fitted final selection.**
A finalist must win in each of k document folds of the search pool (stability selection).
- **Source:** V2 (fit-slice picks about 3% optimistic).
- **Fit/Freeze:** finals.
- **Test:** CPU on recorded rows.
- **Gain/Risk:** removes about 1–3% of optimism.
- **Conflicts:** none.

**G4. A reusable holdout for ver3 reads.**
Thresholdout: answer ver3 reads with calibrated noise so they can be read more often without overfitting.
- **Source:** new to us (Dwork et al.).
- **Fit/Freeze:** method.
- **Test:** simulation on V2.
- **Gain/Risk:** more monitor reads.
- **Conflicts:** V2's ver3 rule.

**G5. Multi-fidelity selection.**
kfit2 prediction → one-third slice → full slice, with acquisition by expected improvement per GPU-second.
- **Source:** new.
- **Fit/Freeze:** search.
- **Test:** replay.
- **Gain/Risk:** 20–40% eval time.
- **Conflicts:** none.

**G6. Refit kfit2 on per-document data.**
Use per-document KL vectors instead of slice means. Each eval then carries more information, and per-category factors become possible.
- **Source:** new.
- **Fit/Freeze:** model.
- **Test:** CPU CV.
- **Gain/Risk:** 0–1%.
- **Conflicts:** #68.

**G7. The cost of a nested size ladder.**
Run knap5 with K monotone across sizes. If that costs ≤0.3% predicted, search one nested family (measurements reused across sizes), which also enables the user-side mix builder.
- **Source:** new.
- **Fit/Freeze:** knapsack.
- **Test:** CPU.
- **Gain/Risk:** cheaper search.
- **Conflicts:** none.

**G8. An uncertainty-aware knapsack.**
Bootstrap kfit2 over documents and measurements; penalize each option by λ × the SD of its price; validate on verify.
- **Source:** BACKLOG B11.
- **Fit/Freeze:** knapsack.
- **Test:** CPU plus 2 evals.
- **Gain/Risk:** avoids V0-style extrapolation picks.
- **Conflicts:** BACKLOG.

**G9. Allocate evals across sizes by marginal value.**
Give evals to the band whose validation-curve slope × importance is highest (2.5 is the key product).
- **Source:** new.
- **Fit/Freeze:** stream.
- **Gain/Risk:** better use of a fixed budget.
- **Conflicts:** none.

**G10. Explore at recipe boundaries.**
Propose moves across the K3.875/K4 H_out boundary and the i8/mul1 boundary, where prices are least certain (B2, B9).
- **Fit/Freeze:** stream.
- **Gain/Risk:** finds pricing errors.
- **Conflicts:** none.

---

## H. Pools, objective and overfitting

**H1. A board-weighted objective.**
The board's mean token KL weights long traces' late positions heavily; our slices cap at 2k/8k. Reweight per-position KL by category × position bin to calib's token distribution, using position ratios from 32k chunks, and check that finalist rankings hold.
- **Source:** B4 (#52: ratios flat for V2 mixes, untested for mixes that differ in GDN).
- **Fit/Freeze:** objective.
- **Test:** CPU plus I1's evals.
- **Gain/Risk:** guards against a hidden misranking.
- **Conflicts:** none.

**H2. Family factors per slice (xcorp, calib, agentic).**
Search accept/reject runs on xcorp plus private agentic text, finals on calib. Score 6 family group swaps on all three goldens (18 evals). If allocation-relevant differences appear, weight the search objective to calib's category mix.
- **Source:** #93 (xcorp keeps 52–91% of each step); #101 (agentic +4%).
- **Fit/Freeze:** objective; before the stream.
- **Test:** about 0.6 card-hour.
- **Gain/Risk:** 0–2%.
- **Conflicts:** eval-data.md.

**H3. A winner's-curse estimate for the whole selection.**
Bootstrap documents and re-run the recorded selection to estimate optimism; report de-biased finalist scores.
- **Source:** V2 (about 3%).
- **Fit/Freeze:** finals.
- **Test:** CPU.
- **Gain/Risk:** honest numbers.
- **Conflicts:** none.

**H4. Split-pool stability of prices.**
Compare kfit2 family factors fitted on fit-pool halves (calA/fit2/fit3) to quantify how much the prices overfit.
- **Source:** V2 slice offsets.
- **Fit/Freeze:** model.
- **Test:** CPU.
- **Gain/Risk:** enabling.
- **Conflicts:** #55.

**H5. Per-category heterogeneity of adopted recipes.**
From existing per-document rows, check whether swj and H_out help code, chat, agentic, math and multilingual alike. This predicts how gains transfer to the private mix.
- **Source:** new.
- **Fit/Freeze:** method.
- **Test:** CPU.
- **Gain/Risk:** enabling.
- **Conflicts:** none.

---

## I. Long context and GDN

**I1. Position sensitivity per family.**
Run group swaps of GDN-in, GDN-out, attention and MLP at K2.5 on 32k trace chunks and compare KL by position bin. A steeper GDN slope means the 2k-capped prices underprice GDN for the board's long traces.
- **Source:** 2609.38169 (STEPQuant on Qwen3.8-27B: state errors accumulate with position).
- **Fit/Freeze:** pricing.
- **Test:** 8 evals on 32k chunks (about 1 card-hour).
- **Gain/Risk:** 0–2%.
- **Conflicts:** #52.

**I2. GDN state-precision parity.**
Confirm that both the golden recording and our evals carry the GDN state in fp32 across 2048-token chunks. Measure bf16 against fp32 carry on one long trace.
- **Source:** 2609.38166 (bf16 per-step state: AIME 87.9→72.1); 2610.39418.
- **Fit/Freeze:** method.
- **Test:** config read plus 1 eval.
- **Gain/Risk:** removes a possible floor.
- **Conflicts:** #33-26.

**I3. GDN half-life as a sensitivity feature.**
Correlate kfit2 residuals of ssm_qkv/z/out with per-layer gate half-life (from A_log/dt_bias). For state errors, half-life against accumulated error has ρ ≈ 0.80 over 2,304 Qwen3.8-27B heads.
- **Source:** 2609.38169.
- **Fit/Freeze:** pricing.
- **Test:** CPU.
- **Gain/Risk:** 0–1%.
- **Conflicts:** none.

**I4. Position bins in every comparison.**
Add posbins output to pcmp reports. This flags recipes with position-dependent effects.
- **Source:** posbins.py.
- **Fit/Freeze:** method, minutes.
- **Conflicts:** none.

**I5. A long-trace guard eval for GDN/attention adoption tests.**
One 32k-chunk eval per arm. This would catch effects like xcorp's "agentic +4%" early.
- **Source:** #101.
- **Fit/Freeze:** rule.
- **Test:** +1 eval per arm.
- **Conflicts:** none.

---

## J. Transfer and coherent-error diagnostics

**J1. A 0.8B→27B transfer table with empirical-Bayes shrinkage.**
Classify each tested item as coherent or noise-shaping: H_out about 0.33, xcorp about 0.3, swj about 0.7–1 at K2.5, i8c null. Fit a per-class prior and shrink both new 0.8B results and noisy 27B pairs with it.
- **Source:** brief §1–2; review idea 10.
- **Fit/Freeze:** method; triage now.
- **Test:** CPU table.
- **Gain/Risk:** better queue choices.
- **Conflicts:** none.

**J2. A coherent-error map of a 27B mix.**
Per layer and junction, measure gain error E[Δy·y]/E[y²] and offset E[Δy] of block outputs against bf16, layer-streamed with the sib27 machinery. This shows where least-squares shrink remains after swj and points the next refit.
- **Source:** brief §1; 2609.11716 (error counteraction).
- **Fit/Freeze:** method.
- **Test:** a few card-hours, or CPU on colo for selected layers.
- **Gain/Risk:** targets coherent fixes.
- **Conflicts:** #108 (that measured refits, not the map).

**J3. The temperature-only share of KL.**
Fit a global logit temperature, and one per entropy class, for a quantized mix: 0.8B in PyTorch, 27B from the F5 dump. If ≥1–2% of KL is removable, the shortlisted head-side gain refit is worth doing.
- **Source:** shortlist "head KL-side gain refit".
- **Fit/Freeze:** method.
- **Test:** 0.8B minutes; 27B 1 dump.
- **Gain/Risk:** decides a refit.
- **Conflicts:** none.

**J4. A census of least-squares shrink.**
From the ladder's refit logs, tabulate per-row c\* against ‖W‖/‖Ŵ‖ by family and K. This predicts where swj-like gain refits pay and how B1-type proxy inflation scales with K.
- **Source:** new.
- **Fit/Freeze:** method.
- **Test:** CPU.
- **Gain/Risk:** enabling.
- **Conflicts:** none.

**J5. Tag proxies with LDLQ order and codebook path.**
Tag every proxy result and refuse comparisons across tags. The settle.py order bug inflated #106 and #77.
- **Source:** w64.
- **Fit/Freeze:** hygiene.
- **Conflicts:** #106, #77.

**J6. A single-block-in-host KL screen.**
A K6 host with only the test layers at K2.5 gives a cheaper 27B screen than whole-model pairs. Depends on A2's per-layer seed noise.
- **Source:** new; hconv overstated H_out at −6% against −2.8% for the whole model.
- **Fit/Freeze:** method.
- **Test:** 6 evals after A2.
- **Gain/Risk:** cheaper 27B screens.
- **Conflicts:** #33-25.

---

## K. Ladder compute allocation (measurement-driven)

**K1. An active-set ladder.**
Price unbuilt levels by log-linear interpolation (#36: fractional levels sit on the line, 0.98–1.04). Run knap5 at every planned size plus a margin. Quantize per tensor only the levels in the active set; rqah jobs are per tensor.
- **Source:** w58 (pass 1's 25 levels came within 0.2–0.8% of all 73).
- **Fit/Freeze:** ladder plan; pre-ladder.
- **Test:** CPU knap5 to list the active sets.
- **Gain/Risk:** 30–50% fewer ladder card-hours, freeing H_out time at K≥4. Interpolation error risk; keep a ±2-step margin.
- **Conflicts:** v3.md's "deliberately finer" ladder.

**K2. Skip H_out on writers at K≥4.**
#82: on the 0.8B, writers carry 3% of the H_out gain. Two-sided LDLQ feedback dominates quant time. Measure the H_out time share per family (PROFILE) and drop it where the gain is negligible.
- **Source:** #82; PROFILE.md.
- **Fit/Freeze:** quantizer cost; pre-ladder.
- **Test:** profiling plus one fixed comparison.
- **Gain/Risk:** −20–30% of K≥4 level time for about 0.1% KL.
- **Conflicts:** #33-28 (adoption unchanged).

**K3. Value-of-information ordering of the ladder.**
Quantize next the level whose measured price would change the predicted knapsack KL at 2.5–4.5 the most.
- **Source:** new.
- **Fit/Freeze:** ladder order.
- **Test:** CPU.
- **Gain/Risk:** earlier useful mixes.
- **Conflicts:** order of use.

**K4. Thin the levels above K5 for low-sensitivity families.**
V2's K5/K6/K8 group swaps are indistinguishable (gate_up L48w16: −0.00069, −0.00090, −0.00095, within about 2 SE). Skip quarter-step levels above K5 for families whose marginal KL is below noise.
- **Source:** check of V2 data.
- **Fit/Freeze:** ladder.
- **Test:** CPU.
- **Gain/Risk:** fewer high-K levels; 7.0 needs care.
- **Conflicts:** v3.md 7.0 plans.

---

## Ranked top 20 (expected gain per unit cost)

| # | idea | cost | why it ranks here |
|---|---|---|---|
| 1 | A11 manifest diff + A12 stale-constant audit | minutes | prevents errors of up to 20% (head mismatch; withdrawn 0.7786 default) |
| 2 | B8 fixed-code swj-in-mix pair | 0.1 card-hour | validates the swj pricing every knapsack size relies on; no seed noise |
| 3 | A1 seed replication on verify | 0.1 card-hour | decides whether the 27B floor is real; changes every rule |
| 4 | B1 pre-refit proxy for swj levels | CPU, minutes | blocks a 15–40% mispricing of gate_up in the planned V3 hproxy |
| 5 | B2 H_out boundary recipe factor | CPU, minutes | the K3.875/K4 discontinuity decides allocation at 3.5–4.5 |
| 6 | E3 price the tie constraint | CPU, under 1 h | input to the user's #105 rollout decision |
| 7 | C8 + F5 in-context head pricing and re-opt | 4–6 evals plus 1 dump | the head is 5–25% of KL; ×0.76 seen from a head swap |
| 8 | K1 active-set ladder | CPU | 30–50% of ladder card-hours |
| 9 | C2 family KKT audit at each finalist | 12–18 evals per size | direct 1–3% if the plan is off |
| 10 | A2 layer-split virtual seeds | 8 evals | σ_seed(K) table and localization; unlocks A3/A5 |
| 11 | A5 + A6 power calculator and sequential pairs | CPU | saves 27B pairs immediately (#106/#112 queue) |
| 12 | B6 uniform-level anchors | about 2 min per level | ties the K-curve to all recipe effects |
| 13 | C13 A8 prices at K2.5 | 8 evals | the fast 2.5 variant is a product |
| 14 | F2/F3 futility and halving evals | replay on CPU, small runner change | −30–50% search eval time |
| 15 | C1 within-window additivity audit | about 1.5 card-hours | V2 hint of strong sub-additivity contradicts kfit2's structure |
| 16 | B3/D1 Σdy² price with calibrated exponent | colo CPU, 2–4 h | prices new recipes; check 1 shows ρ 0.82, slope 0.49 |
| 17 | H2 slice-specific family factors | 18 evals | the search slices are not calib |
| 18 | A4 variance-components model | CPU | cleaner 27B verdicts |
| 19 | C4 lazy measured descent | 100–200 evals per size | proven structured search (Q-Strata) |
| 20 | D2/G1 gradient-screened local search | 0.8B first; about 0.5 h on both cards per round at 27B | screens all options per round |

## Five cheapest diagnostics that would most change this ranking

1. **A1: score base, s1 and s2 at K2.5 on golden-verify27** (about 1 eval plus pcmp). A real seed effect promotes A2/A3/G2 to the top and changes every rule; noise demotes them and loosens rules.
2. **B1 and B2: hproxy ratios on existing tensors** (CPU, minutes): gate/up before and after swj; bn4 against base4 per family. Ratios far from 1 make the pricing fixes urgent before the ladder price table is built.
3. **B8: swj on/off on the w63 2.5 plan with fixed codes** (2 evals). A mismatch with knap5's "gu" prediction means the knapsack's swj model must become per-layer (B4).
4. **A2 halves only: base/s1 K2.5 mixed by layer halves** (2–4 evals). A seed effect concentrated in one half makes localized best-of-N (A3) worth doing; a spread effect kills it.
5. **C8: the V2 3.00 → V2 4.00 head swap on a K2.5 trunk** (2 evals). If the head share is far from the knapsack's head term, F5's exact head re-optimization becomes a top-3 item.
