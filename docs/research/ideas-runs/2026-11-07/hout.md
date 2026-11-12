# Lens **hout**: output-side information on the 27B (ideas pass, 2026-11-07)

The instruction file for this lens is from the 11-02 run. It assumes a 27B diagonal collection is still planned. This report works from the 11-07 state instead:

- **The 27B Σdy² diagonal exists.** 1,923 sequences, block-normalized ("bn"). Published as `Hero422/qwen3.8-27b-hout-diag`; colo copy at `/data/work/hout27/hout-bn`.
- **H_out is adopted for levels ≥ K4.** K4 pairs 0.961 / 0.967 (geo 0.964).
- **It is not adopted for levels ≤ K3.5.** K2.5 geo 0.9801, so it missed the rule by a hair.
- **No more Σdy² collections** (user, 11-05). This lens is also dropped from this pass's lens list in the brief.

So the report ranks ideas that use the existing diagonal, analytic or forward-only output information, and 0.8B research. Anything that needs a new 27B backward pass is marked V4 / research.

The 11-02 workers already prepared a lot of this lens, but most of it never ran on a GPU:

| Prepared work | What it is | Status |
|---|---|---|
| `w04-blockgain` | Output-side predictor. Spearman 0.998 vs measured on 0.8B | CPU only; GPU arms not run |
| `w05-head2s` | Two-sided LM head with the softmax Fisher | Not run |
| `w08-hprice` | Two-sided knapsack prices | Patches not merged |
| `w02` B1 | h-weighted su refit | Seed-0 screen only |
| `w10-longpos` | Long-position re-score of the 0.8B arms | Not run |
| PROFILE.md **T1** | Block-sparse `ldlq_2hess` | Not built |

These prepared items are not presented as new ideas. They are re-ranked below for today's state, and marked "prepared".

---

## 0. New measurements (colo, CPU, ~5 min; files in `/data/work/ideas-hout/`)

I ran the per-tensor predictor from w04's `blk.py` on all 400 27B trunk tensors. It uses the bn diagonal h and sv² from bf16 row RMS, with exllamav3's `prepare_H_out` algebra: 128-row Hadamard blocks, σ_out 0.01, block-16 LDL.

- **R** = two-sided cost / one-sided cost, both in the true h·sv² metric, assuming white rounding noise. 1 − R is the available shaping gain in h-proxy terms. On the 0.8B, R_pred matched R_meas with Spearman 0.998 and slope 0.99 (w04).
- **1-sided ratio** = how the one-sided proxy (what knap4 reads today) prices an H_out quant relative to a plain quant.

| family | **27B R** | 0.8B R (w04) | **27B 1-sided ratio** | 0.8B 1-sided | 27B per-128-slab spread (bits) | 0.8B spread |
|---|---|---|---|---|---|---|
| GDN in_proj_qkv | **0.777** | 0.805 | **1.180** | 1.157 | **0.289** | 0.192 |
| GDN in_proj_z | **0.731** | 0.779 | **1.230** | 1.206 | 0.097 | 0.056 |
| attn q_proj (qg) | **0.783** | 0.738 | **1.136** | 1.165 | 0.174 | 0.125 |
| attn k / v | 0.881 / 0.893 | 0.806 / 0.931 | 1.027 / 1.090 | 1.000 / 1.059 | | |
| mlp gate / up | 0.895 / 0.896 | 0.887 / 0.883 | 1.074 / 1.063 | 1.096 / 1.048 | ≤0.005 | ≤0.03 |
| writers out / o / down | 0.956 / 0.959 / 0.975 | 0.946 / 0.956 / 0.952 | 0.99–1.00 | 0.94–0.96 | ≤0.003 | ≤0.01 |

What follows from these numbers:

1. **The 27B is not flatter.** The h-proxy gain available at 27B is about the same as on the 0.8B. Weighting families by the 0.8B loss shares gives 1 − R = **11.3% at 27B vs 12.4% at 0.8B**.
2. **The shrink is in the step from proxy to KL.**
   - 27B measured KL gains: −2.0% (K2.5) and −3.6% (K4). That is a proxy→KL factor of about **0.18 (K2.5) and 0.32 (K4)**.
   - 0.8B: 0.47 (K2) and 0.66 (K3).
   - The factor rises with K at both sizes, and is about 2.5× lower at 27B.
   - Caveat: this uses 0.8B loss shares and the *predicted* 27B h-proxy, not the realized one. H09 tests the realized value.
   - This fits the brief's hypothesis that fixes which reshape random rounding noise shrink with model size.
3. **The knapsack misprices H_out levels.** The one-sided proxy calls K ≥ 4 H_out quants 14–23% *worse* on GDN qkv/z and q_proj, and 6–9% worse on gate/up/v. Their true h-weighted cost is 10–27% *lower*.
4. **The gain is concentrated in a few tensors.** Lowest R values:

   | tensor | R |
   |---|---|
   | L63 up_proj | **0.367** |
   | L63 gate_proj | **0.540** |
   | L63 v_proj | 0.60 |
   | GDN in_proj_z, layers 12–21 | 0.60–0.64 |
   | GDN in_proj_qkv, L60 / L62 | 0.63 |
   | L3 q_proj | 0.65 |

   L63 gate/up is the known sensitivity spike (#21).
5. **Depth pattern.** GDN z/qkv gain most in layers 0–31. gate/up gain most in layers 48–63. The middle band (32–47) gains least.
6. **The MLP neuron permutation is dead at 27B.** gate/up R moves 0.895 → 0.885 with the snake deal and 0.893 with a random one (the 0.8B found +0.018 bit).
7. **Damping and tempering in the predictor.** σ_out 0.1 costs nothing in the predictor (qkv 0.778 vs 0.777). Tempering the factor to d^0.5 loses about 1/7 of the predicted gain (qkv 0.811). Only KL can show whether tempering helps; the white-noise model can't.
8. **Prior for this lens.** For hout ideas priced in h-proxy, use 27B KL ≈ 0.2–0.3 × the h-proxy gain, or ≈ 0.4 × the 0.8B KL gain.

---

## 1. Consequences of "H_out at K ≥ 4": settle before the ladder

**H01. Block-sparse `ldlq_2hess` (PROFILE T1). Now needed regardless.**
- **Mechanism.** The rotated diagonal H_out is exactly block-diagonal (128×128 blocks), so L_out is too. The current anti-diagonal sweep still multiplies a dense L_out (`bmm(dE, Ln[cols][:, :, :n_hi])`). Restrict each tile's F update to its own 128-column block, and run a block-wavefront: tiles_k + 7 steps per block, with all blocks batched. The algorithm is the same in exact arithmetic.
- **Source.** PROFILE.md T1. FastKron 2608.06291 has the same goal for dense Kronecker factors; here the block structure makes it simple.
- **Why it matters.** An H_out level takes 3.2 h vs 37 min; the F update is 72% of gate_proj's time. The fix cuts the flops 40–90×. There are about 24 levels ≥ K4 (i8 K4–5 and K5.25–7, mul1 K6.5–8), so it saves ≈55–60 card-h.
- **Fit.** Quantizer only. No runtime cost. **Freeze: yes** — before any ≥ K4 level is quantized.
- **Test.** Unit test on 0.8B tensors (4070): codes may differ by float summation order. Judge the two-sided proxy (within the seed spread) and the time. Then one 27B layer on gpubox. ~2 h of work, 1 card-h.
- **Gain / risk.** No KL change; it is time. It also removes the cost objection to H06 and H16. Low risk.
- **Conflicts.** None.

**H02. Two-sided prices for the H_out levels in the knapsack (prepared: w08 `KN_HRATIO`, `ourq-hprice`).**
- **Mechanism.** knap4/knap5 read the one-sided `proxy_err`. Per §0.3, that makes K ≥ 4 levels of qkv/z/q look 14–23% worse than they are. So at the K3.875/K4 boundary the solver avoids exactly the tensors H_out helps.
- **Fix.** Log `proxy_err2` (the h-weighted proxy) at every level. Price oursH options as e1 × r × KN_HF, with KN_HF fitted from the bn4/base4 pair.
- **Source.** w08: on the 0.8B the one-sided ratio was 1.065 while KL went to 0.919, so it got the sign wrong. §0 table for the 27B.
- **Fit.** Knapsack. **Freeze: the logging, yes** (the rxq port of `ourq-hprice`). The pricing can come later.
- **Test.** CPU re-solve at 2.5–4.5 with and without the fix. If more than ~3% of parameters move, run 2 finals evals.
- **Gain / risk.** −0.3…−1% at 2.5 bpw; −0.5…−2% at 3.5–4.5 bpw. Risk: KN_HF is fitted from a single pair.
- **Conflicts.** The w58/w63 re-pricing did not include H_out (reprice.md: "H_out … not in the model").

**H03. Resolve the tile-CD vs H_out conflict on attn_qg at K ≥ 4.**
- **Problem.** `pipeline/rxq/config.py` raises "tile_cd and H_out together". cd3 on attn_qg is adopted at every level, and H_out at K ≥ 4, so the freeze config cannot express both.
- **Recommendation.** Use H_out for attn_qg at K ≥ 4 and cd3 below. Predicted q_proj R is 0.783, a 22% h-proxy cut, against cd3's 0.5–1.2% on the real quantizer. Later option: tilecd scoring the two-sided objective (H55).
- **Fit.** Config. **Freeze: yes.** No test needed for the choice.
- **Gain.** About 0.1–0.2%, and it avoids a hard failure in the ladder run.
- **Conflicts.** #77.

**H04. `apply_k_min` in rxq and the per-level stamp.**
- The config TODO says K ≥ 4 is "not expressible" yet. Trivial to add. **Freeze: yes.**

**H05. Log two-sided costs for every level, including the one-sided levels.**
- Per tensor: `proxy_err1`, `proxy_err2`, and the 128-block cost vector.
- Negligible CPU. It feeds H02, H09, H27 and H63. **Freeze: the logging, yes.**

**H06. Extend H_out to K3–K3.5 once H01 lands.** A user decision, not an experiment.
- **Reasoning.** All three K2.5 pairs were below 1 (0.972 / 0.992 / 0.976). K4 gave −3.6%, and the factor rises with K (§0.2). With H01 the cost is about 1.2×, not 5×.
- **Expected.** −2…−3% on K3–3.5 levels; at 2.5 bpw only the tensors sitting at K3–3.5.
- **Optional check.** One K3.5 27B pair. Its seed floor should lie between K2.5's 1.8% and K4's 0.6%.
- **Freeze: yes.**
- **Conflicts.** #33-28. What is different now: the cost is gone, and the decision was by a hair.

**H07. Keep the residual writers one-sided at every K.**
- Predicted R for out / o / down is 0.956–0.975, so 2–4% of h-proxy. On the 0.8B the writers carried 3% of the gain (#82).
- The writers' asym pass (#105 / #111) is a separate rounding change. "Two-sided + asym" was never tested at 27B. Specify writers = one-sided so there is no untested combination.
- **Freeze: yes.** Gain ≈0. Risk reduced.

**H08. h-weighted input-scale refit (B1), completed (prepared: w02).**
- **Mechanism.** `refit_scales` solves the su vector over all output columns with equal weight. With h: `((Q diag(h) Qᵀ) ∘ H) r = rowsum(Q ∘ HW ∘ h)`. The per-column sv step is unchanged, because h cancels per column. Codes stay fixed.
- **Evidence.** Seed-0 screen on bdiag: K3 0.993, K2 0.975. The CPU proxy predicts it is nearly inert (−0.08…−0.15% h-proxy). KL moved more than the proxy, which is a hint of a coherent (scale) effect.
- **Fit.** Per level, fixed codes. Can be done as a container edit after the freeze (suh only).
- **Test.** 0.8B `B1_FULL` on base and bdiag, 3 seeds (~25 min on the 4070). Pass if K2 ≤ 0.98 on every seed. Then one 27B fixed-code pair at K2.5 (~1 card-h, no requant).
- **Gain / risk.** 27B −0.3…−1%. Risk: the screen was noise.
- **Conflicts.** #85 (B1 pending), #85-02 (wrong order).

---

## 2. Explaining the 27B shrink (existing data only)

**H09. Realized h-proxy of the existing 27B arms.**
- Recompute Σ_r h_r e_rᵀ H_in e_r for base / bn (K2.5, 3 seeds) and base4 / bn4 from the trunk containers. Use w12's pure-torch shard decoder on cpu1.
- If the realized value is ≈ 11% (= predicted), the 27B factor is 0.18–0.32, and every hout idea gets about ×0.4 vs its 0.8B gain.
- If it is well below the prediction, the real quantizer realizes less shaping than it should (σ_out normalization, swj, or refit interactions), which may be fixable.
- CPU 2–4 h. **This decides the shrink prior for the whole lens.**

**H10. Family attribution at 27B by swapping existing bn4 tensors into base4. No requant.**
- Arms: GDN-in (qkv + z), MLP-in (gate + up), attn-in (q, k, v), writers. 4 evals of ~10 min plus packing.
- Gives per-family proxy→KL factors. Decides the scope of H06 / H16.
- Pre-registered expectation from §0: GDN-in ≥ 40% of the gain, writers ≤ 5%.
- The packing must match head/embedding on all arms (the swj K4 lesson).

**H11. Depth-band attribution with the same swaps by layer quartile.** 4 evals. The predictor says the middle band (32–47) gains least.

**H12. Factor-vs-K curve from existing 0.8B numbers.**
- Data: #33-16 (K3, K2), #82 (K4, K5), w04 predictions.
- Fit factor(K), then predict 27B K3 / K3.5 from its two points (0.18 at K2.5, 0.32 at K4). CPU only, minutes. Feeds H06.

**H13. Verify-slice and long-trace check of the K4 gain.**
- bn4 / base4 on `golden-verify27`, plus the ≤ 32k trace chunks. 4 evals.
- The rule decided on the search slice only. Cheap insurance before ~24 levels inherit the setting.

**H14. Category and position split of the K4 gain from existing per-position KL files.** No GPU.
- Does H_out help the agentic and code text that is ~25% of scored positions, or mostly prose?

**H15. Coherence amplification factor (forward-only finite differences). The root-cause test.**
- For family F, scale its outputs by (1 ± ε) and measure ΔKL. Compare with the diagonal prediction ε² Σ_r h_r ‖W_r x‖², using Σdy² and row output energy.
- A ratio ≫ 1 means coherent directions dominate: the diagonal H_out (rows independent) cannot see them. That would explain both the shrink and swj's −6.5%.
- 0.8B: minutes on the 4070. 27B: base = V2 8.0 on one card, ε = ±2%, 6 families → 12 evals, ~2 card-h.

---

## 3. Targeted H_out: few tensors, every K

**H16. H_out at every K for the lowest-R tensors.** *(new)*
- **Mechanism.** About 20 tensors have R < 0.66, so their h-proxy can drop 35–63%: L63 up / gate / v, GDN z L12–21, qkv L60/62, L3 q.
- L63 gate/up is the model's biggest sensitivity spike (#21: K2 +0.0065, 80% of a base's KLD) and IQR's documented exception.
- Two-sided cost on 20 tensors is minutes per level, even without H01.
- **Fit.** Per level, library-safe (bf16 statistics only).
- **Test.** 27B K2.5: partial requant of those ~20 tensors with the base seed and with s1. Swap into base and s1, 2 pairs, ~1–2 card-h. Rule: geo ≤ 0.99 and both pairs < 1.
- **Gain / risk.** −0.5…−1.5% at 2.5 bpw. Risk: the effect hides under seed noise. Re-rounding only 20 tensors keeps that noise small.
- **Conflicts.** #33-28 tested all tensors. What is different: this targets tensors whose predicted gain is 3–6× the average.

**H17. A per-tensor "two-sided" rounding option at ≤ K3.5 for the top-N tensors by (1 − R) × loss share.** N is set by H16. This generalizes H16. With H01, the extra ladder cost is ≈0.

**H18. Early massive-channel readers do not get special treatment** (negative prediction).
- L10 ssm_qkv (R 0.794) and L11 attn_qg (0.778) are ordinary.
- Their coder overhead (faraway M5: 1.35–1.74× an ideal coder's error) is an input-side problem, and H_out does not address it.

---

## 4. Head and embedding (no collection; the Fisher comes from forward goldens)

**H19. Two-sided LM head with the exact softmax-Fisher diagonal (prepared: w05-head2s; never run).**
- **Mechanism.** F_v = Σ p(1 − p), from the calib golden's top-512. Use the block-diagonal `ldlq_2hess_bd` path, keep vocabulary order, no format change.
- **CPU emulation** (count-proxy Fisher, scalar stand-in for the trellis, in-sample): head error 0.37–0.40× in vocabulary order, 0.14–0.16× in strided order, and 0.75× in slab order.
- **Why it should help.** The head is about 5–7% of total KLD. This is exact head KL via headsim, so the 27B proxy→KL factor does not apply.
- **Fit.** Per head donor. No runtime cost. **Freeze: before the head donors are built.**
- **Test.** w05 stages 1–2 on one gpubox card (~3–4 h), then one whole-model swap pair.
- **Gain / risk.** −1…−2.5% total at 2.5 bpw. Risk: rare-token rows (use w05's guard rule).
- **Conflicts.** #34 (slab head, whose plans are suspect — see H21).

**H20. Strided head row order plus two-sided.** The emulated bound is 0.046 vs 0.275 for vocabulary order. Needs the slab head's row-permutation plumbing in the plugin (decode gather). Run only if H19 passes.

**H21. Rebuild the slab-head plans from correct row errors.**
- w04 and w08 both found that `--save-wq` `rowerr.npz` is computed from the regularized weight: per-row Spearman ≈ 0.
- So the `headslab.py` plans (#34, `KN_HEADSLAB`) were optimized on noise. Recompute row errors offline from `wq` + checkpoint + `lm_head.input`; CPU only.
- **Must precede any slab-head use.** Not an H_out idea, but found by this lens's workers.

**H22. Slab head plus two-sided inside each slab.** Combination of H19 and H21. Sorting by Fisher flattens the blocks, so pair it with a within-slab strided order. Run after H19 / H21.

**H23. Add the head's within-block −ppᵀ term.**
- A dense 128×128 correction per block from top-512 co-occurrence. It refines H19's diagonal; F·err drops these cross terms.
- CPU cost to build is small. Try only if H19 wins.

**H24. Category-balanced head Fisher, including long-trace positions.** Robustness against a qad-pool-peaked Fisher. Re-score H19's winner on agentic/multilingual slices.

**H25. Embedding LDLQ along the hidden dimension with the residual-gradient metric.** *(new)*
- **Mechanism.** Embedding rows are trellis-coded after fwht128 with H = I. Replace H with P diag(d_res) P per 128-block. d_res is the 5120-dim residual-gradient diagonal at the layer-0 input, approximated by layer 0's writer Σdy² (exists).
- This shapes error across hidden dimensions within each row, using the same algebra as H_out. No row weighting, so it is not the dropped vocabulary tiering.
- **Fit.** Embedding levels. No runtime cost.
- **Test.** Predicted R for d_res with blk.py (CPU, minutes). Then `embtrellis.py` with H, then one 27B embedding swap.
- **Gain.** −0.2…−0.5% at 2.5 bpw; more at ≤ 2 bpw, where the embedding sits at its floor.

**H26. Choose the head's mean-row shift by Fisher-weighted error** (the t·μ family, 2609.31291). On Qwen3.5 heads it gave nothing with GPTQ. Fold into H19's arms only as a cheap extra.

---

## 5. Pricing and allocation with the diagonal (no rounding change)

**H27. kfit2 / knap prior on the two-sided proxy for every option.** CV on the V2 measurements (hproxy decodes plus h) and the V3 swaps: does e2 beat e1 as the proxy feature (log-RMSE 0.10–0.12 now)? CPU.

**H28. The turboderp-vs-ours mystery (design §1d).**
- Our tensors show 23% lower one-sided proxy error, but 1.25–1.58× more KLD per proxy unit at K4–8.
- Compute e2 for both sets at K5–8 (CPU, hproxy). If our advantage disappears in the h-metric, that explains it, and tells whether his tensors should stay as options.

**H29. A8/A4 activation-cost proxy weighted by h.**
- Use rel_out_h = Σ_r h_r (W e)_r² / Σ_r h_r (W x)_r² on the E1c dumps (colo `/data/work/aq`).
- Refit the aqcal family factors (log-RMSE 0.25 today). Better A8/A4 prices for knap4 and #46-03. CPU.

**H30. Use 1 − R per family as a prior for knap family factors.** Smooths the oursH factor fit when few swaps exist. CPU.

**H31. Per-head K on ssm_qkv / attn_qg** (format; V4).
- The predicted 27B per-128-slab spread is larger than on the 0.8B (qkv 0.29 bit, q 0.17, L0 qkv 1.02).
- #96 put it at ≈0.04 bpw overall. Strengthens the BACKLOG item for V4, not V3.

**H32. Per-tensor input damping chosen by the two-sided proxy.** #38 used the one-sided proxy. Low priority (no damping alternative reached ≥ 1%).

**H33. Do not use h for per-row or per-channel fits** (swj, per-row gains). This is an anti-idea.
- A diagonal output weight cancels in any fit that is independent per row or per channel.
- It only changes decisions that are shared across rows: su, family α, rounding coupling, and allocation.

---

## 6. Output-factor variants for rounding (need KL tests)

**H34. Tempered factor d^p at low K (p = 0.5).**
- The predictor says it loses about 1/7 of the gain under white noise. It may still win in KL where the factor is low (K2.5 at 27B).
- Evidence for tempering under large errors: ShamAN-Q (p = ½ best), KBBQ (λ < 1). Block normalization (#33-27) was a mild version and helped.
- **Test.** 0.8B K2, 3 seeds (~1.5 h, 4070). Gain at 27B ±1%; uncertain.

**H35. σ_out 0.1 instead of 0.01.** Costs nothing in the predictor, so it is free robustness. Test on 0.8B K2/K3. Adopt only if it is not worse.

**H36. K-dependent tempering (p rising with K).** Combine H34 and H35 if either shows a K trend.

**H37. Diagonal + low-rank H_out** (0.8B research; V4 for 27B). Use the existing full Sketch B 8K factors. A 27B version would need a backward collection, which is not allowed now.

**H38. Shrink full gate/up H_out toward its diagonal** (0.8B research). #33-12's mixed arm was best at K3; the full gate/up factor hurt at K2.

**H39. Output-side pivoted processing order of rotated rows** (in 16-row groups; free at runtime).
- Compute the change in Σ pivots with blk.py, CPU, minutes.
- Expect < 1%, because the rotated diagonal is constant.

**H40. Prepare H_out with the post-refit sv.** `prepare_H_out` uses regularize's sv, while refit changes sv by an LS factor. A small mismatch; iterate once only if H09 shows realized < predicted.

**H41. Output factor recentred at a quantized operating point** (0.8B research). GEMQ / contextual-utility evidence says sensitivities should be measured near the actual quant. Low priority.

**H42. MLP neuron permutation for gate/up output coupling.** Measured dead at 27B (§0.6). Record it and close B6.

---

## 7. Coherent / structured output curvature (forward-only)

**H43. Coherent-aware sv refit.**
- Replace the per-row LS (which shrinks gains) with
  min Σ_r h_r ‖W_r − c_r Q_r‖²_H + κ_F (Σ_r c_r ‖Q_r‖_H − Σ_r ‖W_r‖_H)², with κ_F measured in H15. Closed form; fixed codes.
- Overlaps C9 / #112 and swj. What is different: κ comes from a measurement instead of an α grid.
- 0.8B fixed-code test, 1 seed.

**H44. Mean-shift curvature for the writers.** Finite-difference ΔKL along W·E[x]. This prices the shortlisted "writers' mean-shift bias" before anyone builds it. 0.8B, minutes.

**H45. Closed-form head temperature from the output Fisher.**
- At a quantized head, the KL-optimal logit scale follows from Var_p(z) and a cross-term over the golden's top-512.
- Feeds the shortlist item "head KL-side gain refit". CPU plus one forward pass of the quant.

**H46. SPSA family probes on the 0.8B.** Random-sign simultaneous perturbations, 2 forwards per probe, map coherent curvature for every family × depth band at once.

**H47. Split the 27B quant error into coherent and random parts.**
- Project each tensor's error onto per-row gain residuals, then price them with H15's curvature against the h-diagonal for the remainder.
- Gives an estimate of how much of the K2.5 KL the hout family of ideas can address at all. CPU plus H15's numbers.

---

## 8. Input side: why the gradient-weighted H_in hurts

**Working hypothesis.**
- H_in drives a dense, aggressively used feedback metric. H_out acts only inside 128-row blocks, where R is 0.73–0.98.
- Replacing calib x xᵀ with a gradient-weighted, low-participation-ratio metric makes LDLQ push error into input directions the new metric calls cheap. At 2–3 bits that is outside the quadratic regime.
- Supporting evidence: DRQ (lower reconstruction loss can raise PPL); the full gate/up H_out hurt at K2; the proxy→KL factor rises with K.

**H48. Effective-sample-size diagnostic.** Compute the ESS of the per-token ‖dy‖² weights on the 0.8B. Heavy tails mean the weighted H_in comes from a few tokens. CPU or 4070, minutes.

**H49. Error-direction diagnostic.** For the 0.8B hin vs base arms, measure ‖E‖_F and the error energy in calib-H_in's lowest eigen-directions. This tests the hypothesis directly. CPU, existing quants.

**H50. Blend or temper the Sketch B H_in** (α ∈ {0.1, 0.25}, or the matrix geometric mean). #33-07's planned blend never ran. Use the existing 0.8B factors, 3 seeds (~1.5 h). Expect ≤ 1% at 27B.

**H51. KL-Shampoo token weights** dy_tᵀ D⁻¹ dy_t (bounded leverage, ShamAN-Q) for H_in and H_out. ShamAN-Q: 17.04 vs 19.21 PPL vs NKP at 1.7B. 0.8B collection, ~1 h. Research.

**H52. Gen-masked H_in at 27B (forward-only).** #86 was inconclusive (K2 0.976 on every seed). Expected ≤ −0.5% at 27B; low priority.

**H53. H_in from late positions in long traces (forward-only, mkhin27).** For the agentic tail. #101-style blends shrank at 27B; low priority.

**H54. Round on calib H_in, accept CD moves by the Fisher metric** (hybrid). Low priority.

**H55. tilecd scoring the two-sided objective.** Re-enables CD on attn_qg at K ≥ 4 (H03). Expected +0.5% on attn_qg only; low priority.

---

## 9. Cheaper H_out for future models (no or tiny collection)

**H56. Gaussian-surrogate analytic factor for gate/up.** *(new)*
- (g, u) per channel are taken as bivariate normal, with covariance from wᵀ H_in w (published Hessian).
- d_up,r ≈ Σ_j W_d[j,r]² d_res,j × E[SiLU(g)²]; gate uses E[(SiLU'(g) u)²]. Forward-free.
- Cross-fit at 27B: design on the analytic factor, score in the bn metric. CPU ~1 h on colo.
- If R_cross ≈ 0.90 (vs self 0.895), future models need no gate/up backward. #84's A9 failed a cosine gate (0.86), but cross-fit R is the right test.

**H57. A9g (forward statistics plus the measured down-output diagonal) at 27B.** Same cross-fit as H56. Needs one forward-only stats pass; allowed.

**H58. Dose at 27B in the predictor metric.** Cross-fit the 128…1920 snapshots (HF dataset). Gives the minimum collection for future models (the 0.8B plateau was ~384). CPU.

**H59. bf16 hi/lo backward judged by predictor-metric cross-fit.** KL was too noisy to separate fp32 / x2 / bf16 in #33-15. 0.8B.

**H60. Per-family R priors transfer across sizes** (§0: 0.8B ≈ 27B). For a new model, predict share-weighted 1 − R × an assumed factor of ~0.25. Skip the collection if the result is below 1%.

**H61. Collect only the late layers (truncated backward).** The depth table says the early GDN layers carry much of the gain, so this likely fails. Listed for completeness; low priority.

**H62. Better gradient targets** (0.8B research). Top-k teacher expectation instead of one sampled label, to reduce label noise so fewer sequences are needed. A KL-to-quant (recentred) target is the other option.

**H63. Small-n bias correction.** w04 showed a small-n diagonal over-predicts its own gain. Report self and cross-fit for every new estimator; this is method, no cost.

---

## 10. Trellis, Hadamard and fractional K

**H64. Fractional-K drift under two-sided feedback.** `ldlq_drift` was calibrated one-sided, and the K ≥ 4 ladder has eighth and quarter steps. Check the two-sided proxy with the drift on vs off at K4.125 / 4.5 on 3 tensors (GPU, minutes). **Freeze: yes** (cheap).

**H65. Larger output Hadamard blocks** (format; V4). Measure R at 256 / 512 / full with blk.py (CPU, minutes): bigger blocks raise the attainable GM/AM gain. Information for V4 only.

**H66. Hadamard sign seeds do not affect R** (anti-idea; #41). R depends only on h·sv².

**H67. T1's block-wavefront gives bigger trellis batches.** Part of H01; check the GPU occupancy gain separately.

---

## 11. Validation method at 27B

- **H68.** Use K4 for every rounding variant: seed σ 0.6% vs 1.8% at K2.5.
- **H69.** Use family-partial requant arms (cheap after H01): smaller seed noise, and a direct answer.
- **H70.** Pre-screen with the predictor and the 27B factor (0.2–0.3). Only variants predicted at ≥ 3% h-proxy go to KL.
- **H71.** Fixed-code refit variants need one pair; there is no rounding noise.
- **H72.** Run identical non-trunk packing checks in every pair (the swj K4 lesson). Add a tooling assert.
- **H73.** Score finals for any H_out change on verify plus trace chunks, not only on search.

---

## 12. Long context

**H74. Re-score the existing 0.8B base / edy arms on `g08-long1`** (prepared: w10-longpos; never run). Fixed codes, about 1 h on the 4070. Does the gain hold past 2k tokens, where GDN state error accumulates (STEPQuant, LeapQuant)?

**H75. Score 27B bn4 / base4 on the 32k trace chunks.** Part of H13; listed separately because the private set's long agentic traces are ~35% of public scored positions.

**H76. Check whether 2048-token collection windows under-weight GDN long-memory projections** (0.8B research). Compare the diagonal from 8k windows with 2k windows on GDN in_proj. Run only if H74 shows the gain fading with position.

---

## Ranked top 20 (expected gain per unit cost)

| # | idea | cost | expected at 2.5 bpw (27B) | freeze |
|---|---|---|---|---|
| 1 | **H03** tile-CD vs H_out on attn_qg | config | +0.1–0.2%; avoids a crash | yes |
| 2 | **H02 + H05** two-sided prices / logging | CPU | −0.3…−1% (−0.5…−2% at 3.5–4.5) | logging yes |
| 3 | **H01** block-sparse `ldlq_2hess` | ~1 card-h + dev | saves ≈55–60 card-h; enables H06 / H16 | yes |
| 4 | **H19** two-sided head (prepared) | ~4 card-h | −1…−2.5% | before head donors |
| 5 | **H16** H_out on the ~20 lowest-R tensors at every K | ~2 card-h | −0.5…−1.5% | yes |
| 6 | **H09** realized 27B h-proxy | CPU 2–4 h | diagnostic: sets the shrink prior | — |
| 7 | **H10** family attribution swaps | ~1 card-h | diagnostic: sets H06 / H16 scope | — |
| 8 | **H06** H_out to K3–3.5 after H01 | decision (+1 pair) | −1…−2% on those levels | yes |
| 9 | **H21** slab plans on correct row errors | CPU | protects the head plan (≈0.5–1%) | before slab use |
| 10 | **H08** h-weighted su refit (B1) | 25 min 4070 + 1 card-h | −0.3…−1% | no (container edit) |
| 11 | **H15** coherence amplification probe | minutes (0.8B), 2 card-h (27B) | root cause; may redirect the lens | — |
| 12 | **H13 / H75** verify and long checks of K4 | ~1 card-h | insurance for 24 levels | before ladder |
| 13 | **H07** writers one-sided at every K | none | ≈0; removes an untested combination | yes |
| 14 | **H64** fractional-K drift under two-sided | minutes | correctness | yes |
| 15 | **H27** kfit / knap prior on e2 | CPU | −0.2…−0.5% | no |
| 16 | **H25** embedding residual-gradient LDLQ | CPU + 1 eval | −0.2…−0.5% | before embedding levels |
| 17 | **H29** h-weighted A8 / A4 prices | CPU | better fast-variant picks (±) | no |
| 18 | **H28** turboderp e2 comparison | CPU | settles §1d; ±0.5% at ≥ 5 bpw | no |
| 19 | **H34 / H35** tempered or damped factor at low K | 1.5 h 4070 | ±1% | no |
| 20 | **H56** Gaussian-surrogate gate/up factor | CPU ~1 h | future models: removes half the backward | no |

## The 5 cheapest diagnostics that would most change this ranking

1. **H09 — realized h-proxy of the 27B arms** (CPU). If realized ≈ predicted, all output-shaping ideas get ×0.4 and coherent fixes (H15 / H43, swj-like) move up. If realized ≪ predicted, there is a quantizer-side loss to recover first.
2. **H15 on the 0.8B — coherence amplification** (minutes on the 4070). A ratio ≫ 1 says the diagonal H_out is the wrong object at scale and coherent refits should get priority. A ratio ≈ 1 says the diagonal is fine and the shrink is elsewhere.
3. **H10 — family swaps at K4** (~1 card-h). Shows whether GDN-in carries most of the 27B gain as predicted. That decides whether H06 / H16 go ahead, or whether H_out stays at K ≥ 4 only.
4. **H19 stage 1 — two-sided head on headsim** (a few card-hours). The largest single hout gain that needs no collection. If vocabulary order gets ≤ 0.8× head KL, it goes to the top of this list.
5. **H12 — factor vs K from existing 0.8B numbers** (CPU minutes). A steep rise from K2 to K3 makes H06 likely to pay at K3–3.5. A flat curve makes H06 a coin flip.

**Files.** Scripts are in `/tmp/hout-lens/` on the VM (`pred27.py`, `onesided27.py`, `agg*.py`). Results are on colo in `/data/work/ideas-hout/pred27.*.jsonl` and `one27.*.jsonl`: 400 tensors each, with per-tensor R (with and without sv), one-sided ratio, slab spread, top-1% share, tempered / damped variants, and gate/up permutations.
