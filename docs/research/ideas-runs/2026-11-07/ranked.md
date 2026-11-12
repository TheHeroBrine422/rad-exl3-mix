# Ranked ideas — critic pass, 2026-11-07

I merged the 11 lens reports (codes, composable, crossarea, faraway, heal, hout, kernels, lowbpw, measure, open, redteam) into about 95 ideas in 11 themes and ranked them by expected KL gain at 2.5 bpw × probability ÷ cost of the first test that decides. Five lens claims were stale or wrong; §1 has the details.

**Main conclusions**
- **The 27B noise sets the method.** One requant seed pair has σ ≈ 0.018 at K2.5, so it only resolves effects of 4–5% or more. Ideas that can be tested with the codes fixed (no rounding noise, resolves about 0.5%) or by splicing a few tensors rank above ideas that need a full requant.
- **Proxy error does not predict KL.** #106 was dropped this morning: it cut o_proj quantization error by 9% on every seed and KL did not move (K3 0.998, K2 1.003). I discount every idea whose only evidence is proxy error (salts, larger RHT, massive-column extraction, ordering ideas).
- **Coherent fixes do not uniformly grow with size.** swj grew from the 0.8B to the 27B (−6.5% at K2.5). C9 on the writers shrank (27B pair 0.9911 search / 0.9936 verify). Each coherent fix needs its own 27B pair.

Exchange rate: 0.01 bpw ≈ −1.5…−1.7% KL at 2.5 bpw.

---

## 1. What I checked, and corrections to the lens reports

I ran reads on colo (single-thread, under 2 min each) and in the repo. Nothing was written outside my output.

| # | Check | Result | Consequence |
|---|---|---|---|
| 1 | **Super-weight conflict.** Row 3994 of L0/L2/L3 down_proj, L0 out_proj and L3 o_proj in the bf16 checkpoint | The largest single entry holds **0.4%, 2.8%, 2.2%, 0.3%, 0.5%** of the row's energy. The row is dense: its norm is 21× (L0 out_proj) and 18× (L3 o_proj) the median row norm. Entries above 50× the tensor RMS: L0 down 40, L0 out_proj 221, L3 o_proj 201. | Faraway's claim that L2 down row 3994 has 99% in one entry is **wrong**. Open's "≤3%" is right; crossarea's "40 entries" is L0 down. Sparse super-weight escapes (crossarea #45) cannot capture the row; only whole-row exceptions can. |
| 2 | **Row 3994's share of each writer's H_in-weighted row energy** (calib Hessians) | L0 out_proj **0.947**, L3 o_proj **0.523**, L30 out_proj 0.012 | Confirms faraway's L0 figure. The massive-row problem is limited to L0–L3. |
| 3 | **Norm gains at channel 3994** | post_attention_layernorm: 0.0039 (L0), 0.0039 (L3), **0.0000** (L7), 0.074 (L11), 0.445 (L30), 0.94 (L62). input_layernorm: 1.00–1.30 at L0–L11. | The model already shields the **MLP** readers from the channel in early layers. Only attention/GDN readers see it at full size, which matches codes M1. Massive-column work on gate/up before L30 is moot. |
| 4 | **Tile CD × H_out refusal** (redteam C1/C2, hout H03) | **Stale.** The current `rxq/config.py` has no such refusal. w76 implemented CD on the two-sided objective (LOG 11-07 9:25). v3.toml has `apply_k_min = 4.0` and the fallback `k_max = 3.875` if `tools/CDHOUT27.md` fails. | What remains is the prepared 27B K4 no-harm pair (`tools/cdhout27.sh`, not run). |
| 5 | **KN_SWJ_PTS** (measure A12) | The research `v3/levels/reprice/knap5.py` still defaults to `2.5:0.9346,4:0.7786` (the withdrawn K4 ratio). The pipeline passes the corrected curve `2:0.91,2.5:0.935,4:0.989,5:1.0` (v3.toml `swj_pts`). | Only ad-hoc knap5 runs are at risk. Fix the default (one line). |
| 6 | **H_out K4 on verify** | H27.md: "Not checked on verify (rule decides on search)." | Confirms redteam A1 / hout H13. Diagnostic D1. |
| 7 | **GDN in_proj_a/b shape** | `[48, 5120]`. The exl3 quantizer asserts `size_n % 128 == 0`. | Lowbpw A8 ("trellis GDN a/b") cannot be done as stated. int8 per-row (composable D1) is the realistic form. |
| 8 | **Slab-head row errors** (hout H21) | w04's `rowerr.patch` adds `load_w()`; `quantize/ourq.py` has no `load_w`. | The rowerr fix looks unapplied. Apply it before any slab-plan rebuild. |
| 9 | **New LOG facts since the lenses ran** | #106 **dropped** (proxy −9%, KL not following). #112-02 27B C9 on the writers: seed-1 pair 0.9911 / verify 0.9936, base pair pending. #46-03 interim: A8-g128 vs f16 ×1.0001 at K2 (free); A4 gets costlier with depth. | A8-side massive-channel ideas are moot at K2. C9's 27B effect is ≈ −0.6…−0.9%. |
| 10 | **Earlier C9 on the 0.8B** (w02, 11-03; no swj, no H_out; fit slice) | Per-family α\*: attn_in −1, attn_o −1, **gdn_in 2**, gdn_out 2, gate_up 1.55–2, down 0. gdn_in at α 2 alone: K2 0.5028 vs base 0.5325 (**0.944**), K3 0.983. | The strongest single family was the GDN **inputs**, which #108 later dropped because it overlapped with H_out. H_out is not adopted at ≤K3.5 on the 27B, so that reason for the drop does not apply at 2.5 (idea A1). |

Paper numbers I verified by grep:
- 2609.04098: in_proj_a/b are the smallest single-projection effects (2.1%/2.6%).
- 2609.33923: the oracle allocation (6.087) loses to a block-output cosine signal (6.041).
- 2608.06291: FastKron 250 ms vs 23,084 ms at n=8192.
- 2610.32127: E8P 2-bit Wiki2 PPL 15.01 (LDLQ) → 12.76 (LDLQ+CD) → 11.63 (MPQ).
- 2609.38521: tempering p=½ 17.46 / 16.72.
- 2610.09227: composition slope β 0.26–0.77.
- 2609.11716: counteraction cancels 50.2% / 63.4%.

Paper numbers cited from the digests but not checked are marked *(not verified)*.

## 2. Scoring rules

- **Score** = mid-range gain (%) × P ÷ cost of the first decisive test, in card-hour equivalents. Conversions: 4070-hour = 0.3, CPU-hour = 0.1, dev-day = 4. The table gives adoption cost separately.
- **Shrink from the 0.8B:** noise-shaping ideas ×⅓; coherent fixes ×0.5–1.5 (swj ≈ ×2, writers' C9 ≈ ×0.7); proxy-only results ×0.3, with P ≤ 0.3 until KL confirms.
- **Pass rules for 27B tests:**
  - fixed-code pairs: both pairs < 1 and geo ≤ 0.99 (the #112-02 rule);
  - splices of a few tensors: geo ≤ 0.985;
  - full requants: two pairs with geo ≤ 0.98 (the H27 rule).
- **Pass rules for 0.8B tests:** 3 seeds, every seed < 1, verify keeps ≥ ⅔ of the search gain.
- **Fit types:**
  - **L** = per level (fit on bf16 inputs, library-safe);
  - **F** = final pass on the 2–3 shipped mixes;
  - **M** = per mix;
  - **X** = format change;
  - **T** = training (deferred with healing).
- **Hardware:**
  - **G** = gpubox, 2×R9700, cost in card-h;
  - **S** = 4070 (0.8B proxy);
  - **C** = colo (cpu1, 36 cores);
  - **R** = rented.

## 3. Top 25

| # | ID | Idea (lenses) | Fit | First decisive test (hardware, cost) | Pass rule (fixed before data) | Gain @2.5 | P | Freeze? |
|---|---|---|---|---|---|---|---|---|
| 1 | P1 | Log per-row e_n, LS gain vs pre-refit scale, pre/post-swj proxy, and the two-sided cost (h-weighted, even for one-sided levels) in every ladder job (composable C3, open G2, redteam I3, hout H05) | L | Code change in rxq; 0 card-h | n/a (enabler) | enables A1–A3, H5, P2 | 0.9 | **yes** |
| 2 | B1 | GDN a/b off bf16 as int8 per-row (composable D1, redteam D9, kernels G6, BACKLOG 11-02; replaces lowbpw A8) | X, one per model | Splice int8-dequantized a/b into the main 2.5 mix, 1 pair; G, 0.5 card-h | Ratio ≤ 1.003 (the 0.007 bpw saved is worth ≈ −1.1%) | −0.9…−1.5% | 0.7 | no (plugin op, ~1 dev-day) |
| 3 | A1 | GDN-input coherent refit at ≤K3.5 (z-gate swz / C9 gdn_in / #108 GDN arms) (composable A1/A2/B4, lowbpw B1, crossarea note) | L | 0.8B fixed codes on trunk+swj+C9 writers with **no H_out**, K2/K3 × 3 seeds, plus a mixed-K upstream arm (S, 1.5 h); then a 27B fixed-code pair on the swj K2.5 levels (G, ~2 card-h) | 0.8B: K2 ≤ 0.98 and K3 ≤ 0.99, every seed; the mixed-K arm keeps ≥ ½. 27B: both pairs < 1, geo ≤ 0.99 | −1…−3% | 0.5 | no |
| 4 | H2 | In-context price of head and embedding at 2.5, then re-solve (measure C8/F5, redteam D6/D7) | allocation | Diagnostic D6: 4 evals; G, 0.5 card-h | Measured ΔKL/Δbpw within ±25% of knap5's price; otherwise reprice | 0…−3% | 0.35 | before head donors |
| 5 | P2 | Two-sided level positions (KN_LEVELS) for H_out levels plus the K3.875/K4 boundary factor (hout H02/H27/H30, measure B2/B3, redteam D1, composable G1) | allocation | C, a few CPU-h, on the logged e2 | Reprice moves ≥ 3 tensors across K4 at 2.5, or predicted ≥ 0.5% at 3.5–4.5 | −0.1…−0.5% (−0.5…−2% at 3.5–4.5) | 0.6 | no |
| 6 | P3 | Reprice after each coherent fix: add C9 and A1 to the price list; fix the knap5 default (open H1, measure A11/A12) | allocation | C, minutes | n/a (correctness) | protects 0.5–2% | 0.6 | no |
| 7 | H1 | Head temperature τ folded into model.norm; optional V-dim logit bias (crossarea #25/#26, faraway B4/B5, lowbpw A5/B5/B6, composable A11/A12, open A4/E1, heal J1, hout H45, measure J3) | F (τ free); bias needs field K5 | 3 evals of the main 2.5 mix with τ ∈ {0.98, 1.00, 1.02}; G, 20–30 min; hout H45's closed form as prior | Best τ ≠ 1 at ≤ 0.995 on search, verify agrees | −0.2…−1% | 0.4 | no |
| 8 | H3 | Head rank-r deflation and re-spend (lowbpw A1/A2; faraway B1 is the V4 form) | head level + small X (f16 rank-r adder) | headsim on colo (exact head KL, 32,768 inputs) at equal bytes, r ∈ {0,1,4,8,16}; C, ~4 CPU-h | ≤ 0.85× head KL at the K3.5–K4 budget; then a whole-model swap pair ≤ 0.985 | −1…−2% | 0.35 | before head donors |
| 9 | T1 | Block-sparse `ldlq_2hess` (PROFILE T1 / FastKron) (hout H01/H67, composable C1, kernels L1/L2) | quantizer speed | 1–2 dev-days; 1 card-h equivalence check (two-sided proxy ≤ reference, per tensor, within seed spread) | Same proxy ±seed spread, ≥ 3× faster | saves ≈58 card-h (24 levels ≥K4); enables T2/T3 | 0.8 | **yes** (codes change float order) |
| 10 | W1 | Writers-asym rollout support if #105 passes: price the tie constraint (measure E3), plain+heal vs per-mix asym (heal H3), sparse re-encode (crossarea #65) | decision support | C, < 1 h for E3; S, 3 h for H3 | E3: predicted tied-K loss ≤ ⅓ of the asym gain → recommend tied K; otherwise a per-mix pass on the 2.5 mix only | −1…−4% (conditional) | 0.5 × 0.6 | the user's rollout pick |
| 11 | Q1 | i8c codebook at ≤K3.5. #78-04 was null **on top of H_out**; the 27B has no H_out there (codes A1/A3, lowbpw D1/D2) | X (codebook variant, +2 FMA decode) | 0.8B K2/K3 × 3 seeds on trunk+swj (S, 1.5 h) and settle64 on 27B tensors (C, 1 h) | 0.8B ≤ 0.985 at K3 and ≤ 0.98 at K2, every seed; settle64 median ≤ 0.99; then a no-harm 27B pair | −0.5…−1.2% | 0.35 | **yes** |
| 12 | K1 | Per-head K for GDN qkv/z and attn_qg, assembled from existing levels (head-aligned 128-row segments keep the output Hadamard blocks intact) (codes F1/F2, composable D5, lowbpw E1, hout H31; #96 ceiling ≈0.04 bpw) | X (segment descriptor; no requant) | 0.8B fixed-code splice by Σdy² head importance, 3 seeds; S, 2 h | ≤ 0.985 at equal bytes, every seed | −0.5…−2% | 0.25 | format only (after the freeze is fine) |
| 13 | P7 | Localize the seed variance, then pick per-level seeds by KL (measure A1/A2/A3, redteam G1/G2/A7) | L | Diagnostic D5, then a layer split; G, 0.3 + 1 card-h | Seed ordering replicates on verify (per-doc diff correlation ≥ 0.5) before any seed picking | 0…−2% | 0.15 | yes, if adopted |
| 14 | P4 | Composition/additivity calibration on 8 random V3 mixes (measure C1, redteam D4, crossarea #9/#67/#68, composable G3) | allocation | G, ~2 card-h, after the first levels | Fitted κ/β moves predictions by > 3% at 2.5 → refit kfit | −0.5…−2% | 0.35 | no |
| 15 | M1 | Exact row 3994 in L0–L3 writers (faraway A1/A3/A4, open B1, crossarea #44) | L + side-car field (K5) | Diagnostic D3 oracle (decode patch, fixed codes); G, 1.5 card-h | Row arm ≤ 0.985 → requant the 4–8 writers with the row excluded, then a pair | 0…−2% | 0.3 | writers only |
| 16 | A9 | Jensen / mean-shift biases at gates and writers (faraway C1, crossarea #31/#34/#35, open A5/A6/A12, lowbpw B7/B8, heal B8, hout H44) | L + bias field (K5) | 0.8B fixed codes; S, 2 h | K2 ≤ 0.985, every seed | −0.3…−1% | 0.2 | no (field before release) |
| 17 | A4 | Embedding row gain (energy or IV restoring) (crossarea #18, composable A13, lowbpw A10, open E3) | L | 27B fixed-code embedding swap, 2 pairs; G, 1 card-h | Both < 1, geo ≤ 0.995 | −0.2…−1% | 0.3 | no |
| 18 | A3 | Estimator family per family: LS → energy → IV (λ-ladder), Deming, unbiased inner product (crossarea #11–#15, faraway C6/C7, lowbpw B3/B16, open A1, hout H43) | L | 0.8B fixed codes; S, 1 h | Beats per-family C9 by ≤ 0.99 at K2, every seed | 0…−0.5% beyond A1/A2 | 0.2 | no |
| 19 | Q2 | Centered LDLQ / drift-amplified H + γμμᵀ / document-scatter H (crossarea #28/#29/#30, composable C4, lowbpw C5, open A7) | requant (#29 needs no format change) | 0.8B requant K2/K3 × 3 seeds; S, 2 h | ≤ 0.98 at K2, ≤ 0.99 at K3, every seed | −0.3…−1% | 0.15 | **yes** |
| 20 | T3 | H_out on the ~20 lowest-R tensors at every K (L63 up R 0.367, gate 0.540) (hout H16/H17, H10 sets the scope) | requant (per-key switch in the stamp) | Requant 20 tensors at K2.5 with H_out, splice pair; G, 2 card-h | Geo ≤ 0.985 | −0.5…−1.5% | 0.3 | **yes** |
| 21 | T2 | Reopen H_out at K2.5–K3.5 once T1 removes the 5× cost (hout H06, redteam A3) | requant | One more K2.5 or K3.5 pair; G, ~4 card-h | The H27 rule unchanged | −1…−2% | 0.4 | user decision |
| 22 | P5 | KKT audit at finalists and lazy measured descent (measure C2/C4; Q-Strata 2608.30564 *(not verified)*) | search | 12–18 evals per size; G | A family move gives ≤ 0.99 on search and verify | −1…−3% | 0.3 | no |
| 23 | A6 | Closed-form gain rules distilled from the 0.8B healing results (composable A7) | L | C, 1 day + S, 2 h | Rule-applied 0.8B ≤ 0.98 at K2 vs swj+C9 | −0.5…−1.5% | 0.2 | no |
| 24 | A7 | Attention temperature: one scalar per layer on q_norm.weight. This differs from #108, whose pre-norm scales q_norm/k_norm absorbed (crossarea #21/#22, faraway C2, open A2, lowbpw B4) | L/M | Entropy and logit-scale shift diagnostic (crossarea #4), 1 eval with hooks; then 0.8B; S, 0.3 h | Max-logit or entropy shift ≥ 1% before testing; 0.8B ≤ 0.99 | 0…−0.8% | 0.2 | no |
| 25 | W2 | Per-mix closed-loop refit of swj/C9/norms on quantized upstream for the shipped 2.5 mix (crossarea #63/#73, composable J1/B6, open A8/F1, lowbpw F3/F4, heal J3) | M/F | 0.8B per-mix vs per-level; S, 1 h; then G, 2–4 card-h per mix | 0.8B ≤ 0.985 over per-level fits, verify agrees | −0.5…−2% | 0.3 | no |

Next six, in order: A2 (C9 on more families and depth knots), A5 (gen-masked / entropy-weighted fit tokens), H5 (slab head on fixed row errors), A8 (norm re-inflation), M3 (spike-excluded damping), Q4 (cd3 on attn_k).

## 4. Cheapest discriminating diagnostics

| # | Diagnostic | Cost | Rule (fixed now) | Decides |
|---|---|---|---|---|
| D1 | **H_out K4 on golden-verify27**: both pairs (bn4/base4, bn4s1/base4s1); also the 32k trace chunks once the agent-search32 golden is fixed (redteam A1, hout H13/H75) | 4 evals, ~12 min (G) | Both verify pairs < 1 and geo ≤ 0.98. Otherwise hold the 24 K≥4 H_out levels (~60 card-h) until a third pair. | the K≥4 H_out spend |
| D2 | **CD × H_out no-harm pair** `tools/cdhout27.sh` | ~1.3 card-h (G) | As written in CDHOUT27.md; on FAIL, `k_max = 3.875` | CD at K≥4 |
| D3 | **Massive-channel oracle at 27B.** Decode-patch the main 2.5 mix with three arms: row 3994 exact in all writers; column 3994 exact in all attention/GDN readers; both. All other codes fixed. Seeds base and s1. | ~1.5 card-h (G) | An arm funds its theme only at ≤ 0.985. If every arm is > 0.99, close the massive-channel theme for V3 (E1, #41, #44, #49, D6, M2–M4, lowbpw C1/C2). | ~10 ideas and their freeze pressure |
| D4 | **Coherent vs random split.** Average the dequantized weights of 4 rotation seeds and compare with single seeds: 0.8B K2/K3 (S, 1.5 h); 27B layers 0–31 from the existing base/s1/s2 K2.5 trunks (G, ~1 card-h) (composable I1, crossarea #1, faraway C9, open M1/M2, hout H47) | 1.5 h S + 1 card-h | KL(avg)/KL(single) ≥ 0.7 → coherent error dominates: fund theme A. ≤ 0.45 → random rounding dominates: fund codes and rounding. | the A vs K/Q budget |
| D5 | **Seed replication on verify.** Re-score base, s1, s2 (K2.5) on golden-verify27 (measure A1) | 3 evals, 0.3 card-h | Per-doc seed differences correlate ≥ 0.5 between search and verify → seed spread is real (fund P7). Otherwise use a smaller σ in every 27B rule. | P7 and the noise floor in every rule |
| D6 | **In-context head/embedding slope at 2.5.** Main mix with head K4→K3.5 and K4→K4.5, embedding t2→t2.5 | 4 evals, 0.5 card-h | Within ±25% of knap5's slope → no reprice. Otherwise reprice and re-solve (H2). | H2, and the scope of H3/H4 |
| D7 | **swj in the mix, fixed codes** (27B main 2.5 mix with and without swj; measure B8), and A1's mixed-K robustness on the 0.8B (composable B4) | 0.1 card-h + S 0.5 h | Measured within ±1% of the 0.935 price; GDN refit keeps ≥ ½ with upstream at K±1 | swj pricing and A1's library form |
| D8 | **Card-0 clock check.** Max sclk is 2,350 MHz on card 0 vs 3,139 on card 1 (kernels A2) | 5 min | If card 0 is capped, fix it before the ladder (Viterbi is ~80% of quant time) | up to ~25% of card-0 ladder time |
| D9 | **Ladder dry run** through rxq: one H_out K4, one fractional K4.25 and one mul1 K6.5 level on 2 layers, plus CPU checks of fractional-K drift under two-sided feedback (hout H64) and `prepare_H_out` with the post-refit sv (H40) (redteam C3/I1) | ~2 card-h + CPU minutes | All jobs finish; two-sided proxy within ±5% of the research path | prevents failed 3 h levels |
| D10 | **gdn_out α vs the massive rows** (lowbpw C3). Re-run #112's α=3 with the L0–L3 writers excluded. | S, 30 min | If ≥ ½ of C9's gain disappears, C9 is a massive-row fix in disguise and M1 is the cleaner form on the 27B | C9 scope, M1 |

## 5. Full merged list, by theme

Format per row: idea (lenses) · fit · first test and pass rule (hardware, cost) · gain @2.5 · P · notes.

### A. Coherent fits on frozen codes (zero runtime unless noted)

| ID | Idea (lenses) | Fit | Test → pass (hw, cost) | Gain · P | Notes |
|---|---|---|---|---|---|
| A1 | GDN-input refit revival: z-gate, C9 gdn_in, #108's GDN arms (composable A1/A2/B4, lowbpw B1, crossarea) | L | §3 #3 | −1…−3% · 0.5 | The q/k rows of in_proj_qkv pass through l2 norms, and per-head uniform v gains pass through the gated RMSNorm; neither is visible. Fit only the z rows and the within-head shape of the v rows. Overlaps C9's gdn_out (fit after it) and asym writers if #105 passes. |
| A2 | C9 α on more families: attn_in/attn_o −1 (w02), gate_up on top of swj, depth knots (composable A3/A5, open A9) | L | 0.8B fixed codes 1 h → K2 ≤ 0.99 every seed; 27B fixed pair 1 card-h | −0.2…−0.8% · 0.3 | w02 had attention preferring shrink (α −1): only −0.1…−0.3% at 0.8B. gate_up likely overlaps swj. |
| A3 | IV / energy / Deming / unbiased refits (crossarea #11–#15, faraway C6/C7, lowbpw B3/B16/A6, open A1/E3, hout H08/H43) | L | §3 #18 | 0…−0.5% · 0.2 | The LS identity puts the amplitude deficit at ≈1.8% at K2.5 (open). "Restore everything" (α=1 everywhere) is measured worse than per-family α at 0.8B K3 (0.983 vs 0.970), so only per-family forms survive. H08 (h-weighted su refit) belongs here (w02 B1 screen: K3 0.993, K2 0.975). |
| A4 | Embedding row gain (crossarea #18, composable A13, lowbpw A10, open E3) | L | §3 #17 | −0.2…−1% · 0.3 | Layer 0's input RMSNorm hides per-token amplitude; only the residual path sees it. Keep expectations low. |
| A5 | Gen-masked / entropy-weighted fit tokens for swj, C9, A1 (composable A8, crossarea #72) | L | 0.8B 1 h → ≤ 0.99 at K2; 27B swj refit + pair 1.5 card-h | −0.2…−0.8% · 0.3 | #83: gen-masking helped the H_out collection (K3 0.971). |
| A6 | Healing-distilled gain rules (composable A7) | L | §3 #23 | −0.5…−1.5% · 0.2 | Gets part of healing without 27B training. |
| A7 | Attention temperature on q_norm.weight, per layer or per head (crossarea #21/#22, faraway C2, open A2, lowbpw B4) | L/M | §3 #24 | 0…−0.8% · 0.2 | Run the entropy-shift diagnostic first (crossarea #4). |
| A8 | Norm gain re-inflation / depth gain schedule / GDN gated o_norm (crossarea #23/#24, faraway C3/C4, open A2/A3, lowbpw B9/F3/F4) | M (or L) | 0.8B 0.5 h → ≤ 0.99 at K2 | 0…−1% · 0.15 | #89-02: trained norm gains were ≤ 0.2% of healing's gain. |
| A9 | Jensen / mean-shift biases (faraway C1, crossarea #31/#34/#35, open A5/A6/A12, lowbpw B7/B8, heal B8, hout H44) | L + field | §3 #16 | −0.3…−1% · 0.2 | Needs K5. |
| A10 | Quant-time slicer de-biasing; re-test #39 by KL (crossarea #16) | requant | 0.8B 2 h | 0…−0.3% · 0.1 | Post-hoc C9 already covers the KL side. |
| A11 | Seed-averaged scale fits (crossarea #85) | — | — | — | Rejected (§7). |

### H. Head and embedding

| ID | Idea (lenses) | Fit | Test → pass (hw, cost) | Gain · P | Notes |
|---|---|---|---|---|---|
| H1 | Head τ plus bias / vector calibration (cluster: crossarea #25/#26, faraway B4/B5, lowbpw A5/B5/B6, composable A11/A12, open A4/E1, heal J1, hout H45, measure J3) | F | §3 #7 | −0.2…−1% · 0.4 | τ is free (folds into model.norm). The bias is a 248k f16 vector (0.00015 bpw) plus a field. |
| H2 | In-context head/embedding pricing (measure C8/F5, redteam D6/D7) | allocation | D6 | 0…−3% · 0.35 | The head is 0.189 bpw (7.6% of budget) at 2.5. Swapping head+embedding at a K4 trunk gave 0.76 for +0.095 bpw, a steeper slope than the trunk's, but at a different operating point. |
| H3 | Head rank-r deflation (lowbpw A1/A2) | head level + X | §3 #8 | −1…−2% · 0.35 | After centring, the top 8 PCs hold 41% of the H-weighted energy → ≈0.38 bit at equal error ≈ 0.018 bpw on the 1.27B-param head, minus 0.0012 bpw for r=8 in f16. Per-row trellis coding does not exploit cross-row structure. Decode adds ~2M MAC per token. |
| H4 | Two-sided head with the exact softmax-Fisher diagonal; strided order; −ppᵀ term (hout H19/H20/H22/H23) | head level | Prepared w05-head2s; ~4 card-h → whole-model pair ≤ 0.985 | −0.5…−2% · 0.3 | The head's Fisher is exact, not an estimated Σdy², which is a real difference from the trunk H_out that missed at 2.5. |
| H5 | Rebuild slab-head plans on correct row errors; whole-model check (hout H21, redteam D8, lowbpw A3/A7, codes J1) | head level | Apply `rowerr.patch` first (check 8); headsim CPU, then 1 card-h (#34-02 never run) | 0…−1% at 2.5 (more at 4–5) · 0.3 | Plans built from the buggy rowerr were optimized on noise. |
| H6 | Category-balanced head Fisher including long-trace positions (hout H24) | head level | C, CPU | 0…−0.3% · 0.2 | Robustness item. |
| H7 | Token-tree / DPCM head (faraway B1) | X | colo 1 day | −1…−2% · 0.3 → V4 | Subsumed by H3 for V3. |
| H8 | Embedding best-of-16 salts (codes I1) | X (embedding encoding) | embtrellis arm, CPU | −0.1…−0.2% · 0.3 | Low value. |

### B. Bytes outside the trunk codes

| ID | Idea | Fit | Test → pass | Gain · P | Notes |
|---|---|---|---|---|---|
| B1 | GDN a/b int8 (cluster) | X | §3 #2 | −0.9…−1.5% · 0.7 | 48 layers × 2 × 48×5120 bf16 = 0.014 bpw; int8 halves it. Trellis is impossible (48 ∤ 128). |
| B2 | Bundled-tensor counting rule: vision tower + MTP = 885M params ≈ 0.53 bpw if counted (redteam J1) | release | Ask the challenge author; the user's call (outward-facing). 0 compute. | protects | #53 established that the board counts tensor payload. Design §8's "not counted" rests on IQR-V3's 14.99 GB vs 17.57 GB file listing. Confirm before release. |

### M. Massive channel 3994

| ID | Idea (lenses) | Fit | Test → pass | Gain · P | Notes |
|---|---|---|---|---|---|
| M1 | Exact row 3994 in L0–L3 writers | L + field | §3 #15 | 0…−2% · 0.3 | Dense row (check 1); 95% of L0 out_proj's H-energy (check 2). |
| M2 | Massive-column bypass/extraction in L3/L11 attn_qg and L10 ssm_qkv (codes E1/E3/E9, crossarea #41/#47, lowbpw C1/C2, open B3, faraway A2/A5) | X (rank-1 f16 term) | Gated on D3; 4-tensor requant + splice, 2 card-h, ≤ 0.985 | −0.1…−0.6% · 0.2 | Proxy-only (0.90–0.95 on 2–3 tensors), the #106 precedent applies. The A8 side is moot at K2 (#46-03: ×1.0001). |
| M3 | Spike-excluded / Marquardt damping (crossarea #42/#43, lowbpw C6, open B4, codes E7) | requant | settle64 on colo, 1 h: ≥ 3% proxy on L3/L7/L11 readers, no tensor worse; then a 6-tensor splice pair | 0…−0.4% · 0.25 | Excess damping: 24–44% (σ 0.025) and 14% (σ 0.005) of the median curvature at L3 vs 0.9% at L62. #38's re-check used the plain formula. |
| M4 | Residual-stream permutation, 512-RHT, or a seed choice for the spike block (codes D1–D3/D6, crossarea #49/#50, lowbpw C7) | requant (all codes) | settle64 | 0…−0.5% · 0.15 | Any Hadamard block containing 3994 is dominated by it regardless of its companion channels. |
| M5 | L63 MLP spike-channel split (composable D4) | X | 2 card-h | −0.2…−0.6% · 0.25 | The top 1% of channels hold 35.5% of importance. The knapsack already favors L63 gate/up (#21). |

### Q. Quantizer settings that need a requant (decide before the freeze)

| ID | Idea (lenses) | Test → pass | Gain · P | Notes |
|---|---|---|---|---|
| Q1 | i8c at ≤K3.5 | §3 #11 | −0.5…−1.2% · 0.35 | A 27B pair cannot resolve 1%. Decide on the 0.8B + settle64, then a no-harm pair. #100: codebook proxies underpriced real KL by 3–4× (in the other direction). |
| Q2 | Centered LDLQ / drift-amplified H (cluster) | §3 #19 | −0.3…−1% · 0.15 | |
| Q3 | Temporal / structure-exact input metrics: conv-aware GDN qkv H_in, attention-filtered v_proj H, query-covariance metric for k_proj, decay-filtered GDN v/k, sink-weighted k/v (open D1, crossarea #33/#36–#40); also gen-masked and late-position H_in (hout H52/H53) | Forward collection on G (2–4 h; allowed, it is not Σdy²) with write-time certification (crossarea #61); family splice pair | −0.2…−0.8% · 0.15–0.2 | The ceiling is set by full diagonal H_out (−2% at K2.5, not adopted). GDN conv DC gain varies ~10× by channel (open), which keeps D1 alive. |
| Q4 | cd3 on attn_k and early attention/GDN inputs (codes G1, crossarea #58) | Config + a no-harm pair as in CD27 (1.3 card-h) | −0.03…−0.15% · 0.6 | GPU-measured attn_k 0.991/0.994 per tensor (#77 sweep). +1–3 min per level. |
| Q5 | LDLQ variants: feedback damping/clamping, K-best, MMSE regularization, fp64 LDL on spike tensors, hot-block rescoring, more wrap candidates (crossarea #56/#57/#59/#60, codes E4/E5, lowbpw D9) | settle64, colo 1–2 h, ≥ 1% median proxy | 0…−0.3% · 0.2 | MPQ's lesson (2610.32127, verified) is that LDLQ+CD captures most of it; tile CD is adopted. |
| Q6 | Write-time Hessian certification (crossarea #61) | — | hygiene | Mandatory for any new collection (LOG 11-06 corrupt hinX). |

### T. H_out at K≥4 and two-sided items

| ID | Idea (lenses) | Test → pass | Gain · P | Freeze |
|---|---|---|---|---|
| T1 | Block-sparse ldlq_2hess (cluster) | §3 #9 | saves ≈58 card-h · 0.8 | **yes** |
| T2 | Reopen H_out at ≤K3.5 after T1 | §3 #21 | −1…−2% · 0.4 | user decision |
| T3 | Targeted H_out on the lowest-R tensors | §3 #20 | −0.5…−1.5% · 0.3 | **yes** |
| T4 | CD × H_out pair | D2 | +0.1…0.2% at K≥4 · 0.7 | **yes** |
| T5 | Cap H_out at K ≤ 5.5 until a K6 pair (redteam C4) | 0 compute | saves ~36 card-h | **yes** (0.8B evidence reaches K5 only) |
| T6 | Correctness: fractional-K drift two-sided (H64), prepare_H_out with post-refit sv (H40), writers one-sided at every K (H07), C9 × H_out at 27B (redteam F2/H1), H_out × mul1 smoke (redteam C3) | D9 + one fixed-code C9 pair at K4 | correctness | **yes** |
| T7 | Tempered/damped output factor; diag+low-rank and shrunk gate/up H_out (hout H34–H38/H41; ShamAN-Q tempering, verified) | 0.8B 1.5 h, only if T2 reopens low K | ±1% · 0.2 | if T2 |
| T8 | Attribution of the 27B shrink: realized h-proxy, family/depth swaps, coherence probe (hout H09/H10/H11/H15/H47) | H10 ~1 card-h from existing bn4/base4 tensors | sets T3 scope | — |
| T9 | Output-side pivoted row order (hout H39) | — | — | Rejected (§7, #106). |

### K. Codes, trellis and format

| ID | Idea (lenses) | Fit | Test → pass | Gain · P | Notes |
|---|---|---|---|---|---|
| K1 | Per-head K from existing levels | X | §3 #12 | −0.5…−2% · 0.25 | Plugin segment support, 2–4 dev-days. |
| K2 | Per-tile salts best-of-2/4, sparse hot-tile salts, scale-select bit (codes B1–B8) | X + requant | settle64 4 CPU-h → ≥ 4% median proxy at K2.5 after side bits | −0.3…−1% · 0.3 | Viterbi ×1.8–3.4 (≈ +30–80 card-h of ladder). Proxy-only. Low score on cost. |
| K3 | Canonical fractional-mask shape check (codes C1) | X | settle64, minutes → another mask ≥ 1% better at ≥ 2 fractional K | 0…−0.3% · 0.15 | Cheap. Close it if it fails. |
| K4 | Per-K warp strength, overload census → tail-extended codebook, stratified int8 palette (codes A3/A7/A8/G2, M4) | X | only after Q1 | 0…−0.5% · 0.15 | |
| K5 | Reserve optional bias / side-car / segment fields in exl3i (codes L1, crossarea #79/#80, open G1) | X | plugin + importer + selftests, 2–3 dev-days | enables A9, M1, H1-bias, K1 · 0.5 | Optional tensors leave the trellis codes untouched, so they can follow the quantizer freeze. Settle them before the release format is published. |
| K6 | V4 format notes: intra-tensor row DPCM, token-tree head, VVC dependent quantization, tail-biting terminal fix (faraway G1/B1, crossarea #81/#83) | — | — | V4 | |

### W. Library vs mix

| ID | Idea (lenses) | Fit | Test → pass | Gain · P | Notes |
|---|---|---|---|---|---|
| W1 | Asym rollout support (measure E3, heal H3, open C1, crossarea #62/#64–#67, composable B1/B2, lowbpw F1/F2) | decision | §3 #10 | −1…−4% conditional on #105 | 0.8B per-mix pass 0.81–0.90 (#111); 27B size unknown. |
| W2 | Per-mix closed-loop continuous refit | M/F | §3 #25 | −0.5…−2% · 0.3 | Shipped mixes only. |
| W3 | Domain-randomized / noise-aware per-level fits (open C4, lowbpw B7) | L | 0.8B 2 h → mixed-K arm keeps ≥ ⅔ | supports library safety · 0.25 | Pair it with A1. |
| W4 | One Gauss–Newton step per mix (open F4) | M (training-like) | R: 2×H100, ~$30–60 per mix | −1…−4% · 0.3 | Defer with healing. |

### P. Pricing, measurement, search

| ID | Idea (lenses) | Cost | Value |
|---|---|---|---|
| P1 | Ladder-job logging (cluster) | 0 | enabler; pre-ladder |
| P2 | Two-sided level positions (cluster) | CPU | −0.1…−0.5% at 2.5 |
| P3 | Reprice after coherent fixes; stale-constant audit (open H1, measure A11/A12) | minutes | protects 0.5–2% |
| P4 | Composition calibration (cluster; 2610.09227 β 0.26–0.77, verified) | ~2 card-h | −0.5…−2% |
| P5 | KKT audit / lazy descent / gradient-screened local search (measure C2/C4/D2/G1) | search phase | −1…−3% |
| P6 | Σdy² price prior with calibrated exponent (measure B3/D1: ρ 0.82, slope 0.49, attention breaks it) | colo 2–4 h | prices unmeasured options |
| P7 | Seed localization (cluster) | D5 first | 0…−2% |
| P8 | Long-context and category re-scores of adopted pairs (redteam B1/B2/B5/A8/B6, hout H14/H74/H75, measure I1/H2) | ~1–2 card-h after the agent-search32 golden is fixed | protects. ~24% of scored characters lie beyond 8k (character proxy); all 27B decisions used the 8k cap. |
| P9 | Uniform-level anchors (measure B6), A8 prices at K2.5 (measure C13; #46-03 shows A8 free at K2) | minutes per level / 8 evals | low |
| P10 | Power calculator, sequential pairs, variance-components model (measure A4–A6) | CPU | saves 27B pairs |
| P11 | swj grid-edge check at K2.0–2.25 (redteam C6): grid 0.6–1.4, clamp 0.5–2.0 | CPU | 0…−1% at 2.0–2.25 |
| P12 | CVaR objective, engine-floor term at 7–8 bpw, numeric floor audit (crossarea #69/#70/#75/#76) | — | low; #99 covers the floor |
| P13 | i8 vs mul1 KL at K4–6.5 on the 0.8B (redteam F4) | S, 1 h | 0–4% at 5.0–7.0 accurate; not 2.5 |
| P14 | Per-setting score bands for rqah (redteam I4) | CPU | prevents donor-ladder rejections |

### Heal: training, deferred by the user (prep can run beside the ladder)

| ID | Item | Cost | Note |
|---|---|---|---|
| A1 | Cached top-k teacher equivalence | 1.5 h S | enabler for S27 |
| B1/B2 | Granularity / family decomposition | 4 h S | |
| G4 | Log-scale interpolation across K | 1 h S | ×3–4 fewer 27B heals |
| C1 | Warm start from swj / C9 / A1 | 2 h S | Order: coherent fits first, then healing. |
| H3 | Plain + heal vs per-mix asym | 3 h S | Feeds W1 now. |
| C3/C9, C6, C4 | K-dependent lr + fp32 master; starvation regularizer; diagonal GN steps | 1.5–3 h S each | |
| F1, F2, M1 | 27B last-8 pilot; depth replication; 9B bridge | 1–9 card-h | |
| A2/A4/G1 | S27 healing stack | 3–5 dev-days + 60–100 card-h | −5…−11% at 2.5 (heal's prior) |
| E3/E5, B3/B5, L1, G2, G6/G7, J4 | Windows, burn-in, GDN conv/a-b compensators, A8-aware, two-level library, healed knapsack, head-only heal | various | |

### Kernels and eval throughput (outside this pass's scope; card-hour effects only)

- **A2 card-0 clock check:** diagnostic D8.
- **L1/L2:** same item as T1.
- **Eval throughput:** K1 on-device KLD; K2 two evals per card; K6 / measure F2/F3 early stopping and halving (must be pre-registered).
- **Prefill speed:** B1 VOPD back on (−3%), C1 LLC-chunked two-pass, D1 multi-output norm-quant, D4 SwiGLU epilogue, B3 g256 fold (+0.03% KL).
- **Decode / other:** H1/H3 decode GEMV, I2 DFlash2. B4 per-token A8 gate_up is a knap4 option (+0.1% KL).
- **D2 tied-suh** changes codes (#42: siblings differ). It is a speed item; the user decides.

## 6. Decide before the freeze

- **Settings:**
  - P1 logging.
  - T1, or accept the ×5 cost of H_out levels.
  - T3 (per-key H_out switch).
  - T5 (cap H_out at K ≤ 5.5).
  - Q1 (i8c at ≤K3.5).
  - Q4 (cd3 on attn_k).
  - M3, Q2 and Q3 only if run; each needs a requant.
- **Correctness:**
  - D1 (H_out K4 verify).
  - D2 (CD × H_out pair).
  - D8 (card-0 clock).
  - D9 with T6 (ladder dry run and K≥4 checks).
- **Format** (can follow the quantizer freeze; must precede the release format): K5 (fields) and K1 (segments). K3 (mask shape) is cheap enough to close now.
- **Not before the freeze:**
  - A1–A9: fixed-code passes per level, run after the levels exist.
  - H1–H5: before the head donors, not the trunk ladder.
  - B1, W*, P2–P14.

## 7. Rejected ideas

| Idea | Reason |
|---|---|
| Super-weight sparse escapes (crossarea #45); faraway's "99% in one entry" | Measured: the largest entry is ≤ 2.8% of row 3994 (check 1). The row is dense; use whole-row M1. |
| Massive-column work on MLP readers (gate/up) before L30 | post_attention_layernorm gain at 3994 is 0.0039 / 0.0000 / 0.074 at L0–L3 / L7 / L11 (check 3); the model already removes the channel there. |
| Trellis-coded GDN a/b (lowbpw A8/A9) | 48 rows ∤ 128 (check 7). Use int8 (B1). |
| Gradient-, entropy- or leverage-weighted H_in; Sketch B blends; KL-Shampoo token weights (crossarea #51/#52, hout H50/H51) | #33-07 (+31…+44%), #91 and #101 (shrank to < 1% at 27B). H51 also needs a new backward collection. |
| Any new backward / Σdy² collection: embedding residual-gradient LDLQ (hout H25), leverage-weighted Σdy² (crossarea #82), late-layer collection (H61), Sketch B dose (H58/H59) | User ruling 11-05. H25 is further weakened by composable K1 (layer-0 diagonal spread 0.037 bit). |
| Low-rank side paths: rank-r drift correction, drift-aware head SVD (crossarea #77/#78) | open D10 measured that they don't pay. H3 is different: it is the head's own PCA, not an error correction. |
| Column-split K; gate/up K split; q/k/v or q/gate part split; per-part GDN K (codes M2/X, composable K2–K4, lowbpw E2) | Ceilings 0.005–0.043 bit. |
| Output/head permutation orders (hout H39, the codes X list) | #106 dropped 11-07 (−9% proxy, no KL); #103 failed. |
| Category banks with runtime routing (faraway D1) | A rules question and a second set of scales per category; not V3. |
| Predicted-noise Wiener filter (faraway C5) | Repeats the measured Wiener pre-shrink negative (review-02b) with no stated difference. |
| Subtractive-dither trellis (faraway C8) | P ≈ 0.05 by the lens's own estimate. |
| Soft BCJR + VAMP rounding (faraway E6) | Days of work for a trellis-marginal denoiser. MPQ's gain over LDLQ is mostly the CD part (15.01 → 12.76 vs 11.63, verified), and tile CD is adopted. |
| Position-0 / template-prefix state table (crossarea #84) | An evaluation-side trick with rules risk; not a quantizer setting. |
| Dead-vocab head rows (open E2) | #32: rare-token rows matter more than frequency predicts; risk to multilingual users. |
| Head mean-row shift t·μ by Fisher (hout H26) | The mean row (t=1) is adopted (#34). The paper shows ≈0 on Qwen3.5 with a GPTQ head (0.011 → 0.010, digest). |
| Seed-averaged scale fits (crossarea #85) | Per-level fits use the level's own codes; nothing to average. |
| H_out on writers combined with asym; mixed or full gate/up H_out at 27B (hout H37/H38/H41) | V4 / 0.8B research; #82: writers carry 3% of the H_out gain. |
| A4-specific palettes (codes K1/K2); A8 massive-channel bypass for the fast variant (open B3, codes E3) | #46-03 interim: A4 not free at K2 and costlier with depth; knap5 never picks it. A8-g128 is free at K2. |
| Hadamard sign seeds for R or proxy (hout H66; #41) | Seed spread 0.5%, not reproducible. |
| GDN recurrent-state precision (redteam's own kill) | f32 by default. |
| Self-generated healing text (heal E8) | #91. |
| Codes X1–X14, composable K1–K6, faraway F1–F15 / B7/B8 / E9–E11, lowbpw kills, kernels B5/B14/D11 | Already recorded as dead in those reports with evidence. I found nothing that revives them. |
