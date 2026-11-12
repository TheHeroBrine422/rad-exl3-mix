# Lens: measure — pre-freeze ideas (round 2, 2026-11-08)

**What this covers.** I limited this to ideas that change the per-level library (quantizer settings, codebooks, format, level menu) or the knapsack's inputs. Knapsack inputs here means the price tables, the sensitivity model and its fit data, the measurement designs that produce them, and anything the one-time ladder build must log. Search policy, healing/QAD, per-mix asym and kernels are left out.

I checked every idea against the INDEX/LOG, the round-1 review (`review-2026-11-07-ideas.md`), the 11-08 merge (Tier A/B/C, "captured/rejected") and the 11-02 `docs/ideas/…/measure.md`. Where an idea extends a listed one, I name it and say what is new.

**Two new findings from this session:**
1. **The in-quantizer scale refit is still one-sided at H_out levels.** `refit_scales` (exllamav3-frac `quantize.py:1730`, called at `:1985–1990`, `refit = true` in v3.toml) takes only `H_orig`. The input-side `su` refit solves `((Q Qᵀ)∘H) r = rowsum(Q∘(HW))` with no H_out row weight. LDLQ minimises the two-sided objective; the refit that follows then pulls `su` back toward the one-sided optimum. The fix is three lines and can be tested as a fixed-code pair. See **A1**.
2. **Diagnostic on colo: weighted-Viterbi ideas are dead after the 128-Hadamard.** On 17 real 27B tensors (calib H + bf16 weights), the gain ceiling from diagonal D-weighting inside a tile is ≤ 0.0011 bit/weight. Gains from in-tile shaping and tile-level reallocation are small too, except at L3 attn_qg. See §0, F1 and F2.

---

## 0. Diagnostic run for this report (colo, CPU, ~10 min)

**What I computed.** `H_eff` = 128-Hadamard · random signs · diag(rms_in(W)) · H · diag(rms_in(W)) · signs · 128-Hadamard, plus damping at the configured σ. I took a lower Cholesky (exllamav3 quantizes blocks last-to-first). For each 16-column tile I took `D_b = L_bb L_bbᵀ`, and computed:
- the in-tile gain ceilings: ½·log₂(AM/GM) of diag(`D_b`) for a weighted Viterbi, and of eig(`D_b`) for full in-tile shaping;
- the spread of tile traces inside a 128-group and across the whole tensor.

All values are high-rate ceilings in bits/weight, trace-weighted. Script: `/data/work/ideas-measure/danis.py`.

| tensor | diag-weighted Viterbi | full in-tile shaping | off-diag energy of D_b | tiles in 128-group | all tiles | tile tr p90/p10 |
|---|---|---|---|---|---|---|
| L3 attn q_proj | 0.00013 | **0.211** | 0.133 | 0.029 | 0.072 | 1.58 |
| L3 k_proj | 0.00020 | 0.042 | 0.078 | 0.010 | 0.028 | 1.86 |
| L43 / L63 q_proj | 0.00008 / 0.00011 | 0.026 / 0.018 | 0.06 / 0.05 | 0.006 / 0.004 | 0.017 / 0.013 | 1.76 / 1.66 |
| L2 / L10 ssm_qkv | 0.00016 / 0.00013 | 0.023 / 0.017 | 0.06 / 0.05 | 0.005 / 0.003 | 0.031 / 0.015 | 2.21 / 1.73 |
| L10 ssm_z | 0.00015 | 0.028 | 0.057 | 0.006 | 0.018 | 1.68 |
| L10 / L62 gate | 0.00008 / 0.00010 | 0.0026 / 0.0056 | 0.008 / 0.017 | 0.0002 / 0.0005 | 0.008 / 0.008 | 1.54 / 1.50 |
| L10 / L62 up | 0.00007 / 0.00009 | 0.0017 / 0.0029 | 0.005 / 0.008 | 0.0001 / 0.0002 | 0.008 / 0.007 | 1.53 / 1.46 |
| L1 / L10 GDN out_proj | 0.0011 / 0.00037 | 0.044 / 0.022 | 0.09 / 0.06 | 0.013 / 0.008 | 0.070 / 0.038 | 2.92 / 2.36 |
| L39 o_proj | 0.00035 | 0.0092 | 0.024 | 0.002 | 0.037 | 2.26 |
| L3 / L50 down | 0.00001 / 0.00002 | 0.011 / 0.018 | 0.03 / 0.05 | 0.002 / 0.004 | 0.019 / 0.012 | 1.85 / 1.60 |
| lm_head | 0.00032 | 0.021 | 0.050 | 0.004 | 0.019 | 1.67 |

**Reading.**
- Diagonal D-weighting in the branch metric (CurveTQ 2610.09212, XOR-Trellis 2610.00432) has nothing left to gain. This matches CurveTQ's own post-RHT Γ_b (0.038 → 0.00013).
- The only material in-tile structure is early attn_qg (0.21 bit), which is where tile CD pays. gate/up are flat (≤ 0.006).
- Tile-level bit reallocation is worth ≤ 0.07 bit, consistent with #96 and the closed format ideas.

**Caveats.** One sign draw, natural block order (not #117's order), input side only (no H_out), su approximated as per-input RMS of W.

---

## A. Freeze-gating measurements and setting consistency

**A1 · H_out-weighted input-scale refit (fix the refit metric).**
At every H_out level, `refit_scales` refits `su` (input rows) in the one-sided metric. Weighting the refit by the bn diagonal *h* (original output basis) makes it minimise the same Kronecker-diagonal objective LDLQ used. Implementation: call the existing routine on `(W·diag√h, Q·diag√h)` and unscale `Q` afterwards. The per-column `sv` refit is unchanged, because it is separable and invariant to per-column scaling.
- *Source:* new (code finding). Related to #85's B1 (H_out-weighted scale refits as an extra pass on the 0.8B, inconclusive because the "full" stack ran in the wrong order). Here it is not an extra method: it makes the default polish consistent with LDLQ's objective at all 55 levels.
- *Why:* #118 shows one-sided and two-sided optima differ by ~7.6% of the objective at H_out levels (×1.076 one-sided while KL improves). The unweighted refit re-inflates error in high-h columns. The 0.8B B1 screen on bdiag gave K3 0.993 and K2 0.975.
- *Fit:* per-level quantizer setting (the refit runs inside every job). Can be applied post hoc on fixed codes.
- *Test:* (1) colo CPU, fork CPU path: f2 (w81 `f2calc`) after refit with and without weighting on 6 tensors at K2.5/K4; counts if f2 drops ≥ 0.3% on ≥ 4/6. ~1 CPU-h. (2) 27B fixed-code pair: re-refit stored bn/bns1 K2.5 levels (codes unchanged, like C9/swj). Seed-luck-free; rule g ≤ 0.995. ~2 card-h.
- *Gain / risk:* −0.1…−0.8% at 2.5 (0.8B screen, ~⅓ 27B shrink). Risk low: it is a scale polish and cannot change codes.
- *Conflicts:* #85-02 (B1 inconclusive). This differs in scope (input scales inside the job) and order (before swj/C9).

**A2 · Objective-consistency audit of every quantizer sub-step.**
List each sub-step's objective at H_out levels and record per-step f2 on two layers:
- `g_scale_gss`: unweighted tile MSE in the regularized domain;
- LDLQ: two-sided;
- refit: one-sided (A1);
- tile CD: two-sided (w76);
- swj: check that its grid is per-channel separable, in which case weights are irrelevant;
- C9: α chosen by KL;
- donor-scorer bands: one-sided.

Anything still one-sided either gets weighted or gets a written reason why weighting is irrelevant.
- *Source:* new; H-Scale 2608.28113 and Schur replay 2609.36654 (scale choice scored in the right metric).
- *Why:* inconsistent sub-objectives silently undo part of H_out. Merge B1 found the scorer one-sided.
- *Fit:* per-level settings.
- *Test:* code read plus per-step f2 log on L10/L39 at K2.5 through the rxq CPU stand-in. ~2 h, colo.
- *Gain / risk:* 0…−0.3% beyond A1. No risk.
- *Conflicts:* merge B1 (scorer). This adds the quantizer itself.

**A3 · Leave-one-component-out audit of the adopted stack at K2.5.**
Each component was confirmed on a different base:
- H_out on ladder2 without #117;
- #117 without H_out or tile CD (its GPU gate was tile-CD-off, no H_out);
- tile CD + H_out only at K4;
- C9 on swj K2.5.

Score the full stack against stack-minus-X for X ∈ {#117, tile CD, swj, C9}, each on the same rotation seed. Use partial requants: only readers for #117, only attn_qg for tile CD. swj and C9 are fixed-code toggles.
- *Source:* new design. Extends freeze-review item 2 (one full-stack level vs bn+swj) from one pair to an ablation.
- *Why:* untested interactions, for example #117's block order with two-sided feedback (the proxy predicts −1.5%, the KL measured only −0.3% without H_out).
- *Fit:* gates per-level settings.
- *Test:* card 0. Two partial requants (~30 min each), two fixed-code toggles, five evals on a fresh two-part slice. ~5 card-h. A component is dropped if its LOO ratio is > 1.003 with z > 2 on both parts.
- *Gain / risk:* expected 0. ~15% chance it finds a −0.3…−1% negative interaction before 55 levels bake it in.
- *Conflicts:* #117, #77, #104, #112 were each tested in isolation. This tests them in the stack.

**A4 · Low-noise design for the freeze-gate pair.**
Build the gate level with the same rotation seed as the bn+swj reference arm: same-seed pairs share ~¾ of seed luck (#116, implied ρ≈0.76). Report SE as seed term plus the joint-bootstrap document term (0.0104 for a 3-pair mean), with t-quantiles at the real df. Commit the rule file before launch, as the freeze review asks.
- *Source:* #116, freeze review. New: it turns those notes into the gate protocol.
- *Why:* the predicted gate effect (~0.90 vs base) is large, but the bn+swj comparison (≈0.97 expected) is at the noise line.
- *Fit:* gate protocol.
- *Test:* none (design).
- *Gain / risk:* indirect: halves the false-alarm rate on a decision that touches 55 levels.
- *Conflicts:* none.

**A5 · Proxy → KL shrink table as a pre-card gate.**
Tabulate measured proxy changes against KL changes for every recorded pair:
- #117: proxy 0.985 → KL 0.997, i.e. ~1/5;
- H_out K2.5: one-sided ×1.076, two-sided −x%, KL 0.98;
- #106-01: o_proj −9% → KL ≈ 1.00;
- swj: proxy ×1.12 while KL 0.935;
- tile CD: proxy 0.9989 → KL 1.000;
- C9.

Fit a shrink factor per change kind (redistributive vs systematic, reader vs writer). From now on, a pre-freeze idea gets a card slot only if proxy gain × shrink ≥ 2× the pair noise.
- *Source:* round-1 idea 10 (predictions); 2610.11226 DRQ (lower reconstruction loss raised PPL in 73.8% of AWQ refinements). New: an operational gate built from our own pairs.
- *Fit:* gate for per-level settings.
- *Test:* CPU, stored proxy.jsonl files and pair ratios. 2 h.
- *Gain / risk:* saves card-hours. Prevents adopting proxy-only wins.
- *Conflicts:* none.

**A6 · Settings-provenance register weighted by stakes.**
For each frozen setting, record:
- KL-verified at 27B (H_out, swj, C9, #117, tile CD), KL-verified at 0.8B only, or proxy-only (#39 gain, #38/#114 damping, drift interp #107, out_scales #48, refit metric);
- share of params affected;
- the proxy delta of the alternative.

Spend remaining pre-freeze card time on the top entries.
- *Source:* freeze review items 6–8. New: a weighted list.
- *Fit:* per-level settings.
- *Test:* 1 h of writing.
- *Gain / risk:* indirect.
- *Conflicts:* merge B3 (#39), which would sit high or low on this list.

**A7 · One low-K and one high-K stack pair before the freeze.**
H_out, #117, swj and C9 were measured at K2.5 and K4 only. The 1.0–1.75 bpw sizes are now release sizes, and the 6.0–7.0 sizes use K5.5–7. Run one K1.5 stack-vs-plain pair and one K6 pair, both fixed-seed.
- *Source:* LOG 11-08 (sub-1 and low-bpw sizes adopted); #82 (0.8B H_out holds at K4/K5).
- *Why:* if H_out or C9 hurts at K≤1.5 (LDLQ feedback overload) or K6 (A8 floor), the ladder would bake that in.
- *Fit:* per-level settings, level-range scoping (`apply_k_min/max`).
- *Test:* card 0. Two levels' worth of partial requants (attn+GDN+MLP of 16 layers). ~5 card-h.
- *Gain / risk:* 0 at 2.5. Protects 1.0–1.75 and 6.0–7.0 (±1–3% there).
- *Conflicts:* #104 (swj K4 −1.1%), #33-28 (H_out K2.5/K4). It extends their K range.

---

## B. Instrument the ladder (log once; cheap now, expensive later)

These change job outputs and therefore the rxq code stamp. Do them in the stamp-regeneration window (freeze-review item 7).

**B1 · Per-output-channel error vector for every tensor × level.**
Store `e_n = (EᵀHE)_nn` (one-sided error per output channel; `E` is decoded − original) as fp32. Cost is the size of a refit matmul, ~16 MB per level, ~0.9 GB for the whole ladder. Then any diagonal output weighting prices a level as Σ_n h_n e_n at zero cost. That covers bn, plain dy, Sketch B, agentic-weighted Σdy², and the head's softmax-Fisher. It also gives per-head and per-section ceilings, donor two-sided bands, and C2/C3/C14.
- *Source:* new; enables 11-02 O4 (two-sided proxy as kfit2 input) and #118 2(c), which was "not done: needs every container row-weighted".
- *Why:* repricing under a new estimator otherwise needs a decode pass of every level.
- *Fit:* per-level (job output). Knapsack input.
- *Test:* CPU stand-in check that Σ_n e_n equals the logged proxy to 1e-6. 1 h dev.
- *Gain / risk:* indirect (enables −0.3…−1% of pricing gains). None.
- *Conflicts:* none.

**B2 · Coarse 2-D error map (128-row group × 128-column block) per tensor × level.**
Store the cellwise sums of `E ∘ (HE)` (~5k floats for down_proj). Enables:
- column/row-slab ceilings;
- checks on block-order effects (#117) per level;
- detecting localized trellis overload;
- a per-block KL check against the Σdy² diagonal.
- *Source:* new; MosaicQuant 2606.15652 (error is block-hotspotted).
- *Fit:* job output.
- *Test:* CPU stand-in. 1 h.
- *Gain / risk:* indirect. None.
- *Conflicts:* #96 (row-slab ceiling, closed). This is data for later, not a format change.

**B3 · Component trajectory log.**
For each tensor × level, log one-sided and two-sided error at four points: after the trellis before refit, after refit, after tile CD, after swj/C9 (the post pass). Add the RTN-equivalent error (trellis without feedback, sample tiles) as an "LDLQ value" reference.
- *Source:* partial today (`proxy`, `scale refit: e_before → e_after`).
- *Why:* gives per-family, per-K value of each component; feeds A5, C6 and K1.
- *Fit:* job output.
- *Test:* ~2 h dev.
- *Gain / risk:* indirect.
- *Conflicts:* none.

**B4 · Trellis overload counters.**
Per tensor × level, log:
- the share of weights decoded at the codebook's outer values;
- the share of tiles whose Viterbi cost exceeds 3× the tensor median;
- the g_scale chosen against the Gaussian-analytic scale (F9).

This flags mis-set scales (especially at sub-1 and K7+), where proxies and KL disagree most.
- *Source:* merge B16 (LDLQ overload diagnostic, unscheduled); 2609.37416 (analytic scale after HIP).
- *Fit:* job output.
- *Test:* 1 h dev.
- *Gain / risk:* indirect. Guards the 1.0–1.25 and 7.0 sizes.
- *Conflicts:* none.

**B5 · Per-tensor proxy-reliability map, computed once.**
On colo, compute per tensor: Cholesky-pivot AM/GM of `H_eff`, a κ estimate, and §0's in-tile ceilings (all 496 tensors, ~2 CPU-h). Store it with the datapack. Use it to place real calibration swaps where proxies are least trustworthy.
- *Source:* CASA 2609.25916 (a scalar proxy can be off by √(κ(A)κ(B))); transforms survey 2608.25188 (pivot AM/GM); §0.
- *Fit:* knapsack input (measurement design).
- *Test:* run §0's script over all tensors. 2 CPU-h, colo.
- *Gain / risk:* indirect.
- *Conflicts:* none.

**B6 · Store per-tensor swj and C9 objective deltas.**
swj is priced as one KL factor per K (`KN_SWJ_PTS`). The per-layer swj objective reduction from the post pass would let the knapsack price swj by tensor and depth. reprice.md notes the model's swj gain (−8%) overshoots the measured −6.5% at 2.0–2.5.
- *Source:* reprice.md, #104.
- *Fit:* knapsack input.
- *Test:* add fields to the swj/C9 stamp. 1 h.
- *Gain / risk:* −0.1…−0.3% (better gate/up allocation).
- *Conflicts:* none.

**B7 · Second-seed duplicates for 1–2% of jobs.**
Re-quantize a random 1–2% of (tensor, level) cells with a second rotation seed (the donor duplicate rule already exists) and store both. This gives the realization noise of per-tensor error and f2 by family × K. That sets the noise model for robust knapsack prices (plan B11) and calibrates donor bands.
- *Source:* 11-02 S8 (realization noise). New: piggy-backs on donor duplicates at no card cost to us.
- *Fit:* knapsack input.
- *Test:* none beyond configuring the job sampler.
- *Gain / risk:* indirect (feeds B11 robust solve).
- *Conflicts:* #41 (best-of-N seeds null). This measures noise; it does not select seeds.

**B8 · Same rotation seed per tensor across levels, plus a monotonicity audit.**
Confirm every level of a tensor uses the same sign draw, so level-to-level price differences are paired. Flag tensors whose error is not monotone in K, or whose fractional levels sit > 2% off the integer-neighbour curve: bugs or overload.
- *Source:* #36 (fractional levels on the curve within 0.99–1.002).
- *Fit:* per-level setting (seed policy) and job check.
- *Test:* check ladder2/v3levels proxy rows for monotonicity. 30 min CPU.
- *Gain / risk:* indirect.
- *Conflicts:* none.

---

## C. Pricing V3 levels for the knapsack

**C1 · Price parity between H_out/#117/swj levels and plain options.**
knap5 compares V3 levels with plain V2 and turboderp options through one-sided proxies. H_out levels look ×1.076 worse one-sided while being better in KL. #117 readers look 1.5% better while the KL gain is ~0.3%. Before the knapsack reads `KN_LEVELS`, convert each setting's proxy shift to its measured KL equivalent: per family and K from w81's f2 (−2.6% K0.125 … −14% K3/K4) × the A5 shrink factor. Otherwise turboderp options at ≥ 5 bpw look ~7% better than they are.
- *Source:* #118; merge derived idea 5 (H_out priced by verify/agentic factor). New: settings-consistent conversion for *all* settings and for the cross-family comparison with turboderp.
- *Fit:* knapsack input.
- *Test:* re-solve knap5 at 2.0–7.0 with converted prices; count changed picks and predicted ΔKL. CPU 1 h.
- *Gain / risk:* 0…−0.3% at 2.5; −0.2…−1% at 5–7 bpw (turboderp share 10–18%, w63). Low risk.
- *Conflicts:* #63 (ours vs turboderp at high K). This removes a known bias in that comparison.

**C2 · Two-sided error as kfit2's error input.**
Refit kfit2 with `e_w2 = Σ_n h_n e_n` for every option:
- V2 containers: decode on colo (~30 CPU-h is ~1 h wall at 36 cores) × bn diagonal;
- V3 levels: from B1.

Compare CV log-RMSE and pairwise ranking against the one-sided fit.
- *Source:* 11-02 O4; #118 2(c) (deferred as over the 2 CPU-h bar on iceman; colo removes that limit). 2610.40345 H_Z-PTQ (output-feature factor beats YAQA's).
- *Why:* kfit2 lumps output sensitivity into group × knot curves. Two-sided errors separate tensors inside a knot.
- *Fit:* knapsack input.
- *Test:* copy V2 containers to colo, decode, refit. ~1 day wall. Counts if CV log-RMSE drops ≥ 5% or ranking rises ≥ 1%.
- *Gain / risk:* −0.3…−1%. Risk: the Kronecker-diagonal approximation ignores x/dy correlation.
- *Conflicts:* #118 2(c) not done; #33-29 (estimators within 1%; irrelevant to pricing).

**C3 · Absolute Fisher price per tensor.**
`ΔKL_w ≈ ½ Σ_n h_n^abs e_n / N` (raw Σdy² totals, not mean-normalised). This is a parameter-free single-tensor KL prediction. Validate it against V2's ~1,250 swap measurements and the stored 27B arms (H_out, swj, C9, #117). If it is within ±30% (log), use it as an informative prior for every option that has no swap data: sub-1, fractional levels, new families.
- *Source:* 11-02 O3; 2609.28270 (price = ½Tr(HΣ), log-Pearson 0.947/0.925 to measured KL). New: the 27B diagonal now exists.
- *Fit:* knapsack input.
- *Test:* CPU, Spearman and log-RMSE against V2 group swaps. 4 h colo.
- *Gain / risk:* −0.3…−1% (overlaps C2/C4). Risk: miscalibrated in absolute scale; fit one global constant.
- *Conflicts:* #15 (noise injection: different mechanism, no forward needed).

**C4 · Split kfit2's lumped groups with Fisher ratios.**
kfit2 shares one depth curve for attn_qg/k/v (`attn_in`) and for ssm_qkv/z (`ssm_in`). #118 measured family offsets of attn_k ×0.74, attn_v ×0.83, attn_qg ×0.93 and ssm_qkv ×0.95 against knap5 prices. Use per-tensor Fisher price ratios (C3) as fixed within-group offsets, or as priors with partial pooling.
- *Source:* #118; 11-02 S16.
- *Fit:* knapsack input.
- *Test:* kfit2 CV with offsets. CPU 2 h.
- *Gain / risk:* −0.2…−0.6% (attention + GDN-in are ~25% of params). Low risk.
- *Conflicts:* #118 recommended per-family offsets; this supplies them from Fisher, not from levels alone.

**C5 · Untie gate and up in the knapsack (with the swj caveat).**
knap5 prices fused gate_up by geo(gate, up). Their output sensitivities differ: gate goes through SiLU′·u, up through SiLU(g). SACQ 2610.41408 gives cheap row weights, ‖W_d[:,r]‖²·E[(SiLU′(g_r)u_r)²] and ‖W_d[:,r]‖²·E[SiLU(g_r)²], and the Σdy² diagonals give them directly. Untying lets K_gate ≠ K_up. **Caveat:** swj is fitted per level with gate and up at the *same* K, so an untied pick needs swj per (K_g, K_u) pair. That is the #111 problem; measure it first (E6).
- *Source:* #118 (gate/up factor 1.075); 2610.41408.
- *Fit:* knapsack input (plus swj scope).
- *Test:* predicted ΔKL of untied picks with tied-swj penalty. CPU 2 h. Then a 2-eval fixed-code check of swj with mismatched K.
- *Gain / risk:* −0.2…−0.5%. Risk: swj coupling eats the gain.
- *Conflicts:* #111 (asym fails with siblings at another K): same coupling mechanism.

**C6 · Per-family swj and C9 KL factors from fixed-code toggles.**
swj and C9 change scales only on frozen codes, so toggling them per family (gate/up by depth band; attn_o vs gdn_out vs down) gives seed-luck-free pairs. The C9 pair spread was 0.0019. This yields per-family and per-band factors in place of one K-curve.
- *Source:* #112 (C9 z −4.2 at 0.75%), #104.
- *Fit:* knapsack input.
- *Test:* card 0, 6–8 fixed-code evals on pass-1 K2.5. ~3 card-h.
- *Gain / risk:* −0.1…−0.3%.
- *Conflicts:* none.

**C7 · Pass-1 calibration design, written before the freeze.**
Pre-register the group swaps that re-anchor kfit2 on pass-1 levels. Use a resolution-IV fractional factorial (Plackett–Burman) over 12 group × band factors at two levels around K2.5 in 16 evals, plus 4 centre points. A common anchor mix scored on every slice (E3) goes with it.
- *Source:* 11-02 S2 (fractional factorial); TODO "re-anchor on the built ladder". New: the design is fixed now so pass 1 does not wait on it.
- *Fit:* knapsack input.
- *Test:* the design is CPU. Executed after pass 1: ~20 evals, ~7 card-h.
- *Gain / risk:* −0.5…−1.5% through a better pass-2 solve (kfit2 was ~25% off at K2.5, I000099).
- *Conflicts:* #14 (single-tensor swaps too noisy). Group factors avoid that.

**C8 · Cost exponent and tempering per family.**
kfit2 fits one global q (~1.004) in `cost = s·e^q`. NatScale 2610.20353 shows the error-power/bit-decay term matters a lot: removing it raised KLD 0.194 → 0.292 at 2.5 bpw. The K1→K2 KL slope is 2.3–2.5 bits/bit against the proxy's 2.11 (#44). Fit q per family (and per K-range for sub-2) with shrinkage.
- *Source:* 2610.20353; #44; 11-02 S5.
- *Fit:* knapsack input.
- *Test:* kfit2 CV on V2 data with per-family q. CPU 1 h.
- *Gain / risk:* −0.1…−0.5% (more at ≤ 2.0).
- *Conflicts:* #44 (K1 own factor, partly done).

**C9 · Fold the stored 27B arms into kfit2 now.**
~20 whole-trunk 27B arms are scored with per-tensor proxy rows in ourq's proxy.jsonl: base/s1/s2, bn/bns1/bns2, base4/bn4, swj, C9, #117. Add them as measurements with their options. This anchors V3-type levels before pass 1 and tests kfit2's prediction of H_out and swj effects.
- *Source:* new (cheap use of existing data).
- *Fit:* knapsack input.
- *Test:* CPU, kfit2 refit plus CV. ½ day.
- *Gain / risk:* indirect; flags model failures early.
- *Conflicts:* none.

**C10 · Within-knot depth modulation from Fisher traces.**
kfit2's depth curve has knots every 8 layers. Per-layer tr(H_out)·tr(H_in) (KronQ 2607.07964, used as an allocation score in ThinkBit 2610.34886) varies inside a band. Use log of that ratio as a fixed offset or prior inside each knot.
- *Source:* 2607.07964, 2610.34886; 11-02 O7.
- *Fit:* knapsack input.
- *Test:* CV. CPU 1 h.
- *Gain / risk:* −0.1…−0.4%.
- *Conflicts:* #19 (per-layer edge offsets, no gain). This uses a measured prior, not free parameters.

**C11 · H_out factor resolved by family and slice.**
Price H_out per family from #82's attribution (gdn_in 39%, mlp_in 44%, attn_in 13%, writers 3%), scaled to the 27B total, using the verify value (K4: 0.965, not search 0.936).
- *Source:* merge derived idea 5, B20 (captured for the total). New: the family split.
- *Fit:* knapsack input.
- *Test:* CPU.
- *Gain / risk:* −0.1…−0.3%.
- *Conflicts:* B20 captured; this refines it.

**C12 · Block-output sensitivity feature via CPU probe propagation.**
On Qwen3.8-27B, allocation by a block-output signal beat allocation by the per-tensor layer objective: block cosine gave PPL 6.041 and cosine×(1−CKA) 5.991, against 6.087 for the oracle real-activation layer objective at IQ3_M bytes (2609.33923). Compute per tensor the block-output response to its actual quantization error. Do it on colo with layer-streamed CPU forwards on ~4k fit-pool tokens, with Gaussian random probes calibrated as in that paper. Use the result as an extra kfit2 regressor.
- *Source:* 2609.33923.
- *Fit:* knapsack input.
- *Test:* colo, ~2–3 h CPU per level; Spearman against measured group swaps.
- *Gain / risk:* −0.2…−0.8%. Risk: CPU forward plumbing, block signal not better than kfit2's fitted curves.
- *Conflicts:* none.

**C13 · Error-amplitude probes (KL vs α along the real error direction).**
For one group (e.g. down_proj L32–39), splice dense bf16 tensors `W + α(Ŵ − W)` for α ∈ {0.5, 1, 1.5} into a K2.5 mix (`rad-convert --reuse --changed`, as w91 did). Three evals give the in-context slope and curvature: kfit2's s_w and q at the operating point, free of seed noise.
- *Source:* new. TWIG 2610.31931 (second-difference sensitivities); 2609.09867 (midpoint prices: sign agreement 97/98 vs 85/98).
- *Why:* separates error magnitude (proxy) from in-context response.
- *Fit:* knapsack input.
- *Test:* card 0, 8 groups × 3 evals. ~8 card-h.
- *Gain / risk:* −0.3…−1% through better s_w/q.
- *Conflicts:* #15 (noise injection needed a bf16 forward that didn't fit). Here only one group is dense and the direction is the real error.

**C14 · Row-shuffle probe: is two-sided pricing right?**
Permute a group's error rows across output channels within 128-groups. One-sided error is unchanged; the two-sided objective changes by a known amount. If ΔKL tracks the two-sided prediction, two-sided pricing (C2) is validated directly at 27B.
- *Source:* new.
- *Fit:* knapsack input (validation).
- *Test:* card 0, 4 dense-splice evals. ~1.5 card-h.
- *Gain / risk:* diagnostic; decides C2/C3.
- *Conflicts:* none.

**C15 · Antithetic ±E probes.**
Score `W + E` and `W − E` for a group inside a quantized mix. One- and two-sided proxies are identical; the KL difference measures the in-context first-order term. #120-03 found 70% (K2) / 32% (K3) of plain-level KL is seed-coherent on the 0.8B, so this term is plausible. A large asymmetry means isolated (bf16-anchored) prices are biased.
- *Source:* TWIG 2610.31931; #120-03.
- *Fit:* knapsack input.
- *Test:* 3 evals per group, 4 groups. ~4 card-h.
- *Gain / risk:* diagnostic.
- *Conflicts:* none.

**C16 · GDN head-lifetime prior for GDN tensor sensitivity.**
STEPQuant measured ρ≈0.80 between gate half-life and accumulated state error across 2,304 Qwen3.8-27B heads (2609.38169). Compute per-layer half-life statistics from A_log, dt_bias and gate statistics on fit tokens (colo CPU). Test them as a kfit2 regressor for ssm_qkv, ssm_z and out_proj within depth knots.
- *Source:* 2609.38169; 2609.04098 (same model, per-projection GDN ordering).
- *Fit:* knapsack input.
- *Test:* CPU 2 h.
- *Gain / risk:* −0.1…−0.3%.
- *Conflicts:* none.

---

## D. Interactions and non-additivity

**D1 · Order-2 energy certificate on pass-1 levels.**
Score ~24 random mixes with each of ~48 group × band factors at K2.5 or K3 (p = 0.5). Fit an additive model with a free intercept. The residual minus measurement noise is the order-≥2 energy (Walsh certificate, 2607.12266 Thm 1). This tells whether κ compounding is enough or whether pair terms are needed at V3.
- *Source:* 2607.12266 (weight-only error non-saturating; isolate-and-sum error 0.10 vs 2.80 for activations); 2609.09867 (composition error 26–42× pricing error on small models).
- *Fit:* knapsack model form.
- *Test:* card 0, 24 evals. ~8 card-h (or 0.8B first, N2).
- *Gain / risk:* decides the model; 0…−1%.
- *Conflicts:* #11 (symmetric compounding, negative) — this tests the replacement.

**D2 · Within-layer pair factorial.**
Pair candidates: gate/up × down, qkv × o, GDN in × out (2609.02451 reports V–up pairs superadditive). Run them as a fractional factorial across 4 depth bands, ~16 evals, and keep pair terms only where |interaction| > 2σ.
- *Source:* 2609.02451; 11-02 S3.
- *Fit:* knapsack input.
- *Test:* card 0 ~6 card-h, or 0.8B first.
- *Gain / risk:* −0.2…−0.8% if pairs matter.
- *Conflicts:* none.

**D3 · CPU intra-layer additivity check (quadrature test).**
On the same model, GDN per-projection layer-output errors add in quadrature (19.2% measured vs 19.4% predicted, 2609.04098). Repeat for attention and MLP layers on colo: one-layer forward on captured inputs, inject each tensor's (stand-in) quantization error singly and jointly. Cheap evidence on kfit2's "additive within a layer".
- *Source:* 2609.04098.
- *Fit:* knapsack model form.
- *Test:* colo, ~2 h CPU.
- *Gain / risk:* diagnostic.
- *Conflicts:* none.

**D4 · Head × trunk and embedding × trunk additivity at 1.25, 2.5 and 4.0.**
The head share is ~5% on the calib part at 4.0. At low bpw the trunk error may dilute or amplify head effects. Run a 2×2 per size: head u/s, trunk mix.
- *Source:* I002026 (merge tail); #34-02.
- *Fit:* knapsack input (head prices).
- *Test:* card 0, 12 evals. ~4 card-h.
- *Gain / risk:* −0.1…−0.5% at low sizes.
- *Conflicts:* #34-02 measured at 4.0 only.

**D5 · Compressive vs superadditive link by size band.**
Regress measured mix KL on summed single-swap predictions per band. Embedders showed compressive β 0.26–0.77 (2610.09227); kfit2's κ is superadditive. At 1.0–1.25 bpw (KL ≳ 0.3) saturation may flip the sign (KFIT_SAT fitted C≈1.8).
- *Source:* 2610.09227; BACKLOG saturation check.
- *Fit:* knapsack model form.
- *Test:* uses K2/K3 data. CPU.
- *Gain / risk:* −0…−2% at ≤ 1.5 bpw.
- *Conflicts:* the 11-03 saturation check (null, but only 13 mixes < 1.6 bpw).

**D6 · Non-additivity lives in the coherent error component (0.8B test).**
With independent per-tensor sign draws, incoherent errors from two tensors have zero expected cross-term. Coherent (seed-independent, gain-shrink-like) parts do not cancel. Test on the 0.8B with #120-03's 6-seed levels: pair interactions for same-seed dequant vs seed-averaged dequant (mostly the coherent part). If interactions concentrate in coherent parts, add pair terms only for high-coherence families. It also predicts that CASA-type inter-layer cross-terms are ~0 for us.
- *Source:* new; #120-03; CASA 2609.25916 (cross-term bound).
- *Fit:* knapsack model form.
- *Test:* 4070, ~20 evals × 23 s plus splicing. ~2 h.
- *Gain / risk:* diagnostic; narrows D1/D2.
- *Conflicts:* none.

**D7 · Counterfactual band restore at 2.5.**
On a 2.5 mix, restore one 8-layer band to bf16 at a time (8 evals; bf16 band adds ~3.4 GB, which fits). The measured KL attribution by band is compared with kfit2's predicted attribution: a direct operating-point test of the model.
- *Source:* w91's restore oracle (L0–L3 writers).
- *Fit:* knapsack input (validation).
- *Test:* card 0, 8 evals. ~3 card-h.
- *Gain / risk:* diagnostic.
- *Conflicts:* none.

**D8 · Adjacent-pair chain DP in knap5 (conditional on D1/D2).**
If pairs matter, extend the per-layer DP with adjacent-pair costs `K(b′,b) = v(b′,b) − v(ref) − C(b′) − C(b)` (exact chain DP). TWLA's interaction-aware allocation cut PPL 11.92 → 8.31 on activation bits (2606.13054). LoopQuant: interaction-aware 12.47 vs local 13.05 (2610.16368).
- *Fit:* knapsack solver input format.
- *Test:* CPU once D1/D2 data exist.
- *Gain / risk:* −0.2…−0.8% if pairs exist; otherwise 0.
- *Conflicts:* none.

**D9 · Refit compounding κ on V3 levels.**
κ was fitted on V2 one-sided errors. H_out removes error from high-Σdy² directions, so the remaining error may propagate less. Refit κ with C9 plus pass-1 data and compare.
- *Source:* kfit2 docstring.
- *Fit:* knapsack input.
- *Test:* CPU.
- *Gain / risk:* −0…−0.5%.
- *Conflicts:* none.

---

## E. Context and anchor dependence of sensitivities

**E1 · Re-anchor group swaps at the operating point.**
Measure group sensitivities on a pass-1 K2.5 uniform base (and K4) instead of V2 4.00. MixQuant 2607.23047 reports context-aware sensitivity plus a tail penalty: PPL 10.70 vs 12.43 at 1.25 GB. GEMQ 2605.23078: importance estimated from FP / 4-bit / 2-bit / 2-bit+FT model gave 9.66 / 9.39 / 9.33 / 9.01.
- *Source:* 11-02 S1. New: on V3 levels, joined with C7's design.
- *Fit:* knapsack input.
- *Test:* in C7.
- *Gain / risk:* −0.3…−1%. Risk: MixFrag 2607.28589 saw re-estimation hurt (31.91 → 29.73), so verify on held-out data.
- *Conflicts:* none.

**E2 · Size-band transfer test.**
Does a model fitted on 2.5-band data predict 1.25 and 4.0 mixes? GAMMA 2605.18475 found allocation cosine 0.976 across budgets. Run leave-band-out CV once pass-1 data exist.
- *Source:* 11-02 S12/S13.
- *Fit:* knapsack input.
- *Test:* CPU.
- *Gain / risk:* diagnostic.
- *Conflicts:* none.

**E3 · Slice-anchor arms in every rotating two-part slice.**
Two-part slices redraw xcorp each round, and kfit2 fits per-slice offsets. Score 2 fixed anchor mixes on every new slice so offsets are identified. Without anchors, slice effects confound sensitivities: a 12k-token slice shifted absolute KLD by 24% in 2610.20826.
- *Source:* new for rotating slices.
- *Fit:* knapsack input (fit data design).
- *Test:* +2 evals per round.
- *Gain / risk:* indirect.
- *Conflicts:* none.

**E4 · Category-reweighted fit targets.**
Pooled slices are ~65% xcorp, and head/embedding effects are ~15× larger on the calib part (LOG 11-08). Fit kfit2 on per-document KL reweighted to calib's category shares (post-stratification), so head, embedding and family prices match a calib-like private corpus.
- *Source:* 11-02 U2. New: pressing because of the two-part slices.
- *Fit:* knapsack input.
- *Test:* CPU reweighting of stored rows.
- *Gain / risk:* −0.2…−1% (head/embedding bytes at low sizes).
- *Conflicts:* none.

**E5 · Do sensitivities differ between xcorp and calib?**
Compare group-swap sensitivities fitted on the xcorp part vs the calib part. If they differ by more than 2σ for some family, price with the calib estimate (the board's corpus is calib-like).
- *Source:* #93 (xcorp keeps 52–91% of gains per step on the 0.8B).
- *Fit:* knapsack input.
- *Test:* CPU on C7's data.
- *Gain / risk:* diagnostic.
- *Conflicts:* #93.

**E6 · Mix-invariance audit of per-level post passes.**
Asym failed per level because its fit depends on sibling codes (#111). Check whether swj (gate/up joint refit at equal K; bf16 or quantized-prefix inputs?) and C9 depend on anything mix-specific. Then splice swj'd gate/up into a mixed-K mix with fixed codes and compare the swj gain against the uniform-K gain.
- *Source:* #111.
- *Fit:* decides whether swj/C9 stay per-level.
- *Test:* code read plus 2 fixed-code evals. ~1 card-h.
- *Gain / risk:* protects the 6.5% swj gain from a #111-type collapse.
- *Conflicts:* #111 — tests the same failure for swj.

---

## F. Quantizer-setting checks grounded in measurement

**F1 · Close diagonal-weighted Viterbi metrics (done).**
§0 puts the ceiling at ≤ 0.0011 bit/weight on 17 tensors (CurveTQ's post-RHT 0.00013 agrees). Record "not V3, not V4 with RHT".
- *Source:* 2610.09212, 2610.00432; §0.
- *Fit:* none.
- *Test:* done.
- *Gain / risk:* 0; saves a test slot.
- *Conflicts:* merge tier C (I001532 D-weighted branch cost) — closes it.

**F2 · Tile-CD family map from the in-tile shaping ceiling.**
Full in-tile shaping ceilings (what CD approximates): attn_qg L3 0.21; k_proj 0.042; early GDN out_proj 0.044; gate/up ≤ 0.006. Keep CD on attn_qg only and close B12 (attn_k, early ssm_qkv) at their −0.03…−0.15% estimate. Optionally restrict CD to early layers, where the structure sits, to save ~2 card-h of ladder time.
- *Source:* §0; #77; merge B12.
- *Fit:* per-level setting (no change needed).
- *Test:* none more.
- *Gain / risk:* 0; closes B12.
- *Conflicts:* #77.

**F3 · H_out tempering exponent h^p.**
ShamAN-Q 2609.38521 found tempering p = ½ beat raw Kronecker factors (PPL 19.21 → 17.46, though in a sub-1-bit regime). On w87's held-out two-sided settle64 proxy, compare p ∈ {½, ¾, 1}, scored against a held-out Σdy² half (block split-halves) so the judge differs from the fit.
- *Source:* 2609.38521; 11-02 O9.
- *Fit:* per-level setting (H_out preprocessing).
- *Test:* colo CPU, w87 harness. ~3 h. A 27B pair only if the held-out gain is ≥ 1%.
- *Gain / risk:* 0…−0.5%. The bn estimator may already capture it.
- *Conflicts:* #33-29 (estimators within 1%). Tempering is a different knob.

**F4 · Gen-weighted H_in for levels ≤ K2.**
On the 0.8B, gen-weighted H_in was K3 0.998 but **K2 0.976, every seed, z −4…−5** (#86), just under its 0.97 bar. With 1.0–1.75 bpw now release sizes, low levels matter more. Option: a second, gen-weighted H_in (forward-only collection on calib, allowed data) used only for K ≤ 2 levels.
- *Source:* #86; #83 (gen-masked H_out).
- *Fit:* per-level input statistics (level-dependent H_in).
- *Test:* 0.8B 2 more K2 seeds plus a K1.5 arm (4070, 2 h). If positive, mkhin27 with gen weights (~12 card-h).
- *Gain / risk:* 0 at 2.5; −0.5…−1.5% at 1.0–1.75 (with 27B shrink). Risk: the user's "no more collections" applied to Σdy²; this is forward-only H_in.
- *Conflicts:* #86 inconclusive; #91 (self-gen text: different).

**F5 · Activation-denoising Hessian (partial denoising).**
2610.07522 replaces LDLQ's (W, H) by (WF, G), with `G = H + λ²N + λ(C+Cᵀ)` from one noise-injected forward pass, and partial denoising ρ < λ. It closed 18–31% of the parallel–sequential gap at W3 and beat sequential on 8B/13B at W2 QuIP#.
- *Source:* 2610.07522.
- *Fit:* per-level quantizer input.
- *Test:* 0.8B first (4070, ~3 h) with ρ < λ, 2 seeds, K2/K3.
- *Gain / risk:* −0.5…−2% if it transfers. **High risk:** #88 (level-matched denoising target) failed (K3 1.031, K2 1.10).
- *Conflicts:* #88, #88-02. What is different: partial ρ < λ, per-channel γ-scaled injected noise, metric change rather than target swap.

**F6 · Analytic Gaussian scale vs g_scale search.**
After HIP, 2609.37416 finds the analytic Gaussian scale matches searched rules at ≥ 3 bits. Compare `g_scale_gss`'s choice with the analytic scale per K from B4 logs. If they agree, the search (7% of quant time) could be skipped at K ≥ 3. If they disagree at sub-1 or K7+, that flags overload.
- *Source:* 2609.37416.
- *Fit:* per-level setting (speed); diagnostic.
- *Test:* from logs.
- *Gain / risk:* 0 KL; ~5% ladder time.
- *Conflicts:* none.

**F7 · Best-of-2 seeds by two-sided objective for the most sensitive tensors only.**
- *Source:* #41 (spread 0.5%, correlation +0.22 across row samples: null).
- *Fit:* per-level.
- *Test:* B7 data shows the spread for free.
- *Gain / risk:* ≤ −0.2%; probably null.
- *Conflicts:* #41. What is different: full tensors and the real objective, but selection is still in-sample. Low priority.

**F8 · Held-out gate for code-changing polish (DRQ-style).**
Accept tile-CD code flips only if error on a held-out activation subset also drops. DRQ 2610.11226: lower reconstruction loss raised PPL in 47% of GPTQ refinements.
- *Source:* 2610.11226.
- *Fit:* per-level setting.
- *Test:* w77 split Hessians on 27 tensors. CPU 1 h.
- *Gain / risk:* ~0 (CD gain under H_out is ~0.1%).
- *Conflicts:* #77.

**F9 · Scaled-Hessian check for early attn_qg.**
§0's 0.21-bit in-tile anisotropy at L3 q_proj but 0.042 at k_proj (same H) comes from `su` scaling by q's per-input RMS. Compare `scaled_hessian` on vs off for L3/L7 attn_qg on the held-out proxy.
- *Source:* §0; CALIBRATION.md (scaled Hessian helped, held-out ratio 0.88 at K2.5 on the 2B).
- *Fit:* per-level setting (per family/layer).
- *Test:* colo settle64. 1 h.
- *Gain / risk:* ~0…−0.2% (two tensors).
- *Conflicts:* none.

---

## G. Level menu

**G1 · Pick-probability map to skip never-used (tensor, level) cells.**
Run knap5 at every release size and variant under ~200 bootstrap price draws (kfit2 `KFIT_BOOT` plus B7 noise). Record each cell's maximum pick probability. Cells at < 1e-3 under ±50% prices can be skipped in the ladder: per-tensor jobs allow it, group jobs need group-level rules.
- *Source:* new; w58 (pass-1 25 levels within 0.2–0.8% of all 73).
- *Fit:* level menu (per-tensor).
- *Test:* CPU, 1 day.
- *Gain / risk:* 0 KL; saves maybe 10–30 of ~90–115 card-h. Risk: the stream later wants a skipped cell (keep margins).
- *Conflicts:* user keeps the 17 i4 levels; this is about cells, not whole levels.

**G2 · Value-of-level ablation with uncertainty.**
For each level, re-solve all sizes without it. Drop levels whose removal costs less than the model's prediction noise everywhere (candidates: odd 1/8 steps above K4.5, mul1 K7.75).
- *Source:* w58; 6b ("continuous K < 1%").
- *Fit:* level menu.
- *Test:* CPU 2 h.
- *Gain / risk:* 0 KL; saves ~3–6 card-h.
- *Conflicts:* user decisions on i4.

**G3 · Build order by decision value.**
Extend `first_use = [2.5, 4.5]` to cell priorities from G1, so an interrupted donor run still delivers the cells the 2.5 product needs.
- *Fit:* ladder ops.
- *Test:* CPU.
- *Gain / risk:* schedule risk only.
- *Conflicts:* none.

**G4 · Gate the top mul1 levels on the 27B engine floor.**
If the bf16-vs-bf16 floor (H4) is ≥ 30% of V2 8.00's 0.00034, the mul1 K7.25–8 levels can't be priced or selected meaningfully. Build them last or drop them.
- *Source:* I001768; #99.
- *Fit:* level menu.
- *Test:* from H4.
- *Gain / risk:* saves ~3 card-h.
- *Conflicts:* #99 (floor split measured on the head only).

---

## H. KL evaluation noise and cost

**H1 · CUPED / regression adjustment, estimated on the stored 27B rows.**
For each stored pair, regress per-document log-ratios on covariates: control-arm per-doc KL, doc length, category, mean teacher entropy. Then report the SE reduction. The document term (~0.010) is as large as the seed term (#116). A 20–40% cut changes doc counts in every later rule.
- *Source:* 11-02 crossarea F9 / measure U2. New: quantified on the real 27B rows.
- *Fit:* gate and fit protocol.
- *Test:* CPU minutes on `noise27.json` inputs.
- *Gain / risk:* indirect.
- *Conflicts:* none.

**H2 · Source-cluster bootstrap for xcorp parts.**
xcorp documents come from 58 sources. Same-source documents are likely correlated, just as agentic chunks are (cluster SE 5× wider, I001945). Measure the SE inflation of source-clustered over document bootstrap; adopt it if > 1.3×.
- *Source:* merge B4 (agentic clusters).
- *Fit:* protocol.
- *Test:* CPU on golden-search27b-x rows.
- *Gain / risk:* indirect (avoids false passes).
- *Conflicts:* none.

**H3 · Importance-sampled documents with Horvitz–Thompson weights.**
For calibration swaps, sample documents with probability ∝ a cheap predictor of |ΔKL| variance, such as the base mix's per-doc KL, and reweight by 1/p. This stays unbiased and has lower variance per eval than Neyman strata alone. Simulate on stored rows first.
- *Source:* round-1 idea 8 (Neyman strata). New: per-document probabilities.
- *Fit:* fit data design.
- *Test:* CPU simulation. 2 h.
- *Gain / risk:* indirect (1.2–2× efficiency if the predictor is good).
- *Conflicts:* none.

**H4 · 27B engine floor.**
Score bf16 against bf16 with a different reassociation (chunk 512, TP1 vs TP2) on a fit slice. On the 0.8B, reordering one o_proj sum alone gave KL 7e-3 and chunking 0.0007. At 27B this sets the floor under 6.0–8.0 prices and the i8-K7 vs mul1 comparison.
- *Source:* I001768; #33-26; #49.
- *Fit:* knapsack input (top levels); menu (G4).
- *Test:* TP2 bf16 on card 0+1 (card 1 now x8). 2 evals. ~1.5 card-h.
- *Gain / risk:* indirect for 6–8 bpw.
- *Conflicts:* #99.

**H5 · Top-512 truncation check at high bpw.**
Score V2 7.00 against a top-4096 golden vs top-512 on ~50 docs. If KL differs by > 3%, top-level prices (i8 K7 ×1.27 vs mul1) are biased in the board's metric.
- *Source:* I001914.
- *Fit:* knapsack input (top levels).
- *Test:* TP2 golden record plus 2 evals. ~2 card-h.
- *Gain / risk:* indirect.
- *Conflicts:* none.

**H6 · Neyman split between xcorp and calib parts.**
Allocate positions between the two parts by part-specific per-position variance and decision weight. The calib part carries the head/embedding signal and is the scarce pool (583 docs), so it gets only what its variance warrants.
- *Source:* round-1 idea 8 (within-slice strata). New: between parts.
- *Fit:* fit data design.
- *Test:* CPU on stored two-part rows.
- *Gain / risk:* indirect; preserves the calib pool.
- *Conflicts:* none.

**H7 · Entropy- and token-type-stratified ΔKL per family.**
Decompose each family's ΔKL by teacher-entropy bin (stored per row) and by agentic token type (reasoning vs tool-call). Low-entropy positions carry FP4 failures in ReQAT 2606.15682. A family that hurts confident tokens transfers differently to code-heavy and agentic traffic.
- *Source:* 2606.15682.
- *Fit:* diagnostic for E4 weights.
- *Test:* CPU on stored rows.
- *Gain / risk:* diagnostic.
- *Conflicts:* none.

---

## I. Long-context behaviour

**I1 · Position-resolved family sensitivity on agent32.**
Swap one family at a time (attention, GDN-in, GDN-out, MLP) on a K3 base and score 32k chunks by position bucket. On the same model, the weight-quant NLL gap shrank with position in GDN-heavy contexts (+0.081 first 16k → −0.053 last 2k, 2609.04098). Attention-tensor sensitivity may instead grow with context. Weight family prices by the corpus's position distribution (~¼ of scored positions are agentic or long).
- *Source:* 2609.04098; 11-02 L1.
- *Fit:* knapsack input.
- *Test:* card 0, 5 evals × agent-search32. ~4 card-h.
- *Gain / risk:* −0.1…−0.5% on long/agentic positions.
- *Conflicts:* #52, #73 (ratios flat by position for whole mixes; family-level not measured).

**I2 · GDN state dtype parity check.**
Confirm radiance's GDN state dtype (f32 vs bf16) is the same in the golden recorder, the KL eval and serving. LeapQuant 2609.38166 shows a bf16 state alone drops AIME 87.9 → 72.1 on Qwen3.5-9B. A mismatch would add a position-growing floor that looks like quantization error.
- *Source:* 2609.38166; merge tier C (I001830 f16 state unknown).
- *Fit:* measurement validity.
- *Test:* code read. 30 min.
- *Gain / risk:* diagnostic.
- *Conflicts:* none.

**I3 · Is Σdy² from 2k windows representative for long contexts?**
Compare per-layer tr(H_out) proportions from the qad 2048-token windows with the unused 256-sequence self-gen x2 set (longer, model-written) on gpubox. No new collection. If attention or out layers differ > 2×, reprice them (C11).
- *Source:* #98 set (unused); I001877.
- *Fit:* knapsack input.
- *Test:* CPU on gpubox files (lead).
- *Gain / risk:* diagnostic.
- *Conflicts:* user's "no more collections" — this reuses existing data.

**I4 · Position-weighted knapsack objective.**
If I1 shows family ratios moving with position, weight each family's price by the private-like position distribution instead of the 2k-capped search slice.
- *Source:* 11-02 L2.
- *Fit:* knapsack input.
- *Test:* CPU after I1.
- *Gain / risk:* small.
- *Conflicts:* none.

---

## J. Calib pool use and overfitting

**J1 · Size of the Hessian taint on proxy prices.**
H_in includes verify docs. Compare per-tensor one-sided error of one level under calib H against the xcorp-only hinX collection (1.14M tokens, gpubox), as a rank-correlation check. #50 found a 9% in-sample edge at 1.2M tokens/half; at 10M it should be smaller. This sets how optimistic verify-based prices are.
- *Source:* #50 (D2).
- *Fit:* knapsack input interpretation.
- *Test:* CPU decode of one level. 2 h.
- *Gain / risk:* diagnostic.
- *Conflicts:* #101 (H_in blend dropped, not related).

**J2 · Held-out check of swj and C9 fits.**
swj fits ~35k scale parameters per layer on 1M qad tokens. Score the swj objective on fit-pool tokens; if the held-out gain is < 70% of in-sample, regularise (wider clamp prior).
- *Source:* #104 (verify confirmed at whole-model level).
- *Fit:* per-level post-pass setting.
- *Test:* CPU/GPU 1 h.
- *Gain / risk:* ~0 (verify already agreed).
- *Conflicts:* none.

**J3 · Leave-slice-out CV for kfit2.**
With rotating slices, report CV by slice in addition to by document, to catch slice-specific overfit in the sensitivity model.
- *Source:* 11-02 S12.
- *Fit:* knapsack input.
- *Test:* CPU.
- *Gain / risk:* diagnostic.
- *Conflicts:* none.

---

## K. Low-bpw release sizes (1.0–1.75)

**K1 · Zero-tensor bound and shrink guarantee for sub-1 levels.**
In #114's harness, the GPU search at K0.125 had error 1.014 relative, **worse than zeroing the tensor**. Log per tensor × sub-1 level the error after refit against the all-zero error (B3). Guarantee < 1 with an MSE-optimal per-tensor shrink: `refit c_n` can reach 0, so check it actually does. Flag levels the knapsack must never pick.
- *Source:* #114 (1.014, "cell unreliable").
- *Fit:* per-level setting and job check.
- *Test:* from logs, CPU.
- *Gain / risk:* protects 1.0–1.25 (could be −1…−3% there).
- *Conflicts:* #90, #114-02.

**K2 · Real trunk swap anchors at K0.5, K0.875 and K1.25.**
Run one group swap per family on a 1.5 bpw base (6–9 evals). The sub-1 proxy is ~13% optimistic at K0.125, and K1→K2 KL slope is 2.3–2.5 bits/bit against the proxy's 2.11 (#44).
- *Source:* merge B27 (embedding swaps, K0.125 oracle). New: trunk families.
- *Fit:* knapsack input.
- *Test:* card 0. ~4 card-h after the sub-1 levels exist.
- *Gain / risk:* −1…−3% at 1.0–1.5; 0 at 2.5.
- *Conflicts:* #44, #90.

**K3 · Saturation fit at ≤ 1.75 bpw.**
Refit KFIT_SAT with K2's anchors plus 4–6 whole mixes at 1.0–1.5. These are now release sizes, and the coverage model (2607.12266) predicts a bend near mean KL ~0.5.
- *Source:* BACKLOG "Saturation at 0.5–2 bpw".
- *Fit:* knapsack model form.
- *Test:* card 0, 6 evals plus CPU.
- *Gain / risk:* −0.5…−2% at 1.0–1.25.
- *Conflicts:* 11-03 saturation check (null at ≥ 1.6).

**K4 · Per-layer floor guards from single-layer probes.**
Single-layer K0.125 probes at L0, L3 and L63 (the L63 gate_up spike, #21). If a layer blows up, add a minimum-K constraint per layer (GEMQ's "≥ 1 high-bit unit per layer" regulariser, 2605.23078).
- *Source:* #21; 2605.23078.
- *Fit:* knapsack constraints.
- *Test:* card 0, 6 evals. ~2 card-h.
- *Gain / risk:* protects 1.0 sizes.
- *Conflicts:* #21.

---

## L. Head and embedding (library donors)

**L1 · Head softmax-gauge scan.**
Mean-row subtraction is gauge t = 1. Scan t ∈ {0, 1, 2, 4} on headsim gen rows (after the A2b redump). Phi gained a lot (2609.31291); Qwen3.5 GPTQ gained nothing (0.011 → 0.010), so expect little.
- *Source:* 2609.31291; #34.
- *Fit:* per-level (head donors).
- *Test:* CPU headsim. 1 h.
- *Gain / risk:* ~0…−0.2%.
- *Conflicts:* merge A6 (head settings batch) — add t to it.

**L2 · Robustness of the head slab plan to corpus.**
Slab plans rank rows by qad-pool Fisher. Recompute the ranking from the xcorp and agent32 parts and measure plan overlap (top-K row Jaccard). If < 0.8, use a pooled ranking, so the plan doesn't target one corpus.
- *Source:* #34-02 (head share 15× larger on calib).
- *Fit:* head donor plan.
- *Test:* CPU.
- *Gain / risk:* ~0…−0.3% on the calib part.
- *Conflicts:* none.

---

## N. Rehearse the V3 pricing methodology on the 0.8B

**N1 · 0.8B dress rehearsal of the V3 pricing pipeline.**
Build ~12 0.8B levels with the frozen stack (H_out dy, swj, C9, block order). Price them one-sided, two-sided and Fisher-only, solve the knapsack at 3 sizes, and measure the mixes (23 s each). This tests C1–C4, C8, D-cluster model forms and K3 end-to-end for under a day.
- *Source:* new composite.
- *Fit:* knapsack input methodology.
- *Test:* 4070, ~1 day.
- *Gain / risk:* indirect; decides which C-ideas get 27B time.
- *Conflicts:* #93 (0.8B transfer caveats: method rankings transfer, magnitudes shrink).

**N2 · Interaction structure first on the 0.8B.**
Run D1, D2 and D6 there before spending 27B card time. Interaction energy is likely larger on the 0.8B (higher KL), so a null there is informative.
- *Fit:* knapsack model form.
- *Test:* 4070, ~40 evals. 2 h.
- *Gain / risk:* indirect.
- *Conflicts:* none.

**N3 · Pricing A/B with measured outcomes.**
For N1's three price sources, compare the *measured* KL of each knapsack's chosen mix at equal bytes, with 3 seeds. Counts if one source wins by > 2σ at ≥ 2 of 3 sizes.
- *Source:* 2609.33923 (block signal beat layer objective on the 27B).
- *Fit:* knapsack input.
- *Test:* 4070. ~2 h.
- *Gain / risk:* decides C2/C3 adoption.
- *Conflicts:* none.

---

## Top 20 by expected gain per unit cost

| rank | idea | cost | expected effect at 2.5 bpw (other sizes) |
|---|---|---|---|
| 1 | **A1** H_out-weighted su refit | 1 CPU-h + 2 card-h (fixed-code pair) | −0.1…−0.8%, every level |
| 2 | **B1** per-channel error vectors in jobs | ½ day dev, ~0 run | enables C2/C3/C14 (−0.3…−1%) |
| 3 | **C1** price parity across settings and options | CPU 1 h | 0…−0.3% (−0.2…−1% at 5–7) |
| 4 | **C9** fold stored 27B arms into kfit2 | CPU ½ day | indirect; early model check |
| 5 | **B3** component trajectory log | 2 h dev | indirect (A5, C6, K1) |
| 6 | **K1** zero-tensor bound at sub-1 | from logs | 0 (−1…−3% at 1.0–1.25) |
| 7 | **E4** category-reweighted fit targets | CPU hours | −0.2…−1% |
| 8 | **C3 + C4** Fisher price, split lumped groups | CPU 1 day | −0.3…−1% |
| 9 | **N1 + N3** 0.8B pricing rehearsal and A/B | 4070 ~1 day | decides C-cluster |
| 10 | **E3** slice-anchor arms | +2 evals/round | indirect |
| 11 | **A4** same-seed gate design | 0 | halves false alarms on the freeze gate |
| 12 | **G1** pick-probability map, skip cells | CPU 1 day | 0 KL; −10…30 card-h |
| 13 | **H1** CUPED on stored rows | CPU minutes | indirect: doc counts |
| 14 | **D6** coherent-part interactions (0.8B) | 4070 2 h | narrows D-cluster |
| 15 | **E6** mix-invariance of swj/C9 | code read + 1 card-h | protects the 6.5% swj gain |
| 16 | **C14** row-shuffle probe | 1.5 card-h | validates two-sided pricing |
| 17 | **C13** error-amplitude probes | 8 card-h | −0.3…−1% (better s_w, q) |
| 18 | **A3** leave-one-out stack audit | ~5 card-h | 0 expected; ~15% chance to catch −0.3…−1% |
| 19 | **K2 + K3** low-bpw anchors and saturation | ~6 card-h | 0 (−1…−3% at ≤ 1.5) |
| 20 | **I1** position-resolved family sensitivity | ~4 card-h | −0.1…−0.5% (long/agentic) |

Next in line: C2 (colo decode of V2 containers), A7, H4 + G4, D1, F4, B7, C7.

## The 5 cheapest diagnostics that would most change this ranking

1. **A1 CPU check** (~1 CPU-h): f2 after h-weighted vs unweighted `su` refit on 6 tensors. Below 0.1% → A1 drops out of the top 10. Above 1% → also check C9/swj-style post passes for the same inconsistency.
2. **C1 re-solve** (~1 CPU-h): knap5 with H_out, #117 and swj proxies converted to KL equivalents. Count changed picks and predicted ΔKL. If < 0.2% at every size, demote the whole pricing cluster (C1–C4, C10, C11).
3. **Fisher price vs V2 swaps** (~4 CPU-h, colo): log-RMSE and Spearman of ½Σh_n e_n against the ~1,250 V2 measurements. If it beats one-sided proxies inside groups, C3/C4/C10 move to the top; if not, C2 alone.
4. **CUPED + source-cluster bootstrap on stored 27B rows** (minutes): sets how many documents every later gate and calibration swap needs. That rescales the card-hour cost of A3, C7, C13, D1, I1 and K2.
5. **D6 on the 0.8B** (~2 h on the 4070): if interactions sit in the coherent component and are small for incoherent errors, the D-cluster shrinks to per-family pair terms or nothing. If they are large, D1/D2/D8 move up.

The weighted-Viterbi ceiling (§0) has already run and closes F1 and B12 at no further cost.
