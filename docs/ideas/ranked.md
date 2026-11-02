# Critic report: merged, checked and ranked ideas from 10 lenses (2026-11-02)

**Inputs.** All ten lens reports were present: composable, open, crossarea, lowbpw, measure, hout, kernels, heal, codes, redteam. I merged roughly 950 raw ideas into the 120 below.

**What I checked**
- **New LOG entries** after the attached context: #33-15, #33-16, #33-17, and YAQA08 amendments 9–10.
- **Code:** `v3/yaqa/collect.py`, the exllamav3-frac `quantize.py` (`prepare_H_out`, `ldlq_2hess`, `refit_scales`), `quantize/ourq.py`, `v3/nf/knap4.py`, the V2 2.0/2.5 mixes, and the radiance-hero `exl3i` decode.
- **Papers:** about 25 cited numbers, checked by grep in `corpus/full` and `corpus/ref`.
- **One exhaustive CPU check**, of the int8-codebook closed form.

## 0. Facts that change the ranking (checked this run)

### 0.1 New results the lenses did not have
- **#33-16: identity H_out is worse than base.** K3 +4.9% (z 7.5), K2 +2.1% (z 3.9). The svh² row weighting explains none of the gain; the measured diagonal explains all of it. This **kills** the composable lens's #1 idea (C1, "two-sided with H_out = I") and redteam B3/C6. Rejected.
- **#33-16: 3-seed gain.** K3 **−8.1%**, K2 **−5.4%**. The −8% at K2 came from a lucky seed 0. Per-seed 0.8B quant noise is **~1% at K3 and ~2% at K2**. Consequence: any 0.8B arm-vs-arm effect under ~3% (K3) or ~5% (K2) needs ≥3 seeds. Most "follow-up knob" ideas (tempering, damping, structure, order) expect 0.5–2% and cannot be judged on one 0.8B quant pair.
- **#33-17: split halves already exist.** The collector writes even/odd sums and `halves.json`, so hout A1 and composable A2 are done. Amendment 10 sets the stop rule: the first snapshot at ≥1,024 sequences with median cos ≥0.995 and p5 ≥0.964. The 27B run has **not started**, so collector changes are still possible.

### 0.2 Code facts (verified)
| claim | status |
|---|---|
| Collector loss = sampled-label CE summed over **every** position; no gen mask (`collect.py:346–375`) | ✓ |
| Windows only from qad docs ≥2048 tokens; shorter docs are never used (`collect.py:76–79`) | ✓ (window category mix not checked: golden and pools are not on the VM or CPU box) |
| `refit_scales(weight, weight_q, H, su, sv)`: input-side solve weights all output columns equally, ignores H_out | ✓ |
| `ldlq_2hess` updates F over full output width (`Ln[cols][:, :, :n_hi]`) even when L_out is block-diagonal | ✓ |
| hout lens: "dense path ≈1e15 FLOPs, hours per gate/up" | ✗ My count is **≈6e14 FLOPs, tens of seconds** on an R9700 (about +1–2 h per full-trunk level). The fast path is worth having but not urgent. |
| `ourq --hout` silently skips tensors with no file (`os.path.exists`) | ✓ |
| `KN_LEVELS` prices measured levels from one-sided `proxy_err` with the family of the lower integer neighbour | ✓ H_out quants **will be mispriced** (objcheck: H_out raises one-sided error) |
| V2 2.0 and 2.5 mixes ship `hd-K3` (uniform head); knap4 offers slab heads only at K5/5.5/6 | ✓ The low-budget slab head is a real menu gap |
| mul1i8 closed form `q = (max(s+3,4)>>2) − 128` (s = bytesum) | ✓ **0 mismatches / 65,536** against the reference (fp16 FMA → f32 rint → clamp) |
| Production `exl3i_i8_pair` decodes fp16 then rounds through f32 | ✓ |
| Decode `sk = 1`; T2 chunk 192 MiB | ✓ |

### 0.3 Paper spot-checks
✓ = number found in the paper text. ✗ = the lens report misread it.

| claim | status |
|---|---|
| YAQA Qwen3-8B Table 8: KL −28/−30/−29% at K2/3/4 (0.285→0.205, 0.0681→0.0479, 0.0216→0.0153) | ✓ gain flat in K |
| H_Z-PTQ (2610.40345): "YAQA's factor unstable, 19.12 vs 14.26" (composable A1, redteam B9) | ✗ **Misread.** 19.12 is YAQA's **rounding procedure** (per-channel W4, Hadamard). With H_Z's own rounding the two factors **tie** (14.26 vs 14.25). The same table has two-sided YAQA rounding **losing to one-sided GPTQ** under Hadamard (19.12 vs 14.50). That fits our 27B negatives: it is a reason to fear the 27B swap test, not to switch factor. |
| ThinkBit: scales only 0.641→0.282 = scales+norms; norms only 0.418 | ✓ (scalar 2-bit GSQ base, weaker than ours) |
| MPQ: Llama-3-8B E8P12 2-bit LDLQ 15.01 → LDLQ+CD3 12.76 → MPQ 11.63 | ✓ (vector code, more freedom than a trellis) |
| Activation denoising: 2-bit QuIP# 8B 13.76 → 12.70 (sequential 12.87); 3-bit closes 18–31% | ✓ |
| ShamAN-Q: untempered 19.21 → tempered 17.46; KL-Shampoo 17.04 → 16.72 | ✓ (1-bit binary factors) |
| 2609.04098 (Qwen3.8-27B NVFP4): a/b 2.1/2.6%, qkv 10.4%, z 9.9%, out 12.7%, quadrature sum | ✓ |
| Q-Strata: lazy greedy 749.4 vs one-shot ILP 823.4; 12.14 vs 13.72 | ✓ (MoE) |
| RAM (on Qwen3.8-27B): oracle layer-objective allocation 6.087 vs block cosine 6.041 vs cosine×CKA 5.991 | ✓ |
| CurveTQ: Γ_b after RHT median 0.0001 bit; **chained row trellis 13.59 ≈ tail-biting T=256 13.38** | ✓ The second number **counts against** the chaining/ring ideas |
| QTIP L ablation: L12 6.97 → L16 6.83 | ✓ (no data beyond L16) |
| Tetra: trained row scales +3.15 / +3.29 / +1.67 MMLU | ✓ |
| QATFactory: LoRA r4 0.0722 vs full QAD 0.0682 (RTN 0.0778); GPTQ worse than RTN on NVFP4 (0.1399) | ✓ |
| 2609.09867: composition error 25.9× / 41.7× the pricing error | ✓ |
| G²PTQ: no trust region → KL 25.3 | ✓ (its Qwen3.5-2B KL numbers were not found by grep: unverified) |
| EFQ: trace-optimal joint (R,G) loses to RHT+feedback (Llama-2-7B M=8: 9.893 vs 9.120) | ✓ |
| DRQ "73.8% of refinements raise PPL", FidelityQ, SoftWater, ARCHead, TVQ, E8-QAT numbers | ? digest only, not re-verified |

### 0.4 Prior for the baseline itself
- **Planned 27B diagonal H_out:** on the 0.8B, −8.1% (K3) and −5.4% (K2) over 3 seeds.
- **Expect −2 to −5% at 2.5 bpw at 27B; P(≥2%) ≈ 0.5.**
- Why it may shrink:
  - Proxy gains shrink with model size (REAL-Q: −49% at 1.7B vs −5…−9% at 8–70B).
  - Every 27B gate with other estimators lost (#33-02…#33-06).
  - H_Z-PTQ shows two-sided rounding can lose under Hadamard.
  - Head and embedding dilute a trunk-only gain.
- **Why the pre-run diagnostics rank first:** they decide whether the ~8.5 h two-card night is spent well.

## 1. Conventions
- **Fit:** PL = per-level library; FP = final pass on shipped mixes; PM = per-mix; FMT = format change; TR = training; DX = diagnostic; INF = infrastructure.
- **HW:** "4070" = the RTX 4070 that runs the 0.8B proxy; "GPU box (one card)"/"GPU box (both cards)" = one or both R9700s; "CPU box" = a 36-core CPU machine; "rented" = cloud GPU (H100 80 GB ≈ $2.5/h, MI300X ≈ $3/h).
- **E** = expected KL change at 2.5 bpw if the idea works, shrunk for 27B transfer. **P** = probability it delivers at least the low end at 27B.
- **Pass rules:**
  - 0.8B: 3 seeds and "every treated seed beats every control seed", unless the arm keeps the codes fixed (no rounding noise).
  - 27B: paired bootstrap on golden-s2k, z ≤ −3, **plus** the 27B seed-noise term from A7.
- **Fixed-code tests:** refit, scale multipliers, healing and head-only changes keep the trellis codes. Their tests carry no quant-to-quant noise, so they get the cheaper pass rule (one paired comparison, z ≤ −3).

---

## 2. Full merged list by theme

### A. The 27B diagonal-H_out collection (decide before `hdiag27.sh`)

| ID | Idea (lenses) | Fit | HW / cost | Cheapest test → pass rule (fixed before data) | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| A1 | **Gen-masked sampled-label loss:** loss only on scored rows; context still flows (hout B1/A10, open A1, lowbpw A1, measure O1, composable A8/B8, crossarea A6) | PL | 4070 ~2.5 h (2 collections × 26 min + 12 quants) | 1K masked vs unmasked diagonals (calib qad windows with golden gen masks), bdiag K3/K2, 3 seeds. **Pass:** masked mean KL ≤ 0.975× unmasked at K3 and ≤ 1.00 at K2, every masked seed below its matched unmasked seed. Else keep the current loss. | 0 to −2% | 0.3 | Verified mismatch with the metric. Fewer loss positions → re-check the dose (A5). |
| A2 | **Estimator check** through `ldlq_2hess`, all from one 0.8B backward: Sketch-B ‖G_b[r,:]‖², per-token Σ‖x‖²dy² (the estimator the 27B gates used), Σdy² (H_Z's factor), leverage-weighted Σ(xᵀH⁻¹x)dy² (redteam B9/C4, hout B2/B3, composable A1/B1, crossarea A3, lowbpw A4, open B6) | DX/PL | 4070 ~2 h | 3 seeds, K3/K2. **Rule:** (a) if Σ‖x‖²dy² keeps ≥2/3 of Sketch B's gain, the old 27B negatives are *not* estimator-driven → run A6's pilot swap before the full night. (b) If Σdy² or leverage is within one seed SD of Sketch B, collect them too (cheaper or more robust). | DX | — | Corrects the H_Z misreading: no paper evidence that Σdy² *beats* Sketch B. |
| A3 | **Instrument the run:** extra sums in the existing hook — Σdy² diag, leverage diag, per-sequence per-tensor traces, per-category sums, position buckets (0–255 / rest), embedding-output gradient covariance (5120², 100 MB), residual-stream gradient diag (+ full 5120² for the last 8 layers), per-32-sequence partials (composable A1–A6, hout A2–A8, redteam D5, heal A10, lowbpw A20, measure O2) | INF | ~1 day code; run cost ≤5% | 64-seq gate: **pass** if time/seq +≤3% and VRAM +≤1 GB; Σ of logged traces = Σ diag to fp32 rounding | enables ~15 ideas | 0.9 | Cheapest way to avoid a second fp32 night. |
| A4 | **Window composition:** short docs are excluded today. Count window tokens by category vs the corpus gen-token mix; pack short docs (state reset) or reweight (measure L8, hout B19, redteam H3, composable A3/B11) | PL | CPU minutes + hours of code | **Rule:** if any category's window share differs from its corpus gen-token share by >2×, pack or reweight. | 0 to −1.5% | 0.3 | Verified exclusion; the mix itself is unmeasured. |
| A5 | **Dose below 1K + early decision:** 128/256/512 diagonals on the 0.8B; for the 27B, a new amendment allowing a K3 swap from the 256/512 snapshot before resuming (redteam D1/D4, hout B10/A9, crossarea A12) | DX/INF | 4070 ~1.5 h | 3 seeds. **Rule:** if 512 keeps ≥80% of the 8K gain at both K, amend: stop at 512 when halves median ≥0.990, then swap-test before continuing. | saves ≤4 h × 2 cards | 0.5 | The collector resumes bit-exactly (#33-17). |
| A6 | **27B 64-sequence pilot:** halves cos, per-family CV, predicted gain from within-128-block GM/AM of h·sv² (the LDL bound), max/median h on residual writers, plus timing of `ldlq_2hess` on one gate_up (redteam C5/B1/E2, hout D1/D5, crossarea A1) | DX | GPU box (both cards) ~35 min + GPU box (one card) 15 min | Validate the predictor on 0.8B rowerr first (CPU ~1 h; pass if Spearman ≥0.7 vs per-tensor improvement). **Go** if predicted 27B gain ≥½ the 0.8B prediction and pilot halves median ≥0.98. | DX | — | The predictor is my formalisation of hout/crossarea "noise-shaping gain". |
| A7 | **27B quant-noise floor:** requant ladder K3 with `--seed-offset 1`, eval paired (redteam J1, heal A8) | DX | GPU box (one card) 1–3 h | Report the seed delta. Every 27B pass rule uses it. | DX | — | #48 (z 2.1) and every 27B single-quant A/B are uninterpretable without it. |
| A8 | **Fail loudly:** assert collector file count = trunk linears; make a missing `--hout` key in ourq an error (redteam D6) | INF | minutes | — | protects | 1 | Verified silent skip. |
| A9 | **Analytic forward-only gate/up diagonal:** w_gate = ‖W_d[:,r]‖²·E[(SiLU′(g)u)²], w_up = ‖W_d[:,r]‖²·E[SiLU(g)²] (hout B12/B13, open A11, crossarea A7, lowbpw A13, composable F10, redteam K1) | PL | 4070 ~1.5 h | **Pass:** cosine ≥0.95 with Sketch B on gate/up, and ≥2/3 of their gain (3 seeds). Then the 27B gate/up need only a forward pass (half the trunk bytes). | cost cut | 0.35 | Source: SACQ, FastKron local Hessian. |
| A10 | **Gaussian or Rademacher √Fisher logit probe** instead of a one-hot sampled label (composable A9/B9, hout B5, crossarea A13) | INF | 4070 1 h | Dose curve: **pass** if 256 probe seqs ≥85% of the 1K sampled-label gain. | time | 0.3 | |
| A11 | **bf16-GEMM 27B collector:** per-linear A/B inside the gate (redteam D2, hout B14 depth-split) | INF | in the gate | **Pass:** per-layer diag rel-L2 ≤5e-2 and cos ≥0.99 vs fp32 in every depth band. | ~3× faster | 0.35 | #33-03 (27B bf16 backward rows wrong early); #33-15 (KL can't see precision on the 0.8B). |
| A12 | **Operating-point H_out:** collect on a quantized model (composable B7, lowbpw A5, hout B16/B17, crossarea A10, open A13, redteam F6) | PL | 4070 1 h; 27B = a second night | 3 seeds K2. **Pass:** ≤0.97×. | 0 to −2% (K≤2.5) | 0.2 | Mixed evidence (GEMQ +, MixFrag −). |
| A13 | **Diagonal + rank-r Nyström sketch** in the collector (composable A7/B4, crossarea A5, open A9, hout C7) | PL | 0.8B offline from the 8K full snapshot, 3 seeds ~1.5 h | **Pass:** r=32 beats bdiag at K3 by ≥2 seed SD and is not worse at K2. Only then add to the 27B run. | 0 to −1.5% (K3) | 0.2 | bmix's K3 edge (#33-12) is inside the noise. |
| A14 | Full H_out for attention k/v (4 MB each) (hout A4) | PL | — | after A13 | ≈0 | 0.15 | Attention is ~6% of KL. |

### B. H_out inside the quantizer (0.8B, then the Phase-B requant)

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| B1 | **H_out-aware `refit_scales`:** input-side solve ((Q·diag(h)·Qᵀ)∘H)r = rowsum(Q∘(HW)∘h); refit in the undamped H; 2→6 rounds (heal G1/G2, redteam E3, crossarea B1, lowbpw A15, hout G3, composable C2) | PL | 1–2 h code; seconds per tensor | **Fixed codes**: refit the existing bdiag quants. **Pass:** ≤0.99× at both K for all 3 seeds. | −0.3 to −1.5% | 0.5 | Verified gap. Applies to the 27B swap test at no cost. |
| B2 | **Tempered diagonal h^p** (p ∈ {¼, ½, ¾, 1.5}) and output damping per K (σ_out ∈ {0.003…0.1}), per-block normalisation, floor (lowbpw A3/A8, composable B2/B3, hout C1–C4, redteam B7/B8, crossarea B3, open A7/A17) | PL | 4070 ~3–4 h (3 seeds) | **Pass:** best arm's 3-seed mean beats bdiag by ≥2 SE of the mean at both K. | 0 to −2% | 0.3 | Effects are likely ≤ seed noise. Prefer the CPU check: held-out two-sided proxy on split halves. |
| B3 | **Attribution:** H_out on one family at a time (GDN in, attention in, gate/up, residual writers), on one depth third at a time, and at K2.5/K4/K5 (redteam B4/B5/C1, hout F1–F3, lowbpw A2, open A14, composable C6/C8) | DX | 4070 ~4 h (2 seeds; families vs base are large effects) | Report shares. **Rules:** families with <15% of the gain are requantized one-sided. If the gain at K4 is <3%, ≥3.5 bpw levels skip two-sided. | DX (scope) | — | Decides Phase-B cost and backward truncation (A11/D3 in hout). |
| B4 | **Output-scale path** (`sensitivity_out_scales`) re-tested with the Sketch-B diagonal; block-level scales (redteam B10, lowbpw A6/A7, hout G1, open A2/A6, crossarea A2) | DX/PL | 4070 30 min (2 seeds) | **Pass (mechanism):** beats base with every seed. | 0 to −1% | 0.2 | Conflicts with #33-06/#48. The real difference is the estimator (gate 2d used 200-seq ‖x‖²-weighted rows). |
| B5 | **Structure beyond diagonal:** 128-blocks, Ledoit–Wolf shrinkage, diag+low-rank, K-dependent choice (hout C5–C10, composable B4/B5, crossarea B4/B5, open A8, lowbpw A18) | PL | 0.8B offline, 3 seeds; at 27B needs A13 | as A13 | 0 to −1.5% at K3 only | 0.2 | bmix vs bdiag (#33-12) is inside the #33-15 noise. |
| B6 | **Free output permutations** (MLP intermediate interleave or sort) to maximise within-block GM/AM (crossarea B6, hout J5, composable D1/D2/D8, open A3, lowbpw A11) | PL | CPU predictor (A6) first | **Pass:** predicted gain ≥0.02 bit on gate/up, then 3-seed 0.8B arm. | 0 to −1% | 0.15 | MLP neuron order is arbitrary, so within-block spread probably already matches global spread. |
| B7 | **Stored output row permutation or wider output Hadamard** for head-structured tensors (q/k/v, GDN qkv/z) (open A4/A5, crossarea B7, codes D5) | FMT | CPU predictor, then fake-quant | as B6 | 0 to −1.5% | 0.15 | Kernel epilogue scatter needed. |
| B8 | **Shuffle control:** permute hout_diag within blocks (redteam B2) | DX | 4070 20 min | If shuffled-within-block ≥½ of the gain, the gain is regularisation, not information. | DX | — | #33-16's identity result already argues against an artefact. |
| B9 | **Two-sided proxy in all internal choices:** sigma-in sweep, g_scale, logging (composable C3, redteam E4/E5, hout G5, crossarea B2) | PL | free code | Count changed winners. Run arms only if >20% change. | 0 to −0.5% | 0.3 | |
| B10 | **Block-diagonal fast path in `ldlq_2hess`** (hout J1/J2, redteam E1) | INF | ~½ day | Only if A6 timing shows two-sided >1.3× the one-sided level time; KL-identity gate. | time only | 0.8 | Verified dense update; my cost estimate is lower than the hout lens's. |
| B11 | Output-side LDLQ block order (codes H1) | PL | 0.8B 3 seeds | as B2 | 0 to −1% | 0.15 | Same drift caveat as C4. |

### C. Other quantizer changes (per-level library; ride on the one Phase-B requant)

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| C1 | **Activation-denoising LDLQ target:** WF under G = H + λ²N + λ(C+Cᵀ), from one noise-injected forward (open D3, lowbpw B6/B10, crossarea C7, composable F3, heal F6, redteam K3) | PL | 4070 ~3 h (3 (λ,ρ) × 3 seeds); 27B: one forward night, C and N ≈2× Hessian storage | **Pass:** best (λ,ρ) chosen on the fit slice beats base by ≥3% at K2 on search with every seed, and ≥1.5% at K3. | −1 to −3% | 0.35 | ✓ numbers (QuIP#-2-bit beats sequential on 8B/13B). Unknown interplay with scaled Hessian and H_out. |
| C2 | **Post-LDLQ tile coordinate descent:** re-Viterbi each tile against the full-tensor residual target, 1–3 sweeps, accept only if the held-out-half proxy drops (codes C6/C7/C8, crossarea D3/D4, lowbpw B2/B12, open D1/D11, composable F1/F2, heal F3) | PL | CPU box CPU day on real tiles, then 0.8B; quant ×2–3 | **Pass (CPU):** ≥2% proxy at K2.5 on H_B (quantized under H_A) over 16 tensors; then 0.8B 3-seed ≤0.97×. | −0.5 to −2.5% (K≤3) | 0.3 | ✓ MPQ, but on vector codes. Trellis tiles have less freedom. DRQ warns about in-sample proxy. |
| C3 | **Hessian-aware input scales** t ∝ (σ²/H_jj)^{β/2}, β ∈ {0, ¼, ½} (lowbpw B1; KronQ) | PL | 4070 ~1.5 h | **Pass:** 3-seed ≤0.97× at both K, and the E1c A8-g128 activation error rises ≤10%. | 0 to −3% | 0.25 | Massive channels give 800–4,800× diagonal spread. Interacts with the scaled Hessian. |
| C4 | **LDLQ block order:** natural / descending / random, real trellis (codes C1/C3, crossarea D7, lowbpw B5/B16, open D12) | PL | CPU box; first fix the torch/OpenMP harness hang | **Pass:** ≥2% proxy on some family with the real quantizer including the scale search. | 0 to −1.5% | 0.2 | Codes lens measured ascending **3.4× worse** on L3 q_proj despite the proxy predicting a gain. Use the real quantizer only. |
| C5 | **Per-level quantized-prefix (GPTAQ) targets** with a control-variate blend toward the published H (open D2/D4, crossarea C5/C6, lowbpw B7/B9, hout E10, heal F5, composable F4) | PL | 0.8B ½ day; 27B a forward pass per level (nights or rented) | 3 seeds K2/K3. **Pass:** ≤0.96×. | −1 to −4% | 0.3 | ROADMAP V3.5; the control variate (crossarea C5) addresses the token-count caveat. |
| C6 | **Intra-layer sequential:** down/ssm_out/o_proj quantized on inputs from already-quantized siblings (lowbpw B8) | PL | 0.8B 2 h | as C5 | −0.5 to −2% | 0.3 | Cheap subset of C5. |
| C7 | Scale ↔ code alternation (refit, then re-Viterbi on the moved grid) (codes G2, lowbpw B3, open D7) | PL | CPU then 0.8B | **Pass:** ≥0.5% held-out proxy. | 0 to −1% | 0.3 | JARQ: ≤1% on strong hosts. |
| C8 | **LDLQ-aware g_scale search** (Schur / proxy-static instead of raw tile MSE) (composable C4, lowbpw B4, codes G3) | PL | CPU 3 layers | **Pass:** proxy gap >1% at K2–3. | 0 to −1.5% | 0.3 | Unblocks #39 with a better criterion. |
| C9 | **KL-optimal svh multiplier per (level, family):** zero bytes, folds into svh (lowbpw B13, composable F7) | PL/FP | 4070 30 min, **fixed codes**; 27B by container edit (G8 of heal) | **Pass:** one ε ≠ 0 beats 0 at z ≤ −3 on search, same sign on a second slice. | 0 to −1.5% | 0.3 | Tests the LS shrink bias directly. |
| C10 | SwiGLU-joint gate/up scale fit per level (composable F9, heal G4) | PL | 0.8B 1 h, fixed codes | as C9 | 0 to −1.5% | 0.3 | |
| C11 | Per-block rotated-domain scales inside LDLQ (drift / overload) + tail-extended codebook (codes G1/B7) | FMT | CPU after the C4 harness | **Pass:** ≥1% proxy at K2. | 0 to −1.5% | 0.15 | Speculative mechanism. |
| C12 | Null-space ridge shrink before the trellis (codes D8, lowbpw B11) | PL | CPU: energy in the bottom eigenspace | Only if >3%. | 0 to −0.8% | 0.15 | OSAQ's gain on QuIP was small. |
| C13 | **Gen-weighted H_in** with bounded weights (no gradients) (hout E5, measure O16, crossarea C4, composable E5) | PL | 0.8B 1 h; 27B own forward + blend | 3 seeds. **Pass:** ≤0.97×. | 0 to −2% | 0.2 | Differs from #33-07 (no gradient weights). |
| C14 | Tempered or blended gradient-weighted H_in, KL-Shampoo H_in (composable E3/E4, hout E7/E8/E11, open B2/B3, crossarea C2/C3, redteam K5) | PL | 0.8B | only after A2/E1 diagnostics | 0 to −1% | 0.1 | Repeats #33-07/#33-10 with small changes. Low. |
| C15 | **H_in factorial diagnosis** (token-level / gen-masked / tempered / domain / ESS) (hout E1/E2, composable E1/E2, open B4/B5, crossarea C1) | DX | 4070 3 h | explanatory | DX | — | Value only if it unlocks C13/C14. |

### D. Codebook and trellis format

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| D1 | **Per-tile codebook salt (1–2 bits per tile), chosen by the block metric** (codes B5/B5b/B6) | FMT | CPU day; side plane 0.008 bpw; trellis ×2–4 | Real LDLQ tiles, int8 cb. **Pass:** best-of-4 ≤0.98 proxy at K2.5 net of bits. | −0.5 to −1.5% | 0.35 | Gaussian-tile result only (×0.974). |
| D2 | **Multi-tile tail-biting rings / chained trellis** (codes A1/A2/A3, crossarea D1/D2, lowbpw H1) | FMT | CPU hours (extend `viterbi_tiles_tb`) | **Pass:** ring-512 recovers ≥30% of the 5.2% free/TB gap at K2 on real tiles. | −1 to −2.5% | 0.2 | ✓ CurveTQ: chained ≈ TB at T=256. #57 measured the gap. |
| D3 | **Gaussianized mul1** (cubic / piecewise map, int8-rounded) (codes B1/B2, lowbpw H3) | FMT (cb flag) | CPU sweep, 33 tensors | **Pass:** ≤0.995 at K2.5–3, no loss at K4. | −0.2 to −0.6% (more ≥5 bpw) | 0.35 | |
| D4 | Longer state L18/20 (codes A4, lowbpw H2, open D9) | FMT | CPU, quant ×4–16 | ≥1% at K2–3 | −0.5 to −1.5% | 0.3 | Ladder time is the cost. |
| D5 | Edge-emitting values from a (16+K)-bit window (codes A5) | FMT | CPU | ≥1% at K2.5 | −0.3 to −1% | 0.25 | CurveTQ: 1.27 PPL at 2 bit, ~0 at 4. |
| D6 | **Row-slab K for the trunk driven by diag H_out:** ceiling first, then sorted sub-MLPs / head groups / q-k-v sections at separate K (lowbpw A10/A12/F1/H4, codes F2, open E1/E2/E5, hout H4–H6, composable D5–D7/L3, crossarea B8/B9, redteam B12) | FMT | ceiling: CPU minutes on 0.8B diagonals; build: 2–4 days | **Pass:** across-block ½log₂(AM/GM) ≥0.1 bit on gate/up (residual between-block part, after within-block feedback), then a 0.8B fake-quant 2-slab mix, 3 seeds ≤0.96× at equal bytes. | −1 to −4% | 0.2 | v3.md §6b dropped this because "output weighting hurt"; #33-12/16 make that reason stale. |
| D7 | Column-split K per k-block (codes F1, open E3) | FMT | CPU | ≥2% proxy | −0.5 to −1.5% | 0.15 | A substitute for ordering. |
| D8 | **Spike spreading** (Householder / 512-point / two-level input Hadamard) and top-1 channel extraction (codes D1–D4/E1, crossarea D8, open E4/E7) | FMT | CPU real trellis | ≥1.5% proxy on attention/GDN inputs | −0.3 to −1.5% | 0.2 | Conflicts with #47's lesson for A8 tensors (spreads the outlier over every activation block). f16-mode tensors only. |
| D9 | Within-tile weighted Viterbi (codes C10, lowbpw A19, crossarea B10, hout J4, open D8) | PL | CPU Γ_b check | — | ≈0 | 0.05 | Codes lens measured Γ_b 0.0001–0.0007 bit. Nearly dead. |

### E. Head and embedding (counted in bpw; the head is ~9–11% of KL at 2.5 with `hd-K3`)

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| E1 | **Low-budget slab heads in knap4 for 2.0/2.5:** budgets 2.25–3.75 in ⅛ steps, K2 floor, mean-row donors, rare-token guard (lowbpw C1/C8/C9) | PL | GPU box (one card) donor quants 1–2 h each ×~5 (≈22 GB host RAM each) + headsim CPU + 1 whole-model eval | knap4 at 2.5 with the new menu. **Pass:** one whole-model swap shows ≥1.5% at z ≤ −3, guard KL ≤1.3×. | −1.5 to −3% | 0.7 | ✓ menu gap; ✓ lcurve (budget 3.0: 0.0060 vs uniform K3 0.0079). |
| E2 | **Two-sided head with the exact softmax-Fisher diagonal** (within-128-row-block feedback); compare with slab sorting, which flattens blocks (lowbpw C2/C10, open A16, crossarea G1/G2, hout I1–I3, codes J3, composable G2, redteam E9) | PL | headsim, 2–3 head quants | **Pass:** headsim −10% head KL at equal bytes vs the best slab plan; then one whole-model swap. | −0.5 to −1.5% | 0.45 | No backward needed. Needs a 128-panel path (dense 248k² impossible). |
| E3 | **Head healing on dumped head inputs vs the golden:** head row scales, final-norm gain, global τ, optional logit bias (heal K1/K2/A4, lowbpw C5/C6, open F11, crossarea G?) | FP | 1 dump eval (~1 h) + minutes on GPU | Fit on qad-pool dumps. **Pass:** whole-model ≤0.985× at z ≤ −3 (fixed trunk codes), multilingual guard ≤1.0. | −0.5 to −2% | 0.5 | Cheapest 27B healing; no S27 stack. |
| E4 | Head statistics from gen rows only (head H_in, Fisher ranking, headsim) (composable G1, lowbpw C11; already in TODO) | PL | hours | headsim, then requant one donor | 0 to −0.5% | 0.4 | |
| E5 | Softmax shift t-grid / Fisher-weighted shift vector (all lenses) | PL | headsim, ~10 head quants | ≥5% head KL | 0 to −0.4% | 0.3 | 2609.31291: Qwen3.5 + GPTQ ≈0. |
| E6 | Low-rank head residual in the C^p metric (ARCHead) (lowbpw C3, crossarea G6, codes J4, open E11) | FMT | CPU SVD first | >5% weighted error in top-16 directions | −0.2 to −0.8% | 0.2 | LDLQ residuals are likely white. |
| E7 | **Embedding with a downstream column metric** E[ggᵀ] (shared across rows, not frequency tiering) (composable G6/A5, lowbpw C7, crossarea G4, hout I6, redteam E10, open E12) | PL | needs A3 accumulator; CPU embtrellis + 1 swap | ≥1% at 2.0 on search + golden-ml guard | −0.3 to −1% (more at 2.0) | 0.3 | Differs from #32/#35. |

### F. Allocation, pricing, search

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| F1 | **Re-price H_out quants:** own family "oursH" fitted from 2–4 uniform-trunk swaps, plus log both proxies (redteam F1/F4/F9/E4, measure O5, hout H8, composable H6) | PL | CPU + the swaps the 27B test runs anyway | CPU check on 0.8B: one-sided proxy ratio bdiag/base per tensor. If >1.05 where KL fell, the fit must switch. | protects 1–3% | 0.7 | ✓ KN_LEVELS reads one-sided `proxy_err`. |
| F2 | **Two-sided (h-weighted) proxy for all ~8,100 options → refit kfit2;** includes ours vs turboderp at K5–8 (measure O4/O6, composable H1/H10/H11, hout H1/H2, open C3, lowbpw A16, crossarea F1/F3, redteam F2/B11) | PL | CPU box CPU ~1 day after the 27B diagonal | **Pass:** grouped-CV log-RMSE ≤0.105 (now 0.118), or pair ranking +1.5 pts beyond bootstrap SD. | −0.5 to −2% | 0.4 | ✓ RAM on Qwen3.8-27B: layer-local objective allocates worse. |
| F3 | **Fisher-predicted absolute ΔKL as a prior** (and the FISL option library) (measure O3/O7/O14/O15/O19, open C1/C2/C12, crossarea F1/F4, composable H2, redteam F3) | PL | 0.8B: 48 group swaps ~1 h, then CPU | **Pass:** Spearman ≥0.8 and log-RMSE ≤0.3 vs measured swaps. | −0.5 to −2% (indirect) | 0.3 | Second order fails at K≤2. |
| F4 | **Re-anchor sensitivities at 2.5 with a fractional-factorial swap design** (measure S1/S2/S6/S11, lowbpw E1, open C6) | PL | the GPU box ~1.5 card-h (32 evals) | **Pass:** kfit2 error on held-out 2.5 measurements −25%. | −1 to −3% | 0.4 | All 256 V2 cal swaps sit on the 4.00 base. |
| F5 | Intra-layer pair interaction terms in the layer DP (measure S3, lowbpw E2, composable H4, crossarea F5, open C2/C5, redteam B6) | PL | CPU on V2 residuals first; ~12 evals | interaction > 2 SE | 0 to −1.5% | 0.2 | 2609.04098: GDN projection errors add in quadrature. |
| F6 | Lazy-greedy measured-KL descent as a stream kind (measure R1, composable H9, lowbpw E5, open C9) | PM | ~10–17 card-h (less with G3/G4) | kind-specific win rate on ver3 | −1 to −3% | 0.3 | ✓ Q-Strata (MoE). |
| F7 | **Cap changes per accepted move (2–8), recenter every ~16 moves** (measure R10/R11, composable H5) | search | free | CPU replay on V2 logs | 0 to −1% | 0.4 | ✓ 2609.09867. |
| F8 | **Posterior-mean selection; Thresholdout for ver3 reads** (measure U1/U6) | search | CPU replay | **Pass:** posterior picks score ≥1% better on ver2 than measured-best picks. | −0.5 to −1.5% | 0.5 | #72: fit-slice picks were ~3% optimistic. |
| F9 | Separate K for gate vs up (per-side rate in the stacked kernel) (lowbpw E3, kernels 44) | FMT | knap4 offline first (CPU minutes) | ≥0.5% predicted | −0.3 to −1% | 0.3 | |
| F10 | GDN decay half-life as a kfit2 feature (lowbpw F2, composable L1, open C11, crossarea F6) | PL | CPU | correlation with fitted GDN sensitivities | ≈0–0.5% | 0.25 | |
| F11 | Tail-aware K1 pricing and per-layer K1 caps (lowbpw E8) | PL | CPU | 2.0 only | 0 at 2.5 | — | Matters at 2.0. |
| F12 | ⅛-step trunk levels K2.125–2.875 (lowbpw E6) | PL | knap4 interpolation first | ≥1% predicted | −0.3 to −1% | 0.2 | Conflicts with #36/#37 ("extra levels don't pay" at higher K). |
| F13 | Category- or tail-weighted aggregation, robust allocation (measure O17/S19, composable H7, crossarea?) | PL | CPU | — | 0 to −0.5% | 0.15 | |

### G. Evaluation statistics and throughput (no KL directly; more search per GPU-hour)

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | Effect | P | Notes |
|---|---|---|---|---|---|---|---|
| G1 | **Seed-aware z, retire g08-search1, new 0.8B search2/verify slices, mark #33-12/13 sub-results unresolved** (redteam A1/A2/A6/A7, heal A8, measure U7) | process | ~1 h | — | protects | 1 | #33-15/16. |
| G2 | **GPU-side top-512 KL + head only on scored rows** (kernels 63/64/74, measure E5; in TODO) | INF | ~1–2 days | KL equal to 1e-7 on 20 candidates; card 1 wall time | card 1 −22%, card 0 −5–10% | 0.8 | Dead ends: card 1 152 vs 119 s from the logits copy. |
| G3 | **Group-sequential early rejection / successive halving** (kernels 69, measure E2/E3/R3, crossarea F8, composable I1/I2, open H3) | search | CPU replay on V2 per-row scores (2–4 h) | **Pass:** ≥30% eval time saved with <2% wrong rejections. | +1.4–2× candidates/h | 0.6 | |
| G4 | Prefix-boundary residual cache / layer-major eval / sibling fan-out (kernels 65/66/71, measure E1/E10) | INF | 2–5 days engine | per-row scores byte-identical to a full eval | +20–40% local search | 0.5 | |
| G5 | Pipelined candidate loading / device-ready blobs (kernels 67/68) | INF | gated by resident-base step 1 (time split) | — | up to +40% | 0.4 | |
| G6 | CUPED / control variates, Neyman-stratified search slices, post-stratification (measure E4/U2/U3/U4, crossarea F9/F10, open H2/H4) | search | CPU replay | **Pass:** SE −20% at equal positions. | indirect | 0.5 | |
| G7 | Last-layer suffix simulator (generalised headsim for L60–63) (measure E7) | PL | 1 dump + torch layer | matches 3 whole-model swaps | exact prices for the L63 spike | 0.4 | |
| G8 | A/A cross-card byte-identity check (measure S9) | DX | 3 evals | — | protects | — | |

### H. Healing / QAD (final pass on shipped 2.0/2.5; library variant if H4 transfers)

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| H1 | **0.8B healing gates:** (a) oracle headroom (overfit all format-free continuous params on the scoring slice); (b) does "last quarter > all layers" reproduce; (c) bf16 vs fp32 healing-gradient cosine per layer (heal A1/A2/A3) | DX | 4070 ~3–4 h | **Rule:** ceiling ≤5% at K3 → defer the S27 stack and do H3/E3. ≥15% → build S27 now. | DX | — | Fixes the order of everything in this theme. |
| H2 | **Scale + norm KL healing with frozen codes** (S27 stack: torch exl3i decode op, golden teacher, metric-exact gen-masked top-512 KL, log-scale trust region, best-held-out checkpoint, gauge fixing) (heal B1/B2/C4/C9–C11/D1/D2/A6/A7, lowbpw D1/D2/D9/D10, open F1/F3/F4/F8–F10, crossarea H1–H5/H8, composable J1/J2/J3/J5/J7/J8/J10, hout L1/L2, codes K1) | FP/TR | 3–5 days engineering + ~3 h/epoch/size on GPU box (both cards) (or rented H100 ≈ $10–20/run) | 0.8B pilot first: ≥5% at K2.5-equivalent, held-out stop. 27B: **pass** ≤0.97× at z ≤ −3 on search (fixed codes), ver3 confirm, multilingual and long guards. | −3 to −10% | 0.55 | ✓ ThinkBit, Tetra. Our base is already refit, so expect the low end. Largest single EV. |
| H3 | **Forward-only asymmetric sequential scale refit** (GPTAQ for scales, SwiGLU-joint, h-weighted) (heal G3/G4/H4, lowbpw D5, crossarea H3) | FP | 1–2 days code + 2–4 h/mix on one card (host-RAM bound) | 0.8B first, fixed codes. **Pass:** ≤0.98× z ≤ −3; held-out token check per layer. | −1 to −4% | 0.35 | No backward. Usable before S27 exists. |
| H4 | **Level → mix transfer:** heal uniform K2/K3, compose a mix, compare with mix-specific healing (heal I1/I5/I6, lowbpw D3) | DX | 4070 2 h | **Pass:** per-level scales keep ≥70% of mix-specific gain → healing becomes a library option. | DX | — | Decides library vs per-mix. |
| H5 | Depth curriculum, per-layer LR, fp32 early-layer backward (heal C1/C2/C3/C5/C7, lowbpw D2) | TR | 4070 | only if H1(b/c) fires | +⅓ of H2's gain | 0.3 | Field report: 22% last-16 vs 6–13% all. |
| H6 | GDN small parameters as compensators: conv1d taps, A_log/dt_bias, bf16 a/b, gated-norm gains (heal B3–B6, open F5, crossarea H8) | TR | 4070 arms on top of H2 | ≥1.5% extra, long-position guard | −0.5 to −2% | 0.3 | Overfitting risk. |
| H7 | Tail-KL window selection (KIC) + importance-sampled batches (heal D4, lowbpw D7, open F2, crossarea H6, composable J4) | TR | in H2 | ≥1% extra | 0 to −1.5% | 0.35 | |
| H8 | Re-rank finalists after healing (heal I4, lowbpw D13, crossarea F11) | FP | 1 night/size | — | 0 to −2% | 0.4 | |
| H9 | Code refresh after healing: scale-conditioned re-encode; PV-lite; STE latent QAD (heal F1/F2/F4, lowbpw D11, open F7, crossarea D6, codes K2/K3, composable J9) | TR | re-encode 35 min + 1 h; STE: rented 100s GPU-h | re-encode first | −1 to −3% beyond H2 | 0.2 | TVQ: ≤0.2 PPL at 8B for 8× cost. |
| H10 | Small residual adapters / spike-tensor LoRA, priced in bytes (heal B11/B12, lowbpw D6) | FMT/TR | after H2 | net of ~0.5% byte price | 0 to −2% net | 0.25 | ✓ QATFactory r4. |
| H11 | Residual-writer biases (closed-form init) (heal B9/B10, lowbpw D12, open E9, crossarea E3) | FMT | small kernel change | ≥1% net of 0.0004 bpw | −0.3 to −1.5% | 0.25 | |
| H12 | Heal with A8 on; tie sibling suh then heal (speed) (heal J1–J3, lowbpw D10, kernels 13) | TR | in H2 | KL-neutral speed | 0 KL, faster prefill | 0.5 | |
| H13 | Long windows / TBPTT in healing + by-position guard (heal E1/E2/A9/D6, lowbpw D8) | TR | in H2 | every position bin ≤1.0× | protects long traces | 0.5 | |

### I. Long context, GDN, numerics

| ID | Idea (lenses) | Fit | HW / cost | Test → pass | E @2.5 | P | Notes |
|---|---|---|---|---|---|---|---|
| I1 | **Mine the 17 existing long-trace evals** for family × position effects (measure L3) | DX | CPU 2 h | A family whose position ratio CI excludes 1 by >25% → run I2. | DX | — | Free; ~35% of public scored positions are long traces. |
| I2 | Long-position group-sensitivity probe; position-weighted objective (measure L1/L2) | PL | ~1 card-h | as I1 | 0 to −3% | 0.25 | Conflicts with #52 (2 docs, whole-mix ratios). |
| I3 | Engine floor and GDN state precision at long positions (state is f32 in the plugin; chunk math bf16) (measure L4, open H1, lowbpw F3, composable I6) | DX | 2 evals | — | ≈0 at 2.5; matters ≥4 bpw | — | |
| I4 | 0.8B long-context H_out gain and cold-start windows (redteam H1/H2, hout B20/A7, measure L5–L7) | DX | 4070 1–2 h | gain at positions >2k ≥⅔ of the gain at <2k | DX | — | |
| I5 | fp64 LDL audit for H_out and massive-activation H_in (crossarea E1) | DX | 30 min | — | insurance | — | |

### J. Kernels and speed (KL 0; the V3 speed goal)

| ID | Idea (kernels unless noted) | Fit | Cost | Test | Effect | P | Notes |
|---|---|---|---|---|---|---|---|
| J1 | **Integer mul1i8 decode** (fp16 and int8 forms; unsigned operand; hash ISA choice) (1–4) | kernel | hours | ntbench M=1/16, output-hash equality | decode +15–40% vs current exl3i; wdec VALU ×0.5 | 0.7 | ✓ closed form exact; ✓ the current path rounds through f32. |
| J2 | Split-K at decode for N=5120 (32) | kernel | hours | ntbench M=1, sk 1/2/4 | decode +5–10% | 0.6 | ✓ sk=1. |
| J3 | Fused rotation+int8 prep; down's int8 input from the gate_up epilogue; gated-norm fusion (9–12, 46, 48) | kernel | days | profile-ops | prefill −5–12% | 0.5 | |
| J4 | T2 chunk sized to the 64 MB Infinity Cache (22) | kernel | 30 min | chunk sweep | prefill −2–5% | 0.5 | ✓ 192 MiB. |
| J5 | Cheaper g128 fold (magic bias, packed f32), static exponent groups (14–17) | kernel | 1–2 days each | wmmagrp; CPU aqproxy for groups | A8 GEMM +5–15% | 0.4 | |
| J6 | Decode profile → per-token overhead fusions (33, 35) | kernel/engine | 1 h profile, then days | profile-ops M=1 | decode +20–60% (very uncertain) | 0.4 | |
| J7 | DFlash2 port + quantized drafter; MTP via spechub (54–60) | serving | 1–2 weeks | acceptance, tok/s | decode 3–4× | 0.6 | |
| J8 | Real-prefill time table including prep / wdec / fold / epilogue (75) | solver input | card-hours | — | correct speed choices | 0.8 | |
| J9 | Tie sibling suh at the requant (13) | PL | CPU proxy | tie cost ≤0.3% proxy | prefill −1–2% | 0.5 | #42. |

### K. Process items (all lenses, mainly redteam)
- **K1.** Write the 27B swap-test pass rule now, including the A7 seed term. Use bdiag + B1 refit at K3 and K2.5 against ladder levels of the same settings.
- **K2.** Update v3.md §2/§6b/§7: they still say YAQA was "dropped" and output weighting "hurt in every form".
- **K3.** Ledger entries for qad-pool use by the 27B collection and for healing. Reserve one long qad trace, never trained, as a final long check.
- **K4.** Requant scope: run the H_out decision (B3) before rebuilding the lost K3.5/K4 ladder levels (#58).

---

## 3. Top 25 (expected gain × probability ÷ cost)

| # | ID | Idea | Fit | Main cost | E @2.5 | P | Why here |
|---|---|---|---|---|---|---|---|
| 1 | A1 | Gen-masked Fisher loss | PL | 4070 2.5 h | 0 to −2% | 0.3 | Must be decided before the 8.5 h night; zero marginal cost after that. |
| 2 | A2 | Estimator check (‖x‖²-weighted vs Sketch B vs Σdy² vs leverage) | DX | 4070 2 h | — | — | The one result that most changes P(27B success). |
| 3 | A3 | Instrument the 27B run (extra sums, traces, embedding/residual gradients) | INF | 1 day code | enabler | 0.9 | Saves a second fp32 night for E7, F2, F3, B2, B5. |
| 4 | E1 | Low-budget slab heads at 2.0/2.5 | PL | ~1 day | −1.5 to −3% | 0.7 | Verified menu gap; uses existing machinery. |
| 5 | F1 | Re-price H_out quants in knap4 | PL | CPU ½ day | protects 1–3% | 0.7 | Verified: the solver reads the one-sided proxy. |
| 6 | B1 | H_out-aware, undamped, multi-round `refit_scales` | PL | ~2 h | −0.3 to −1.5% | 0.5 | Free; fixed-code test; joins the 27B swap test. |
| 7 | A5 | Dose < 1K + early-decision amendment | INF | 4070 1.5 h | saves ≤8 card-h | 0.5 | |
| 8 | A6 | 27B 64-seq pilot + GM/AM predictor + `ldlq_2hess` timing | DX | GPU box (both cards) 35 min | — | — | Go/no-go before the full night. |
| 9 | A7 | 27B quant-noise floor | DX | GPU box (one card) 1–3 h | — | — | Needed to read every 27B A/B. |
| 10 | H1 | 0.8B healing oracle / depth / gradient-precision gates | DX | 4070 3–4 h | — | — | Decides whether to build the largest-EV item. |
| 11 | E3 | Head healing on dumped inputs (+τ, final-norm gain) | FP | 1 eval + minutes | −0.5 to −2% | 0.5 | 27B healing without the S27 stack. |
| 12 | C9 | KL-optimal svh multipliers per level/family | PL/FP | 30 min (fixed codes) | 0 to −1.5% | 0.3 | Zero bytes; noise-free test. |
| 13 | I1 | Mine the 17 long-trace evals | DX | CPU 2 h | — | — | Free; long traces carry a large share of the score. |
| 14 | F7+F8 | Cap moves + posterior-mean selection | search | CPU replay | −0.5 to −1.5% | 0.45 | |
| 15 | E2 | Two-sided head with the exact Fisher diagonal | PL | headsim | −0.5 to −1.5% | 0.45 | |
| 16 | B3 | Attribution by family / depth / K (2.5, 4, 5) | DX | 4070 4 h | — | — | Halves Phase-B scope or shows ≥3.5 bpw gains nothing. |
| 17 | H2 | Scale + norm KL healing (S27) | FP/TR | ~1 week + 3 h/size | −3 to −10% | 0.55 | Largest EV; ranked here only for cost and the H1 gate. |
| 18 | C3 | Hessian-aware input-scale exponent | PL | 4070 1.5 h | 0 to −3% | 0.25 | |
| 19 | F4 | Re-anchored factorial swaps at 2.5 | PL | the GPU box 1.5 card-h | −1 to −3% | 0.4 | |
| 20 | G2+G3 | GPU-side KL scoring + early-rejection racing | INF | 2–3 days | ~1.5–2× search/h | 0.6 | Converts to KL at ≤2.5 bpw, where V2's search had not flattened. |
| 21 | C1 | Activation-denoising LDLQ | PL | 4070 3 h + 27B night | −1 to −3% | 0.35 | ✓ numbers; mix-independent. |
| 22 | D6 | Diag-H_out row-slab ceiling (build only if ≥0.1 bit) | DX→FMT | CPU minutes | −1 to −4% | 0.2 | Cheap ceiling; reopens a dropped direction on new evidence. |
| 23 | C2 | Tile coordinate descent with split-half accept | PL | CPU box day + quant ×2–3 | −0.5 to −2.5% | 0.3 | |
| 24 | H3 | Forward-only asymmetric scale refit | FP | 2 days + 2–4 h/mix | −1 to −4% | 0.35 | |
| 25 | A9 | Analytic SwiGLU diagonal (replaces gate/up backward) | PL | 4070 1.5 h | cost cut | 0.35 | |

Just outside the top 25: D1 per-tile salt, B2 tempering (needs 3 seeds), C5 quantized-prefix targets, F2 kfit2 on the two-sided proxy (waits for the 27B diagonal), E7 embedding metric.

**Speed-track top 4 (KL 0):** J1 integer decode, J2 decode split-K, J4 T2 chunk size, J3 fused prep.

## 4. Cheapest discriminating diagnostics

1. **Estimator arm on the 0.8B (A2)** — 4070, ~2 h, 3 seeds.
   - If the ‖x‖²-weighted diagonal also wins, the old 27B negatives are unexplained by the estimator. Then do a 27B pilot swap (A6) before any full collection.
   - If it loses, the plan is consistent, and A1/A3/A5 are just tuning.
2. **Gen-masked vs all-position diagonal (A1)** — 4070, ~2.5 h. Sets the 27B collector loss.
3. **Dose at 128/256/512 (A5)** — 4070, ~1.5 h. Decides whether the 27B night can be about ¼ as long.
4. **27B pilot plus timing (A6)** — 35 min on both cards + 15 min on one. The GM/AM predictor (validated on 0.8B rowerr) forecasts the 27B gain; `ldlq_2hess` timing sets whether B10 is needed.
5. **Fixed-code refit test (B1) on the existing 0.8B bdiag quants** — minutes. No rounding noise, so a clean yes/no for a free improvement that also enters the 27B swap.
6. **Healing oracle headroom + depth replication (H1)** — 4070, ~3–4 h. Decides whether H2 (largest EV) is built now or after H3/E3.
7. **Head battery in headsim** (E1 low-budget slab curve, E2 Fisher two-sided, E5 shift grid) — a few donor quants + CPU. Cheapest exact-to-score wins.
8. **Diag-H_out across-block ceiling (D6) plus the within-block predictor (A6)** — CPU box, minutes, on the 0.8B 8K diagonals. Decides the whole row-slab / sorted-MLP / permutation cluster (D6, B6, B7).
9. **CPU replay on V2 per-row scores** (G3 racing, F8 posterior-mean, G6 variance reduction) — CPU box, 2–4 h. Sizes all search-efficiency ideas.
10. **27B quant-noise requant (A7)** — one card, 1–3 h. Without it no 27B pass rule is interpretable.

## 5. Rejected (with reason)

- **Identity H_out / "sv-aware two-sided"** (composable C1/M1, redteam B3/C6): measured worse in #33-16 (K3 +4.9%, K2 +2.1%).
- **Best-of-N rotation seeds** (#41; #33-16 spread 1–3.5% is mostly noise on the picking slice).
- **Within-tile D-weighted Viterbi after RHT:** CurveTQ (Γ_b median 0.0001 bit) plus the codes lens's 0.0001–0.0007 bit. Inert.
- **Stored full start state per tile:** 0.055 bpw at K2 buys less than the same bits spent on K. Rings (D2) are the only termination idea kept, at low P.
- **Trace-optimal joint rotation + feedback (GMD):** loses 16/16 to RHT + feedback (2610.18139, verified).
- **Learned rotations (HARP, SpinQuant-style):** ≤0.7% on QTIP 3-bit for +0.05 bpw; #47 also argues against residual-stream rotations for A8.
- **E8 / Leech / lattice codebooks, trained LUTs, entropy coding:** the trellis already beats E8P at equal rate; lattice decode is slower; the trellis stream is fixed-rate and near-uniform.
- **Gradient-weighted H_in as a replacement** (#33-07, #33-10): only bounded-weight variants (C13) and the diagnosis (C15) remain, at low priority.
- **On-policy / reverse-KL distillation as the objective:** the score is teacher-forced forward KL; matched off-policy forward KL is right. A mixed FKL/RKL loss may be tried only as an optimiser aid, judged by forward KL.
- **RL / Shapley / QUBO allocators:** cost, and no evidence over measured-sensitivity knapsack plus search (#16, WINDQuant 49 h at 8B).
- **GDN recurrent-state or KV-cache quantization for the board:** state is f32, KV must stay bf16. Serving only.
- **A4 / W4A4, FP8 activations, 2:4 activation sparsity, A16 via two int8 passes:** #46, plus the kernels lens's own rejections.
- **Low-bpw self-drafting while decode is VALU-bound:** a full step per draft.
- **Residual ladders (MosaicQuant / GateRQ-style) on the trunk:** 0.3–0.9 dB worse than a direct code at equal rate.
- **Per-group scales (#40), V=2 trellis, per-tile K maps, row-chained trellis across k-blocks:** measured or argued dominated.

## 6. Critic's overall read

- **Before the 27B collection, spend about a day of 4070 time** on A1, A2 and A5, and add the A3 instrumentation and A8 asserts.
  - The A2 result alone could move P(27B success) from about 0.5 to 0.3 or 0.7.
  - A5 can cut the night by up to 4×.
- **Then run the 27B swap test with B1 applied** and a pass rule that includes A7's seed noise.
- **In parallel, at low risk:** the head (E1, E3, E2) and the solver fixes (F1, F7/F8). Head plus solver fixes are probably worth −2 to −4% at 2.5 bpw regardless of H_out.
- **Largest single lever: frozen-code scale/norm KL healing (H2).** Its case is better than its rank suggests. Gate it with H1 in one afternoon, and consider a rented H100 for development so it doesn't compete with the ladder for the GPU box.
- **Proxy test design (follows from #33-16):**
  - Prefer fixed-code tests, which carry no rounding noise.
  - Otherwise use CPU proxy tests on real 27B tensors with held-out Hessian halves.
  - Otherwise use ≥3 seeds.
  - Do not trust any single 0.8B quant pair under ~3%.
