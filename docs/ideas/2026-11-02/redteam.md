# Red-team report: H_out on the 27B, the 0.8B evidence and the V3 plan

**Bottom line:** I found five concrete risks. Three have cheap checks that should run before the ~8.5 h two-card collection.

1. **The 27B failures and the 0.8B win used different estimators.** This is the most likely way to waste the night.
2. **The 1K / fp32 collection settings rest on noise-level evidence.** The run could probably be 2 h instead of 8.5 h, but no test has looked below 1K sequences.
3. **The knapsack would misprice H_out quants.** It reads a one-sided proxy that H_out deliberately makes worse.
4. **Two code paths ignore H_out.** The scale refit sums all output rows equally, and two-sided LDLQ does far more work than a diagonal H_out needs, which matters for a full Phase-B requant.
5. **The 0.8B proxy was never calibrated,** and its K2/K3 error level is far from the 27B's.

**What I checked on disk** (read-only; the ideas below rest on these):

- **#33-15 (logged after the attached LOG).** Quant-to-quant spread is about ±2–3% at K2 and ±2% at K3, with |z| up to 5, and the paired per-document SE does not include it. #33-16 (3 seeds, plus `bident` with H_out = I) is prepared but not run.
- **`prepare_H_out`** (`exllamav3-frac/.../quantize.py:780`) turns H_out into diag(sv)·H_out·diag(sv), applies the block Hadamard on both sides, mean-normalises, adds sigma_reg_out and takes a block-16 LDL.
  - With a diagonal H_out, the rotated factor is exactly block-diagonal in 128-row blocks.
  - So "diagonal H_out" really means: error feedback across rows inside each 128-row output block, weighted by h·sv².
- **`ldlq_2hess`** (`:710`) runs the dense YAQA anti-diagonal schedule. Its F-update uses `Ln[cols][:, :, :n_hi]` and a full-width `addmm`, so the zeros of a block-diagonal L_out are not exploited.
- **`refit_scales` runs after `ldlq_2hess` with H_in only.** Its input-side scale solve sums all output columns equally. The `proxy_err` it writes to proxy.jsonl is the one-sided error after refit (it overwrites the two-sided proxy).
- **`knap4.py` KN_LEVELS** prices measured levels from that `proxy_err`, using the family factor of the lower integer neighbour.
- **`ourq.py --hout` skips tensors without a file silently** (`os.path.exists`).
- **The 27B gates (2b/2c/2d) used a different estimator, Σ_t‖x_t‖²·dy_t·dy_tᵀ.** It was built either from sampled rows or from every token. The new collector uses the Sketch-B row norms ‖G_b[r,:]‖². The 0.8B never tested the 27B estimator: yaqa08.py computed it but was stopped at 525 sequences and the data was never used.
- **Gate 2d also used a different mechanism:** `sensitivity_out_scales` through `--hout-diag`, not two-sided LDLQ.
- **`ladder2.sh` and `q2b.sh` both use a fixed `--sigma-reg-in`**, not the per-tensor sweep.

Gains below are "KL % at 2.5 bpw". For pure checks, the gain is "0 directly" and the value is the mistake it prevents.

---

## A. How solid the 0.8B evidence is

**A1. Put the seed noise into every z.** Report z_eff = diff / √(se_doc² + 2σ_seed²), with σ_seed taken from #33-16, and refuse decisions with z_eff < 3.
- *Source:* #33-15; Tetra (2609.35465), where encoder re-draws alone spread MMLU with SD 2.9 points.
- *Why:* a −8% gain against σ_seed ≈ 2.5% gives z_eff ≈ 2.3–4, not the −12 to −13 reported.
- *Fit:* process. *Test:* pcmp change, CPU, minutes.
- *Gain/risk:* 0 directly; stops noise-level conclusions from steering 27B plans.
- *Conflicts:* every z in #33-10…#33-15.

**A2. Retire g08-search1 as a decision slice.** It has been read by nine amendments (3–9), and every winner was chosen on it. Build g08-search2 (search pool, new seed) and a small 0.8B verify slice, then rescore base vs bdiag with 3 seeds each.
- *Source:* new; same logic as V2's ver1/ver2 split, where best-of-many fit-slice picks were ~3% optimistic (#72).
- *Why:* winner's curse across arms and amendments.
- *Fit:* process.
- *Test:* mkgold ≈ 15 min, 12 evals × 23 s on the 4070. Passes if the sign holds and the gain keeps ≥ 2/3 of its size.
- *Gain/risk:* 0; it guards the whole programme.
- *Conflicts:* #33-10…#33-16.

**A3. Break the gain down by category** (code / chat / agentic / math / multilingual), using the per-document deltas pcmp already has.
- *Source:* new.
- *Why:* the private corpus is token-weighted and heavy in long agentic traces. If the gain is about 0 on agentic or multilingual, the board gain shrinks.
- *Fit:* diagnostic. *Test:* free, CPU.
- *Gain/risk:* 0. A fired check would mean a smaller real gain than the headline.
- *Conflicts:* none.

**A4. Break the gain down by position** (0–256, 256–1k, 1k–2k) from per-row scores.
- *Why:* 27B KLD doubles past 32k (#52). If the H_out gain falls with position, long traces get less of it.
- *Test:* free.
- *Conflicts:* #52 rests on only 2 documents.

**A5. Check the tails.** Compare per-document p99 KL, max KL and top-1 agreement, base vs bdiag.
- *Source:* FidelityQ (2610.14934): the worst decile of tokens carries 45% of KL.
- *Why:* H_out moves error into rows that are insensitive *on average*; rare-context tokens could get worse.
- *Test:* free. Fires if p99 worsens beyond seed noise.

**A6. Check whether base seed 0 was an unlucky draw.** Every arm since #33-10 was paired against one `base` quant. If that draw was 2–3% bad, every gain is inflated by the same amount.
- *Test:* covered by #33-16. Additionally report the rank of base_s0 among the seeds.

**A7. Mark the secondary 0.8B conclusions as unresolved in INDEX.** "Diagonal ≈ full" (K3 +1.9%, z 2.4), "1K is enough" (85% / 98% / 99% / 84% shares, not monotone) and "full gate/up H_out hurts at K2" are all inside the #33-15 noise band.
- *Fit:* process.
- *Why:* "diagonal only at 1K" is fine as a *cost* decision. It should not be treated as a measured fact or used as a 27B gain estimate.

**A8. Write the 27B pass rule with a 27B seed term in it** (this needs J1). Otherwise a ±2% rounding draw decides V3.

---

## B. What the gain actually is (mechanism and attribution)

**B1. A within-block AM/GM predictor.**
- **Mechanism.** In the rotated basis, a diagonal H_out becomes Had·diag(h·sv²)·Had inside each 128-row block. Without feedback the weighted error is set by the arithmetic mean (AM) of those weights. Sequential feedback can at best reach their geometric mean (GM) — the high-rate LDL bound, det^{1/n}.
- **Prediction.** The per-tensor reduction in output-weighted error should be about 1 − GM/AM of h·sv² within each block. GDN `in_proj_z` (CV 3.96) should gain a lot; residual writers (CV ≈ 0.2) should gain about nothing.
- **Use.** Correlate the prediction with per-tensor rowerr changes on the 0.8B. Then apply the same predictor to a 27B pilot (C5) to forecast the 27B gain.
- *Source:* QuIP/LDLQ tr(D) bound; 2608.25188 (AM of Cholesky pivots vs geometric mean); E3c's AM/GM machinery.
- *Fit:* diagonal predictor.
- *Test:* CPU, 1–2 h, using the existing `dv8192` diagonals, svh from the quant tensors, and rowerr.npz.
- *Gain/risk:* 0 directly; it turns the 27B go/no-go from a guess into a forecast.
- *Conflicts:* none.

**B2. Shuffle control.** Permute the hout_diag entries at random, once within each 128-block and once across the whole tensor. The distribution stays the same; the row identities are wrong.
- *Why:* if the gain survives, it comes from heterogeneous weighting acting as a regulariser, not from gradient information. That would also make the 27B collection pointless.
- *Test:* 2 arms × K3/K2, ≈ 20 min on the 4070.
- *Risk:* none.
- *Conflicts:* none.

**B3. Read `bident` as a possible fix to the one-sided objective.**
- The one-sided LDLQ in the rotated basis assumes all rotated rows are equally weighted. The true objective (and hproxy) weights original rows by sv². H_out = I gives the correctly specified one-sided quantizer.
- If #33-16 shows bident ≥ 1/3 of the gain, adopt it in the ladder right away (no backward pass needed) and re-run #48 (output scales off) under bident.
- *Fit:* per-level.
- *Test:* in flight.
- *Gain:* 0 to −3%.
- *Conflicts:* #48, whose +2.5% (z 2.1) is itself within 27B quant noise.

**B4. Which tensor families carry the gain.** Apply H_out only to one family at a time: GDN in_proj_qkv/z; attention q/k/v; MLP gate/up; residual writers (o_proj / out_proj / down).
- *Why:* tells which families must be requantized at 27B and lets B1 be checked per family.
- *Fit:* scope cut for Phase B.
- *Test:* 4 arms × 2 K × 2 seeds ≈ 1.5 h on the 4070.
- *Expected:* gate/up and the input projections carry it; residual writers do not.

**B5. Which depth carries the gain.** H_out on layers 0–7, 8–15 or 16–23 only.
- *Why:* if late layers dominate, the 27B collector can cut the backward pass short. Gate 2c cut it short at layer 22 and went from 30 to 22 s per sequence. Collecting only layers ≥ 32 roughly halves the backward.
- *Note:* #33-09 shows the early-layer full H_out is noisy (halves cos 0.69).
- *Test:* 3 arms × 2 K × 2 seeds ≈ 1 h.

**B6. Is the gain additive?** Put H_out on a random half A of the tensors, on the complementary half B, and on all. Compare KL(A) + KL(B) − KL(base) with KL(all).
- *Source:* 2607.12266 (weight-only isolate-and-sum error ≈ 0.10; isolated sensitivities mispriced by 71–376% on the activation channel); 2609.09867 (composition error 25.9–41.7× the pricing error).
- *Why:* the knapsack will mix H_out tensors with per-tensor pricing.
- *Test:* 3 arms × 2 seeds at K3, ≈ 40 min.
- *Conflicts:* the additivity assumption behind #69.

**B7. Tempered diagonal h^p,** p ∈ {0.25, 0.5, 1, 1.5}.
- *Source:* ShamAN-Q (2609.38521): tempering with p = ½ beat p = 1 when the error is large (Qwen3-1.7B PPL 19.21 → 17.46). Here the factor is a diagonal and the quantizer is a trellis.
- *Why:* K2 on the 0.8B is a large-error regime. The full gate/up H_out "hurting at K2" may be over-trust in curvature.
- *Fit:* per-level knob.
- *Test:* 4 arms × 2 K × 2 seeds ≈ 1 h.
- *Gain:* maybe −1 to −2% on top of the diagonal.

**B8. sigma_reg_out sweep** {0.003, 0.01, 0.03, 0.1}.
- *Why:* 0.01 is the paper's value for a *full* H_out. After mean-normalisation, a diagonal with CV 4 has a mean set by a few rows, so the effective damping on most rows depends on the spread, and the 27B spread will differ.
- *Test:* ≈ 40 min.
- *Conflicts:* 27B gate 2b, where more damping (0.3) helped. This does not mean more damping helps here.

**B9. Estimator test — the conflict that could waste the night.**
- Collect three diagonals in one pass on the same 1K 0.8B windows:
  - (a) Sketch-B ‖G_b[r,:]‖² (the current one);
  - (b) Σ_t‖x_t‖²·dy_{t,r}², the estimator of 27B gates 2b/2c/2d;
  - (c) Σ_t dy_{t,r}², the output-feature factor E[G_Zᵀ G_Z] of H_Z-PTQ.
- Quantize with each through the same `--hout` path.
- **How to read it:**
  - If (b) loses, the 27B negatives are explained (massive-activation tokens dominate ‖x‖² weights).
  - If (b) also wins on the 0.8B, the 27B negatives are *not* estimator-driven, the proxy fails to reproduce a known 27B result, and P(27B failure) rises sharply.
- *Source:* H_Z-PTQ (2610.40345): YAQA's factor was unstable (Llama-3.2-1B W4 per-channel PPL 19.12 vs 14.26 for E[G_Zᵀ G_Z]); ShamAN-Q: quartic-in-activation token weights are dominated by massive activations.
- *Test:* ≈ 30 min collection plus 6 quants × 2 seeds, ≈ 1.5 h on the 4070.
- *Conflicts:* #33-04, #33-05, #33-06.

**B10. Mechanism test.** Feed the Sketch-B diagonal through `--hout-diag` (sensitivity_out_scales, gate 2d's path) instead of two-sided LDLQ.
- If the out-scales path also wins on the 0.8B, gate 2d's 27B failure (K2 +10–15%, K3 +4%) is a transfer failure, not a wrong mechanism.
- *Test:* 2 K × 2 seeds, 20 min.
- *Conflicts:* #33-06.

**B11. Which proxy tracks KL.** Weight the per-row errors in rowerr.npz by h. Over the arms base / bident / bdiag / bhout / shuffled, does this h-weighted proxy rank arms like KL? Does the one-sided proxy rank them backwards?
- *Why:* this validates F2. It is the 0.8B version of "the input-side proxy cannot see output directions".
- *Test:* CPU, free.

**B12. Heterogeneity between blocks that LDLQ cannot use.** Feedback only works inside 128-row blocks, so differences between block means of h are left untouched.
- Compute the ceiling ½·log₂(AM/GM of block means) per tensor, as E3c did for the input side.
- If it reaches ≥ 0.05–0.1 bit on gate/up or z, row-slab K for the trunk is back on. v3.md §6b dropped it because "output-side weighting hurt in every form", and that is now stale.
- *Fit:* format change.
- *Test:* CPU on 0.8B diagonals now, on 27B diagonals later.
- *Gain:* unknown, at most a few %.
- *Conflicts:* #29-02 (E3c, input side only).

---

## C. Transfer from 0.8B to 27B

**C1. Match the error level, not the K.** On the 0.8B, base KL is 0.099 at K3 and 0.535 at K2 (top-1 about 69%: barely working). The 27B 2.5–3 bpw mixes run most tensors at much lower relative error. Run base vs bdiag at K4, K5 and K2.5 on the 0.8B.
- *Why:* if the gain drops to ≤ 3% at K4, the 27B mix gain comes mostly from its K2–K3 tensors and is diluted.
- *Test:* ≈ 30 min plus seeds.
- *Gain/risk:* 0; it resets the expected gain.

**C2. Width trend on the 2B.** Collect a bf16-GEMM diagonal for the 2B; #33-15 showed the bf16 collector matches fp32 by KL (0.982 / 1.005). Run base vs bdiag, 2 seeds, K3 and K2.
- *Why:* two width points (1024 → 2048) give a trend to extrapolate to 5120.
- *Test:* 3–5 h on the 4070, if the 2B backward fits in 12 GB with a bf16 forward.
- *Conflicts:* #60-04.

**C3. Run the calibration study the proxy never passed.** #60-04 was stopped, and the 0.8B was never in it. Run the 4 pre-written tests on the 2B, plus an 0.8B column. Tests 3 (output scales) and 4 (scaled Hessian) matter most, since bhout "stacks" with the scaled Hessian.
- *Test:* ≈ 1 h on the 4070.
- *Gain/risk:* 0. A failed calibration would downgrade all of #33-08…#33-16.

**C4. Reproduce a 27B negative on the 0.8B.** Use the gate-2c protocol: full-token Σ‖x‖²·dy·dyᵀ H_out, two-sided, sigma_out 0.025 / 0.3. If the 0.8B reproduces the 27B loss, the proxy is calibrated for exactly this question.
- *Test:* ≈ 45 min (overlaps B9).

**C5. A 64-sequence 27B pilot of the diagonal** (≈ 35 min with both cards).
- Measure per-family CV, split-half cosine, the B1 predicted gain, and the max/median h of residual writers on massive-activation channels.
- **Go/no-go:** predicted gain ≥ half of the 0.8B's prediction, and split-half cos ≥ 0.98 at 64 sequences.
- *Fit:* diagonal.
- *Gain/risk:* 0; a low prediction could avoid ~8 h.

**C6. 27B `bident`, whole trunk at K3** — no collection, one card.
- Use one quant from ladder2 settings plus H_out = I, and pair it against ladder K3.
- This is the first 27B evidence on two-sided machinery. It also tests ldlq_2hess time and memory at 27B scale (see E2).
- *Test:* one 27B quant (35 min–3 h) plus one eval.
- *Gain:* 0 to −3%. If #33-16 shows a large bident share, this may be most of the gain.

**C7. Do the dilution arithmetic before forecasting.** The 0.8B ran a bf16, tied head and embedding, so its numbers are trunk-only. At 27B 2.5 bpw, the head is about 5–7% of KLD and the embedding a few %. A −8% trunk gain becomes ≤ −7% overall, before any width or error-level loss.
- *Test:* CPU, using headsim and the embedding swaps.

**C8. Residual writers and massive channels.** On the 27B, rows carrying the massive-activation channel may have extreme h. Mean-normalisation inside `prepare_H_out` then pushes all other rows close to sigma_reg_out.
- *Test:* check in C5. If max/mean > 100, cap or temper (B7) before quantizing.

**C9. Compare the two model generations (3.5 vs 3.8).** Compare H_in eigen-spread, out-scale CV and massive-activation size between the 0.8B and the 27B, from the existing Hessians. Large differences would discount the proxy evidence.
- *Test:* CPU, 1 h.

---

## D. Keeping the 8.5 h run from being wasted

**D1. Dose below 1K on the 0.8B:** 128, 256 and 512 windows, 2 seeds.
- Only 1K–8K has been measured. If 256 keeps ≥ 80% of the gain, the 27B run takes about 2.1 h.
- *Test:* ≈ 1 h on the 4070.
- *Gain:* frees about 6 h of both cards.

**D2. bf16-GEMM collector at 27B,** in an A/B during the gate (64 sequences each).
- Accept if per-layer relative error on the diagonal is within the 0.8B bf16 band (median ≤ 5e-2) and cos ≥ 0.99. The 0.8B KL test could not separate bf16 from fp32.
- *Caveat:* the old 27B "bf16 unusable" result had bf16 *activations*; this keeps everything except the GEMMs in fp32.
- *Gain:* about 3× faster.
- *Conflicts:* #33-03, #33-14, #33-15.

**D3. Cut the backward short** if B5 shows late layers dominate.

**D4. Early kill switch.**
- STOP at the 256-sequence snapshot (≈ 2.1 h).
- Requant the K3 trunk from it, run the paired swap with 2 seeds, and resume (the collector is resumable) only if the gain clears the pre-written bar.
- Cost of resuming: ≈ 15 min of Triton warm-up.

**D5. Record more in the same run.** It costs almost nothing extra:
- even/odd halves;
- per-32-sequence partial sums, for post-hoc bootstrap and dose;
- the alternative estimators (b) and (c) from B9;
- per-category accumulators and position buckets (0–255 vs the rest);
- E[g·gᵀ] at the embedding output (5120², 100 MB, used by E10);
- per-tensor scalars tr(H_out) and E‖G_b‖² (used by F3).

This turns one expensive run into many cheap analyses.

**D6. Fail loudly on missing keys.**
- After collection, assert that the file count equals the number of trunk linears.
- In `ourq`, make a requested `--hout` key that is missing an error, not a silent skip.
- Then check that the `hout: true` count in proxy.jsonl equals the expected count.

**D7. Ledger.** Log the qad-pool window use (non-golden) and the xcorp 0.8B collections before the run.

**D8. Order the work.** If H_out is adopted, every level has to be requantized. Do not rebuild the lost one-sided K3.5/K4 ladder levels before the H_out decision.

---

## E. Quantizer implementation

**E1. Panel-decomposed `ldlq_2hess` for a diagonal H_out (exact).**
- L_out is block-diagonal in 128-row blocks, so the 128-wide output panels are independent of each other.
- Batch all panels and run the anti-diagonal schedule inside each panel. The F-update shrinks by about n/128 (136× for gate/up).
- *Source:* FastKron (2608.06291) reports 23,084 ms for the dense anti-diagonal schedule vs 250 ms recursive at n = 8192.
- *Fit:* per-level tooling.
- *Test:* code ≈ 2–3 h. Gate on KL-identity, not byte-identity: fp32 summation order changes.
- *Gain:* 0 KL; protects the Phase-B schedule.

**E2. Measure 27B cost first.** Time and memory of `ldlq_2hess` on one 27B gate/up and one down at K3:
- the dense `Ho` is 17408² fp32 = 1.2 GB, plus LDL temporaries;
- about 1,400 anti-diagonal launches per tensor;
- two ourq processes per card may OOM.
- *Test:* 15 min on card 0.

**E3. H_out-aware `refit_scales`.** In the input-side solve, weight the output columns by h:
A = (Q·diag(h)·Qᵀ) ∘ H, b = rowsum(Q ∘ HW ∘ h). The per-output-column refit does not change. Add a no-refit arm.
- *Source:* ThinkBit (2610.34886): KL-relevant scales correlate only 0.07 with weight-space MSE-optimal ones.
- *Fit:* per-level, free at inference.
- *Test:* ≈ 1 h code plus 20 min on the 0.8B.
- *Gain:* −0.5 to −1.5%.

**E4. Log both proxies** (one-sided and h-weighted) per tensor in proxy.jsonl. KN_LEVELS currently reads a number whose meaning will change silently.

**E5. Select by KL or the h-weighted proxy, never the one-sided one,** whenever a sweep is combined with H_out (sigma-in-sweep, gain, seeds). The sweep keeps H_out in `qs` but picks by one-sided `rel`, which systematically favours settings that undo the output shaping.
- *Source:* DRQ (2610.11226): lowering layer reconstruction loss raised PPL in 73.8% (AWQ) and 47.1% (GPTQ) of refinement cases.

**E6. Check `g_scale` under H_out.** The global scale search runs on unweighted tiles. Sweep the gain ±10% with H_out on the 0.8B.
- *Gain:* ≤ 1%.
- *Conflicts:* #39 (never logged).

**E7. Re-tune input damping under H_out:** sigma_reg_in ∈ {0.001, 0.005, 0.025}.
- *Test:* ≈ 30 min.
- *Conflicts:* #38 (per-tensor winners agree only about 53%).

**E8. Fractional and low K through `ldlq_2hess` + qtf2:** K2.5, K2.25 and K1.5 on the 0.8B — correctness first, then gain.
- *Why:* the ladder is mostly fractional levels, and only K2/K3 have been tested.
- *Test:* ≈ 20 min.

**E9. Head with the exact Fisher diagonal.** Σ p(1−p) is already computed for the slab head; no backward is needed. It needs the panel path from E1, since a dense 248k² matrix is impossible.
- **Interplay:** the slab head sorts rows by Fisher (`head_perm`), which makes 128-blocks homogeneous and kills the feedback gain.
- *Test:* in headsim, compare uniform half-step head + H_out, slab head, and slab head with interleaved rows inside each slab.
- *Source:* SoftWater (2608.12026), but fixed-rate here; 2609.31291 found a near-zero gain on Qwen3.5 heads with GPTQ.
- *Gain:* −0.3 to −1% total.

**E10. An output-side metric for the embedding.** Use E[g·gᵀ] at the embedding output (5120², from D5) as the trellis metric per dimension instead of H = I. It weights dimensions, not tokens, so it does not starve rare or multilingual rows the way vocabulary tiering did (#32).
- *Source:* OPTIC (2610.31309): a path-weighted metric beat Euclidean, 92.8 → 94.4% retention.
- *Test:* embtrellis CPU proxy, then one swap.
- *Gain:* −0.5 to −1% at ≤ 2.5 bpw.

**E11. Do not D-weight the tile Viterbi.**
- *Source:* XOR-Trellis and CurveTQ: after the randomized Hadamard, within-block anisotropy Γ_b drops from 0.038 to 0.00013 bit, so the weighting does nothing.
- *Fit:* negative prior; nothing to test.

**E12. Interleave output rows across blocks (speculative).** Permute gate/up output rows (folded into down's input columns) so every 128-block spans the full range of h. This maximises within-block AM/GM.
- *Gain:* bounded by B12. Low priority.

**E13. h-weighted A8 activation cost.** E1c's `rel_out` treats output rows equally; weighting by h would price per-tensor A8 better.
- *Gain:* < 0.5%.

---

## F. Library, knapsack and search

**F1. Fix the KN_LEVELS mispricing.** H_out quants raise the one-sided proxy: Sketch-B gave 1.4–3.5× in `objcheck`, the diagonal less. With the inherited family factor, the solver will under-select them.
- *Fix:* add a family "oursH" whose factor is fitted from whole-trunk uniform-K swaps (K2.5, K3).
- *Test:* CPU; compare predicted vs measured on C6 and the swap test.
- *Gain:* protects −2 to −5% that would otherwise be lost silently.

**F2. An h-weighted proxy for every option.** Compute tr(E·H_in·Eᵀ·diag(h)) from hproxy per-row errors for old and new quants alike, then refit kfit2 on V2's ~1,250 measurements. Compare cross-validated log-RMSE (now 0.118) and pair ranking (now 79.9%).
- *Source:* 2609.33923 on Qwen3.8-27B: the isolated layer objective has Spearman −0.15 against the real-activation objective, and block-cosine allocation (6.041 PPL) beat even the oracle layer-objective allocation (6.087).
- *Also:* this directly tests why turboderp's quants win at high K despite worse input-side proxy error (#63 / #50).
- *Fit:* knapsack. *Test:* CPU, once the diagonal exists.
- *Gain:* −1 to −3% from better allocation.
- *Conflicts:* #20 / #18 (generic re-rankers failed); this is a physics feature, not a black box.

**F3. An absolute Fisher-predicted ΔKL as the prior.** Use ½·tr(E·H_in·Eᵀ·diag(h)) with matched normalisation as the prior for unmeasured options (sub-2 K, fractional, new levels), instead of depth-curve family factors.
- *Source:* KronQ (score tr(H_G)·tr(H_X)); the RCO "one backward pass" spike.
- *Test:* cross-validation on V2 data.
- *Gain:* fewer wasted stream evals.

**F4. Do not offer an H_out and a plain variant at the same K** unless both are priced by measurement; the proxy will pick the plain one.

**F5. Refit the compounding term κ for H_out mixes.** H_out pushes error into directions the *bf16* downstream ignores; with a quantized downstream, the compounding can change.
- *Source:* 2609.11716 (downstream blocks cancel 50–82% of the incoming error).
- *Test:* kfit2 residuals by depth on the first ~30 stream evals.

**F6. Re-estimate the diagonal at the operating point.** Collect the 0.8B diagonal with the K3 bdiag model's fake-quant weights, requantize and compare.
- *Source:* for: GEMQ (9.66 → 9.01), Q-Strata (12.14 vs 13.72). Against: MixFrag (re-estimation hurt, 31.91 → 29.73).
- *Test:* ≈ 1 h on the 4070.
- *Gain:* ±1–2%, uncertain sign.

**F7. Requant only what pays.** Use B4 and C1 to requantize only the families and levels that gain (likely input projections and gate/up at K1.5–K4). This roughly halves Phase B.

**F8. Run D3 (#51) on the H_out-era library.** Until it runs, knap4's −15% at 2.5 bpw (#45) is a model prediction, not a measurement.

**F9. Partial pooling for the new family** (tracker B11 plan): start the oursH factor shrunk toward the "ours" factor.

---

## G. Interactions

**G1. H_out × A8.** On the 0.8B, run base vs bdiag at K3 under fake A8-g128. The gain must survive the A8 penalty (+5% at K6, #26).
- *Test:* port the E1b hook to evalq, ≈ 1 h.

**G2. H_out × scale-only healing.** On the 0.8B, KL-train suh/svh with frozen codes for both base and bdiag.
- *Source:* ThinkBit recalibration cut KL 0.641 → 0.282.
- *Why:* if healing closes the gap, H_out matters less for the 2–3 sizes that will be healed.
- *Test:* 2–3 h on the 4070.

**G3. H_out × i8 codebook at K5–6:** fold into C1's K5 arm.

**G4. Head / embedding / turboderp tensors:** independent of trunk H_out. Note only, no test.

---

## H. Data and corpus mismatch

**H1. Long context on the 0.8B.** Build 8k/16k-capped goldens from long docs, mostly agentic traces, and score base vs bdiag on positions > 2k.
- *Why:* the private corpus has traces up to 250k tokens, and every H_out number so far stops at 2k.
- *Test:* ≈ 1 h on the 4070.

**H2. Cold-start windows.** Every collection window starts with no context, so sink and massive-activation tokens recur every 2048 tokens. Use the position buckets from D5: if late-position diagonals differ (cos < 0.98), weight long-doc windows differently or use 4096-token windows for them.

**H3. Category weighting.** Compare the token mix of the qad-pool windows with the corpus. Re-weight the per-category accumulators post hoc. Check that the 11 moved docs did not skew the agentic share.

**H4. Multilingual guard.** Read golden-ml once for the final 27B H_out K3 vs the ladder. Vocabulary tiering showed multilingual text takes the biggest hits.

**H5. Data-source invariance.** The 0.8B diagonals came from xcorp; the 27B ones will come from calib qad windows. Collect the 0.8B diagonal from calib qad windows and compare.
- *Test:* ≈ 45 min.

---

## I. Process hygiene

**I1.** Retire g08-search1 for new decisions (A2), and record its reads.

**I2.** Make seed replicates the default in the proxy method README; fold the #33-15 caveat into the INDEX header.

**I3.** Update v3.md §2, §6b and §7. They still say YAQA was "dropped" and that output weighting "hurt in every form", which stops row-slab K (B12) from being reconsidered.

**I4.** Document what KN_LEVELS reads (one-sided proxy after refit), so a requant with a new objective cannot silently change the solver's inputs.

**I5.** Freeze the 27B swap slice and log every read, as with ver3.

---

## J. Other plan items with thin evidence

**J1. Measure 27B quant-to-quant noise.** Requant ladder K3 with `--seed-offset 1` and eval. Every 27B single-quant A/B is affected: #48, chain1's ours-vs-turboderp K6, #38, and the coming H_out swap.
- *Test:* one quant plus eval on card 0.

**J2. Re-judge #48** (output scales off +2.5%, z 2.1) against J1's noise. "Keep them on" survives either way, but stop citing it as evidence about output-side models.

**J3. Slab head (#34).** The whole-model check never ran, and the docs disagree on the head's share of KLD (1.5–3% vs 5–7%). Recompute the share from V2 2.50 swaps.

**J4. Fractional levels in KL, not just proxy.** Compare whole-trunk ladder K2.5 KL with the K2–K3 KL curve on one 27B slice (#36 / #37 are proxy-level).

**J5. A8 on a low-K base** (#26 open item): one E1b batch on a K2/K3 base.

**J6. Trellis-embedding price at low bpw** (#35): extrapolated from a 6.00 base on a fit-pool slice of 67 docs. Do one swap on a 2.5 base.

**J7. The K1 family factor** (#44, #71) comes from 3 swaps; it matters for 2.0 bpw. Add 3–5 K1-dominated swaps.

---

## K. Cheaper substitutes if 27B H_out falters

**K1. An analytic MLP diagonal, forward-only.**
- up_proj: h_r = E[SiLU(g_r)²]·‖W_down[:, r]‖².
- gate_proj: h_r = E[(SiLU′(g_r)·u_r)²]·‖W_down[:, r]‖².
- *Source:* SACQ (2610.41408; structure-aware fit 47.5 vs Euclidean 0 before refinement); FastKron's local MLP Hessian.
- *Test:* on the 0.8B, compare against the Sketch-B diagonal for gate/up only. At 27B this needs only a forward pass on one card.
- *Gain:* if it keeps ≥ 2/3 of the gate/up gain, it replaces most of the collection.

**K2. Consumer-norm diagonals** for q/k/v and GDN qkv/z (‖W_out column‖² × gate statistics). Weaker; test after K1.

**K3. Activation Denoising** (2610.07522): a noise-propagated metric (WF, G) that closes 18–31% of the gap between parallel and sequential calibration, at the cost of one extra forward pass. An output-aware objective with no backward.
- *Fit:* per-level.
- *Gain:* −1 to −3%.

**K4. MPQ-style block coordinate-descent repair after LDLQ** (2610.32127; LDLQ + CD 15.01 → 12.76 at 2 bits on Llama-3-8B). For a trellis, "CD" means re-running the Viterbi on a chunk with the rest fixed. Orthogonal to H_out.

**K5. A small blend of Jacobian-weighted H_in,** H_in + α·s·H_J, choosing α per tensor by KL on the search pool.
- *Source:* EFQ (2610.18139): improved logit MSE in 24/24 cases, but a fixed α was unsafe.
- *Conflicts:* gate 2e (#33-07) was effectively the pure weighted Hessian (α → ∞) and lost; the 0.8B H_in swap lost +20%. Low priority.

---

## INDEX experiments resting on the thinnest evidence

| # | conclusion | why it is thin |
|---|---|---|
| 33-12 | diagonal ≈ full; full gate/up H_out hurts at K2 | differences of 1.9–2.5%, inside the #33-15 noise |
| 33-13 | 1K sequences are enough | shares 85 / 98 / 100% (K3) and 99 / 84% (K2) are not monotone; inside the noise; < 1K never measured |
| 33-11 | H_out stacks with our settings | one seed, one base quant, same slice |
| 33-04/05/06 | backward Hessians and diagonal output scales hurt on the 27B | blocks 22/23 only, swapped into a 6.00 base; one estimator (Σ‖x‖²·dy·dyᵀ); 200 sequences; now in tension with the 0.8B |
| 48 | output scales off is worse (+2.5%, z 2.1) | single quant pair; 27B seed noise unknown, likely ~1–2% |
| 60-04 | (proxy used without calibration) | the calibration study was stopped; the 0.8B was never in it |
| 45 / 51 | knap4: −15% at 2.5 bpw, −10% at 3.0 | model predictions; the D3 check is blocked |
| 46 | A4 not worth building | timing slot invalid; core-matmul table only |
| 34 | slab head −36…−40% head KL | head-only; whole-model check not run; head share inconsistent |
| 35 | trellis embedding pricing | fit-pool slice of 67 docs on a 6.00 base, extrapolated |
| 36/37 | fractional levels on the curve | proxy-level; end-to-end KL never checked |
| 44/71 | K1 factor 1.35 | 3 K1 swaps |
| 52 | KLD vs position rises alike for all quants | 2 documents |
| 38 | input damping 0.005 | proxy winners agree about 53% across levels; KL effect about 0 |
| 41 | rotation seeds are dead | 9 tensors, proxy only; whole-model seed spread is real (#33-15) though unproven exploitable |
| 39 | gain vs K | GPU result never logged |
| 26 | A8 costs | K6 base only |
| 21 / 50 | layer-0 spike; proxy edge over turboderp | single probes; 17 tensors over 3 layers |

---

## Top 20 by expected gain per unit cost

| rank | idea | cost | expected gain / value |
|---|---|---|---|
| 1 | **B9** estimator test (Sketch-B vs ‖x‖²-weighted vs Σdy²) | 1.5 h 4070 | decides whether the 27B night is likely wasted |
| 2 | read **#33-16** with **A1** (seed-aware z) | in flight | confirms or kills the −8% |
| 3 | **D1** dose at 128 / 256 / 512 | 1 h 4070 | may cut 8.5 h to about 2 h |
| 4 | **C6** 27B bident whole-trunk K3 | 1 card, 1–3 h | 0 to −3%, plus 27B machinery check |
| 5 | **B1** within-block AM/GM predictor | CPU 1–2 h | forecasts the 27B gain from the C5 pilot |
| 6 | **F1** separate H_out family pricing | CPU | protects −2 to −5% |
| 7 | **D5** instrument the run | 1–2 h code | many free post-hoc diagnostics |
| 8 | **E1 + E2** panel `ldlq_2hess` and 27B timing | 3 h dev + 15 min | protects the Phase-B schedule |
| 9 | **B4** family attribution | 1.5 h 4070 | halves the requant scope |
| 10 | **C5** 27B 64-sequence pilot | 35 min TP2 | go/no-go before 8.5 h |
| 11 | **E3** H_out-aware refit | 1.5 h | −0.5 to −1.5% |
| 12 | **C1** K4 / K5 / K2.5 error-level check | 30 min | resets the expected gain |
| 13 | **F2** h-weighted proxy refit of kfit2 | CPU 3 h | −1 to −3% via allocation |
| 14 | **B2** shuffle control | 20 min | rules out an artefact |
| 15 | **A3 / A4 / A5** category / position / tail breakdowns | free | private-corpus risk |
| 16 | **B7** tempered diagonal | 1 h | −1 to −2% |
| 17 | **J1** 27B quant-noise measurement | 1 card, 1–3 h | makes every 27B A/B readable |
| 18 | **K1** analytic MLP diagonal | 2 h | could replace the backward |
| 19 | **E9** head H_out from the Fisher diagonal | 1–2 h headsim | −0.3 to −1% |
| 20 | **D2** bf16-GEMM 27B collector A/B | inside the gate | about 3× faster collection |

## The 5 cheapest diagnostics that would most change this ranking

1. **The ‖x‖²-weighted diagonal arm on the 0.8B** (the core of B9): one collection of ~25 min plus 2 quants.
   - If it *wins*, the 27B negatives are unexplained and C5/C6 must come before any collection.
   - If it *loses*, the plan is consistent and 1, 4 and 10 move down.
2. **The `bident` share in #33-16.** At ≥ 2/3 of the gain, drop the collection and promote C6 to rank 1.
3. **The B1 AM/GM predictor against per-tensor rowerr on the existing 0.8B data** (CPU, about 1 h). If it predicts well, C5 becomes a reliable 35-minute forecast of the 27B gain.
4. **The 0.8B dose at 256 windows** (≈ 7 min collection plus 4 quants). It sets the run length: about 2 h vs 8.5 h.
5. **`ldlq_2hess` timing on one 27B gate/up** (15 min on card 0). If two-sided costs ≥ 2× one-sided per level, E1 becomes a prerequisite and F7 (requant only what pays) moves up.
