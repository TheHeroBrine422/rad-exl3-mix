# Research ideas, "open" lens: Qwen3.8-27B EXL3-V3

This report has 86 ideas in nine themes, each with source, mechanism, fit, cheapest test, expected gain and conflicts. After them come a short "checked, not recommended" list, a ranked top 20 and the five cheapest diagnostics that would most change that ranking.

## What I checked in the code first, and why it shapes the ideas

1. **The diagonal-H_out win is output-axis error feedback, and only within each 128-row block.**
   - `ourq --hout` with a `hout_diag` file builds `torch.diag(d)`.
   - `prepare_H_out` multiplies it by `sv`, applies the blockwise 128-point Hadamard on both sides and block-LDLs the result.
   - So in the quantizer's basis, diag(d) becomes H·D·Hᵀ: dense inside each 128-row block, zero between blocks. `ldlq_2hess` then feeds error back across rotated rows within a block. Sensitivity differences *between* 128-row blocks are invisible to this mechanism. Several ideas below (A3–A6, E1–E2) exist only because of this.
2. **Gate 2d (#33-06) used a different mechanism.** `--hout-diag` goes to `sensitivity_out_scales`: per-channel scales t ∝ (σ²/s)^¼ applied *before* the rotation. Those scales reallocate across all rows. Gate 2d also used a 200-sequence, ‖x‖²-weighted estimate, so its failure does not condemn the diagonal itself.
3. **The Fisher collectors put the sampled-label loss on every position** (`collect.py`, `yaqaB.py`). The board scores gen-masked rows only (about 55% of positions in g2b-search1).
4. **The `qwen35_exl3` SSM state cache is already f32.** The chunk-scan intra-chunk math is bf16 WMMA.
5. **QTIP's L ablation** (2406.11235, Table 10, 2-bit Llama-2-7B): L=12 gives 6.97 Wiki2 PPL, L=16 gives 6.83. At L=16 the Gaussian MSE is 0.069 against a bound of 0.063.

Shorthand: "lognormal ceiling" is ½·log2(AM/GM) of a sensitivity vector, which equals ½·log2√(1+CV²) under a lognormal assumption. For the 0.8B diagonal CVs this gives:

| Tensor family | CV | Ceiling (bits) | Weighted-error cut at equal bits |
|---|---|---|---|
| gate/up | 1.0 | 0.25 | −29% |
| attention q | 2.3 | 0.6 | |
| GDN z | 3.96 | 1.0 | |
| residual writers (down, o, ssm_out) | 0.2 | 0.01 | |

These are ceilings for *perfect* per-row allocation. The measured 8% total gain shows how much the current feedback captures.

Hardware shorthand:
- **4070**: the 0.8B/2B proxy. A 1K-sequence 0.8B collection takes about 26 min; a whole-trunk quant plus eval takes 2–4 min.
- **iceman**: 2× R9700. The 27B fp32 1K diagonal collection takes about 8.5 h on both cards.
- **CPU**: the CPU boxes.
- **rented**: H100/H200/MI300X.

---

## A. Output-side information (H_out): mechanism, estimation, exploitation

**A1 · Gen-masked Fisher for the 27B diagonal collection.**
The metric is mean KL on gen rows, but our Fisher includes the loss at prompt, tool-output and system positions. Put the sampled-label CE only on scored positions. Gradients still flow back through unscored context via attention and recurrence, which is what the metric needs.
- *Source:* new. YAQA (2505.22988) puts loss on all positions; a metric-matched Fisher is the obvious correction for a gen-masked score.
- *Why:* tool outputs and user turns can load different heads and neurons than assistant tokens (copying versus generation), so the row weights d may be mis-set. Roughly half of all positions are unscored.
- *Fit:* per-level (changes only the H_out the requant uses).
- *Test:* 0.8B, 1K chat-templated xcorp windows, masked vs unmasked diagonal; bdiag K3/K2 on g08-search1. Cost: 2 × 26 min plus 4 quants, about 1.5 h on the 4070. Counts if masked beats unmasked at z ≤ −2 at either K and is not worse at the other. Re-check the 1K dose, since masking halves the samples.
- *Gain @2.5:* 0 to −3% (honestly unknown). *Risk:* low; the worst case is a tie.
- *Conflicts:* #33-13 (the planned run). Decide before launching `hdiag27.sh`.

**A2 · Separate mechanism from data in gate 2d's failure.**
Run exllamav3's `sensitivity_out_scales` (the gate-2d path) on the 0.8B with the precise 8K Sketch B diagonal, next to bdiag and base.
- *Source:* exllamav3; KronQ (2607.07964) uses the same (h/a)^¼ scaling successfully.
- *Why:* if out-scales win with good data, gate 2d's loss came from its 200-sequence ‖x‖²-weighted estimate, and between-block reallocation (A6, E1) becomes attractive. If they still lose, scaling un-flattens what the Hadamard flattened, and only feedback and exact splitting should be pursued.
- *Fit:* diagnostic, per-level.
- *Test:* 0.8B, K3/K2, 3 arms, about 20 min on the 4070.
- *Gain:* diagnostic. *Risk:* none.
- *Conflicts:* #33-06 (same mechanism, different data quality), #48.

**A3 · Interleave MLP neurons across 128-row output blocks.**
Permute the intermediate dimension (gate/up rows and down columns, an exact and free transform) so that each 128-row Hadamard block holds a sample of the whole sensitivity range, for example round-robin over sorted d. That maximizes what within-block feedback can reach.
- *Source:* new. Analogues: NOVA-KV's equal-volume interleave (2608.04074, sorted 37.2 vs random 74.4 vs interleaved 75.4) and CurveTQ's within-block anisotropy bound Γ_b (2610.09212).
- *Why:* feedback captures only within-block AM/GM. Neuron order in a trained MLP is arbitrary, though, so within-block spread probably already matches the global spread. I expect a null for MLPs; this is mostly the control for A4.
- *Fit:* per-level; the permutation is folded into the stored weights.
- *Test:* CPU first: split the log-variance of d into within- and between-block parts for every 0.8B tensor (minutes). Quantize only if the between-block share exceeds 25%.
- *Gain:* 0 to −1%. *Risk:* low. *Conflicts:* none.

**A4 · Stored output permutation (head interleaving) for q/k/v and GDN qkv/z.**
In head-structured tensors a 128-row block is half a head (head_dim 256), so cross-head sensitivity spread is not reachable by the feedback. Store a per-tensor row permutation and apply it in the GEMM epilogue (and as a scatter in decode), so that each Hadamard block mixes rows from heads of different sensitivity.
- *Source:* new.
- *Why:* 0.8B diagonal CVs are high exactly here (z 3.96, qkv 1.69, q 2.3, k 1.5; lognormal ceilings 0.4–1.0 bit). Even 20% capture would be 0.1+ bit on these families.
- *Fit:* format change (a u16 permutation per tensor, an epilogue scatter; the input side is untouched).
- *Test:* 0.8B fake-quant, where `evalq` applies the permutation trivially; bdiag K3/K2 with vs without, after the CPU decomposition from A3. About 1 h.
- *Gain:* −1 to −4%. *Risk:* medium (kernel plumbing at every M band).
- *Conflicts:* none.

**A5 · Wider or strided output Hadamard for high-anisotropy tensors.**
An alternative to A4: a 256/512-point, or strided (rows i, i+N/128, …), output rotation makes the rotated H_out dense over more rows.
- *Source:* QuIP#/QTIP use full-dimension RHT; our 128 is a speed choice.
- *Fit:* format change (output FWHT size in the encoding).
- *Test:* 0.8B fake quant at 128/256/512, K3 bdiag, on tensors flagged by the decomposition. About 1 h.
- *Gain:* −1 to −3% (overlaps A4; pick one). *Risk:* decode FWHT cost at M=1; the int8 epilogue.
- *Conflicts:* none.

**A6 · Out-scales for the between-block part, feedback for the within-block part.**
Apply sensitivity scales at block granularity (one t per 128-row block from the block mean of d), or per row with a tempered exponent of 1/8, together with bdiag feedback.
- *Source:* exllamav3 `sensitivity_out_scales`; KBBQ (2609.08135) found that partial flattening (λ = 0.5–0.75) beats full; KronQ BiIP.
- *Why:* gate 2d may have failed by over-scaling (exponent ¼ on noisy d). A block-level t cannot destroy within-block incoherence.
- *Fit:* per-level.
- *Test:* 0.8B after A2: 4 arms × 2 K, about 30 min.
- *Gain:* 0 to −3%. *Risk:* a repeat of gate 2d.
- *Conflicts:* #33-06, #48 (here the scales are modified, not removed).

**A7 · Tempered H_out.**
Use d^p for the diagonal, and a trace-preserving H_out^p via eigendecomposition for the full factor, with p ∈ {0.5, 0.75, 1.25}.
- *Source:* ShamAN-Q (2609.38521): tempering p = ½ took 19.21 → 17.46 PPL, and KL-Shampoo plus tempering reached 16.72 when the error is large. Also KBBQ's brake.
- *Why:* the full gate/up H_out hurt at K2 (bhout −5.6% vs bmix −7.7%), the classic over-confident-curvature symptom.
- *Fit:* per-level.
- *Test:* 0.8B on the existing 8K snapshots, 12 arms, about 1 h.
- *Gain:* 0 to −2% over bdiag. *Risk:* low.
- *Conflicts:* #33-12.

**A8 · Shrink the full H_out toward its diagonal.**
Use H_γ = (1−γ)·H_out + γ·diag(H_out), γ ∈ {0.5, 0.8}, offline from the 8K snapshots. The diagonal is precise (halves cos ≥ 0.997); the off-diagonal is noisy (0.66–0.85 for o/out/gate/up).
- *Source:* TaSQ (2610.03027): the dense metric won in-sample and lost 4× out of sample, while the diagonal transferred. Ledoit–Wolf shrinkage (statistics).
- *Fit:* per-level.
- *Test:* 0.8B, about 40 min. Counts if any γ beats bdiag at both K with z ≤ −2.
- *Gain:* 0 to −2% over diag; useful at 27B only together with A9 or A10.
- *Conflicts:* #33-12.

**A9 · Diagonal plus low-rank H_out from a streaming sketch.**
Approximate H_out ≈ diag(d) + U_r·Λ_r·U_rᵀ. U_r comes from a randomized range sketch Y = Σ_b G_b·(G_bᵀ·Ω), accumulated during collection (Ω is out × (r+p) Gaussian, about 70 MB total at r = 64 for the 27B) and finished by Nyström. It can be tested offline right now by truncating the 0.8B 8K full snapshots.
- *Source:* new combination; randomized numerical linear algebra (Nyström; Clean 2610.04204).
- *Why:* participation ratios of 0.04–0.6 mean a few directions dominate. bmix beat diag at K3 by about 2%. A rank-r part may carry that without the 155–206 GB of full accumulators, and drop the noisy tail.
- *Fit:* per-level; adds a sketch accumulator to the planned 27B run at a few % of its runtime.
- *Test:* CPU plus 4070: diag+rank-{8, 32, 64} from 8K snapshots, K3/K2, about 1 h. Adopt if it matches bmix at K3 and diag at K2.
- *Gain:* −1 to −2% over diag. *Risk:* low.
- *Conflicts:* #33-12.

**A10 · Rented full H_out for the non-gate/up tensors at 27B (the bmix arm).**
This needs 51 GB of fp32 accumulators on an 80–192 GB card, with 1–2K sequences.
- *Source:* #33-12 (bmix: K3 −9.9% vs diag −8.0%).
- *Fit:* per-level.
- *Test:* rented MI300X/H200, about 20–40 GPU-h. Do A9 first; if A9 matches bmix, skip this.
- *Gain:* −1 to −2% at K3, about 0 at K2. *Risk:* 27B full-factor noise is higher than at 0.8B.
- *Conflicts:* #33-05 (gate 2c used a different estimator and 200 sequences).

**A11 · Analytic local H_out for gate/up from forward statistics only.**
For SwiGLU, the Gauss–Newton output factor at the down output is:
- gate row r: d_r = ‖W_down[:,r]‖²·Σ_t (SiLU′(g_{t,r})·u_{t,r})²·‖x_t‖²
- up row r: ‖W_down[:,r]‖²·Σ_t SiLU(g_{t,r})²·‖x_t‖²

The global factor at residual writers is nearly flat (CV 0.17–0.2), so local ≈ global up to a per-layer scalar, which LDLQ normalizes away. It needs one forward pass with per-neuron accumulators: no labels, no backward, no fp32 problem, and it can run over all 10M calib tokens.
- *Source:* SACQ (2610.41408, the formula); FastKron (2608.06291, local MLP Hessians). Using it as the 27B factor is new.
- *Why:* gate/up are about 47% of trunk weights. If local d gives ≥80% of the bdiag gain, the 27B work for them shrinks to a forward-stat pass, and A3/E1 can start before the 8.5 h backward.
- *Fit:* per-level.
- *Test:* 0.8B: forward stats over 1K windows (about 5 min), corr(local d, Sketch B d) per tensor, then K3/K2 with local d for gate/up and Sketch B elsewhere. About 40 min. Counts at ≥80% of bdiag's gain.
- *Gain:* the diagonal's gate/up share at a fraction of the cost. *Risk:* misses residual-stream anisotropy (massive channels).
- *Conflicts:* none.

**A12 · Local H_out for GDN z and the attention/GDN projections.**
Extend A11:
- GDN z row r: ‖W_out[:,r]‖²·E[(norm(o)_r·SiLU′(z_r))²].
- Attention v rows: o-column norms × an attention-weight factor (SACQ's ω_s = Σ p²).
- q/k: KᵀK and QᵀQ per head (OSCAR).
- GDN q/k/v through the recurrence: readout-side term only.
- *Source:* SACQ, OSCAR (2605.17757), 2609.04098 (per-projection GDN output error on our model: qkv 10.4%, z 9.9%, out 12.7%; a/b ~2–3%).
- *Fit:* per-level.
- *Test:* 0.8B after A11, about 1 h.
- *Gain:* the remaining families if A11 works. *Risk:* the GDN recurrence makes q/k crude.
- *Conflicts:* none.

**A13 · Operating-point H_out: re-estimate on the quantized model.**
Collect the diagonal Fisher on a K3-bdiag quant (fake-quant weights in the backward) and requantize: one Sketch-A-like refresh at the perturbed point.
- *Source:* GEMQ (2605.23078: importance from FP / 4-bit / 2-bit / 2-bit+router-FT models gives 9.66 / 9.39 / 9.33 / 9.01 PPL); MixQuant. Contra: MixFrag (2607.28589) found re-estimation hurt on a ViT (31.91 → 29.73).
- *Fit:* per-level (one collection per uniform level).
- *Test:* 0.8B, 1K sequences (about 30 min) plus K3/K2.
- *Gain:* 0 to −3%. *Risk:* noise, and a second 27B run.
- *Conflicts:* none.

**A14 · Per-family attribution of the diagonal gain.**
On the 0.8B, apply diag H_out only to input projections vs only residual writers vs attention-only vs GDN-only.
- *Why:* narrows A3–A12 (and A18) and tells which local estimators must be good.
- *Fit:* diagnostic.
- *Test:* 6 arms × 2 K, about 1 h on the 4070.
- *Conflicts:* none.

**A15 · H_out-weighted refit of input scales.**
`refit_scales` couples rows through suh (input scales). With a diagonal H_out, that least-squares fit should weight rows by d. Check whether the bdiag path does; add it if not.
- *Source:* JARQ (2609.38599), H-Scale (2608.28113).
- *Fit:* per-level.
- *Test:* 0.8B K3/K2, about 20 min.
- *Gain:* 0 to −1%. *Conflicts:* none.

**A16 · Two-sided LM head with the exact softmax Fisher diagonal.**
The head's output factor is known in closed form: E[p(1−p)] per vocab row, already computed by `headsens.py`. Pass it as diag H_out into the head's two-sided LDLQ, so its 128-row output Hadamard gives within-block feedback on top of the slab head's between-slab allocation. Optionally interleave rows within slabs.
- *Source:* SoftWater (2608.12026: 6.5–8.3× lower head KL at 2 bit, variable rate); our slab head (#34).
- *Why:* the slab head already gives −36…−40% head KL at K5–6 from between-slab allocation; within-slab feedback is untouched. The head is about 5–7% of total KL.
- *Fit:* per-level (head donors).
- *Test:* `headsim` exact head-only KL, 2–3 donors with and without the diagonal; CPU hours.
- *Gain:* −10…−20% head KL, i.e. −0.5…−1% total. *Risk:* low.
- *Conflicts:* #34 (complementary).

**A17 · Per-tensor output damping sweep.**
Pick sigma_reg_out ∈ {0.003, 0.01, 0.03, 0.1} per tensor by two-sided proxy error against a split-half H_out, like `--sigma-in-sweep`.
- *Source:* #38.
- *Fit:* per-level.
- *Test:* 0.8B K2 vs fixed 0.01, about 40 min.
- *Gain:* 0 to −1%. *Risk:* proxy overfit, mitigated by split halves.
- *Conflicts:* #38.

**A18 · Truncated-backward hybrid collector.**
Take gradients for layers ≥ 32 from a backward truncated at layer 32 (exact; gate 2c already did this). Take layers 0–31 from the local estimators (A11/A12). Early layers are the least KL-sensitive (kfit2) and the hardest numerically.
- *Source:* #33-05's truncation.
- *Fit:* per-level.
- *Test:* the 0.8B analogue: Sketch B for layers 12–23, local for 0–11, K3/K2.
- *Gain:* the 27B run gets about 40–50% cheaper at equal KL if A11 holds.
- *Conflicts:* #33-13 plan.

**A19 · Rate-distortion headroom map.**
A one-off CPU table per tensor of:
- the total, within-128-block and between-block ½log2(AM/GM) of d;
- the same for block-averaged LDL pivots of the rotated H_in (E3);
- CurveTQ's Γ_b (D8);
- the captured fraction: measured bdiag proxy reduction ÷ within-block ceiling.

It bounds A3–A6, E1–E3 and D8 before any GPU time.
- *Source:* new (reverse water-filling; WaterSIC AM/GM argument in 2608.25188).
- *Fit:* diagnostic.
- *Test:* CPU minutes on 0.8B snapshots; repeat on 27B when its diagonal lands.
- *Conflicts:* E3c (#29-02, input side only).

## B. Why the gradient-weighted H_in hurts, and which factors are right

**B1 · Exact-Fisher ground truth for the factor choice.**
For each existing 0.8B arm (base, ldlq, hin, hout, hdiag, bdiag), compute q(E) = mean_b ⟨E, G_b⟩² per linear, with E = Ŵ − W, in one sampled-label backward. G_b is just the weight gradient, so this is an elementwise product sum. It is the Gauss–Newton second-order KL of each tensor's actual error without any Kronecker assumption. Compare Σq with measured KL, and per tensor against tr(E·H_in·Eᵀ·H_out) for each factor choice.
- *Source:* new (YAQA's objective without its approximation).
- *Why:* settles whether calib H_in is "right" or merely luckier, explains #33-10's +20% H_in swap, and validates C1/C2 before any engineering.
- *Fit:* diagnostic.
- *Test:* 0.8B, 1K sequences, the arms' weights streamed per layer (6 × 1 GB), about 40 min on the 4070.
- *Risk:* second order misses compounding at K2.
- *Conflicts:* none.

**B2 · Leverage-weighted (matrix-normal ML) Kronecker factors.**
Replace Sketch B's factors with the KL-Shampoo fixed point:
G = (1/n)·Σ_t (x_tᵀ·A⁻¹·x_t)·δ_t·δ_tᵀ, A = (1/m)·Σ_t (δ_tᵀ·G⁻¹·δ_t)·x_t·x_tᵀ, iterated 3 times.
The token weights become bounded leverages instead of Sketch B's roughly quartic weights, which massive-activation tokens dominate.
- *Source:* ShamAN-Q (2609.38521: KL-Shampoo untempered 17.04 vs NKP 19.21 PPL; condition number about 40 vs several hundred).
- *Fit:* per-level.
- *Test:* 0.8B, 3 passes × 1K (about 1.5 h). Arms: B2-H_in alone; B2-H_out with calib H_in.
- *Gain:* speculative, 0 to −3%. *Risk:* may just reproduce #33-10.
- *Conflicts:* #33-07, #33-10 (different weighting).

**B3 · Small blend of Sketch B H_in into calib H_in.**
H_in = calib + α·SB_norm, α ∈ {0.05, 0.15, 0.3}, on bdiag settings.
- *Source:* the gate-2e blend plan (never run on the 27B because of a pre-set rule).
- *Fit:* per-level.
- *Test:* 0.8B, 6 arms, about 30 min.
- *Gain:* 0 to −2%. *Conflicts:* #33-07 (new pass rule needed).

**B4 · Domain check: Sketch B H_in on calib windows vs xcorp.**
The H_in swap changed both weighting and domain. Collect 1K Sketch B on qad windows and rerun the `hin` arm.
- *Source:* #33-09 (H_in vs calib cosine 0.35–0.98).
- *Fit:* diagnostic.
- *Test:* 26 min plus 2 arms.
- *Conflicts:* #33-10.

**B5 · Effective sample size and tail-eigenvalue noise.**
Compute the ESS of the per-token weights ‖δ_t‖², and the split-half out-of-sample LDLQ proxy (quantize on half A, score on half B, as D2 did) for calib H_in vs Sketch B H_in.
- *Why:* if Sketch B's out-of-sample gap far exceeds calib's, the H_in loss is estimation noise in the small-eigenvalue directions that LDLQ's feedback relies on. The halves cosine of ≥0.96 only checks the top of the spectrum.
- *Fit:* diagnostic.
- *Test:* CPU plus GPU, about 1 h.
- *Conflicts:* #33-09.

**B6 · Which output estimator: per-sequence G_b·G_bᵀ vs token-level Σδδᵀ vs Σ‖x‖²δδᵀ vs Sketch A.**
Accumulate all four diagonals in one 0.8B backward:
- Sketch B;
- H_Z-PTQ's E[G_Zᵀ·G_Z];
- gate-2c's ‖x‖²-weighted form;
- Sketch A's (xᵀ·H_in·x)-weighted form.

Then run bdiag with each.
- *Source:* H_Z-PTQ (2610.40345: their factor beat YAQA's on Llama-3.2-1B, where YAQA was unstable at 19–70 PPL); our #33-04/05.
- *Why:* tells whether the 27B gate failures were the estimator, and picks the cheapest good estimator for the 27B run.
- *Fit:* per-level.
- *Test:* one 1K pass with four accumulators plus 8 arms, about 1.2 h.
- *Conflicts:* #33-04, #33-05.

## C. Measuring sensitivity and allocating bits

**C1 · Fisher inner-product sensitivity library (FISL).**
In a sampled-label backward, compute s_b(i,K) = ⟨Ŵ_{i,K} − W_i, G_b⟩ for every library option: decode each option tile-wise on the GPU, take the inner product with the transient weight gradient, discard. Store per sequence a vector of about 337 × 30 floats. Then for any mix, 2·KL ≈ mean_b(Σ_i s_b(i,K_i))²/T. That gives per-option sensitivities *and* every pairwise interaction (Gauss–Newton cross terms) from one data product, with no evals.
- *Source:* new. Related: RCO (2605.00649, true-KL over options), YAQA, Q-PACE (2610.09183, forward-only linear sensitivities).
- *Why:* kfit2's input-side proxy needs fitted family factors (1.15–1.58× disagreements) and a compounding term; FISL is metric-exact to second order, includes cross-tensor terms, and is mix-independent.
- *Fit:* per-level.
- *Test:*
  - 0.8B first: 13 levels × 0.5B params streamed per layer, 1K sequences (about 1–2 h on the 4070). Validate on 30 random 0.8B mixes (about 1 min each): does predicted KL beat a kfit2-style fit on ranking?
  - 27B: piggyback on the diagonal run with on-GPU decode (+3–6 h), or rent.
- *Gain:* −3 to −8% via allocation if validated. *Risk:* second order breaks at ≤K2 (2.0-bpw KL 0.164); non-GN compounding is missing.
- *Conflicts:* #15 (no forward per option here), #17 (one pass, not 800+).

**C2 · Interaction-aware knapsack.**
With FISL's option covariance, minimize Σc_ii + Σc_ij under bytes. Use a Lagrangian DP on the diagonal plus iterated linearization of cross terms (CASA's majorizer), or a chain DP over adjacent layers (TWLA), then verify with evals.
- *Source:* TWLA (2606.13054: pairwise chain DP 8.31 vs unary 11.92 PPL), LoopQuant (2610.16368), CASA (2609.25916: cross-term screening 22.41 vs 28.15).
- *Fit:* per-level allocation.
- *Test:* 0.8B: FISL solve vs additive solve at matched bits, 5–10 mixes.
- *Gain:* −1 to −4% over additive. *Risk:* cross terms may be small (2609.01587: pairs ≈ sum of singles, 0.91–1.10).
- *Conflicts:* #11 (that term was a global link; this is per-pair), #16 (not Shapley).

**C3 · Two-sided proxy in kfit2.**
Recompute every proxy row as tr(E·H_in·Eᵀ·diag(d)) / tr(W·H_in·Wᵀ·diag(d)) with the 27B diagonal on CPU (`hproxy` already decodes options), then refit. Hypothesis: our quantizer's 1.25–1.58× KLD per unit input proxy at K4–K8 (turboderp's ~1.0) is error pushed into high-d output directions that the input proxy cannot see. If so, the family factors collapse toward 1.
- *Source:* #44 notes; KronQ (2607.07964: s = tr(H_G)·tr(H_X)).
- *Fit:* per-level.
- *Test:* CPU, now on the 0.8B library, later on the 27B. Counts if CV log-RMSE falls below 0.118 and family factors land within ±10%.
- *Gain:* −1 to −3% (more above 4 bpw; −0.5…−1% at 2.5). *Risk:* low.
- *Conflicts:* #13 (explains the factors instead of fitting them).

**C4 · Quantizer variant as a knapsack dimension.**
Library options become (K, variant ∈ {base, bdiag, bdiag+lowrank}); the knapsack picks per tensor. Residual writers with flat d may lose slightly to H_out noise.
- *Fit:* per-level (2× requant cost).
- *Test:* 0.8B per-tensor two-sided proxy comparison, then one mixed quant vs all-bdiag.
- *Gain:* −0.5 to −2% over all-bdiag. *Risk:* pricing variants by proxy can mislead (#48).
- *Conflicts:* #48.

**C5 · Additivity audit on the 2B.**
Measure 20 random tensor-group pairs (adjacent and distant, K2/K3) against their singles. Keep C2 only if pairs deviate from the sum.
- *Source:* 2609.01587 (0.91–1.10); 2610.09227 (compressive β ≈ 0.53); 2609.09867 (composition error 26–42× the pricing error).
- *Fit:* diagnostic.
- *Test:* 2B, about 60 fake-quant evals × 23 s, about 1 h.
- *Conflicts:* #14 (27B singles too noisy; 2B evals can be averaged cheaply).

**C6 · Re-anchor probes at the 2.5 base.**
V2's group probes sit on a 4-bpw base. Measure about 12 group swaps (K2↔K3, several families and depths) on the current best 2.5 mix and check kfit2's directional predictions.
- *Source:* GEMQ, MixQuant (2607.23047), Q-Strata (in-context marginals 12.14 vs separable 13.72 PPL).
- *Fit:* allocation.
- *Test:* 27B, 12 evals, under 1 card-h.
- *Gain:* 0 to −3%. *Risk:* MixFrag found re-measuring hurt.
- *Conflicts:* #68.

**C7 · Weight search slices by corpus token share.**
The score is mean *token* KL, so long agentic traces carry a big share. Search slices cap documents at 2k positions. Reweight per-document scores by category token share ÷ slice share, and check whether V2's top candidates reorder.
- *Source:* 2610.20826 (slice KLD 0.0193 vs corpus 0.0147; candidate rankings did not transfer).
- *Fit:* selection.
- *Test:* CPU on stored per-row scores.
- *Gain:* 0 to −2%. *Risk:* variance from up-weighting few documents.
- *Conflicts:* #55, the capstudy.

**C8 · Position-bin family sensitivity on long traces.**
Swap one family (attention layers, GDN in-projections, MLP) K3→K2 and measure ΔKL per position bin on fit-pool long documents. If the attention share grows with position (GDN errors plateau and decay per 2609.04098), raise attention K for the long-trace-heavy score.
- *Fit:* allocation.
- *Test:* 27B, 3–4 evals, about 1 h.
- *Gain:* 0 to −3%.
- *Conflicts:* #52 (whole mixes looked flat; family groups were not tested).

**C9 · Interaction ceiling: measured-KL lazy descent on the 0.8B.**
Run Q-Strata's lazy-greedy descent with real evals at group granularity, 4-bit → 2-bit, and compare its 2.5-equivalent mix with knapsack plus local search. This prices *all* interaction modeling at once.
- *Source:* Q-Strata (2608.30564: JSD 749 vs one-shot ILP 823).
- *Fit:* diagnostic.
- *Test:* 0.8B, 1–2k evals of ~10 s, 4–6 h on the 4070.
- *Conflicts:* #70.

**C10 · Allocation after healing.**
Healing helps low-K tensors most, so post-heal sensitivities differ. After F1 on a 2.5 mix, re-measure a dozen groups, re-solve, and heal again.
- *Source:* 2606.09012, 2609.22238 (post-recovery ranking ≠ static).
- *Fit:* per-mix, training.
- *Test:* 0.8B: heal → reallocate → heal vs heal once.
- *Gain:* −1 to −3% on top of healing. *Risk:* doubles healing cost.
- *Conflicts:* none.

**C11 · GDN head half-life as a prior.**
Per-head retention from A_log/dt_bias statistics on calib correlates with error persistence (ρ ≈ 0.80 over 2,304 heads of Qwen3.8-27B). Use it as a kfit2 feature for GDN groups and as the grouping signal for E2.
- *Source:* STEPQuant (2609.38169).
- *Fit:* per-level prior.
- *Test:* CPU correlation with fitted GDN sensitivities.
- *Gain:* fit stability; small.
- *Conflicts:* #19 (that was a free offset; this is a measured feature).

**C12 · FISL bootstrap as the uncertainty term.**
FISL's per-sequence s vectors give each option's sensitivity uncertainty directly (resample sequences). That feeds the planned uncertainty-aware solve (B11) without refitting kfit2 on bootstrap draws.
- *Fit:* allocation.
- *Test:* with C1.
- *Gain:* fewer V0-style extrapolation picks.
- *Conflicts:* B11 (planned; this is a cheaper source).

**C13 · Two-sided activation-cost proxy for the per-tensor A8/f16 decision.**
The KL cost of activation error at a linear's input is tr(Wᵀ·diag(d)·W·Σ_err), not E1c's family-scaled rel_out (log-RMSE 0.26). Use the 27B diagonal plus the weights to price A8 group options.
- *Source:* E1c (#27); KBBQ's noise law.
- *Fit:* per-level (V3 solver).
- *Test:* CPU refit of `aqcal` against the 45 E1b runs.
- *Gain:* KL ≈ 0 at 2.5 (A8 is nearly free), but a better speed/KL trade at ≥4 bpw.
- *Conflicts:* #27.

**C14 · Joint Bayesian optimization of quantizer knobs on the 0.8B.**
A7, A17, B3, D3 (λ, ρ, γ) and sigma_reg interact. A small GP-BO over about 6 knobs, with 2–4 min evaluations on the 4070, beats one-at-a-time sweeps.
- *Source:* BO (cross-area).
- *Fit:* process.
- *Test:* about 60 evaluations, about 4 h.
- *Gain:* finds the joint optimum (−1 to −3% over separate sweeps). *Risk:* proxy-to-27B transfer.
- *Conflicts:* #18 (that was BO over 340 categorical dims; here 6 continuous knobs).

## D. Rounding and the trellis solver

**D1 · Trellis coordinate-descent sweeps after LDLQ.**
After one LDLQ pass, run Gauss–Seidel over tiles: re-run each tile's Viterbi with the target corrected by *all* other tiles' current errors under the full H (two-sided for bdiag). Accept a tile only if the objective drops on a split-half H. Two or three sweeps.
- *Source:* MPQ (2610.32127: LDLQ+CD3 on Llama-3-8B 2-bit E8P12 took PPL 15.01 → 12.76; at 4 bit 6.54 → 6.46); JARQ.
- *Why:* LDLQ is one-pass and greedy; a 256-weight tile cannot see later columns' needs.
- *Fit:* per-level (2–3× quant time).
- *Test:* 0.8B K2/K3, proxy first then KL, about 1 h.
- *Gain:* −2 to −5% at K2, about −1% at K3. *Risk:* lower proxy can raise KL (DRQ: 47–74% of reconstruction-driven refinements raised PPL), hence the split-half gate.
- *Conflicts:* #57 (tail-biting is a different gap).

**D2 · Asymmetric (quantized-prefix) targets per level.**
For level K, capture x̃ (inputs through the uniform-K quantized prefix) and fit Ŵ to ‖Ŵx̃ − Wx‖²: target W' = W·E[x·x̃ᵀ]·E[x̃·x̃ᵀ]⁻¹. The library stays valid, since each level is a uniform quant.
- *Source:* GPTAQ (2504.02692); G²PTQ (2609.31009) on Qwen3.5-2B W3, KL: GPTQ 0.228, GPTAQ 0.215, G²PTQ* 0.177.
- *Fit:* per-level.
- *Test:* 0.8B/2B, about 2 h. The 27B needs a layer-sequential capture (256k tokens ≈ 2.6 GB per hidden state), about 10–20 GPU-h rented per level.
- *Gain:* −3 to −8% at K2–K3. *Risk:* a mix's upstream ≠ the uniform-K upstream.
- *Conflicts:* ROADMAP sequential Hessians (this is the stronger asymmetric variant).

**D3 · Activation-denoising LDLQ (mix-independent).**
One noise-injected forward pass collects C = E[x·ηᵀ] and N = E[η·ηᵀ] (η = Gaussian noise injected at every linear input and propagated, σ_c = γ·max|x_c|). Quantize W·F under G = H + αI + λ²N + λ(C + Cᵀ). Upstream error is modeled statistically, so the library stays mix-independent.
- *Source:* 2610.07522 (closes 18–31% of the parallel→sequential gap; with QuIP# E8 at W2 it beat sequential on 8B/13B).
- *Fit:* per-level.
- *Test:* 0.8B: one noisy pass (about 10 min), 4 (λ, ρ) settings at K2/K3, about 1 h. The 27B needs two extra d×d accumulators per linear: one night of forward.
- *Gain:* −2 to −5% at K2. *Risk:* four knobs; tune on the search pool only.
- *Conflicts:* none.

**D4 · First-order term with trust region.**
Add G²PTQ's g·H⁻¹ correction (row-wise β trust region) to LDLQ feedback. At bf16 the KL gradient is about 0, so this only means something together with D2's quantized upstream.
- *Source:* 2609.31009 (without the trust region KL collapsed to 25.3; exact vs approximate gradient 0.497 vs 0.710).
- *Fit:* per-level.
- *Test:* 0.8B with D2.
- *Gain:* −1 to −3% beyond D2. *Risk:* complexity.
- *Conflicts:* none.

**D5 · SwiGLU-coupled gate/up rounding.**
Gate and up errors meet in h = SiLU(g)⊙u: δh_r = a_r·δg_r + b_r·δu_r, with a = SiLU′(g)·u and b = SiLU(g). Quantize gate first, then shift up row r's target by −ρ_r·δg_r, with ρ_r = E[a·b·‖x‖²]/E[b²·‖x‖²]. A matrix version: give the stacked gate_up its 2×2-per-neuron cross entries in H_out.
- *Source:* new (SACQ has only the diagonal terms).
- *Fit:* per-level.
- *Test:* CPU first: corr(a_r, b_r) per neuron on 0.8B forward stats. Quantize only if |corr| > 0.3 is common.
- *Gain:* 0 to −2%. *Risk:* likely small correlation.
- *Conflicts:* none.

**D6 · Coupled q/k rounding in the 16 attention layers.**
δ(q·k) = δq·k + q·δk. A joint metric for q and k rows of a head with QᵀQ / KᵀK factors and cross terms.
- *Source:* OSCAR.
- *Fit:* per-level.
- *Test:* after A12.
- *Gain:* small. *Risk:* medium.
- *Conflicts:* none.

**D7 · Scale refit ↔ re-round alternation.**
After `refit_scales`, re-run Viterbi on the moved grid; repeat 2–3 times; accept on proxy.
- *Source:* JARQ (an independent per-group scale fit leaves a 21–25% objective gap; ≤1% gains on strong hosts).
- *Fit:* per-level.
- *Test:* 0.8B K2, about 30 min.
- *Gain:* 0 to −1%.
- *Conflicts:* #39.

**D8 · Within-tile weighted Viterbi: check headroom first.**
Compute CurveTQ's Γ_b after our 128-point rotation. Implement ω = 1/d_i² in the branch metric only if Γ_b ≳ 0.01 bit for many tensors.
- *Source:* CurveTQ (Γ_b 0.038 → 0.00013 bit under full RHT; weighting inert with rotation, +0.36 ± 0.51).
- *Fit:* per-level.
- *Test:* CPU minutes (part of A19).
- *Gain:* likely about 0.
- *Conflicts:* #57.

**D9 · Longer trellis state (L = 20).**
Raise the bitshift state from 16 to 20 bits: the computed codebook takes a 20-bit input and decode reads a 20-bit window. The Viterbi state space grows 16×.
- *Source:* QTIP Table 10 (L 12 → 16: 6.97 → 6.83 at 2 bit; Gaussian MSE 0.069 vs bound 0.063, so at most about 9% MSE remains at K2).
- *Fit:* format change (all decode kernels and the quantizer).
- *Test:* CPU first: `trellis.py` at L=20 on sampled rows. A 27B requant would then cost about 10+ card-h per level, rented.
- *Gain:* −1 to −2% at K2. *Risk:* large engineering; int8 decode cost.
- *Conflicts:* none.

**D10 · Soft trellis front end, then hard Viterbi and CD.**
Use BCJR forward–backward posteriors over tile paths (finite temperature) to pick a basin, project, then run D1.
- *Source:* MPQ (VAMP+CD 11.63 vs LDLQ+CD 12.76); BCJR-QAT (2605.10655, hours per layer at 1B).
- *Fit:* per-level; heavy.
- *Test:* CPU sampled rows.
- *Gain:* −1 to −3% at K2 beyond D1. *Risk:* very costly.
- *Conflicts:* none.

**D11 · Held-out gating for every refinement.**
Accept any post-LDLQ refinement (D1, D7, A17) only if the proxy drops on an independent split-half H.
- *Source:* DRQ (2610.11226).
- *Fit:* process (split-half Hessians exist from the D2 work).
- *Gain:* protects other ideas from proxy overfit.
- *Conflicts:* none.

**D12 · LDL processing order by curvature.**
Order column blocks by Fisher or H trace before rounding (an act-order analogue).
- *Source:* XOR-Trellis (8.06 → 8.00 PPL); H_Z-PTQ.
- *Fit:* per-level.
- *Test:* CPU proxy on 6 tensors.
- *Gain:* about 0 after the 128-point Hadamard.
- *Conflicts:* none.

## E. Format: exact splitting, permutations, side branches

**E1 · Sensitivity-sorted sub-MLPs with independent K.**
Permute the intermediate dimension by a neuron sensitivity (H_out diagonal of gate+up, or A11's local version), split into G = 2–4 contiguous neuron groups (multiples of 128), and store each group's gate/up/down as separate tensors. y = Σ_g W_down,g·(SiLU(W_gate,g·x) ⊙ W_up,g·x) is exact. Each group becomes a knapsack item with its own K.
- *Source:* new for EXL3. Reverse water-filling (rate-distortion); AlignQuant (2610.07457: sensitivity placement 12.61 vs random 12.88 PPL); WINDQuant (sub-tensor granularity helps).
- *Why:* the per-neuron ceiling is about 0.25 bit at CV 1 on the family that is about 70% of trunk weights. Between-block allocation is exactly what the current feedback cannot reach. E3c's 0.05-bit ceiling saw only the input side, which cannot tell rows apart.
- *Fit:* per-level (each group at every K once; per-group probes for kfit2), plus a kernel change (E5).
- *Test:* 0.8B: G = 2/4 by Sketch B diagonal, K per group from a small proxy knapsack, vs uniform at equal bits, about 3 h. Read it against the A19 between-block share.
- *Gain:* −2 to −6% at 2.5 (uncertain; overlaps the diagonal feedback). *Risk:* some within-group feedback is lost; more launches (prefill −1–3% unless E5); kfit2 needs group terms.
- *Conflicts:* #29-02 (E3c), the BACKLOG row-slab item (dropped for an output-side reason that no longer holds).

**E2 · Head-group splitting for attention and GDN.**
The same idea along heads:
- attention by KV group (q/k/v/o slices: 6 q heads, 1 KV head, 6 o-column blocks);
- GDN by head groups (qkv, z, a, b, out slices).

Group heads by summed H_out diagonal or by half-life (C11). Exact.
- *Source:* E1; 2609.26173 (V carries 38–51% of attention sensitivity); STEPQuant.
- *Fit:* per-level plus kernel.
- *Test:* 0.8B, 2 head classes, about 2 h.
- *Gain:* −1 to −3%. *Risk:* GQA limits the freedom.
- *Conflicts:* none.

**E3 · Column-block K on LDL pivots.**
The loss is Σ_j D_j·‖ẽ_j‖², so uniform K costs ∝ AM of the pivots, while per-block rates could approach GM. Split the input dimension at 128-boundaries (W·x = W₁·x₁ + W₂·x₂, exact) into high-pivot and low-pivot blocks, each a knapsack item.
- *Source:* the WaterSIC argument in 2608.25188 (a full rotation closes most of it, about 0.1 bit; ours is only blockwise).
- *Fit:* per-level plus splitting.
- *Test:* CPU (A19). Proceed only if ½log2(AM/GM) > 0.1 bit on big tensors.
- *Gain:* 0 to −3%. *Risk:* my estimate says LDLQ already removes about 127/128 of a rank-1 massive-channel spike from the loss (first pivot ≈ λ/128), so likely small.
- *Conflicts:* #29-02.

**E4 · Global residual-stream permutation.**
RMSNorm commutes with a permutation of the residual stream (gains permuted), so one model-wide permutation is exact and free. It can group the massive channels into one 128-block or spread them out. It enables E3 and changes every residual reader's and writer's Hadamard-block contents.
- *Source:* new (an exact symmetry; R1 failed because it folded norm gains, which a permutation doesn't need).
- *Fit:* per-level (a full requant).
- *Test:* CPU proxy on 3 layers (grouped vs spread), then a 0.8B quant, about 1 h.
- *Gain:* 0 to −2%. *Risk:* changes A8 activation scales; R1's lesson says keep the outlier inside one block, i.e. group.
- *Conflicts:* #47.

**E5 · Grouped-GEMM kernel with per-group K.**
Reuse the MoE expert path from the Qwen Flash backlog for E1/E2: one launch over groups with a per-group trellis pattern.
- *Fit:* kernel.
- *Test:* microbench gate_up with G = 4 vs 1 at M = 2048 on card 0.
- *Gain:* speed only.
- *Conflicts:* BACKLOG grouped GEMM.

**E6 · No output Hadamard plus per-16-row K for gate/up (radical).**
Drop the output-side rotation on gate/up, sort neurons, and vary K per 16-row tile strip by water-filling on d.
- *Source:* CurveTQ ("weighting and rotating are substitutes"), XOR-Trellis.
- *Fit:* format change.
- *Test:* 0.8B uniform K without vs with the output Hadamard. Abandon if no-Hadamard loses more than 5%.
- *Gain:* speculative, 0 to −5%. *Risk:* high (row incoherence).
- *Conflicts:* none.

**E7 · Top-pivot full-precision side columns.**
Keep the first r rotated input columns (the largest LDL pivots) at K8/bf16, as in augmented-Hessian "intrinsic LoRA".
- *Source:* 2606.01412.
- *Fit:* per-level/format (a column split with a K8 block).
- *Test:* CPU: the fraction of tr(E·H·Eᵀ) carried by the top-r pivots for real library errors.
- *Gain:* likely <1% (see E3).
- *Conflicts:* none.

**E8 · Whitened low-rank residual correction for spike tensors.**
For the few extreme tensors (L63 ffn_gate_up at K2 costs +0.0065, 80% of a 4-bpw base; L60–63 down), offer a rank-r bf16 correction from the SVD of D^½·(W − Ŵ)·H^½ as a knapsack option. At r = 32 on L63 gate_up that is about 2.5 MB, or 0.0008 bpw.
- *Source:* ARCHead (2608.02703), ResCom (2610.35955).
- *Fit:* per-level option.
- *Test:* CPU: the energy captured at r ∈ {8, 32, 128}. LDLQ error is close to white, so expect about r/min(m,n); implement only if >10%.
- *Gain:* likely about 0. *Risk:* an extra kernel branch.
- *Conflicts:* none.

**E9 · Mean (bias) correction of residual writers.**
Closed-form per-channel c = E[Y − Ŷ] for down/o/ssm_out, layer-sequential and damped, added as an epilogue bias.
- *Source:* SSDi8 (2608.21952, +0.2 pt); ProQuant.
- *Fit:* final pass, or per level.
- *Test:* 0.8B one pass.
- *Gain:* 0 to −1%. *Risk:* exl3i needs bias support.
- *Conflicts:* none.

**E10 · Softmax-shift sweep for the head.**
The mean-row correction is t = 1 (−30% head KL). Sweep W − t·μ·1ᵀ, t ∈ {0.5, 2, 4}, in `headsim`.
- *Source:* 2609.31291 (Phi picks t = 4; about 0 gain on Qwen3.5 with GPTQ).
- *Fit:* per-level.
- *Test:* CPU hours.
- *Gain:* ≤0.3% total.
- *Conflicts:* #34.

**E11 · Activation-metric low-rank head residual.**
ARCHead's rank-6 INT8 correction in the C^0.75 metric, on top of the slab head.
- *Source:* 2608.02703 (1.0058 vs 1.0104 relative PPL against a GPTQ head at about 4.1 bits).
- *Fit:* per-level option.
- *Test:* `headsim`.
- *Gain:* ≤0.3% total.
- *Conflicts:* #34.

**E12 · Embedding quantized with a residual-channel weight from the layer-0 Fisher.**
The trellis embedding uses H = I. A 5120-vector of layer-0 input sensitivity (from the diagonal collection) as a column weight, rotated by the row's 128-point Hadamard, gives within-row feedback.
- *Source:* new (a twin of A16).
- *Fit:* per-level (embedding donors).
- *Test:* CPU embedding sweep, then one 27B swap.
- *Gain:* about −0.3% total (more at 2.0, where the embedding hits its floor).
- *Conflicts:* #35, #32 (not frequency weighting).

## F. Healing / QAD

**F1 · Scale-only KL healing with zero storage cost.**
Freeze trellis codes. Train log-offsets of suh and svh (s = s₀·e^r), all RMSNorm gains (block, q/k-norm, GDN gated norm), and head/embedding row scales. Loss: forward KL to bf16 on cached top-512 teacher logits, gen-masked (exactly the score), with an r² trust region; keep the best held-out checkpoint.
- *Source:* ThinkBit (2610.34886: scales+norms KL 0.641 → 0.282 at ~2 bpp; scales do most of it; weight-MSE-optimal scales uncorrelated with good ones), Tetra (2609.35465: +1.7–3.3 MMLU in 1–2 GPU-h), RoleQ (0.024% of params), ProQuant (norm folding).
- *Fit:* training, a final pass per shipped size.
- *Test:* 0.8B first (hours on the 4070). 27B: a 2.5-bpw student (~8.4 GB) plus checkpointed activations on one R9700 with cached teacher logits; 1–3 days per size, or 10–20 rented GPU-h.
- *Gain:* −5 to −20% at 2.5 (field report: full QAD on EXL3 gave −6…−13%; scale-only is a subset). *Risk:* data rules (qad pool only).
- *Conflicts:* ROADMAP V5. The proposal is to run it early on 2.5.

**F2 · Healing data: gen-masked and tail-selected.**
Score qad documents by the mean of their top-25% token KL under the shipped mix (one forward), and heal on the top-k plus a random share.
- *Source:* FidelityQ (2610.14934: the worst decile holds 45% of KL; 17.05 → 16.69 vs random selection).
- *Fit:* training.
- *Test:* 0.8B A/B within F1.
- *Gain:* 0 to −2% extra.
- *Conflicts:* none.

**F3 · Block-wise healing with cached teacher block outputs.**
A memory-cheap first stage: per segment (widths 1→2→4 blocks), match bf16 block outputs given quantized-prefix inputs, with a normalized (RMSE) loss so early blocks are not starved; then a short end-to-end KL stage.
- *Source:* JustQuant (2609.33601: full QAD collapsed, staged worked), 2610.00983 (RMSE vs MSE on Qwen3.5-27B W2A16: 37.51 → 22.20 PPL), 2609.19213 (4.2 vs 20.1 GB).
- *Fit:* training.
- *Test:* 0.8B.
- *Gain:* makes F1 fit our cards; quality ≈ F1.
- *Conflicts:* none.

**F4 · Late-layer-only healing.**
Heal layers 48–63 only, vs all layers.
- *Source:* the field report (22% when training only the last 16 vs 6–13% for all); the kfit2 spike at L63.
- *Fit:* training.
- *Test:* 0.8B, last quarter of layers.
- *Gain:* within F1, cheaper.
- *Conflicts:* none.

**F5 · GDN parameter healing.**
Add A_log, dt_bias, conv1d, the GDN norm gains and the a/b projections (all bf16 and tiny) to F1's trainable set. They set state retention and readout and can compensate for in_proj errors.
- *Source:* 2609.04098 (log-space gates are robust; GDN errors don't compound).
- *Fit:* training.
- *Test:* 0.8B F1 with and without.
- *Gain:* 0 to −3% extra.
- *Conflicts:* none.

**F6 · One-shot Gauss–Newton scale healing (no training loop).**
One sampled-label backward on the quantized mix gives per-sequence gradients with respect to svh, suh and norm gains. Take one damped GN step, Δr = −(JᵀFJ + λI)⁻¹·Jᵀ∇KL, using the per-sequence gradient outer products, followed by a line search on the search pool.
- *Source:* new (Gauss–Newton); the closed-form refits in ARCHead and JARQ.
- *Fit:* final pass per mix.
- *Test:* 0.8B: one 1K pass on a 2.5-equivalent mix, step, eval. About 1 h.
- *Gain:* −2 to −8% (a fraction of F1, at near-zero cost). *Risk:* a single step can overshoot.
- *Conflicts:* none.

**F7 · STE index QAD with periodic re-Viterbi.**
Keep latents in the rotated domain, forward on the decoded trellis, STE backward, re-encode every ~16 steps; then run F1.
- *Source:* TVQ (2610.41822: Llama3-70B 4.97 → 4.48 PPL, but ≤0.2 PPL at 7–8B and none on Llama2-70B, at 8× QTIP compute); E8-QAT (2610.27699: +24 LCB at 2 bit); the BCJR-QAT drift budget.
- *Fit:* training, per mix.
- *Test:* rented multi-GPU (fp32 latents plus Adam ≈ 300+ GB), roughly 200–500 GPU-h.
- *Gain:* −3 to −10% beyond F1 (uncertain). *Risk:* cost; lr × steps must exceed the code's Voronoi radius.
- *Conflicts:* V5.

**F8 · Long-window healing.**
Spend part of the budget on 16–32k windows; long traces dominate the score's tokens.
- *Source:* QATFactory (2609.39223: 32K beat 4K by +1.9 at equal tokens).
- *Fit:* training (rented for 27B at 32k).
- *Gain:* protects long-context positions.
- *Conflicts:* none.

**F9 · Heal with int8 activations on.**
Fake-quantize A8-g128 in F1's forward so scales adapt to it.
- *Source:* CanonQ (2610.09202).
- *Fit:* training.
- *Gain:* small at 2.5 (A8 is about free).
- *Conflicts:* #26.

**F10 · Don't heal in FP and then re-quantize.**
Guidance, not an experiment. FP fine-tuning followed by re-PTQ did nothing at W2–W3 (DeiT: 7.328 vs GPTQ 7.313 loss), while STE-QAT at the deployed point recovered.
- *Source:* 2606.09012.
- *Fit:* process.
- *Conflicts:* the ROADMAP V5 idea "QAD before the trellis step" (deprioritize it).

**F11 · Head-only scale healing via `headsim`.**
Train the head's row scales (and the slab boundary scales) against *exact* head-only KL on the 32,768 cached head inputs. No model forward is needed: minutes on a GPU or hours on CPU.
- *Source:* new (healing restricted to where we have an exact cheap objective).
- *Fit:* per-level (head donors).
- *Test:* `headsim`, paired vs uniform, then one 27B swap.
- *Gain:* −10…−30% head KL, i.e. −0.5…−1.5% total. *Risk:* overfitting the cached inputs; use the held-out head inputs.
- *Conflicts:* #34.

**F12 · Final-hidden-state adapter.**
A small bf16 linear map (rank 16–64, residual form) on the final normed hidden state before the head, trained with KL. It fixes systematic logit shifts accumulated through the trunk for about 0.6 MB.
- *Source:* new; 2609.11716 (the final error is mostly an angular rotation of about 12.5°).
- *Fit:* training, final pass.
- *Test:* 0.8B, about 1 h.
- *Gain:* 0 to −3% (per-token rotations aren't linear-fixable; systematic ones are). *Risk:* low.
- *Conflicts:* none.

## G. Per-mix final passes

**G1 · Late-layer sequential requant for shipped mixes.**
For each shipped size, requantize layers 48–63 with GPTAQ targets using the mix's *actual* quantized prefix (D2, exact for the mix).
- *Fit:* per-mix.
- *Test:* 27B, 16 layers, about 1–2 h per size if a 256k-token capture fits host RAM; otherwise rented.
- *Gain:* −2 to −5% at 2.5.
- *Conflicts:* none.

**G2 · FISL-screened local search.**
Rank candidate swaps by FISL-predicted ΔKL including cross terms; spend evals only on the top few per round.
- *Source:* CASA (cross-term screening 22.41 vs self-term 28.15 PPL at 2.25 bpw).
- *Fit:* per-mix search.
- *Gain:* faster convergence (indirect).
- *Conflicts:* #70.

**G3 · REAL-Q-light on late layers.**
Block-output Fisher (5120² per block) plus one Adam step on unquantized columns after each column block, layers 48–63 only.
- *Source:* 2609.00049 (−49% KL at 1.7B, −5…−9% at 8–70B).
- *Fit:* per-mix.
- *Test:* rented, 10–20 GPU-h per size.
- *Gain:* −2 to −5%.
- *Conflicts:* ROADMAP (REAL-Q).

## H. Numerics and evaluation efficiency

**H1 · Engine floor audit.**
Run radiance's `qwen35_bf16` (TP2) against the golden on a short slice and one long trace, with KL by position bin. The GDN state is f32, but chunk-scan intra-chunk math and attention accumulation are bf16 at 250k positions.
- *Source:* B4 (#52: the reference quant's own KL doubles past 32k), #49.
- *Fit:* diagnostic.
- *Test:* both cards, about 1 h.
- *Gain:* 0 to −5% at 4–5 bpw (a floor of 3e-4 is 4% of 0.0083); about 0 at 2.5.
- *Conflicts:* #49, #74.

**H2 · Control variates for paired evals.**
Estimate ΔKL(A,B) with a control variate: per-document KL of a fixed reference mix whose corpus mean is known from one big eval, applied as −β·(slice mean − population mean). This removes slice-composition variance.
- *Source:* Monte Carlo control variates (cross-area).
- *Fit:* eval.
- *Test:* CPU on V2's stored per-row scores: variance reduction over 100 mix pairs. Adopt above 30% (≈1.4× evals/h).
- *Gain:* indirect, −0.5 to −2%.
- *Conflicts:* none.

**H3 · Successive halving over nested slices.**
Score many local-search candidates on the first quarter of the slice, keep the top half, extend, reuse prefix scores.
- *Source:* Loopy (2610.05265, 4.2× less selection time).
- *Fit:* search. Needs the resident-base runner.
- *Gain:* indirect.
- *Conflicts:* none.

**H4 · Fragility-stratified search slices.**
Stratify documents by reference-mix KL and sample per stratum with weights. Per-document KL is heavy-tailed (mean ≈ 20× median in 2610.20826).
- *Fit:* eval.
- *Test:* CPU on V2 scores.
- *Gain:* indirect.
- *Conflicts:* none.

**H5 · Truncation consistency.**
Fisher collections (A1, B1, C1) and healing (F1) must use the scorer's exact top-512 renormalization and gen mask. The head-tail mass (the slab-head guard) is where a mismatch would bite.
- *Fit:* process.
- *Conflicts:* none.

## I. Data and calibration

**I1 · Self-generated calibration for H_in.**
Build H_in from the bf16 model's own continuations of calib prompts (turboderp-SC-style), blended with calib.
- *Source:* turboderp SC notes; ThinkBit/SAGE use self-generated data.
- *Fit:* per-level.
- *Test:* 0.8B: 2M self-generated tokens, blend, K3, about 1.5 h.
- *Gain:* 0 to −3%.
- *Conflicts:* the §1d notes.

**I2 · More H_in tokens (xcorp), with shrinkage.**
D2 measured about 9% in-sample fit at 1.2M tokens per half. Extend H_in with xcorp after a rules check.
- *Fit:* per-level.
- *Test:* 0.8B calib vs calib+xcorp at K3. The 27B needs 1–2 nights of forward.
- *Gain:* 0 to −2%. *Risk:* data rules.
- *Conflicts:* #61.

**I3 · Gen-span-aligned collection windows.**
Draw Fisher windows so that scored assistant spans sit late in the window behind real context. That gives more scored positions per backward (A1) and realistic attention lengths. Add some 8k windows for the full-attention layers.
- *Fit:* per-level.
- *Gain:* efficiency of A1.
- *Conflicts:* none.

---

## Checked, not recommended

- **Stored start state per trellis tile:** #57's gap is 4–6% proxy error; a 16-bit start state costs 0.0625 bpw, which buys about 9% error if spent on K instead.
- **Entropy-coded or variable-rate weights** (EntroPack, SoftWater's rate mode): the trellis bitstream is near-uniform, and rANS decode cost is prohibitive.
- **Learned rotations or seeds** (HARP, ThinQuant, best-of-N): #41 found a 0.5% spread. HARP on QTIP 3-bit gained about 0.7% for +0.05 bpw.
- **Reviving A4 via PrismQuant-style alignment:** #46. The solver put ≤14% of FLOPs on A4 even under maximum speed pressure.
- **On-policy distillation** (OnQ, OnlineQAT): our metric is teacher-forced on fixed text; forward KL is the matched objective.
- **Shapley or RL allocators:** #16, and WINDQuant took 49 h at 8B.

---

## Ranked top 20 (expected gain per unit cost)

| # | Idea | Gain @2.5 (KL) | Cost | Why this rank |
|---|---|---|---|---|
| 1 | **A1** gen-masked Fisher | 0 to −3% | 1.5 h 4070; free in the 27B run | Must be decided before the 8.5 h run |
| 2 | **A11** local analytic H_out (gate/up) | ≈ diagonal's share, cheaply | 1 h 4070; 27B forward stats 2–3 h | Could shrink the 27B run and unblocks E1/A4 now |
| 3 | **A9** diagonal + low-rank sketch | −1 to −2% over diagonal | 1 h now; +few % to the 27B run | Offline test on existing snapshots |
| 4 | **F11** head scale healing in `headsim` | −0.5 to −1.5% | GPU/CPU hours | Exact objective, no model forward |
| 5 | **A16** two-sided head with exact Fisher diagonal | −0.5 to −1% | CPU hours | The factor is known in closed form |
| 6 | **F6** one-shot Gauss–Newton scale healing | −2 to −8% | ~1 backward per size | Most of healing's lever for minutes of compute |
| 7 | **A4** head-interleaving output permutation | −1 to −4% | 1 h proxy; small kernel work | Targets the highest-CV families |
| 8 | **D3** activation-denoising LDLQ | −2 to −5% | 1 h proxy; 1 night 27B | Mix-independent, fits the library |
| 9 | **F1** scale-only KL healing | −5 to −20% | days per size (or ~20 rented GPU-h) | Largest single lever; planned anyway |
| 10 | **E1** sorted sub-MLPs with own K | −2 to −6% | 3 h proxy; library + kernel | Reaches between-block spread on 70% of weights |
| 11 | **D1** trellis CD sweeps | −2 to −5% at K2 | 1 h proxy; 2–3× quant time | Proven big at 2 bit elsewhere |
| 12 | **C3** two-sided proxy in kfit2 | −0.5 to −1% (more ≥4 bpw) | CPU | Tests the "error moved to output directions" story |
| 13 | **H2** control variates for evals | indirect | CPU | Free search speedup if variance drops >30% |
| 14 | **B6** output-estimator comparison | picks the estimator | 1.2 h | Cheapest way to choose the 27B estimator |
| 15 | **A7 + A8** tempering and shrinkage | 0 to −2% | 1.5 h | Fixes K2 full-factor harm |
| 16 | **C1 + C2** FISL + interaction knapsack | −3 to −8% | 4 h proxy + dev; +3–6 h at 27B | High value only if B1/C5 validate it |
| 17 | **D2** asymmetric targets per level | −3 to −8% | 2 h proxy; rented at 27B | Strong evidence on our proxy family |
| 18 | **G1** late-layer sequential requant per mix | −2 to −5% | 1–2 h per size | Concentrates on the sensitive end |
| 19 | **A13** operating-point H_out | 0 to −3% | 1 h proxy; a 2nd 27B run | Mixed literature evidence |
| 20 | **E2** head-group splitting | −1 to −3% | 2 h proxy; kernel | Complements A4 and E1 |

## The 5 cheapest diagnostics that would most change this ranking

1. **Headroom map (A19).** CPU, minutes, on the 0.8B snapshots now and the 27B diagonal later.
   - Inputs: per-tensor within- vs between-128-block ½log2(AM/GM) of d, pivot AM/GM (E3), Γ_b (D8), and the fraction of the within-block ceiling that bdiag already captures.
   - If between-block headroom is large on q/k/GDN, A4/A5/E2 rise. If it is large on MLPs, E1 rises. If small everywhere, all three fall and A3/E3/D8 are dropped.
2. **Gate-2d mechanism on the 0.8B with the 8K diagonal (A2).** 20 min on the 4070.
   - If out-scales now win, gate 2d's failure was data quality, and A6 plus between-block scaling rise.
   - If they lose, only feedback and exact splitting remain.
3. **Gen-masked vs all-position 1K diagonal (A1).** 1.5 h on the 4070. Decides how the 8.5 h 27B run is configured.
4. **Local analytic vs Sketch B diagonal (A11).** Under 1 h: per-family correlation plus one bdiag arm.
   - If local d gives ≥80% of the gain, the 27B work reshapes into a forward-stat pass plus a truncated backward (A18), and E1/A4 start before the run.
5. **Exact Fisher check (B1).** About 40 min. Compute Σ_b⟨E, G_b⟩² for the existing #33-10/11 arms and compare with measured KL and with each Kronecker prediction.
   - If it tracks KL, FISL (C1/C2/C12/G2) moves into the top five, and the "which H_in" question is settled empirically.
   - If it doesn't, the interaction-modeling ideas drop and the effort goes to healing (F1/F6).

Runners-up, if a sixth slot opens: per-family attribution (A14) and the 2B additivity audit (C5).
