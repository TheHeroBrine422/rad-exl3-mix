I've mined about 130 cross-area abstracts plus selected full texts, and checked the quantizer and plugin code. Writing the report now.

# Cross-area research ideas: crossarea lens (2026-11-02)

## Two things from the code that the ideas below rely on

1. **How the diagonal H_out actually helps.** In `prepare_H_out` (exllamav3-frac `quantize.py`), the diagonal $h$ is multiplied by the output scale vector ($s\,h\,s$, with $s$ = `sv`). It is then rotated on both sides by the 128-point output Hadamard. So a diagonal H_out reaches `ldlq_2hess` as dense 128×128 blocks.
   - The win in #33-12/#33-13 is therefore **error feedback across output channels inside each 128-row Hadamard block**. In signal-processing terms it is noise shaping, like sigma-delta: error is pushed from high-$h$ rows onto low-$h$ rows.
   - The high-rate ceiling per block is $\mathrm{GM}(s^2h)/\mathrm{AM}(s^2h)$ in error, i.e. $\tfrac12\log_2(\mathrm{AM}/\mathrm{GM})$ bits. This follows from $\mathrm{tr}(D)\ge n\det^{1/n}$ for LDL pivots.
   - `block_ldl(Ho,16)` only feeds back between 16-row sub-blocks. The Viterbi metric inside a tile stays Euclidean.
   - Gate 2d (#33-06) applied the same information as output *scales*, which is a different mechanism. That explains how "diagonal output scales hurt" and "diagonal H_out helps" can both be true.
   - Many of the ideas below exploit or stretch this mechanism.
2. **GDN recurrent state is already fp32.** `qwen35_exl3.cpp` declares it as `sd.dtype = RAD_F32`. The ~10 state-quantization papers (DAMP, LeapQuant, STEPQuant, PACE, …) therefore have nothing to fix in our eval or serving path, and I left them out.

Gains are quoted as KL change at 2.5 bpw. Proxy-to-KL conversion uses kfit2's KLD ∝ proxy^1.004.

---

## Theme A: what to add to the 27B diagonal-H_out collection (decide before the 8.5 h run)

### A1. Noise-shaping gain map (diagnostic)
For every linear, compute the predicted feedback gain $\sum_b 128\,\mathrm{GM}_b/\sum_b 128\,\mathrm{AM}_b$ of $s_i^2h_i$ per 128-block. Compute it two ways: (a) for whole 128-blocks; (b) restricted to inter-16-sub-block feedback, i.e. the log-det captured by 16×16 pivots compared with the full 128 block. Correlate both with the measured per-tensor two-sided proxy reduction on the 0.8B.
- **Source:** transform-coding gain; sigma-delta/error diffusion (2609.17276, 2610.34566); the det-bound in 2610.18139 Prop. 1.
- **Why:** 0.8B diagonal CV ranges from 3.96 (in_proj_z) and 1.69 (qkv) down to ~0.2 (residual writers). For a lognormal, CV 1 corresponds to AM/GM ≈1.4 (≈0.25 bit). The map tells us which tensors carry the gain and how much the 16-sub-block structure leaves unused. That decides A6, B6–B10 and G1.
- **Fit:** diagnostic.
- **Test:** CPU on the 0.8B 8K/1K diagonals plus the `sv` dumps; minutes. It counts if predicted and measured per-tensor gains correlate with ρ > 0.6.
- **Gain/risk:** 0 directly; it reorders the theme. No risk.
- **Conflicts:** none.

### A2. Re-run the gate-2d mechanism on the 0.8B with the Sketch B diagonal
Run `sensitivity_out_scales` (output scales from diag(H_out), one-sided LDLQ) with the *same* 8K Sketch B diagonal that works through `ldlq_2hess`.
- **Source:** KronQ output pre-scaling $S_G=(h/a)^{1/4}$ (2607.07964, BiIP, which helped there); C-PTQ diagonal Fisher scaling (2607.21076).
- **Why:** this separates "the 27B gate-2d estimator was bad" (200 sampled sequences, |x|-weighted rows) from "scaling is the wrong mechanism". If scaling also wins on the 0.8B, the 27B failure was the estimator. That changes how far to trust gates 2b/2c too, and revives B8.
- **Fit:** diagnostic.
- **Test:** 0.8B K3/K2, two arms, ~10 min on the 4070. Counts if output scales beat base with z ≤ −2.
- **Gain/risk:** information only. Low risk.
- **Conflicts:** #33-06 (D1). This time the estimator is the one known to work.

### A3. Token-wise (K-FAC-expand) vs sequence-summed (Sketch B, K-FAC-reduce) diagonal at equal data
Collect, in the same backward, both $\mathrm{diag}\sum_b G_bG_b^\top$ (Sketch B) and $\sum_t \|x_t\|^2 dy_{t}^{\odot2}$ (token-wise). Feed each into the `bdiag` arm.
- **Source:** KFAC-SVD (2610.32878): token-wise factors avoid the rank collapse of sequence-gradient Fisher. KFAC-expand/reduce is the standard weight-sharing distinction.
- **Why:**
  - Gates 2b/2c (#33-04/05, 27B, 200 sequences, token-wise) failed; Sketch B at 1K+ on the 0.8B wins. Nobody has compared the two estimators at equal data.
  - Token-wise drops cross-position terms (attention couples label noise across positions) but has ~2M samples instead of 1K. If it matches Sketch B, the 27B collection gets a cheaper, lower-variance option.
  - Either way it tells us whether the old 27B failures were estimator or sample size.
- **Fit:** collector.
- **Test:** 0.8B collector variant, 1K windows (~20 min), plus 4 arms (16 min).
- **Gain/risk:** could replace or augment the 27B estimator. Low risk.
- **Conflicts:** #33-04/05, #33-13.

### A4. Store per-sequence diagonals in the 27B run
Write each sequence's diagonal contribution (≈10 MB × 1024 ≈ 10 GB), not only snapshot sums.
- **Source:** robust statistics (median-of-means, winsorizing); bootstrap; 2605.20756 cross-fitting.
- **Why:**
  - YAQA notes that H_out is "dominated by few high-gradient tokens".
  - Per-sequence storage makes the following possible after the fact, with no new 8.5 h run: robust means, tempering (B3), split-half reliability per tensor, n_eff, bootstrap pricing uncertainty (F2).
- **Fit:** collector.
- **Test:** none needed; cost is disk only.
- **Gain/risk:** option value. Nil risk.
- **Conflicts:** #33-13 (add-on).

### A5. Diagonal + rank-r H_out sketch in the collector (spike-and-flat)
Stream a frequent-directions or randomized-Nyström sketch of each linear's per-sequence gradient rows. Keep $H_{out}\approx D + U_rS_rU_r^\top$ with r = 32–64: 17408×64 fp32 ≈ 4.5 MB per linear, versus 1.2 GB for full.
- **Source:** Pro-KLShampoo (2605.06316): Kronecker factors are spike-and-flat. Frequent directions (2610.16792). Nyström SOAP (2610.04204).
- **Why:**
  - On the 0.8B, full beats diagonal at K3 (−9.7% vs −8.0%) but loses at K2 (−5.6% vs −8.0%), because the off-diagonal is noisy (early-layer halves cos 0.69).
  - The low-rank spike is the precisely estimable part of the off-diagonal. It should keep the K3 surplus without the K2 loss.
  - Full gate/up H_out costs 155 GB at 27B; the sketch costs MB.
- **Fit:** per-level.
- **Test:** first on the 0.8B. Eigendecompose the 8K full H_out, keep $D+$top-r for r ∈ {8, 32, 128}, run K3/K2 arms (~30 min). If r = 32 lands within ~1% of the best of full/mixed at both K, add the sketch to `collect.py` before the 27B run.
- **Gain/risk:** −1 to −2% beyond the diagonal. Risk: collector engineering, ~1 day.
- **Conflicts:** #33-12 (`bmix`/full arms). This is a new parametrization.

### A6. Loss only on scored positions (gen mask) in the Fisher collection
Backpropagate the sampled-label loss only from positions the board scores. Our evals mask with `gen.u8`; I'm assuming the board does too. Prompt and tool tokens still contribute as context through attention and state.
- **Source:** importance-weighted ERM / covariate shift; 2610.12327 SparseDecoding (Hessians from the evaluated distribution differ from natural text).
- **Why:** H_out should be the Fisher of *our* objective. The current collector weights every position, while ~half of calib tokens may be unscored prompts and tool outputs.
- **Fit:** collector.
- **Test:** 0.8B, 1K windows with masked vs unmasked loss → `bdiag` K3/K2 arms scored on the gen-masked golden. ~40 min.
- **Gain/risk:** 0 to −3%. Risk: fewer loss positions means more noise.
- **Conflicts:** none.

### A7. Forward-only local output metrics for gate/up (GLU local Hessian)
For SwiGLU, the output-side metric of gate/up row $i$ follows in closed form from forward quantities and down's columns:
- $h^{up}_i \approx \|W_d[:,i]\|^2\,\mathbb E[\mathrm{SiLU}(g_i)^2 w]$
- $h^{gate}_i\approx\|W_d[:,i]\|^2\,\mathbb E[(\mathrm{SiLU}'(g_i)u_i)^2 w]$

Here $w$ is down's own output sensitivity, which is flat (CV 0.17).
- **Source:** SACQ (2610.41408); FastKron's local MLP Hessian $aa^\top\otimes(cc^\top\odot W_d^\top W_d)$ (2608.06291); BoA.
- **Why:** gate/up are ~45% of trunk FLOPs and bytes and dominate the diagonal's gain. If a forward-only (bf16-safe) estimate matches Sketch B, the 27B gate/up diagonal costs a forward pass (~6× cheaper than the fp32 backward, no precision issue) and can use the whole qad pool.
- **Fit:** per-level.
- **Test:** 0.8B, CPU/GPU: cosine of the local diagonal vs the 8K Sketch B diagonal per layer, then `bdiag` arms with local gate/up diagonals and Sketch B for the rest. ~1 h.
- **Gain/risk:** enabler; frees the backward budget for attention/GDN. Risk: misses downstream nonlinearity.
- **Conflicts:** none.

### A8. Gate–up cross term (joint SwiGLU output metric)
$y_i=\mathrm{SiLU}(g_i)u_i$, so errors in gate row $i$ and up row $i$ interact. The joint Fisher of stacked [gate; up] has a per-channel 2×2 block with off-diagonal $\mathbb E[\mathrm{SiLU}'(g)u\,\mathrm{SiLU}(g)\,d^2]$. Collect the cross diagonal $\mathrm{diag}(G^{gate}_bG^{up\top}_b)$ (one extra dot per row). Quantize stacked gate_up two-sided with the 2×2-block H_out.
- **Source:** SACQ GLU metrics; KFAC for tied/coupled layers.
- **Why:** separate per-tensor H_out treats correlated gate/up errors as independent. When the correlation sign is consistent, feedback can make them cancel. Correlation size is unknown; this test measures it.
- **Fit:** per-level (the stacked-gate_up path exists).
- **Test:** 0.8B collector variant, correlation statistics, then stacked arms. ~2 h.
- **Gain/risk:** 0 to −2% on MLP-heavy KL. Medium risk.
- **Conflicts:** none.

### A9. Deep-position H_out via truncated backprop (diagnostic)
For a few long agentic traces, run the forward over 32k–64k tokens of prefix with state/KV detached. Take the loss and backward on a 2k window deep inside; compare the per-tensor diagonal cosine with 2k-window diagonals.
- **Source:** truncated BPTT from RNN training; DASC (2608.30386): retention horizons differ per head.
- **Why:** the collection uses 2048-token windows, while a large share of scored tokens sits deep in 250k traces. If deep-position sensitivity differs (attention layers especially), the collection should mix in deep windows.
- **Fit:** diagnostic → collector.
- **Test:** 0.8B, 20 traces, ~1 h.
- **Gain/risk:** 0–2% if a difference exists. Low risk.
- **Conflicts:** #52 (B4: KL ratio flat with position — evidence against, but it measured quants, not sensitivity shape).

### A10. Operating-point Fisher (H_out of the quantized model)
Collect H_out with the model at uniform level K (dequantized library weights) instead of bf16. The KL expansion is around bf16, but at 2.5 bpw (KL ≈ 0.07) the curvature seen by later layers already includes upstream damage.
- **Source:** GEMQ progressive re-estimation (2605.23078: Mixtral 1.5 bpe PPL 9.66 → 9.01 using a 2-bit quantized model's statistics); 2609.36828 on-policy calibration.
- **Fit:** per-level (one H_out per level band).
- **Test:** 0.8B diagonal from a K3 model vs bf16 → K2/K3 arms. ~1.5 h.
- **Gain/risk:** 0 to −3% at K ≤ 2.5. Risk: doubles collection cost at 27B.
- **Conflicts:** ROADMAP sequential Hessians (input side; this is the output side).

### A11. Stratified window selection for the collection
Pick the 1K qad windows by stratified sampling over category × document length × agentic/long. Use Neyman allocation by the per-stratum variance of diagonal contributions from the 0.8B.
- **Source:** survey sampling.
- **Why:** with only 1K windows, a random draw under-represents rare categories (multilingual, long agentic) whose gradient patterns differ.
- **Fit:** collector.
- **Test:** CPU, from the 0.8B per-sequence diagonals (needs A4 on the 0.8B too). Counts if the variance of the stratified estimator is < 0.7× random.
- **Gain/risk:** same quality from fewer sequences. Nil risk.
- **Conflicts:** none.

### A12. Adaptive budget split in the 8.5 h run
Use the 256/512/1024 snapshots to stop collecting for tensor families whose split-half diagonal cosine already exceeds 0.995. The 0.8B diagonal reaches ≥0.997 at 4K; residual writers need almost nothing. Spend the rest on the sketch (A5) for gate/up/qkv.
- **Source:** sequential estimation / early stopping.
- **Fit:** collector.
- **Test:** none; a design rule.
- **Gain/risk:** better use of a fixed night. Nil risk.
- **Conflicts:** #33-13.

### A13. Variance-reduced Fisher via multiple label samples per forward
Back-propagate k = 2–4 independent sampled labels per forward (or Gaussian logit probes with covariance diag(p) − ppᵀ), reusing the forward. In the 27B fp32 profile the forward is 8.8 s and the backward 13.8 s; k = 2 gives ~1.6× effective sequences per hour.
- **Source:** Monte Carlo Fisher (influence-function literature: EK-FAC, Traceprop 2609.32380).
- **Fit:** collector.
- **Test:** 0.8B dose comparison (1K × 2 labels vs 2K × 1).
- **Gain/risk:** ~1.3–1.6× sample efficiency, mostly relevant to full/low-rank H_out. Low risk.
- **Conflicts:** #33-13 dose.

---

## Theme B: using H_out inside the quantizer

### B1. Two-sided `refit_scales`
`refit_scales` refits `su`/`sv` in the input-Hessian metric only. With H_out:
- the `su` step becomes $((Q\,\mathrm{diag}(h)\,Q^\top)\circ H_{in})r=\mathrm{rowsum}(Q\circ H_{in}W\,\mathrm{diag}(h))$;
- the `sv` step with full/low-rank H_out becomes $((Q^\top H_{in}Q)\circ H_{out})c=\mathrm{diag}(Q^\top H_{in}WH_{out})$.

Closed form, zero inference cost.
- **Source:** SVD-Surgeon (2606.23568: OBS compensation of retained parameters); JARQ (2609.38599: joint scale refit leaves a 21–25% gap if done per group); ThinkBit (2610.34886: KL-optimal scales correlate 0.07 with weight-MSE-optimal ones).
- **Why:** the polish step currently optimizes a different objective from the rounding.
- **Fit:** per-level.
- **Test:** 0.8B `bdiag`/`bhout` arms with two-sided refit, K3/K2, ~20 min.
- **Gain/risk:** −0.5 to −2%. Low risk.
- **Conflicts:** none.

### B2. Two-sided metric for the damping and g_scale sweeps
`--sigma-in-sweep` and the g_scale search choose by the one-sided proxy. With H_out, choose by $\mathrm{tr}(E^\top H_{in}EH_{out})$.
- **Source:** 2610.40345 H_Z-PTQ (Fisher-weighted scale search matches the best tuned L_p without tuning).
- **Fit:** per-level.
- **Test:** 0.8B, sweep-on arms, 20 min.
- **Gain/risk:** −0 to −1%. Low risk.
- **Conflicts:** #38/#39 (same knobs, new metric).

### B3. Tempered H_out ($h^p$, $p\in\{\tfrac14,\tfrac12,\tfrac34,1\}$)
Raise the diagonal (or the eigenvalues of the low-rank part) to a power with a trace-preserving rescale.
- **Source:** ShamAN-Q (2609.38521): tempering p = ½ helped when the error is far outside the quadratic regime (Qwen3-1.7B NKP 19.21 → 17.46 PPL, KL fit + tempering 16.72). Spectral curvature exponents (2606.02596): α varies by layer type.
- **Why:** K2 errors are large; full Fisher weighting over-trusts dominant directions.
- **Fit:** per-level (p may depend on K).
- **Test:** 0.8B `bdiag`, 3 p × 2 K, ~25 min.
- **Gain/risk:** 0 to −3% at K ≤ 2.5. Low risk.
- **Conflicts:** none.

### B4. Ledoit–Wolf shrinkage between diagonal and full H_out
$H=D+\alpha(H_{full}-D)$, with α from the analytic LW intensity estimated from the two 4K halves (per tensor or per family).
- **Source:** shrinkage covariance estimation; 2610.08534 (finite-sample error in Kronecker fits; KL fit lowest).
- **Why:** this does "mixed" properly. The 0.8B shows full winning at K3 and losing at K2, the signature of estimation noise.
- **Fit:** per-level.
- **Test:** 0.8B, α ∈ {LW, 0.25, 0.5}, K3/K2, 25 min.
- **Gain/risk:** −1 to −2% over the diagonal on the 0.8B. At 27B only usable through A5. Low risk.
- **Conflicts:** #33-12.

### B5. Inverse-bias (jackknife) correction of full H_out
LDLQ consumes $H^{-1}$ through LDL, and $\hat H^{-1}$ is biased upward at few samples. Use the split-half jackknife $2\hat H^{-1}-\tfrac12(\hat H_A^{-1}+\hat H_B^{-1})$ projected to PSD, or nonlinear eigenvalue shrinkage.
- **Source:** 2605.20756 (variance-corrected inversion of preconditioners).
- **Why:** the dose curve (full H_out at 1K = 40% of the 8K gain, 2K = 70%) is the expected finite-sample inversion-bias pattern.
- **Fit:** per-level.
- **Test:** 0.8B full H_out at the 1K/2K snapshots ± correction; the result counts if 1K moves above 70% of the 8K gain. 30 min.
- **Gain/risk:** makes full/low-rank viable at small N. Medium risk.
- **Conflicts:** #33-12 dose.

### B6. Stratified interleaving of the MLP intermediate dimension
Permuting the 17408 intermediate channels (gate/up rows plus down columns, same permutation) is function-preserving. Sort by $s^2h$ and deal round-robin into 128-blocks, so each block's GM approaches the global GM. That maximizes the within-block noise-shaping gain from the code note above.
- **Source:** sigma-delta single-trajectory argument (2609.17276); permutation-symmetry codecs (2605.24754).
- **Why:** the shaping gain is capped by within-block heterogeneity. If the natural order clusters $h$ at all, interleaving recovers the gap.
- **Fit:** per-level (one permutation shared by all levels; down's input LDLQ is unaffected in principle).
- **Test:**
  - A1 first: predicted gain natural vs interleaved.
  - If > 0.02 bit, run 0.8B `bdiag` with permuted MLPs, K3/K2. Half a day plus 20 min.
- **Gain/risk:** 0 to −2%. Risk: changes down's input Hadamard grouping.
- **Conflicts:** #42 (siblings share signs, not order; unaffected).

### B7. Larger output-side transform blocks per tensor
The output Hadamard acts on the output vector in the epilogue, not per weight, so its size is a kernel choice. Options: 512 (128⊗4) or full-n via a Kronecker 128 ⊗ (random orthogonal mixer) as in SoloQ's K-RPBH, for tensors with high between-block variance of $h$. Then shaping can work across 512+ channels.
- **Source:** 2610.07121 K-RPBH (block energy imbalance 67.9% → 22.1%); transform-coding gain.
- **Why:** no permutation constraints (works for q/k/v and GDN heads, where B6 is not free).
- **Fit:** format change (exl3i transform field per tensor).
- **Test:** CPU (A1 numbers at 128/512/full) → 0.8B arms by fake-quant with a 512 transform.
- **Gain/risk:** −1 to −3% if between-block variance is real. Risk: kernel work; transform cost in decode.
- **Conflicts:** none.

### B8. Row-slab K driven by the output-side diagonal (homogeneous slabs)
The opposite design to B6: sort rows by $h$ so 128-row slabs are homogeneous, and give slabs different K (reverse water-filling, $b_j=\bar b+\tfrac12\log_2(v_j/\mathrm{GM})$).
- **Source:** SPECTRA/AATC reverse water-filling (2608.07915, 2608.14191); F-WANDA per-row budgets by Fisher (2608.00481); the slab head (#34).
- **Why:**
  - E3c's ceiling (0.05 bit on big tensors) was input-side only.
  - Gate/up output CV ≈1.0 on the 0.8B suggests ≈0.25 bit across slabs.
  - The design note dropped trunk row-slab K because "output-side weighting hurt in every form"; #33-12 removes that reason.
  - B6 and B8 compete for the same heterogeneity; A1 decides which captures more.
- **Fit:** format change (a K per slab; the head machinery exists).
- **Test:** CPU water-filling gain from the 0.8B diagonal vs the shaping gain (A1) → 0.8B fake-quant arms mixing two K by slab.
- **Gain/risk:** −2 to −5% if the 27B $h$ spread matches the 0.8B. High engineering cost.
- **Conflicts:** #29-02 (E3c), design §6b ("row-slab K dropped").

### B9. Per-head K for attention and GDN input projections
For q/k/v and in_proj_qkv/z, natural slabs are heads (128 rows). Aggregate $h$ per head and allocate K per head.
- **Source:** DASC (2608.30386: retention horizons differ by head); STEPQuant (2609.38169: per-head half-life vs error ρ ≈ 0.80).
- **Why:** slab boundaries match the format's 128-row units with no permutation, and head sensitivity is plausibly bimodal (long- vs short-memory GDN heads).
- **Fit:** format (as B8).
- **Test:** CPU, AM/GM across heads from the 0.8B diagonal.
- **Gain/risk:** −0.5 to −2%. Format risk.
- **Conflicts:** #29-02.

### B10. Weighted Viterbi metric from the output-side tile pivots
Inside each 16×16 tile, weight squared error by $D^{in}_{ii}D^{out}_{jj}$ (diagonals of the block pivots).
- **Source:** CurveTQ (2610.09212: $1/d_i^2$ weighting gave +2.4 points rotation-free but was inert after RHT, $\Gamma_b$ 0.038 → 0.00013 bit); XOR-Trellis (2610.00432).
- **Why:** after rotating the diagonal H_out, the 16-sub-block pivots are *not* flat, and that within-sub-block structure is the share of AM/GM the current feedback cannot use (A1-b).
- **Fit:** per-level (quantizer kernel change).
- **Test:** CPU, $\Gamma_b$ on rotated 0.8B H_out blocks; proceed only if > 0.01 bit.
- **Gain/risk:** 0 to −1%. Low risk.
- **Conflicts:** none.

### B11. Structure-averaged H_out for heads (weight-space symmetries)
Average gradient outer products over within-head permutation symmetry (dot-product invariance of q/k under a joint permutation; RoPE pairs restrict this to pair-preserving permutations). The result is a per-head "isotropic + rank-few" H_out from few sequences.
- **Source:** 2606.00442 (symmetry-averaged curvature recovers Shampoo-like estimates).
- **Fit:** per-level.
- **Test:** 0.8B attention/GDN projections, structured vs full vs diagonal, 30 min.
- **Gain/risk:** small. Medium risk.
- **Conflicts:** none.

### B12. Damped feedback late in the sweep (error decay)
Scale LDLQ feedback by $\gamma_i=\exp(-\lambda\,[H^{-1}]_{ii}/\overline{[H^{-1}]}\cdot i/B)$.
- **Source:** QTEA (2609.00224: +0.46 avg points at ternary; noisy).
- **Fit:** per-level.
- **Test:** 0.8B K2, 10 min.
- **Gain/risk:** ≤ 1%. Low priority.
- **Conflicts:** none.

---

## Theme C: input-side Hessian (why gradient weighting hurts, and fixes)

### C1. n_eff and split-half generalization of H_in variants (diagnostic)
For gradient-weighted H_in (gate 2e) and Sketch B H_in, compute the effective sample size $(\sum w)^2/\sum w^2$ of token weights. Quantize with half A and score the proxy under half B and under the calib H_in (D2-style).
- **Source:** 2610.08534 (finite-sample underestimation of small eigenvalues); ShamAN-Q (quartic token weights dominated by massive-activation tokens).
- **Why:** this directly tests the leading explanation for #33-07/#33-10: heavy-tailed weights → tiny n_eff → noisy small eigenvalues that LDLQ trusts.
- **Fit:** diagnostic.
- **Test:** CPU on 0.8B snapshots plus one GPU quant per half, ~1 h.
- **Gain/risk:** settles the "why" question. No risk.
- **Conflicts:** #33-07, #33-10.

### C2. Leverage-bounded token weights (matrix-normal MLE / KL-Shampoo fit)
Fit $H_{in}=\tfrac1m\sum_t(g_t^\top H_{out}^{-1}g_t)x_tx_t^\top$ jointly with H_out by 3 flip-flop iterations, plus shrinkage.
- **Source:** ShamAN-Q (2609.38521): KL-Shampoo's inverse-quadratic token weights are bounded leverage scores; Bregman analysis (2606.00542, 2610.08534): LogDet/KL compensates finite-sample underestimation best.
- **Why:** this is the principled Kronecker fit of the KL Fisher. It keeps gradient information while bounding any token's influence.
- **Fit:** per-level.
- **Test:** 0.8B collector variant storing per-token $(x,g)$ summaries for 1K windows (~40 min) → K3/K2 arms with calib H_out held fixed.
- **Gain/risk:** 0 to −3%. Likely still behind the 10M-token calib H_in alone, so test inside C3.
- **Conflicts:** #33-07, #33-08/10. Different weighting and estimator.

### C3. Shrinkage blend of calib H_in with gradient-aware H_in
$H=(1-\alpha)\bar H_{calib}+\alpha\bar H_{grad}$ (both trace-normalized), α ∈ {0.05, 0.1, 0.25}. Use Sketch B or C2 for $\bar H_{grad}$.
- **Source:** 2610.18139 (objective-aware Hessian $\bar C+\alpha sH_{Jac}$: Llama-2-7B M=8 PPL 9.120 → 8.704; any fixed α > 0 is unsafe, α must be per network).
- **Why:** the blend test was written for gate 2e and never run. The 0.8B now has all factors on disk.
- **Fit:** per-level.
- **Test:** 0.8B K3, 3 arms, 12 min. Counts if any α beats `bhout`/`bdiag` with z ≤ −2.
- **Gain/risk:** 0 to −2%; possibly null. Low risk.
- **Conflicts:** #33-07 (`gate2e-blend.sh` written, unused).

### C4. Scored-position-weighted H_in (bounded weights)
$H_{in}=\sum_t w_tx_tx_t^\top$ with $w=1$ on scored positions and $w_0\in\{0.2,0.5\}$ elsewhere (prompts and tool outputs still enter).
- **Source:** DiffAQ (2610.22395: floor α = 0.2 weights so static frames don't dominate); turboderp's SC calibration (model-written text only).
- **Fit:** per-level (27B needs an H_in recollection: forward only, both cards, a few hours).
- **Test:** 0.8B/2B: H_in from gen-weighted calib (~15–35 min collection) → K3/K2 arms on the gen-masked golden.
- **Gain/risk:** 0 to −3%. Risk: fewer effective tokens (use C5).
- **Conflicts:** #33-07 (not gradient-norm weights).

### C5. Difference (control-variate) estimator for every re-collected Hessian
Estimate $H_{new}=H_{calib}^{(10M)}+(\hat H_{new}-\hat H_{calib})$, with both hatted terms on the *same* sampled windows. The paired difference has low variance.
- **Source:** control variates / difference estimators; cross-fitting (2605.20756).
- **Why:** this directly answers the ROADMAP caveat that our collections have far fewer tokens than the published 10M. It unlocks C4, C6 and C7 at 27B with 0.5–1M tokens.
- **Fit:** per-level.
- **Test:** 0.8B: C4 at 0.5M tokens naive vs difference estimator vs 10M truth. Counts if the difference estimator halves the proxy gap.
- **Gain/risk:** enabler. Low risk.
- **Conflicts:** none.

### C6. Level-K prefix Hessians with the GPTAQ asymmetric target (sequential, per level)
For each library level K, collect $\hat X\hat X^\top$ and $X\hat X^\top$ with upstream tensors at K. Run LDLQ on target $WX\hat X^\top(\hat X\hat X^\top)^{-1}$ (drift compensation). This stays library-compatible because sequential means "per level", not per mix.
- **Source:** GPTAQ (2504.02692 ref); 2610.07522 (sequential vs parallel gap: W3 GPTQ closes 18–31%); G²PTQ (2609.31009).
- **Fit:** per-level (27B: forward of a dequantized level-K model on two cards, per level).
- **Test:** 0.8B K2/K2.5 with C5. ~2 h.
- **Gain/risk:** −2 to −4% at 2.5 bpw. Medium risk (6+ collections at 27B).
- **Conflicts:** ROADMAP "Sequential (quantized-prefix) Hessians" (V3.5). C5 removes the token-count caveat noted there.

### C7. Activation denoising (noise-injection C, N) as the cheap version of C6
Pre-transform $W\to WF$ and use metric $G=H+\lambda^2N+\lambda(C+C^\top)$, with $C=\mathbb E[x\eta^\top]$ and $N=\mathbb E[\eta\eta^\top]$ from one noisy forward. Inject the *actual* level-K quantization errors of upstream tensors from the library, not Gaussian noise.
- **Source:** 2610.07522: QuIP# E8 2-bit on Llama-3-8B, parallel 13.76 → denoised 12.70 vs sequential-asymmetric 12.87.
- **Fit:** per-level.
- **Test:** 0.8B K2/K3, (λ, ρ) grid on the fit pool; KL on search.
- **Gain/risk:** −1 to −3%. Risk: 2–4 knobs (pick per family, not per tensor).
- **Conflicts:** none.

### C8. Sink-token share of H_in (diagnostic)
Measure the fraction of $\mathrm{tr}(H_{in})$ per tensor and per 128-block contributed by the first ~4 positions of each document (massive-activation sink tokens).
- **Source:** attention sinks / massive activations (2609.09085); E3c outliers (L0 ssm_out, L3 attn_output).
- **Why:** if a few repeated near-identical vectors dominate some directions of H_in, LDLQ is effectively calibrating to them. That decides I1 and whether to down-weight or treat them separately.
- **Fit:** diagnostic.
- **Test:** 0.8B/2B Hessian collection with position bins, 30 min.
- **Gain/risk:** information. No risk.
- **Conflicts:** none.

---

## Theme D: trellis and coding theory

### D1. Chained tail-biting: one trellis trajectory per tile row instead of per tile
Today each 256-weight tile is its own tail-biting trellis; the decoder wraps reads with `% words` (`dq8_half`, `wrap_idx`). Instead, let tile c's initial state come from the last bits of tile c−1 along n (one wrap per chain), and run Viterbi jointly over the chain with the existing tail-biting heuristic applied at chain level.
- **Source:** 2609.17276 (error diffusion on the torus: one closed trajectory leaves a single boundary mismatch whose size is ∝ 1/length, instead of a mismatch per row).
- **Why:**
  - #57 measured the tail-biting penalty as free/TB = 0.948 at K2 and 0.962 at K4, i.e. 5.2% / 3.8% proxy error.
  - Trying wrap candidates recovered only 4–12% of that gap. A chain of C tiles leaves ≈1/C of it.
  - Chaining along n stays within one LDLQ step, since all tiles of a 16-row block are quantized together, and Viterbi parallelism is over 2¹⁶ states, not tiles.
- **Fit:** format change: an exl3i decode address change, a reads-previous-tile flag, and a quantizer kernel change. It slots into the planned Phase-B requant.
- **Test:** CPU first: extend `viterbi_tiles_tb` to chains of C = 4/8/all on real tensors from layers 10/43 at K2, K2.5, K3 (hours on CPU box). Counts if ≥ 70% of the free-start gap is recovered. Then a GPU quantizer instance plus a decode selftest.
- **Gain/risk:**
  - **−2 to −4% KL** at 2.5 bpw, roughly 0.03 bpw; similar at every K ≤ 4.
  - Risks: GEMM tile layout (previous tile must be addressable), fractional/sub-1 rates, speculative M-bands.
- **Conflicts:** #57 (judged "not worth a kernel change" for wrap search, which recovers ≤ 12%; chaining is a different, much larger lever).

### D2. Fixed-start chaining along k (cheaper variant)
Tile (a, c)'s start state is the end state of tile (a+1, c), already quantized in an earlier LDLQ step. This needs no joint Viterbi, only a known-start Viterbi.
- **Source:** as D1.
- **Why:** smaller quantizer change; the CPU test shows how much of D1 it keeps (greedy at boundaries).
- **Fit:** format change.
- **Test:** same CPU harness as D1.
- **Gain/risk:** likely half of D1. Low risk.
- **Conflicts:** #57.

### D3. Block coordinate-descent re-Viterbi after LDLQ
Sweep tiles. Re-quantize each against the non-causal target $z=w-H_{tt}^{-1}H_{t,\neg t}e_{\neg t}$ (and the H_out-side analogue). Accept if the total two-sided proxy drops. 2–3 sweeps.
- **Source:** MPQ (2610.32127: Llama-3-8B E8P12 2-bit PPL LDLQ 15.01 → LDLQ+CD 12.76; 4-bit only 6.54 → 6.46); DRQ (2610.11226) warns refinement can overfit H, so validate on split-half H.
- **Fit:** per-level (+2–3× quant time).
- **Test:**
  - 0.8B K2/K2.5/K3 with CD.
  - Check proxy under held-out H_B (D2 split-half) before KL.
  - ~1 day of code plus 1 h of runs.
- **Gain/risk:** −1 to −3% at 2.5 bpw. Risk: overfitting the 10M-token H is low; quant time.
- **Conflicts:** BACKLOG "Quantizer knobs: … re-Viterbi" (untested).

### D4. Delayed decision across LDLQ steps (list/M-algorithm)
Keep the top-2 trellis solutions per tile, measured by the full LDLQ cost including next-block feedback. Choose after quantizing the next 16-row block (M-algorithm / list Viterbi).
- **Source:** list decoding and the M-algorithm in coding; AQLM beam search.
- **Fit:** per-level (~2× quant time).
- **Test:** CPU emulation on a few tensors at K2.
- **Gain/risk:** −0.5 to −2%. Medium risk.
- **Conflicts:** none.

### D5. Soft-decision (BCJR) initialization before CD
Compute trellis posteriors by forward–backward over the rotated tile with a temperature. Use marginal-MAP as the CD start.
- **Source:** MPQ (VAMP + CD 11.63 vs CD 12.76); BCJR-QAT (2605.10655).
- **Fit:** per-level.
- **Test:** CPU on 64 rows at K2.
- **Gain/risk:** −1 to −2% beyond D3. High compute; low priority.
- **Conflicts:** none.

### D6. Event-driven re-encoding for trellis QAD (error-diffusion accumulator)
In index-level QAD, accumulate gradient-driven target drift per tile and re-run Viterbi only when a tile's accumulated drift crosses a threshold aligned with the code's local spacing. This replaces re-encoding every tile every 16 steps.
- **Source:** 2610.34566 (error diffusion replaces latent weights; the threshold-alignment rule); TVQ (2610.41822: re-Viterbi every 16 steps, ~8× QTIP cost, ≤ 0.2 PPL gain on 7–8B).
- **Fit:** training (V5).
- **Test:** 0.8B K2 QAD, cost and KL vs scale-only healing.
- **Gain/risk:** enables index QAD at 27B. Gain unclear beyond scale healing (TVQ is weak). High risk.
- **Conflicts:** ROADMAP V5.

### D7. Block energy ordering (act-order analogue) for LDLQ
Order 16-blocks by $\mathrm{tr}(H_{bb})$ before LDLQ.
- **Source:** XOR-Trellis (2610.00432: 8.06 → 8.00 PPL, ~0.7%); H_Z-PTQ orders by Fisher trace.
- **Fit:** per-level.
- **Test:** 0.8B K3, 5 min.
- **Gain/risk:** 0 to −0.5%. Low risk.
- **Conflicts:** BACKLOG "column order".

### D8. Post-RHT non-Gaussianity flags
Compute $\rho_3$ and crest factor $\|x\|_\infty^2/\|x\|_2^2$ per rotated 128-block. Flag tensors where one RHT leaves spikes; consider a second sign-Hadamard pass for flagged tensors only.
- **Source:** 2605.06014 (two RHTs give Gaussian marginals; three give block decorrelation).
- **Fit:** diagnostic → format (extra input pass) for flagged tensors.
- **Test:** CPU, minutes.
- **Gain/risk:** probably 0. Low priority.
- **Conflicts:** #41 (seeds; a different question).

---

## Theme E: numerics and linear algebra

### E1. fp64 / preconditioned LDL audit for H_out and massive-activation H_in
Compare `block_ldl` pivots in fp32 vs fp64 for: (a) normalized, rotated full/low-rank H_out (gate/up, 1K–8K sequences); (b) early-layer H_in with massive channels.
- **Source:** MRCQR (2606.18411: Gram forms square κ, so fp32 breaks near κ ≈ 1e8); mixed-precision Newton (2607.04828).
- **Fit:** diagnostic.
- **Test:** CPU/GPU, 30 min.
- **Gain/risk:** 0 unless a numerical bug; cheap insurance before full/low-rank H_out. No risk.
- **Conflicts:** none.

### E2. Round-to-nearest and store-dtype audit of V3 int8 epilogues
Check that the int8 path's bf16/f16 downcasts round to nearest (no truncation) and match the reference store.
- **Source:** HEAL (2606.21023: kernel-boundary truncation drives divergence).
- **Fit:** kernels.
- **Test:** selftest diff.
- **Gain/risk:** 0 at 2.5 bpw; matters at ≥ 5 bpw. Nil risk.
- **Conflicts:** design §1c.9 (already partly planned).

### E3. Mean-error (bias) correction for residual writers
Store $c=(W-Q)\mu_x$ per down/o_proj/ssm_out as an epilogue bias, where μ is the input mean (dominated by sink/massive tokens).
- **Source:** SSDi8 mean correction (2608.21952: only +0.2 points); SHIFT-LLM closed-form LS correction (2608.25068).
- **Fit:** format (bias add).
- **Test:** 2B fake-quant, 30 min.
- **Gain/risk:** 0 to −0.5%. Low priority.
- **Conflicts:** none.

---

## Theme F: allocation, pricing and search

### F1. Absolute Fisher pricing of options
Price each (tensor, option) as $\Delta\widehat{KL}=\tfrac1{2N}\mathrm{tr}(E^\top H_{in}EH_{out})$, using the *un-normalized* diagonal Fisher from the 27B run (per-token units) and existing library quants. Compare with V2's ~1,250 measured swaps and kfit2's fitted sensitivities. If log-RMSE ≤ 0.2 at K ≥ 2.5, use it as the per-tensor prior in hierarchical kfit2 and to price new options without evals: fractional, H_out-quantized, slab, sub-2 with care.
- **Source:**
  - KronQ ($S=\mathrm{tr}H_G\mathrm{tr}H_X$; down_proj-first allocation 7.21 vs 7.40 PPL);
  - 2609.28270 (quantization price ½Tr(HΣ): ρ = 0.92–0.95 with measured KL);
  - IO-SVD eq. 4;
  - 2609.33923 on Qwen3.8-27B: layer-local objective allocation (6.087) lost to downstream-aware (6.041), which argues for output-side weighting.
- **Why:** this attacks eval cost and model uncertainty. It is analytic, needs no extra forwards, and could explain the per-family factor puzzles (§1d).
- **Fit:** knapsack.
- **Test:** CPU after the 27B diagonal exists, ~half a day.
- **Gain/risk:** indirect, −1 to −3% via better pricing and fewer wasted evals. Risk: second order fails at K ≤ 2 (KL ≈ 0.16).
- **Conflicts:** #15 (noise injection: needed extra bf16 forwards; this does not), #67/#68.

### F2. Fisher-priced activation-quantization cost per tensor
Price A8/A4 per tensor as $\tfrac12\mathrm{tr}(H_{out}W\Sigma_{\delta x}W^\top)$ with $\Sigma_{\delta x}$ from the E1c activation dumps. This replaces E1c's family factors (log-RMSE 0.26 over 4 orders of magnitude, ±30%).
- **Source:** as F1; ResCom (2610.35955: error $W\,d$ from activation rounding).
- **Fit:** knapsack (speed-aware V3 solve).
- **Test:** CPU, refit on the 45 E1b runs; counts if log-RMSE < 0.2.
- **Gain/risk:** better A8/A4 placement; small KL change at 2.5 bpw (A8 nearly free), larger for A4 and V3.5. Low risk.
- **Conflicts:** #27 (E1c), #46.

### F3. Re-score ours vs turboderp at K5–K8 with the two-sided proxy
Check whether his raised-to-K7/K8 tensors, which V2 picks despite ~23% worse input-side proxy, are better under $\mathrm{tr}(E^\top H_{in}EH_{out})$.
- **Source:** IO-SVD / KronQ (one-sided proxies miss output-direction error).
- **Why:** this explains the open puzzle in design §1d. If yes, H_out-quantized high-K levels of ours would replace his tensors.
- **Fit:** diagnostic → per-level.
- **Test:** CPU, existing tensors plus the 27B diagonal.
- **Gain/risk:** matters ≥ 5 bpw; ~0 at 2.5 bpw. Low risk.
- **Conflicts:** §1d, #50 (D2).

### F4. Fisher-propagated pricing uncertainty for the uncertainty-aware solve
Use the per-sequence diagonals (A4) to bootstrap F1 prices per option and feed the variance into B11's penalty. Options priced from noisy Fisher estimates pay more.
- **Source:** Laplace/GLM predictive uncertainty (2607.00214); SOAP-Bubbles structured uncertainty (2606.23357).
- **Fit:** knapsack.
- **Test:** CPU.
- **Gain/risk:** guards against V0-style extrapolation misses. Low risk.
- **Conflicts:** BACKLOG B11.

### F5. Intra-layer pairwise interaction terms
Add $\lambda q_iq_j\Gamma_{ij}$ for pairs within a layer (gate/up × down; qkv × o; GDN in × out), measured by ~20 targeted joint-vs-single swaps. Solve by enumerating per-layer Pareto combinations inside the layer DP.
- **Source:** correlation-aware BQP (2609.22131); TWLA chain DP; LoopQuant Shapley interactions (Huginn: 12.47 vs 13.05 local / 13.67 marginal); 2609.09867 (composition error 26–42× pricing error at 2.5–3.4 bpw; pair quadratic cuts MAE 5–8×). Counter-evidence: 2609.01587 (pairs ≈ additive, 0.91–1.10) and 2604.04098 (GDN projection errors add in quadrature).
- **Fit:** knapsack.
- **Test:** CPU on V2 residuals by pair type first; GPU swaps only if residuals cluster.
- **Gain/risk:** 0 to −2%. Medium risk.
- **Conflicts:** #11 (symmetric compounding), #16 (full Shapley skipped). These are targeted pairs only.

### F6. Data-free GDN decay-horizon prior (diagnostic)
Per GDN head, compute the retention half-life from `A_log`/`dt_bias` and the gate statistics. Correlate with measured per-tensor sensitivities, and with per-head $h$ (B9).
- **Source:** STEPQuant ρ ≈ 0.80 (half-life vs state error); DASC; 2609.04098 (same model: in_proj_a/b least sensitive, 2.1%/2.6% vs qkv 10.4%, z 9.9%, out 12.7%).
- **Fit:** diagnostic → kfit2 feature.
- **Test:** CPU, minutes.
- **Gain/risk:** small prior improvement. No risk.
- **Conflicts:** #19 (layer offsets; a different feature).

### F7. Super-weight scan from the 27B backward (WAG)
During the collection, track the top-k $|W\odot G_b|$ per tensor (cheap running top-k). Check those weights' quantization errors in library quants. Protect them by raising the containing tensor's K or with a sparse fp16 patch.
- **Source:** WAG (2607.10803: < 0.5 ppm of parameters cause collapse); Seed-Q (four fp16 columns remove 6% of excess loss).
- **Fit:** diagnostic → knapsack/format.
- **Test:** collector add-on; CPU check.
- **Gain/risk:** 0, or a large fix if something is mangled. With RHT spreading, probably 0. Low risk.
- **Conflicts:** none.

### F8. Successive halving / racing in the V3 stream
Score each candidate on a nested prefix of the search slice (~25% of documents) and drop it if a paired sequential test (SPRT) says it is worse than the incumbent. Continue survivors to the full slice. ver3 reads stay as planned.
- **Source:** Loopy (2610.05265: 3.5–4.2× less selection time at the same quality); Hoeffding races.
- **Fit:** search.
- **Test:** CPU replay on V2's per-document scores: fraction of evals cut vs decisions changed.
- **Gain/risk:** 1.5–2× candidates per GPU-hour if the resident-base runner lands; indirect −0.5 to −1.5%. Risk: small-slice bias (2610.20826 saw 24% offset on a 12k slice). Keep prefixes ≥ 75k positions.
- **Conflicts:** #70, design §5 eval-runner proposal.

### F9. Regression-adjusted (CUPED) paired comparisons on log ratios
Model per-document log(KL_candidate / KL_base) and regress on a covariate (the kfit2-predicted per-document change, or base per-document KL) to cut variance beyond pairing.
- **Source:** CUPED / regression adjustment from online A/B testing.
- **Fit:** search/eval.
- **Test:** CPU on V2 per-row data. Counts if SE drops ≥ 20% at equal positions.
- **Gain/risk:** indirect, as fewer documents per decision. Nil risk.
- **Conflicts:** #14 (single-tensor swaps too noisy; this lowers the floor).

### F10. Neyman-stratified search slices
Allocate search-slice documents across category strata by the per-stratum SD of paired log-differences (from V2), rather than proportionally.
- **Source:** survey sampling (Neyman allocation).
- **Fit:** eval.
- **Test:** CPU.
- **Gain/risk:** SE −10 to −30%. The score target stays proportional because strata are reweighted. Nil risk.
- **Conflicts:** #72 (pools unchanged).

### F11. Re-rank finalists after healing
Choose the shipped mix among the top 3–5 measured *after* scale healing (H3/H4), not before.
- **Source:** recovery-heterogeneity results (2610.05504; 2610.15574; QUBO 2609.22238: post-recovery rankings change).
- **Fit:** final pass.
- **Test:** process rule; 2–3 extra heals per size.
- **Gain/risk:** 0 to −2%. Low risk.
- **Conflicts:** none.

---

## Theme G: head and embedding

### G1. Two-sided, noise-shaped LM head
Quantize lm_head with `ldlq_2hess` and $H_{out}=\mathrm{diag}(\mathbb E[p(1-p)])$ (the Fisher already used for slabs). Rows stay in natural token-id order, so each 128-row vocab block is heterogeneous: ~400 rows hold half the Fisher, and one block holds 39%. Feedback can then push error off high-Fisher rows. Compare by headsim against the slab head at equal bytes, and combine: slab-K with Fisher-interleaved order inside slabs.
- **Source:** SoftWater (2608.12026: class-aware head allocation, 6.5–8.3× lower head KL at 2 bits); the noise-shaping note above.
- **Why:** the head is ~5–7% of total KL; slab heads already gave −36 to −40% head KL. Shaping is a second, partly independent lever.
- **Fit:** per-level (head donors).
- **Test:** headsim (exact), two head quants (K4, K5), ~22 GB host RAM each.
- **Gain/risk:** head KL −20 to −50% → **total −1 to −3%** at 2.5 bpw. Risk: the high-rate shaping assumption fails on the extreme spread (regularize via `sigma_reg_out`).
- **Conflicts:** #34 (complements the slab head).

### G2. Full softmax-GGN head factor (diagonal minus low-rank)
Use $H_{out}=\mathrm{diag}(\mathbb E p)-\mathbb E[pp^\top]$ restricted to the top ~2k vocab rows as rank-r, plus the mean-row null direction.
- **Source:** IO-SVD ($C=\mathbb E[J^\top(\mathrm{diag}p-pp^\top)J]$); the softmax-gauge paper (2609.31291).
- **Fit:** per-level.
- **Test:** headsim, as G1.
- **Gain/risk:** a few % of head KL. Low risk.
- **Conflicts:** #34.

### G3. Softmax-gauge t-grid for the head
Search $W_t=W-t\mathbf 1\mu^\top$ over t ∈ {0.5, 1, 2, 3, 4} by headsim, instead of the fixed mean-row correction (t = 1).
- **Source:** 2609.31291 (t = 1 is Frobenius-optimal, not KL-optimal; Phi selected t = 4, W4 GPTQ head KL 0.158 → 0.059; Qwen3.5 GPTQ only 0.011 → 0.010).
- **Fit:** per-level.
- **Test:** headsim, an afternoon.
- **Gain/risk:** 0 to −10% head KL → ≤ −0.5% total. Nil risk.
- **Conflicts:** #34 (mean-row −30% done).

### G4. Gradient-metric embedding (token-agnostic H)
Add one accumulator to the 27B collector: $H_{emb}=\mathbb E[g g^\top]$ of the residual-stream gradient at layer-0 input (5120² fp32, ~100 MB). Quantize the trellis embedding with LDLQ under this metric (rows = tokens, columns = hidden dims) instead of H = I.
- **Source:** "rounding in the consumer's space" (2610.12444); KL-aware whitening (IO-SVD).
- **Why:**
  - The metric is shared by all tokens, so it is not frequency tiering (#32's multilingual concern does not apply).
  - It exploits per-dimension sensitivity, e.g. RMSNorm-damped and massive dims.
  - Trellis embedding vs Lloyd was worth ~1% KL at 3 bpw, and the embedding sits at its floor at ≤ 2 bpw.
- **Fit:** per-level (embedding menu).
- **Test:** 27B: collector add-on, then embedding quants at t2/t3 with H vs I, swap into V2 2.5 on search + golden-ml.
- **Gain/risk:** −0.5 to −1.5%. Low risk.
- **Conflicts:** #32, #35 (neither used a dimension metric).

### G5. Inner-product-preserving embedding row scale
Set the per-row scale to $c=\|z\|^2/\langle q,z\rangle$ instead of MSE-optimal.
- **Source:** SoloQ (2610.07121: scale-only bias correction).
- **Fit:** per-level.
- **Test:** CPU sweep on rows.
- **Gain/risk:** ~0. Low priority.
- **Conflicts:** #35 run5/run6 (scale search null).

### G6. Whitened low-rank head residual
Add a rank-6 correction in the $C^{p}$ metric (p = 0.75).
- **Source:** ARCHead (2608.02703: vs a GPTQ-style head only 1.0104 → 1.0058 rel. PPL).
- **Fit:** format (extra GEMV at decode).
- **Test:** headsim.
- **Gain/risk:** < 0.3% total. Low priority.
- **Conflicts:** #34.

---

## Theme H: healing, QAD and distillation objectives

### H1. Match the loss to the score exactly
The board's metric is *teacher-prefix, forward KL, top-512 support, gen-masked positions*. That is off-policy forward KL in the 2605.16826 decomposition. Use exactly that, including the golden's renormalization convention, and avoid on-policy or reverse-KL objectives (OnlineQAT, OnQ, OPD), which target generation behaviour.
- **Source:** 2605.16826; 2609.33791; FidelityQ (2610.14934).
- **Fit:** training design.
- **Test:** none.
- **Gain/risk:** avoids a mismatch that can cost several %. Nil risk.
- **Conflicts:** ROADMAP V5.

### H2. The golden as a free teacher
Train scale healing on qad-pool documents against the published bf16 top-512 golden logits. This removes the teacher forward; the bf16 27B doesn't fit one card anyway.
- **Source:** cached-teacher KD (CKA-QAD top-8192; LLM-QAT).
- **Fit:** training.
- **Test:** check golden coverage of the qad pool, and that its rules allow this.
- **Gain/risk:** halves QAD cost. Nil risk.
- **Conflicts:** ROADMAP V5, data rules (the qad pool is the designated pool).

### H3. Gauss–Newton sequential scale refit (closed form, per shipped mix)
Per layer in order, refit `su`, `sv` and RMSNorm/gated-norm gains to minimize $\|WX-\hat Q\hat X\|^2$ in the two-sided metric. $\hat X$ comes from the already-refit quantized prefix: one forward per layer collecting $\hat X\hat X^\top$ and $X\hat X^\top$. No backward, no optimizer.
- **Source:** GPTAQ asymmetric objective; SHIFT-LLM closed-form LS adapters (2608.25068); SVD-Surgeon OBS compensation; Layerwise LQR view of propagated error (2605.04230).
- **Fit:** final pass on shipped mixes.
- **Test:** 0.8B uniform K2.5 and a 2.5 bpw-like mix, ≤ 1M qad tokens; 27B: about one eval's compute per mix.
- **Gain/risk:** −2 to −5% at 2.5 bpw. Risk: low with few DOF; check on verify.
- **Conflicts:** V5 (zero-training precursor).

### H4. K-FAC-preconditioned KL scale healing with a trust region
Train log-scales $s=s_0e^r$ (su, sv, norm gains; ~0.02% of parameters) with Adam preconditioned by the collected Kronecker factors, an $r^2$ penalty, and best-held-out-KL checkpointing. This is for sample efficiency on a 3.7M-token pool.
- **Source:**
  - ThinkBit (2610.34886): KL 0.641 → 0.282 at 2 bpw with scales and norms; scales-only 0.282 vs norms-only 0.418; 268M tokens.
  - RoleQ (scale-only KL distillation, 400 steps).
  - Tetra: row-scale KL healing +1.7–3.3 MMLU in 1–2 GPU-hours at 4–14B.
  - SQuAT: gain LR ~1000× the direction LR.
- **Fit:** training (V5; 2–3 shipped sizes).
- **Test:** 0.8B K2.5 uniform: qad-pool-only data, Adam vs preconditioned, 1–2 GPU-hours on the 4070.
- **Gain/risk:** **−5 to −15% KL** at 2.5 bpw (our base is far stronger than ThinkBit's GSQ start, so expect the low end). Risk: overfitting the small pool.
- **Conflicts:** ROADMAP V5.

### H5. Memory plan for 27B healing on 2×32 GB
- Layer-streamed forward/backward with trellis codes decoded on the fly (only scales trainable).
- Activation checkpoints compressed unbiased int8 (safe for the linear parts per 2605.01255).
- A fused chunked top-512 KL kernel (StreamKL-style, 2606.20005) so 248k-vocab logits never materialize at 8k positions.
- **Source:** as cited.
- **Fit:** training infrastructure.
- **Test:** memory dry run on one layer group.
- **Gain/risk:** enabler. Medium risk.
- **Conflicts:** none.

### H6. Unbiased importance-sampled healing batches
Sample windows with probability ∝ current per-window KL (worst decile ≈45% of KL per FidelityQ) and reweight by 1/p, so the objective stays the corpus mean.
- **Source:** importance sampling for SGD variance reduction; FidelityQ's KIC (biased variant: Wiki2 17.05 → 16.69).
- **Fit:** training.
- **Test:** 0.8B, convergence speed at equal tokens.
- **Gain/risk:** 1.3–2× faster convergence. Low risk.
- **Conflicts:** none.

### H7. Teacher-input block warm-up before end-to-end KL
Warm up with blocks fed teacher inputs (widths 1 → 2 → 4 layers, geometric step budget), then quantized-prefix inputs, then end-to-end top-512 KL.
- **Source:** JustQuant Theseus γ curriculum (2609.33601; full-model QAD collapsed without it on DiT); layer-wise curriculum (2609.19213).
- **Fit:** training.
- **Test:** 0.8B, with vs without warm-up at equal tokens.
- **Gain/risk:** stability; −0 to −2%. Low risk.
- **Conflicts:** none.

### H8. Heal norm gains including the GDN gated-norm gains
Make the gated RMSNorm gains in GDN blocks and the q/k norms trainable alongside su/sv. Fold back into stored gains.
- **Source:** ProQuant RoCo (2610.30779: 64.77 → 67.30 at 1.5 bit on a GDN hybrid); ThinkBit (norms help less than scales but add).
- **Fit:** training.
- **Test:** inside H4.
- **Gain/risk:** −0.5 to −1% on top of H4. Low risk.
- **Conflicts:** none.

### H9. Drift monitoring only (CKA/MSE of block outputs)
Log per-block linear CKA vs bf16 on a verify slice during healing. Add an auxiliary loss only if verify KL improves.
- **Source:** CKA-QAD (2606.05682; thin evidence).
- **Fit:** training diagnostic.
- **Test:** one hook pass.
- **Gain/risk:** information. Nil risk.
- **Conflicts:** none.

---

## Theme I: attention, GDN and activation specifics

### I1. Sink-token exactness via a rank-1 correction
For tensors where the sink/massive-activation token dominates (C8, E3c outliers), store $u=(W-Q)x_{sink}$ and $v=x_{sink}/\|x_{sink}\|^2$ and add $u(v^\top x)$. This cancels error on the sink direction and is negligible elsewhere, at ~0.0003 bpw.
- **Source:** attention sinks / massive activations (2609.09085); KV-cache work keeps sink tokens bf16 (OSCAR: without the sink window accuracy collapses); D-SLR verbatim rows (2610.10636).
- **Fit:** format (side GEMV), or emulate by forcing high K on those small tensors.
- **Test:** 2B fake-quant: replace the sink positions' layer outputs with bf16 for the first 4 layers and measure the KL drop. 30 min.
- **Gain/risk:** 0 to −3%; unknown. Kernel risk.
- **Conflicts:** E3c note ("a few rows at bf16").

### I2. Radial-invariance-structured H_out for q/k slices
GDN L2-normalizes q/k and attention RMS-normalizes q/k, so error along $\hat k$ is projected out. Model the per-head metric as $c_h(I-\mathbb E[\hat k\hat k^\top])$ instead of estimating it.
- **Source:** first-order normalization Jacobian; KVarN magnitude/direction split (2606.03458).
- **Fit:** per-level.
- **Test:** 0.8B, check the cosine with the full H_out per head.
- **Gain/risk:** small. Low priority.
- **Conflicts:** none.

### I3. If GDN internals ever go int8: the LinQuant recipe
Use per-(head, value-channel) static scales, fp32 state carry plus an int8 read copy, a Hadamard on q/k after the conv (valid for GDN's scalar decay), and keep the WY solve in fp32.
- **Source:** 2610.39418 (GDN INT8 PPL 15.019 vs FP 15.004).
- **Fit:** kernels (not planned for V3).
- **Test:** n/a.
- **Gain/risk:** speed only. —
- **Conflicts:** none.

---

## Theme J: low-rank and pruning-derived structure

### J1. Whitened residual spectrum (diagnostic)
For a few tensors per family, compute singular values of $H_{out}^{1/2}EH_{in}^{1/2}$ for the library quant error. If they are flat, as LDLQ + RHT should make them, rank-r corrections (IO-SVD/ARCHead/BiLoCo-style) are dead for us; if a spike exists, price rank-r as a knapsack option.
- **Source:** IO-SVD, ARCHead, BiLoCo (2610.32243: flat spectra, gains 0.02–0.2 PPL).
- **Fit:** diagnostic.
- **Test:** CPU, 1 h.
- **Gain/risk:** decides a whole family of ideas. No risk.
- **Conflicts:** none.

### J2. Whitened low-rank vs trellis K0.25–0.75 for insensitive tensors (V3.5)
At equal bytes (16r(k+n) bits vs K·kn), compare the proxy of a two-sided-whitened rank-r approximation with sub-1 trellis for tensors like layer-0 ffn_down (K2 costs +0.00006).
- **Source:** IO-SVD/AIR/SVD-Surgeon; GA-SVD.
- **Fit:** per-level (V3.5 menu).
- **Test:** CPU on 5 tensors.
- **Gain/risk:** only < 2 bpw. Low risk.
- **Conflicts:** V3.5 sub-1 K plan.

### J3. Verbatim massive columns for E3c outlier tensors
Store the few massive-activation input columns of L0 ssm_out and L3 attn_output in fp16 (D-SLR "rows verbatim") and quantize the rest.
- **Source:** D-SLR (2610.10636: closed form, never worse than truncated SVD); PolyQ (~1% of channels at 8–16 bit rescues 2.x-bit budgets).
- **Fit:** format (sparse column patch); knapsack alternative: give the tensor more K.
- **Test:** hproxy on the two tensors.
- **Gain/risk:** ≤ 1%. Low priority.
- **Conflicts:** #29-02.

---

## Considered and not recommended (each with the reason)

- **Trace-optimal joint (rotation, feedback) / geometric-mean decomposition:** lost to RHT + feedback in 16/16 comparisons at finite levels (2610.18139).
- **Learned rotations (HARP, SpinQuant-style):** ≤ 0.7% at 3 bits on QTIP, and +0.05 bpw (2605.29843). Related: #41, #47.
- **E8 / Leech / lattice codebooks:** the trellis already beats E8P (Gaussian MSE 0.073 vs 0.089 at 2 bits), and lattice formats decode slower (2609.02652).
- **Residual trellis ladders for upgradable bitrates:** cost 0.3–0.9 dB vs direct TCQ (2610.35985).
- **Entropy coding of trellis indices:** the stream is near-uniform.
- **LoRA-then-requantize healing:** requantization deletes the adaptation when the update is small vs the grid (Scale-QLoRA, 2609.04526).
- **On-policy or reverse-KL distillation:** targets generation, not our teacher-forced forward KL (H1).
- **RL / evolutionary allocators:** WINDQuant took 49 h on an 8B; EvoPress is surpassed by an exact DP with measured costs.
- **GDN state or KV-cache quantization:** state is fp32 and KV bf16 in our path.

---

## Ranked top 20 (expected gain per unit cost)

| rank | idea | expected KL @2.5 bpw | cost | why it ranks here |
|---|---|---|---|---|
| 1 | A4 per-sequence diagonals in the 27B run | option value | disk only | Unlocks B3, F4, A11 and robust estimators without a second 8.5 h run |
| 2 | B1 two-sided `refit_scales` | −0.5…−2% | ~1 h code, 20 min 0.8B | Closed form, free at inference |
| 3 | A5 diag + rank-r sketch (test on 0.8B, then collector) | −1…−2% beyond diag | 0.5 day + 30 min | Must be decided before the 27B night |
| 4 | D1 chained tail-biting | **−2…−4%** | CPU sim hours; kernel + quantizer days (inside the Phase-B requant) | Removes a measured 4–5% proxy penalty from every level |
| 5 | G1 two-sided shaped head | −1…−3% | headsim, ~1 day | Exact evaluator already exists |
| 6 | B3 tempered H_out | 0…−3% (K2) | 25 min 0.8B | Pure knob |
| 7 | B4 LW shrinkage diag↔full | −1…−2% (0.8B) | 25 min | Uses data on disk |
| 8 | A6 gen-masked Fisher loss | 0…−3% | collector flag + 40 min test | Objective match |
| 9 | H3 Gauss–Newton sequential scale refit | −2…−5% | ~2 days + ~1 GPU-h per mix | No training loop |
| 10 | F1 absolute Fisher pricing | indirect −1…−3% | CPU, 0.5 day after collection | Cuts evals; per-tensor prior |
| 11 | A7 forward-only GLU metrics | enabler | 1 h on 0.8B | Could make gate/up collection forward-only |
| 12 | D3 CD re-Viterbi | −1…−3% | ~1 day + quant time | In backlog, untested |
| 13 | C6 + C5 level-K prefix Hessians with control variate | −2…−4% | days + several 27B forwards | Removes the token-count caveat |
| 14 | G4 gradient-metric embedding | −0.5…−1.5% | collector accumulator + 2 quants | Token-agnostic, multilingual-safe |
| 15 | B6 MLP interleave permutation | 0…−2% | 0.5 day (gated by A1) | Free at inference |
| 16 | H4 preconditioned KL scale healing | −5…−15% | weeks (V5) | Biggest lever, highest cost |
| 17 | F8 + F9 racing + CUPED | indirect −0.5…−1.5% | CPU replay | Search efficiency |
| 18 | C7 activation denoising | −1…−3% | ~1 day 0.8B | Cheap stand-in for sequential |
| 19 | G3 softmax-gauge t-grid | ≤ −0.5% | an afternoon | Trivial |
| 20 | B8 output-side row-slab K | −2…−5% | format work | Gated by A1/A2; competes with B6/B7 |

## The 5 cheapest diagnostics that would most change this ranking

1. **A1, noise-shaping gain map** (CPU, minutes): predicted per-tensor feedback gain (whole 128-block vs inter-16-sub-block; natural vs interleaved order) against the measured 0.8B gains. Decides B6/B7/B8/B10 and how much head shaping (G1) to expect.
2. **D1 CPU chained-trellis gap** (hours on CPU box): chained (C = 4/8/all) vs per-tile tail-biting proxy at K2/K2.5/K3 on real tensors. Recovering ≥ 70% of the 0.948/0.962 gap puts D1 first.
3. **A2, gate-2d mechanism on the 0.8B with the Sketch B diagonal** (10 min): whether output *scales* also work with a good diagonal. Tells us if the 27B gate-2b/2c/2d failures were the estimator, which reshapes trust in the 27B run and B8.
4. **A3, token-wise vs Sketch B diagonal at equal data** (~1 h on the 4070): picks the 27B estimator; may enable cheaper or longer collections.
5. **A5 test, rank-r H_out arms from the 8K snapshot** (r = 8/32/128, 30 min): whether the 27B run should carry a sketch. Combined with B4 (same session), it decides whether anything beyond the diagonal is worth collecting before the 8.5 h night.
