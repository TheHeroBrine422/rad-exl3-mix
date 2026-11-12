# Faraway lens: replacing large parts of the EXL3 pipeline (2026-11-02)

**Bottom line:** I measured on the real 27B before proposing anything. The measurements kill most "different codec" and "cross-layer" directions, and they found one sizeable piece of structure the current pipeline does not use: in each MLP, a neuron's write vector (down_proj column) is aligned with its read vector (up_proj row). Predicting down from the already-quantized up and gate saves 0.09–0.23 bit/weight on down_proj.

All numbers below come from CPU jobs on colo, single-threaded, against the bf16 27B and the calib Hessians. I wrote only to `/data/work/ideas-faraway/` (scripts, JSON) and `/data/logs/faraway-*.log`; no GPU box was used. None of the ideas has been tested end to end. "P" is my probability that the stated gain is real at 2.5 bpw on the 27B.

## 0. What I measured today

Units: bits per weight of the tensor named. Six layers were sampled: 10 and 30 and 50 (GDN), 11 and 31 and 51 (full attention).

| # | Question | Result | Consequence |
|---|---|---|---|
| M1 | Do neurons repeat across layers? For each gate row in layer L+1, the best \|cos\| against all 17,408 gate rows of layer L | Median 0.10–0.16 for adjacent layers, 0.064 for L10 vs L50; random baseline 0.057. For the matched neuron, up rows 0.03–0.06 and down columns 0.07–0.08 | **Kills** cross-layer delta / motion-compensated coding, layer merging and recursive conversion |
| M2 | Spread of each weight's own column covariance (rows normalized, norm gains folded), 128-block high-rate coding gain | gate/up 0.19–0.32, GDN qkv 0.32–0.51, q 0.45–0.53, down 0.15–0.28, GDN out_proj 0.56–0.77, o_proj 0.69–0.94 | Much of this is a sampling artefact (next row) |
| M2b | The same number for a random matrix of the same shape (Marchenko–Pastur) | 0.12 (gate/up/down), 0.18 (q), 0.22 (qkv), 0.46 (o/out) | Real structure is roughly 0.05–0.3 bit, and it is per tensor |
| M3 | Does a basis from elsewhere transfer? | Same type, other layers: 0.002–0.09. Neighbour layer: 0.03–0.07. **Global basis from other layers: 0.001–0.10, mean ≈0.035** (L30/31 0.04–0.10, L50/51 0.001–0.04). Gate's basis applied to up: 0.04–0.06 | A single global rotation is worth about 0.035 bit; the rest is tensor-specific |
| M4 | Per-128-block rate allocation with the real Hessian (LDLQ pivots), against a 128-Hadamard uniform-rate baseline | Hadamard basis, per-block rates: **0.009–0.013**. Global basis: 0.03–0.14. Own basis: 0.21–0.33 | Column-split K without a rotation (ranked D7) is ≈0 |
| M5 | Headroom audit: measured `ours-K` error vs an ideal coder (LDLQ pivots × Gaussian D(R), uniform rate) | gate_up K2/K3/K4 at **1.13–1.19× ideal (0.09–0.13 bit)**. That is mostly the trellis's own ≈0.07 bit gap at 2 bits plus the 4–6% tail-biting loss (#57). Ideal vs own-basis bound: 1.31–1.39×. Attention/GDN inputs: L10 ssm_qkv 1.35×, **L11 attn_qg 1.74×**, L30 1.20, L31 1.15, L50 1.10, L51 1.07 | For gate/up, EXL3 is close to the floor of its own model class. The early-layer input projections are the exception |
| M6 | How non-Gaussian are the raw weights (entropy-power deficit after row/column normalization)? | 0.0002–0.0065 bit; kurtosis 3.04–4.0 | **Kills** non-Gaussian codebooks, rotation-free codecs and entropy-coded scalar quantization for the trunk |
| M7 | Same-neuron read/write alignment | gate↔up median \|cos\| 0.06–0.12. **up↔down 0.27–0.47 (p90 0.40–0.65)**. gate↔down 0.08–0.17 | New structure; see A1 |
| M7b | Prediction gain: down column predicted from its neuron's up and gate rows, least squares per neuron, predictors carrying K2-level noise, weighted by the Hessian diagonal | down←up+gate: 0.089 / 0.104 / 0.232 / 0.225 / 0.174 / 0.135 (L10/11/30/31/50/51). down←up alone: 0.07–0.17. GDN out_proj←v: 0.029 / 0.106 / 0.162. Attention o_proj←v: 0.059 / 0.088 / 0.030 | A1 / A3 / A4 |
| M8 | Embedding and head | Own-basis block gain 0.090 (embedding), 0.101 (head; 0.115 with H); sampling artefact ≈0.007; natural basis ≈0. Embedding row ↔ head row \|cos\| median 0.012 | Head/embedding transforms only pay through a shared rotation; head-from-embedding prediction is dead |

**Exchange rate used throughout.** From the board (2.0: 0.164, 2.5: 0.0666, 3.0: 0.0306), d ln KL / d bpw ≈ −1.7 at 2.5, so 0.01 effective bpw ≈ −1.7% KL and 0.04 ≈ −6.5%.

Parameter shares of the 26.9e9: gate+up 42%, down 21%, GDN qkv 9.4%, GDN z 5.6%, GDN out 5.6%, attention q 3.7%, o 1.9%, embedding 4.7%, head 4.7%.

**Overall read.**
- The current basin is close to its floor where most of the bytes are (gate/up within ~0.1 bit of an idealized LDLQ + Gaussian coder).
- The weights are Gaussian, and there is no cross-layer redundancy.
- Large wins from replacing the codec are therefore unlikely. What remains is:
  1. cross-tensor structure inside a layer (read/write alignment, found today);
  2. model-level reallocation of bits (dead units, pruning);
  3. function-level repair of the residual stream;
  4. the early-layer input projections.

---

## A. Cross-tensor prediction inside a layer (new, measured)

**FA-A1 · Down-from-up+gate closed-loop prediction ("chroma from luma" for MLPs).** Code the down residual $R=D-\hat U^\top\mathrm{diag}(\alpha)-\hat G^\top\mathrm{diag}(\beta)$, with Û and Ĝ the *decoded* up and gate and α, β two scalars per neuron fitted by least squares. Because the prediction uses decoded weights, the decoder reconstructs $\hat D=\hat R+\hat U^\top\alpha+\hat G^\top\beta$ and the error is $\hat R-R$ only (DPCM; no error propagation from up). In the container, store R̂ and α, β (70 KB per layer). At load, rebuild D̂ and keep it in a runtime format (see D2).
- **Source:** new. Analogues: AV1 chroma-from-luma, DPCM. 2605.24754 (MCWC) predicted across layers; M1 shows that fails here, whereas the within-layer pairing exists.
- **Why:** M7b gives 0.09–0.23 bit on down_proj, mean ≈0.16 with the predictors at K2 noise. That is 0.16 × 21% ≈ 0.034 effective bpw; with A3/A4 about 0.04 bpw.
- **Fit:** two-dimensional library. Down options become (K_down × K_gate_up), because R depends on which up/gate level the decoder holds. Restricting to K_down ∈ {K_gu−1, K_gu, K_gu+1} keeps it at about 3× the down quant time.
- **Cheapest test:**
  - Colo CPU: on L10/30/50, quantize D and R with the real trellis (`trellis.py` or qtf2 emulation), using Û and Ĝ from the existing ours-K2/K3 levels. Compare H-weighted proxy error at equal K. **Pass:** ratio ≤0.85 at K2 and K3 on L30.
  - Then 0.8B fake-quant end to end on the 4070, 3 seeds, D̂ formed in torch. **Pass:** ≤0.97× KL at K2.5-equivalent with every seed better.
  - About 1 CPU day plus 3 h of 4070 time.
- **Gain:** −3 to −8% KL at 2.5; P 0.4. **Risk:** KL may not follow residual energy; the runtime format (D2) and the rules question; library complexity.
- **Conflicts:** none in the INDEX. Differs from residual ladders (rejected in ranked §5) because the prediction comes from a *different tensor*, not a second code of the same one.

**FA-A2 · Online-prediction serving (no load-time transcoding).** Compute $y=\hat Rh+\hat U^\top(\alpha\odot h)+\hat G^\top(\beta\odot h)$ with transposed GEMMs that re-read the up/gate trellis tiles.
- **Source:** new.
- **Why:** same gain as A1, with no runtime format change and no rules question.
- **Fit:** kernel change; transposed EXL3 GEMV/GEMM.
- **Cheapest test:** ntbench the transposed decode at M=1/2048, then a 2-layer end-to-end check.
- **Gain:** same as A1. **Risk:** costs +67% MLP GEMM work (+1 GEMM with up only, which keeps ~75% of the gain), against V3's speed target. Restrict to layers ~20–45, where the gain is 0.22 bit; P 0.35 for the gain, speed cost certain.
- **Conflicts:** V3 speed goal (§1).

**FA-A3 · GDN out_proj predicted from in_proj v rows (channel-paired).** Value channel c of in_proj_qkv and input column c of out_proj are paired through the state recurrence, the gated norm and the z gate, all channelwise. Predict out_proj columns from the decoded v rows.
- **Source:** new.
- **Why:** M7b gives 0.03 / 0.11 / 0.16 bit (L10/30/50) on 5.6% of bytes, ≈0.006 bpw (−1%).
- **Fit:** as A1.
- **Cheapest test:** with A1's CPU harness, same pass rule.
- **Gain:** −0.5 to −1.5%; P 0.35.
- **Conflicts:** none.

**FA-A4 · Attention o_proj predicted from v_proj (GQA-paired) and from the q_proj gate half.** In Qwen3.5 attention, q_proj carries an output gate. Gate row c and o_proj column c are a second read/write pair, besides v.
- **Source:** new.
- **Why:** o←v measured 0.03–0.09 bit; o←gate not measured.
- **Fit:** as A1.
- **Cheapest test:** a 10-minute cosine census on colo (extend `pred.py`).
- **Gain:** −0.1 to −0.5% (attention is 16 layers, o_proj 1.9% of bytes); P 0.3.
- **Conflicts:** none.

**FA-A5 · Shared neuron direction coded once.** For the aligned part of neuron j, write $u_j=a_js_j+r^u_j$ and $d_j=b_js_j+r^d_j$, and code $s_j$ once at a higher K. Unlike A1's asymmetric closed loop, this lets the knapsack split bits between the shared part and the private parts.
- **Source:** new.
- **Why:** symmetric version of A1; may matter at 2.0 bpw, where up is coarse.
- **Fit:** format change (a third plane).
- **Cheapest test:** CPU proxy against A1 at K2.
- **Gain:** ≤A1 + 1%; P 0.15.
- **Conflicts:** A1.

**FA-A6 · q↔k row prediction** (full-attention heads and GDN q/k sections, paired by head dimension). QK circuits may have $W_q\approx W_k$ structure in "self-matching" heads.
- **Source:** new.
- **Why:** untested.
- **Fit:** as A1.
- **Cheapest test:** a minutes-long cosine census. **Kill** if median \|cos\| <0.15.
- **Gain:** 0 to −1%; P 0.15.
- **Conflicts:** none.

**FA-A7 · GDN z (output gate) rows ↔ out_proj columns / v rows.**
- **Source:** new.
- **Why:** untested; z_c gates channel c, so this is the GDN analogue of A4's gate pair.
- **Fit:** as A1.
- **Cheapest test:** the census from A6.
- **Gain:** 0 to −1% (z is 5.6% of bytes); P 0.15.
- **Conflicts:** none.

**FA-A8 · up predicted from gate (same neuron). KILLED by M7.** Median \|cos\| 0.06–0.12 gives ≤0.02 bit.

**FA-A9 · Head rows predicted from embedding rows (identity pairing). KILLED by M8.** Median \|cos\| 0.012. A full 5120² linear map between the two was not tested; I expect it to be weak.

**FA-A10 · Prediction-aware knapsack and pricing.** Each down option carries its predictor level; price the residual quants by the measured proxy (as F1 in ranked, re-pricing), and let knap4 pick (K_gu, K_down) jointly.
- **Source:** new; required by A1.
- **Fit:** per-level, two-dimensional.
- **Cheapest test:** knap4 with interpolated residual options on CPU.
- **Gain:** enabler.
- **Conflicts:** #45.

**FA-A11 · Predictor-aware rate for up.** Spend extra bits on up because its errors also reduce down's prediction gain. M7b says the noisy-predictor loss is only 0.01–0.02 bit, so this is **≈0**. Recorded only so it isn't retried.

## B. Transform coding with shared bases (measured: mostly tensor-specific)

**FA-B1 · Global residual-stream KLT rotation + variance-sorted 128-blocks + per-k-block K for readers and per-row-slab K for writers.** Fold one orthogonal Q into every residual reader and writer, the embedding and the head, after folding all RMSNorm (1+w) gains into the weights; this is exact. Permute the residual coordinates so each 128-block holds similar variance. Keep EXL3's within-block Hadamard, and let K vary per k-block (readers) or row slab (writers).
- **Source:** SpinQuant/QuaRot (2405.16406, 2404.00456) fold rotations into the residual stream; SPECTRA/AATC (2608.07915, 2608.14191) use reverse water-filling for KV. This applies water-filling to weights.
- **Why:** M3/M4 give an honest 0.035 bit average, up to 0.10–0.14 bit mid-depth; ≈0.03 bpw (−5%).
- **Fit:** format change plus a full requant; all or nothing, like R1.
- **Cheapest test:** CPU real-trellis proxy on L30 gate/up/qkv/down in the LOLO basis with water-filled block K vs uniform at equal bits. **Pass:** ≥8% proxy reduction. Then the E1c-dump A8-g128 activation error in the Q basis (#47 saw a 6.6× blow-up for a random R1).
- **Gain:** −2 to −5%; P 0.2. **Risk:** the A8 path; kernel work for per-k-block K and row-slab K.
- **Conflicts:** #47 (R1 for activations failed; this is a data-derived Q for weights), #29-02 (row-slab ceiling measured input-side only), D7/D6 in ranked.

**FA-B2 · Depth-banded residual re-rotation.** Same as B1 with 4–8 bases. At band boundaries, multiply the residual by the orthogonal $Q_b^\top Q_{b-1}$: 26M MACs per token and 26M parameters per boundary (fp16 0.016 bpw each, int8 0.008).
- **Source:** new.
- **Why:** neighbour-layer bases keep about 2× the global basis's gain (M3).
- **Fit:** format plus engine (an extra op).
- **Cheapest test:** CPU with a band basis built from layers L±1…±4, excluding L. **Pass:** ≥0.02 bit over LOLO net of the boundary bytes.
- **Gain:** B1 + 0 to −2%; P 0.1.
- **Conflicts:** B1.

**FA-B3 · Per-tensor learned structured KLT (butterfly or Kronecker $V_1\otimes V_2$) maximizing block coding gain.** Objective: minimize $\sum_b\log\bar v_b$ (log-det of the block variances, with H pivots), instead of HARP's nearest-code proxy, then per-block K. Online cost is comparable to the Hadamard.
- **Source:** 2605.29843 (HARP), 2605.00422 (Kronecker Procrustes); new objective.
- **Why:** after removing the sampling artefact, 0.05–0.3 bit of per-tensor structure remains (M2/M2b); the fraction a butterfly captures is unknown.
- **Fit:** format and kernel.
- **Cheapest test:** CPU fit on 4 tensors, measuring captured gain against the artefact-corrected own-basis gain. **Pass:** ≥0.05 bit net.
- **Gain:** −1 to −3%; P 0.1.
- **Conflicts:** ranked §5 rejected learned rotations under a *uniform-rate* objective; here the rate is allocated per block.

**FA-B4 · Implement per-block K as 2–4 column-group sub-tensors, each at uniform K, accumulated into one output.** The cheapest engine path for B1/B3: no per-k-block pattern dispatch, just split-K over groups.
- **Source:** new; mechanics like D7.
- **Fit:** format (a tensor becomes a list of slabs).
- **Cheapest test:** knap4 offline with 3-group options.
- **Gain:** enabler; speed cost of extra launches.
- **Conflicts:** D7.

**FA-B5 · Embedding and head in their own KLT basis with per-column-block K.** Gain 0.09–0.115 bit (M8; no sampling artefact at 248k rows). But a private 5120² basis costs 0.008–0.016 bpw, about equal to the 0.005–0.010 bpw saved. It only pays through B1's shared rotation. P 0.1 standalone.
- **Conflicts:** #35, #34.

**FA-B6 · SliceGPT-style residual truncation (zero rate for tail coordinates). Effectively KILLED.** The water-filled block rates in the LOLO basis never fall below 1.3 bit at a 2.5 mean, so no coordinate earns 0 bits.
- **Source:** SliceGPT.

**FA-B7 · Per-tensor own-basis transform with a stored V. KILLED** by the sampling artefact plus storage: n² = 26M parameters per tensor is about 0.6–1.2 bpw at 4 bits.

**FA-B8 · Rank-r subspace split with a stored V_r. KILLED** by cost. The best split is r = 2048 (0.13–0.5 bit including the artefact), which needs 21 MB per tensor.

**FA-B9 · Per-128-block K in the existing Hadamard basis. ≈ KILLED** by M4 (0.009–0.013 bit, Hessian included). This re-ranks ranked D7 down.

**FA-B10 · Attention per-head V/O orthogonal rotation (exact, free) chosen to concentrate o_proj input variance, then per-block K.** Only the 16 attention layers; o_proj is 1.9% of bytes.
- **Fit:** per-level plus per-block K.
- **Cheapest test:** CPU gain census.
- **Gain:** −0.1 to −0.5%; P 0.1.
- **Conflicts:** B1.

## C. Headroom audit and implementation gaps

**FA-C1 · Gap-to-ideal audit over all 496 tensors.** Extend `rd.py` to every tensor and K: measured hproxy error against LDLQ-ideal and own-basis bounds. This is the map of where any codec or quantizer change could pay.
- **Source:** new; the method is today's M5.
- **Fit:** diagnostic.
- **Cheapest test:** colo, ~4 CPU-h in parallel.
- **Gain:** decides C2, H1 and D-theme priorities.
- **Conflicts:** none.

**FA-C2 · Early-layer attention/GDN-input repair.** L11 attn_qg is at 1.74× ideal and L10 ssm_qkv at 1.35×, against 1.07–1.20× in later layers. Candidates for those few tensors:
- per-block transform skip (identity instead of Hadamard for the block holding a massive channel, chosen per tensor);
- a 512-point input Hadamard;
- massive-channel isolation at a higher K;
- undamped re-quantization.
- **Source:** M5; massive-activation literature; 2610.09212 (rotation-free weighting).
- **Why:** the measured gap of 0.2–0.4 bit sits on tensors with large KL shares at shallow depth.
- **Fit:** per-level.
- **Cheapest test:** CPU proxy on L3/L7/L10/L11 inputs with each variant. **Pass:** ≥15% proxy.
- **Gain:** −0.5 to −2%; P 0.3.
- **Conflicts:** D8 in ranked (spike spreading, P 0.2; this supplies the evidence), #38 damping, #47.

**FA-C3 · Ideal-model pricing for new formats.** Use the LDLQ-ideal and own-basis bounds to price unbuilt options (prediction residuals, KLT blocks) in knap4 before any quant run.
- **Fit:** per-level pricing.
- **Cheapest test:** check against V2's measured levels (log-RMSE).
- **Gain:** enabler.
- **Conflicts:** F3 in ranked.

**FA-C4 · Non-Gaussian codebooks / rotation-free codecs / entropy-coded scalar quantization for the trunk. KILLED** by M6 (≤0.0065 bit of entropy-power deficit).

**FA-C5 · Entropy-coded TCQ at rest (ECTCQ), transcoded at load to a 4-bit index runtime format.** A 256-state ECTCQ sits about 0.2 dB from rate–distortion, against about 0.43 dB for QTIP at 2 bits, i.e. ~0.03 bit, plus no tail-biting loss.
- **Source:** Fischer–Wang ECTCQ; 2609.34185 (EntroPack).
- **Fit:** new codec and runtime format.
- **Cheapest test:** CPU ECTCQ vs our trellis on real LDLQ tiles. **Pass:** ≥0.03 bit at K2–3.
- **Gain:** −2 to −4%; P 0.15.
- **Conflicts:** ranked §5 rejected entropy coding of the *trellis stream*; this is a different code.

## D. Decoupling storage from compute

**FA-D1 · Rules question: does the board count only container payload, letting the plugin expand at load to a larger runtime format?** B10 (#53) shows the board counts tensor-plane bytes.
- **Fit:** process; free.
- **Cheapest test:** ask the challenge author (as with the open MTP question).
- **Gain:** enables A1 without a speed cost, plus C5 and D2. Treat it as **the single most valuable question** for this lens.

**FA-D2 · Load-time reconstruct and re-encode.** The loader rebuilds D̂ (A1/A3), then stores it at runtime as K6 trellis (≈12 s per tensor with qtf2, ~13 min total) or as int8 per-row. Runtime memory +2.5–5 GB (fits TP1 with 262k KV). The K6 re-encoding error (~0.1% relative) is negligible.
- **Fit:** plugin change.
- **Cheapest test:** a byte-identity / KL-parity harness on 2 layers.
- **Gain:** enables A1 at zero speed cost; P (allowed) unknown.
- **Conflicts:** v3 §1c.

**FA-D3 · Entropy-coded or KLT-domain embedding rows, decoded at gather.** Rows decode independently; an offset table makes variable-rate rows trivial. Targets the 0.09 bit own-basis gain (M8) plus ECVQ shaping.
- **Fit:** embedding format plus gather kernel.
- **Cheapest test:** CPU embtrellis vs a KLT plus ECSQ row coder at equal bytes on 4096 rows.
- **Gain:** −0.3 to −1% at ≤2.5; P 0.25.
- **Conflicts:** #35, #32 (frequency tiering dropped; this is frequency-agnostic).

**FA-D4 · Runtime dense int8 weights behind a 2.5-bpw container (speed angle).** Prefill without trellis decode. KL-neutral, so it serves V3's speed goal only. P 0.5 that it is faster; memory 27 GB means TP2.
- **Conflicts:** v3 A2-C.

## E. Model-level restructuring

**FA-E1 · Cross-layer delta / motion-compensated weight coding. KILLED** (M1).
- **Source:** 2605.24754.

**FA-E2 · Relaxed recursive / looped conversion (shared block + per-layer LoRA). KILLED** (M1, plus training cost).

**FA-E3 · Layer merging (averaging aligned neighbours). KILLED** (M1).

**FA-E4 · Layer drop plus closed-form linear residual adapter to free bits.** One layer is ≈0.036 bpw at 2.5 (−6% KL if spent elsewhere). Dropping pays only if a layer plus its adapter costs <6% KL.
- **Source:** 2608.25068 (SHIFT-LLM), 2609.09883.
- **Fit:** per-mix.
- **Cheapest test:** 2B: KL of dropping each layer with a least-squares adapter fitted on the fit pool. 27B: the 47 single-layer probes (#21) give priors. **Pass:** any layer <3%.
- **Gain:** 0 to −2%; P 0.08.
- **Conflicts:** user decision (K0 dropped, 2026-11-01) concerned tensors at near-zero bits; this drops whole blocks with an adapter.

**FA-E5 · MLP width pruning plus higher K at equal bytes (a smaller model of the same architecture).** Prune 5–15% of neurons by H-weighted importance, compensate down via OBS least squares, requant the rest.
- **Source:** 2606.07819, 2607.18280, 2610.02462.
- **Fit:** per-mix-lite (shape change, multiple of 128).
- **Cheapest test:** 2B, 3 seeds, at a 2.5-bpw equivalent. **Pass:** ≤0.97×.
- **Gain:** 0 to −3%; P 0.1.
- **Conflicts:** #32 lesson (rare-token / multilingual units matter); use the ml guard slice.

**FA-E6 · GDN value-head and attention-head pruning plus higher K.** As E5 at head granularity (48 value heads per GDN layer).
- **Cheapest test:** 2B head-importance census, then a swap.
- **Gain:** 0 to −1.5%; P 0.07.
- **Conflicts:** as E5.

**FA-E7 · Within-layer neuron VQ (clustered neurons share a centroid).** Untested; M1's near-random nearest neighbours suggest rows are close to orthogonal.
- **Cheapest test:** a 2-minute colo check of within-layer max \|cos\|. **Kill** if median <0.2.
- **Gain:** ≈0; P 0.03.

**FA-E8 · Neuron-type split: aligned neurons (\|cos\| >0.5) predicted, the rest coded plainly.** A1 with per-neuron α=0 already covers this; listed so the alignment distribution is used in pricing.
- **Conflicts:** A1.

**FA-E9 · MoE-ification for bit allocation (neuron clusters as "experts" with their own K).** This is ranked D6's sorted sub-MLPs.
- **Gain:** as D6; P 0.15.
- **Conflicts:** D6, #29-02.

**FA-E10 · Dead-unit census and exact removal.** Neurons with ≈0 calib energy $E[h_j^2]$ (diagonal of the down_proj input Hessian), GDN value channels or heads with ≈0 energy at out_proj input, and attention heads likewise. LDLQ already gives them ≈0 error weight, but uniform K still spends bits on them. Remove them exactly (shapes padded to 128) and spend the bits elsewhere.
- **Source:** new framing; F-WANDA-style statistics.
- **Fit:** per-level (shape change) or column K0.
- **Cheapest test:** colo, minutes: the distribution of $E[h_j^2]$ from the existing Hessians, the share of trunk bytes below 1e-4 × median, then a multilingual guard check. **Pass:** ≥1% of bytes dead.
- **Gain:** 0 to −3%; P 0.25.
- **Conflicts:** user decision on K0 (tensors; this is exact removal of unused units); #32 lesson.

## F. Residual-stream repair ("in-loop filters")

**FA-F1 · Closed-form low-rank drift correction at layer boundaries.** At each layer (or every k layers) add $h\leftarrow h+UV^\top h+b$. Fit by ridge regression of $h_{bf16}-h_q$ on $h_q$ over fit-pool positions, layer-sequentially with quantized-prefix inputs. Forward-only, no backprop. Rank 32 is ≈0.01 bpw for all 64 sites.
- **Source:** 2608.25068 (LRA for pruning); codec in-loop filters; 2610.35955 (closed-form RRR).
- **Why:** absorbs correlated errors from all tensors of a layer, including compounding (about 1/6 of KL at 2.5 per the lowbpw lens).
- **Fit:** final pass (per mix).
- **Cheapest test:** 0.8B, 3 seeds, r ∈ {0, 8, 32}. **Pass:** ≤0.97× net of bytes.
- **Gain:** −1 to −4%; P 0.2.
- **Conflicts:** H10/H11/H3 in ranked (tensor-level adapters, biases, scale refit); this one sits on the residual stream.

**FA-F2 · Diagonal affine at layer boundaries (per-channel scale and bias, 10,240 parameters per layer), closed form.** The zero-byte version of F1.
- **Fit:** final pass.
- **Cheapest test:** 0.8B, minutes per fit.
- **Gain:** −0.5 to −2%; P 0.3.
- **Conflicts:** C9 and H11 in ranked (svh multipliers, writer biases).

**FA-F3 · Multi-layer readout corrector for logits.** Logits get a learned rank-r term from a combination of the last k residual states, trained by forward KL on the qad pool; deep supervision repairs the head input.
- **Fit:** final pass plus a head-side op.
- **Cheapest test:** headsim-style dumps of L60–63 states.
- **Gain:** −0.5 to −2%; P 0.15.
- **Conflicts:** E3 in ranked (head healing).

**FA-F4 · Exact full output Hessian for the last layer's writers via the head Fisher.** For L63 (and L60–62 approximately), $H_{out}=J^\top FJ$ through the final norm and head, computed analytically from the softmax Fisher on dumped hidden states. Needs no backward pass and gives a full 5120² matrix, not a diagonal.
- **Source:** new; headsim machinery.
- **Why:** L63 gate_up at K2 alone costs 80% of the base KL (#21).
- **Fit:** per-level (last layers only).
- **Cheapest test:** 0.8B last layer, two-sided LDLQ, 3 seeds.
- **Gain:** −0.5 to −2%; P 0.3.
- **Conflicts:** #33 (the diagonal from backprop); the full matrix is exact and noise-free here.

## G. Objectives and data

**FA-G1 · Per-document runtime adaptivity (precision chosen per document or token). REJECTED:** bytes are fixed, so adaptivity saves compute, not KL.

**FA-G2 · Test-time dynamic evaluation (adapt norm gains on the document's own text by cross-entropy). Crazy.** It moves the model toward the data, not toward bf16, so it probably hurts KL; the engine also has no backward pass.
- **Cheapest test:** 0.8B torch, 30 minutes.
- **P:** 0.03.

**FA-G3 · Ship bf16 KV/state for a shared chat-template preamble. REJECTED unless explicitly approved:** it is benchmark-specific and probably against the spirit of the board.

**FA-G4 · Pre-quantization conditioning fine-tune.** Briefly fine-tune bf16 → W′ with KL(bf16‖W′) plus a compressibility penalty (read/write alignment, block-variance concentration), then quantize W′.
- **Source:** QAT literature; 2609.32060.
- **Fit:** training, rented GPUs (100s of GPU-h), and a data-rules check.
- **Cheapest test:** 0.8B with 20M tokens of calib.
- **Gain:** −5 to −15% if it works; P 0.05.
- **Conflicts:** H9 in ranked.

**FA-G5 · Calibrate H_in on the quantized model's own errors near the KL tail (tail-position-weighted H_in).** Probably a duplicate of A1/H7 in ranked; listed for completeness. P 0.05.

## H. Rounding as inference (statistical physics and coding)

**FA-H1 · Soft-decision trellis LDLQ: BCJR forward–backward posteriors per tile plus VAMP-style message passing over the whole matrix, then a hard Viterbi polish.**
- **Source:** 2610.32127 (MPQ: E8P 2-bit Llama-3-8B 15.01 → 11.63 PPL, against 12.76 for coordinate descent alone), 2605.10655 (BCJR).
- **Why:** LDLQ commits greedily column by column. M5's gap to ideal (0.09–0.13 bit, ≈0.07 of it trellis-intrinsic) bounds this at roughly −3 to −6%.
- **Fit:** per-level; quant time ×10–30.
- **Cheapest test:** 0.8B, one family, K2, 3 seeds. **Pass:** ≤0.97×.
- **Gain:** −1 to −4%; P 0.15.
- **Conflicts:** C2 in ranked (hard tile coordinate descent).

**FA-H2 · List-Viterbi with cross-tile beam search under the true matrix objective.** Keep the top-L paths per tile and choose the combination with a beam (AQLM-style).
- **Fit:** per-level.
- **Cheapest test:** CPU on real tiles, L=4.
- **Gain:** −0.5 to −2%; P 0.12.
- **Conflicts:** C2 in ranked.

**FA-H3 · Annealed stochastic LDLQ ensembles with split-half accept.** Run many stochastic-rounding variants and keep the best on held-out Hessian halves.
- **Fit:** per-level.
- **Gain:** 0 to −1%; P 0.08.
- **Conflicts:** #41 (seed spread 0.5%, mostly noise).

**FA-H4 · Lattice basis reduction (LLL) before Babai/GPTQ. KILLED:** the trellis codebook is not a lattice.

## J. Symmetries and reparameterizations

**FA-J1 · SwiGLU up/down per-neuron scale symmetry (cross-layer equalization). REJECTED:** with per-row and per-column scales (svh/suh) the relative errors are invariant to the rescaling.

**FA-J2 · Residual-stream global scale symmetry. REJECTED:** inert for the same reason.

**FA-J3 · Free permutations aligning read/write-aligned neurons into the same k-blocks** for A1's selective prediction. Minor; folds into A1.

## K. Crazy but cheap

**FA-K1 · Implicit neural representation / hypernetwork weight generator.** Fit an MLP from (row, column) to weight on one 5120×128 slice and compare bits at equal MSE.
- **Source:** 2610.16890, 2610.11115.
- **Expectation:** hopeless (M6: Gaussian, near full rank); P 0.01. 30 minutes on colo to retire it for good.

**FA-K2 · Noise filling for sub-1 / zero-rate components (variance-preserving reconstruction).**
- **Source:** audio-codec noise substitution; #39's best gain factor rises above 1 with K.
- **Fit:** V3.5 sub-1.
- **Cheapest test:** 0.8B K0.125 tensors with dither vs zero.
- **Gain:** 0 to −1% at ≤2.0; P 0.05.

**FA-K3 · Spectral shrinkage (Gavish–Donoho, in the H metric) before quantization.** Shrink the sampling-noise bulk to lower the coded energy.
- **Cheapest test:** CPU H-weighted functional change vs energy removed.
- **P:** 0.05.

**FA-K4 · Seed/LFSR and dynamic-system codes. REJECTED** (2609.38477: worse than AWQ at 4 bpw).

**FA-K5 · Logit mixture with a small bf16 model. REJECTED:** a 0.8B bf16 model costs 0.48 bpw.

**FA-K6 · "Self-adjoint MLP neurons."** For neurons with \|cos(u_j,d_j)\| >0.7 (p99 reaches 0.78 at L30), store one vector plus a scalar. This is A1 at the high-alignment tail; a quick count of such neurons sizes it.
- **P:** 0.1 as a standalone format.

**FA-K7 · Per-block transform choice side bit ({Hadamard, identity, DCT} per 128-block, chosen by proxy).** Mostly subsumed by C2; M4 suggests little outside massive-channel blocks.
- **P:** 0.05.

---

## Top 20 by expected gain per unit cost

| Rank | Idea | E @2.5 | P | Cost to first decision |
|---|---|---|---|---|
| 1 | E10 dead-unit census + exact removal | 0 to −3% | 0.25 | Colo minutes |
| 2 | A1 down←up+gate prediction | −3 to −8% | 0.40 | Colo 1 day + 4070 3 h; build 1–2 weeks + D1 |
| 3 | D1 rules question (container vs runtime) | enabler | — | An email |
| 4 | F2 boundary diagonal affine (closed form) | −0.5 to −2% | 0.30 | 4070 2 h |
| 5 | A3 GDN out_proj←v (rides A1) | −0.5 to −1.5% | 0.35 | With A1 |
| 6 | A6/A7/A4 remaining read/write pair census | 0 to −2% | 0.30 | Colo 10 min |
| 7 | F4 exact last-layer output Hessian | −0.5 to −2% | 0.30 | 4070 3 h |
| 8 | C2 early attention/GDN-input repair | −0.5 to −2% | 0.30 | Colo ½ day |
| 9 | F1 low-rank residual drift correction | −1 to −4% | 0.20 | 4070 4 h |
| 10 | A2 online prediction (fallback if D1 is "no") | as A1, restricted | 0.35 | Kernel days |
| 11 | B1 global residual KLT + per-block K | −2 to −5% | 0.20 | Colo 1 day; build weeks |
| 12 | H1 soft BCJR + VAMP | −1 to −4% | 0.15 | 0.8B days |
| 13 | E5 width pruning + higher K | 0 to −3% | 0.10 | 2B 1 day |
| 14 | D3 KLT / entropy-coded embedding rows | −0.3 to −1% | 0.25 | CPU 1 day |
| 15 | C5 ECTCQ at rest | −2 to −4% | 0.15 | CPU 2 days |
| 16 | E4 layer drop + adapter | 0 to −2% | 0.08 | 2B ½ day |
| 17 | F3 multi-layer readout corrector | −0.5 to −2% | 0.15 | Dumps + days |
| 18 | B3 learned butterfly per-tensor KLT | −1 to −3% | 0.10 | CPU 2 days |
| 19 | K3 spectral shrinkage (crazy, cheap) | 0 to −1% | 0.05 | Colo 1 h |
| 20 | G4 pre-quantization conditioning | −5 to −15% | 0.05 | 0.8B week; rented for the 27B |

## The 5 cheapest diagnostics that would most change this ranking

1. **Dead-unit census (E10).** Colo, minutes: the distribution of $E[h_j^2]$ at down_proj, out_proj and o_proj inputs from the existing Hessians, and the byte share below 1e-4 × median, with the multilingual slice as a guard. It moves E10 from rank 1 to dead or to "ship".
2. **Real-trellis proxy test of A1.** Colo, ~1 day: quantize D and R = D − αÛ − βĜ at K2/K3 on L10/30/50 with the actual quantizer and the existing Û/Ĝ levels. A proxy ratio ≤0.85 at K2 on L30 confirms the top new direction; ≥0.95 kills the A cluster.
3. **Read/write pair census (A4/A6/A7).** Colo, 10 minutes: q-gate↔o, z↔out, GDN q↔k, attention q↔k. It decides whether A1's infrastructure covers ~25% of the bytes or ~35%.
4. **496-tensor gap-to-ideal audit (C1).** Colo, ~4 CPU-h. It decides whether C2 is local to a few early tensors or wider, and caps H1/C5 for all the other tensors.
5. **0.8B error-shape test.** 4070, ~1 h, 3 seeds: inject noise with equal tr(EHEᵀ) but different shapes (prediction residuals, global-KLT tail vs head directions). If KL does not follow the proxy for these shapes, A1/B1 lose most of their value before any format work.

Question 0 is free: ask the challenge author about load-time expansion (D1). A "yes" removes A1's only speed cost.


---
**Note (2026-11-02, after the run):** the challenge author confirmed the target is **VRAM size**; load-time expansion is acceptable only as a temporary workaround. A1 / A5 / C5 and any storage-only gain therefore need a kernel that runs from the stored form (A2 for A1). Tracked in `docs/experiments/BACKLOG.md` (Faraway section).
