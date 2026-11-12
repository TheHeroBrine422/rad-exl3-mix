# Faraway lens, second pass (2026-11-07)

This builds on the 11-02 faraway report (`local/papers/runs/2026-11-02-faraway/ideas/faraway.md`). Its measurements M1–M8 already ruled out cross-layer, generic transform/low-rank, codec-swap and layer-drop work. I don't repeat those ideas; they are listed in §F with what would revive them. Before proposing anything I ran six new CPU measurements on colo against the bf16 27B, the calib Hessians and the published block-normalised Σdy² diagonal (scripts in `/data/work/ideas-faraway/w2/`, logs in `/data/logs/faraway2-*.log`). None of the ideas below has been tested end to end.

**Bottom line**
- **Zero-runtime structure is dead.** Predicting one projection's rows from a sibling that reads the same input (so its outputs come free in the same GEMM) saves at most 0.08 bit per channel. The per-head versions cost more in coefficient bytes than they save.
- **The embedding has no usable neighbour structure, and predicting the head from the embedding doesn't pay.**
- **Two new, concrete pieces of structure turned up:**
  1. Residual dimension **3994 is the massive channel**. In the early writers its single row carries most of the tensor's output energy: 94.7% of L0 GDN out_proj, 52% of L3 o_proj, and 42% of L2 down_proj, where one entry holds 99% of that row (a super weight). On a two-sided diagonal proxy (input energy × Σdy²) that one row is **82% / 28% / 16.5%** of those tensors' weighted error. kfit2 independently ranks exactly these L0–L3 writers as the 7 most sensitive tensors per unit proxy error, about 1000–2000× the median. A bf16 row exception costs ~0 bytes.
  2. **The LM head has a token-neighbour structure.** With the mean row removed, 48% of head rows still have another row at |cos| > 0.5, worth 0.25 bit (one parent) to 0.35 bit (four parents). Exploiting it costs one gather-add pass over the logits.
- Beyond that, the measured evidence of the week points this lens at **systematic error at nonlinearities and through depth**: swj grew from −3% to −6.5% between the 0.8B and the 27B, and C9 picked α at the edge of its grid on gdn_out. The open questions are how allocation interacts with healing, and per-document adaptivity of tiny parameters. A different codec is no longer where the bits are.

**Exchange rate** (from the 11-02 report): 0.01 effective bpw ≈ −1.7% KL at 2.5 bpw.

**Byte shares:** gate+up 42%, down 21%, GDN qkv 9.4%, GDN z 5.6%, GDN out 5.6%, attention q (incl. gate) 3.7%, o 1.9%, embedding 4.7%, head 4.7%.

---

## 0. What I measured today

| # | Question | Result | Consequence |
|---|---|---|---|
| **M9** | Zero-runtime sibling prediction. Rows predicted from a sibling that reads the same input (GDN z←v, q←k; attention gate←v, q←k, gate←q; MLP up←gate). 10 layers (2/10/30/50/62 GDN, 3/11/31/51/63 attention), in the coder's metric (plain, and columns scaled by diag(H)^¼). | **Per channel (free):** z←v 0.02–0.045 bit; GDN q←k 0.05–0.08; attention gate←v 0.003–0.055, q←k 0.006–0.09, gate←q 0.001. **Per-head full map** (128–256 predictors): matched 0.03–0.21 bit vs an unrelated head 0.02–0.14. The coefficients cost 0.4–0.8 bit/weight in fp16, so the net is negative. **Top-8 sibling rows** (cost 0.045 bit): z←qkv 0.06–0.12, attention gate←k,v 0.013–0.14, up←gate 0.04–0.06 (control 0.02–0.03). | Kills free sibling prediction. This was 11-02 diagnostic #3 (FA-A6/A7/A4-gate), never run until today. |
| **M10** | Intra-tensor row prediction: a row from its 8 best other rows of the same tensor (needs a 2-pass GEMM) | q_proj 0.12–0.32 bit, GDN z 0.07–0.20, gate 0.04–0.10, up 0.03–0.06; overhead 0.045 | Only q_proj and z are positive, at ≤0.27 net on ≤9% of bytes. Kills within-layer neuron VQ/merging (FA-E7). |
| **M11** | The same census in the **input-Hessian metric** | Absurdly high: at L3, a gate head "explained" by an unrelated v head gives 2.56 bit; head rows give 0.88 bit | **Methodology warning:** in readers' H the massive direction dominates every row's H-energy. Prediction or low-rank gains computed in the H-metric are artefacts; rate saving follows the residual in the coder's basis. (The old A1 numbers were on down_proj inputs, where there is no massive channel, so they stand.) |
| **M12** | Embedding and head | **head ≈ A·embedding** (one 5120² map): 17% of plain energy explained, 0.14 bit/row; the map itself costs ≈0.008 bpw. **Embedding nearest-neighbour row**, mean removed: 0.054 bit (top-1), 0.086 (top-4 least squares); median \|cos\| 0.22, 2.9% of rows >0.5. **Head nearest neighbour, mean row removed: 0.248 bit (top-1), 0.349 (top-4).** Median \|cos\| 0.49; 48% >0.5, 11% >0.7, 0.01% >0.9 (no duplicates). Even across norm quartiles (0.21–0.27); stronger for low token ids (0.40 for ids <50k, 0.18 for >150k); parent and child ids ~28k apart (semantic, case and space variants). | Embedding DPCM and tied tables: dead. **Head token-tree coding: alive** (§B). |
| **M13** | Writer row-energy census, $e_i=\sum_j w_{ij}^2H_{jj}$, 14 layers | **Row 3994 is the top row in every writer sampled.** Share of the tensor's $e$: L0 out_proj **94.7%**, L3 o_proj **52%**, L2 down **42%** (99% of that row in one entry: a super weight), L1 out 24%, L2 out 18%, L7 o 11%, L4–6 out 6–7%, L63 down 4.9%, otherwise <3%. L0 out_proj has 31 rows with >50% of their energy in one entry. | New structure, on the writer side. |
| **M14** | The same rows weighted by the published block-normalised Σdy² ($d_i$) | $d_{3994}$ is only 0.31–0.65× the median in L0–L15 (consistent with the norm partly absorbing scale), and 0.95–1.57× at L31/L63. **Share of the diagonal two-sided proxy $\sum_i d_ie_i$ in row 3994:** L0 out **82%**, L3 o **28%**, L2 down **16.5%**, L1 out 8.6%, L2 out 6.5% (plus row 310 with 16%), L63 down 5.9%, L7 o 5.6%; ≤3% elsewhere. kfit2's top-7 sensitivities per unit proxy error are exactly L0–L3 down/ssm_out/attn_output (s 0.0015–0.0049 vs median 2.2e-6), and κ is 52 at L0. | §A |

(Pure M9–M12 statistics. KL effects are estimated, not measured. M11 also explains the "head_H 0.88 bit" number.)

**What changed since 11-02 for this lens:**
- A1 (down from up) was dropped because it needs an extra GEMM. Today's M9 shows the zero-GEMM variants are empty too, so the whole "cross-tensor prediction" cluster is closed except the head (§B).
- At 27B, the fixes that grew are systematic: swj (−6.5%), C9 on the writers with α at the grid edge, H_out only at ≥K4. That is where §C points.
- #111 showed per-level fits that depend on neighbours' codes break in mixes. Every idea below says whether it is per-level (library-safe), a final pass on a shipped mix, or a format change.

---

## A. The massive channel and super weights (new, measured)

**A1 · bf16 row exception for the massive row in early writers.** Store row 3994 of L0–L3 out_proj/o_proj and L2 down_proj (5–8 rows in total) in bf16, outside the trellis: set its svh to 0 so the trellis codes the other rows, and compute the row as one extra 5120–17408-MAC dot product per token. Per-row output scales make every row's relative error equal, so this row's absolute error dominates the tensor's.
- **Source:** new; super-weight literature (2609.38477 found one column carried 30% of SeedLM's excess loss on L2-7B). **Why:** M14 says it removes 82% / 28% / 16.5% of the weighted proxy error of L0 out / L3 o / L2 down, and these are kfit2's most sensitive tensors with the highest compounding (κ≈52).
- **Fit:** per-level (the codes of the other rows are unchanged if the row is zeroed before quantization; or just requant those 5 tensors). Small format plus plugin change (row list + side dot product).
- **Cheapest test (27B, ~1 card-hour, gpubox):** fixed-code pair on the K2.5 swap base (golden-search27, 103 docs).
  - Arm 1: those 5 tensors fully bf16 (+≈0.011 bpw). This bounds what any fix of them can give.
  - Arm 2 emulates A1 without engine work: decode the level's tensor (w12's pure-torch decoder), overwrite row 3994 with bf16, store the dense result as a bf16 tensor. Same function as A1; bytes ignored for the test.
  - **Pass:** arm 2 ≤0.985× base. Kill if arm 1 is ≥0.995×.
- **Gain:** −0.5 to −2% at 2.5 (plus bytes freed when the knapsack lowers those tensors' K). **P 0.35.** **Risk:** Σdy² for dim 3994 is low (norm invariance), so KL may care less than input energy says.
- **Conflicts:** #96 row-slab K (param-weighted median over slabs; a single row in 5 small tensors is invisible there). Shortlist "massive-channel handling on residual readers" is the reader side; this is the writer side.

**A2 · Super-weight scalar exceptions.** Any entry carrying >50% of its row's energy (L2 down row 3994: 99%; 31 rows in L0 out_proj) goes to a sparse fp16 list and is zeroed before the Hadamard. Then the trellis no longer spreads that one large value across 127 neighbours of its 128-block.
- **Source:** 2609.38477 (four fp16 columns removed 6% of excess loss), SpQR. **Why:** M13. With the entry removed, the rest of the row codes at a far lower scale.
- **Fit:** per-level; format: a ≤100-entry COO list per tensor, applied as a sparse FMA. **Test:** inside A1's arm-2 emulation, restore only the >50% entries; CPU proxy for the remaining row. **Gain:** included in A1; P 0.3. **Conflicts:** none.

**A3 · Exact massive channel end to end.** bf16 for row 3994 of all 192 writers and column 3994 of all readers (≈0.0005 bpw). The channel sets every RMSNorm denominator downstream, so an error in its value rescales all post-norm features at once: a coherent, depth-wide gain error of the kind swj fixes locally.
- **Fit:** per-level format (row and column exceptions). **Test:** extend A1's emulation to all layers (decode, patch, bf16). **Pass:** ≤0.98×.
- **Gain:** −1 to −3%; **P 0.2.** **Conflicts:** shortlist massive-channel readers (subsumes its column half).

**A4 · Top-proxy row exceptions in every tensor, generalised.** For each tensor, rank rows by $d_ie_i$ and give the top r (r ≤ 4) bf16 or a higher-K side row. Example: L2 out_proj row 310 is 16% of its proxy. Price r per tensor in knap4.
- **Fit:** per-level. **Test:** CPU census of the top-4 rows' proxy share over all 496 tensors (minutes, from today's scripts); keep tensors >10%. **Gain:** 0 to −1% beyond A1; P 0.2. **Conflicts:** #96.

**A5 · Massive-channel re-anchor.** After each of the first 8 layers, a closed-form affine on the residual's dim 3994 only (two scalars per layer), fitted on fit-pool positions of the quantized mix against bf16.
- **Fit:** final pass (per mix, seconds). **Test:** 0.8B (check it has a massive dim first), 3 seeds. **Gain:** −0.2 to −1%; P 0.15. **Conflicts:** old F2 (this is F2 on one channel).

**A6 · Massive-row-aware H_out weighting for the ≥K4 levels.** The adopted Σdy² H_out gives row 3994 *below-median* weight, while its error dominates because of its input energy. Weight rows by $d_ie_i$-consistent factors, or pair H_out with A1 so H_out stops spending effort on a row that is handled separately.
- **Fit:** per-level (K ≥ 4 only). **Test:** CPU proxy on L0–L3 writers with/without A1. **Gain:** ≈0–0.5%; P 0.1. **Conflicts:** #33-28.

**A7 · Re-price the L0–L3 writers after A1** (enabler). kfit2's huge s for these tensors may partly reflect the massive row. Refit with A1 options and let the knapsack lower their K.
- **Gain:** bytes (≈0.002–0.005 bpw, −0.3 to −0.8%). **P 0.3** given A1 passes.

**A8 · (Rejected, crazy) RMSNorm that excludes dim 3994 from the denominator.** It changes the function relative to the bf16 reference.

---

## B. Vocabulary tables (new, measured)

**B1 · Token-tree DPCM for the LM head.** Choose a parent row for each child (nearest neighbour on the mean-removed head; depth-1 forest, mutual pairs broken by Fisher). Code $r_t=w_t-\alpha_t\hat w_{p(t)}$ with the same trellis, closed loop (the parent is the *decoded* row). Runtime: $z=\hat R h$ (same GEMV bytes as today), then one gather-FMA pass $z_t \mathrel{+}=\alpha_t z_{p(t)}$ over 248k logits. Side info is 34 bits per row (0.007 bit/weight).
- **Source:** new; DPCM / intra prediction. Differs from the dead FA-A9 (identity pairing to the *embedding*, \|cos\| 0.012) and M12's linear map.
- **Why:** M12 gives 0.25 bit (top-1) on 4.7% of bytes; ~0.15–0.2 bit after the forest constraint and parent noise ≈ 0.008 bpw.
- **Fit:** per-level (the head is quantized as its own levels); composes with the slab head's row order (parents in a higher slab are better predictors).
- **Cheapest test (colo, ~1 day CPU):** real trellis proxy on 4096 head rows at K3/K4 with ourq/trellis emulation, parents from decoded parent levels, head H; then headsim exact head-only KL on the 32k head inputs. **Pass:** ≥0.12 bit equal-error saving and headsim KL ≤0.9× at equal bytes.
- **Gain:** −1 to −2%; **P 0.3.** **Risk:** the eval engine needs the logit pass (cheap, but at full M in KL mode).
- **Conflicts:** #34 slab head (orthogonal; it decides K per slab, this changes what is coded), #32 (frequency-agnostic, so no multilingual risk).

**B2 · Multi-parent head prediction (top-4 least squares).** 0.35 bit bound; four gathers per logit, side info 0.03 bit/weight. Do it only if B1 passes and per-row gains are skewed. **P 0.2** beyond B1.

**B3 · Prototype dictionary for head rows.** k-means centroids (e.g. 4096 × 5120 at higher K) as parents instead of real rows. That removes the forest constraint; centroid bytes cost ~1.6% of the head. Test against B1 on the same proxy. **P 0.15.**

**B4 · Jensen logit bias for the head (closed form, per level).** Head row t's quantization error adds logit noise of variance $\varepsilon_t=e_t^\top H_{head}e_t$, which over contexts inflates $\mathbb E[e^{\text{logit}}]$ by $e^{\varepsilon_t/2}$, a systematic leak of mass toward rows with large error (rare tokens at floor K). Store a bias $b_t=-\varepsilon_t/2$ (248k fp16, 0.00015 bpw).
- **Source:** new (Jensen correction of a known noise variance). **Why:** the head is 5–7% of total KL at 3–5 bpw; this is the systematic part of it.
- **Fit:** per-level (depends only on the head's own codes and H). Runtime: one add.
- **Test (CPU, minutes):** headsim on existing head levels with and without $b$. **Pass:** head-only KL ≤0.95×.
- **Gain:** −0.2 to −1%; **P 0.3.** **Conflicts:** shortlist "head KL-side gain refit" (a gain, not a per-row bias); #34 mean-row (compose).

**B5 · Closed-form logit bias + temperature + unigram mixing on the shipped mix.** Fit $z'=\tau z+b$ (bias over the vocab) and $p'=(1-\lambda)p+\lambda\bar u$ on fit-pool KL. Final pass, seconds of CPU on cached logits. **Gain:** −0.2 to −0.8%; P 0.2. **Conflicts:** shortlist head temperature (B5 adds the vocab bias and the mixture).

**B6 · Slab order by tree clusters.** Sort head rows so 128-row slabs hold whole token families (Fisher-ordered within families). Then slab K and B1 parents align. Minor; P 0.1. **Conflicts:** #34.

**B7 · Embedding token-tree DPCM. KILLED** (M12: 0.054 / 0.086 bit).

**B8 · Head ≈ A·embedding linear map. KILLED** (M12: 0.14 bit for a 0.008 bpw map plus an extra GEMV).

**B9 · Shared row for dead tokens** (ids that never get p > 1e-6 under bf16 on calib): one shared row plus a per-token scalar bias. **P 0.05**; the #32 multilingual lesson applies; listed so it is not re-derived.

---

## C. Systematic error at nonlinearities and through depth

Library-safe closed forms come first. These follow the 27B lesson: swj (−6.5% at K2.5) and C9 on writers (α at the grid edge for gdn_out) grew with scale, while noise-shaping fixes shrank.

**C1 · swj-b: Jensen bias at every gated product.** Noise in a pre-activation $g$ shifts $\mathbb E[\mathrm{silu}(g+n)]$ by $\tfrac12\mathrm{silu}''(g)\sigma_n^2$. Add a per-channel additive term $b$ beside swj's gains $(c_g,c_u)$ at three sites: the MLP ($\mathrm{silu}(g)\cdot u$); the attention output gate (`output_gate_type: swish` on the q_proj gate half); and GDN's $\mathrm{silu}(z)\cdot\mathrm{norm}(o)$. Fit by least squares on bf16 inputs, like swj.
- **Source:** swj (#85/#104), DFQ-style bias correction. **Fit:** per-level (bf16 inputs). Needs a bias in the GEMM epilogue (or folds into down's input path via a constant activation channel).
- **Test:** 0.8B, fixed codes, w60's jfit harness with an extra parameter (1 h, 4070). **Pass:** ≤0.98× on top of swj at K2 and K3.
- **Gain:** −0.5 to −2%; **P 0.25.** **Conflicts:** #108 (gains at q/k, v/o, GDN z failed in the full stack; a *bias* at the nonlinearity is a different degree of freedom), shortlist "writers' mean-shift bias" (that one is at writer outputs).

**C2 · Per-head attention temperature via the q_norm/k_norm weights.** Logit noise from quantized q/k flattens attention in expectation (softmax is concave around the top key). Fit a per-head (or per-dim) multiplier on the `q_norm.weight` vector, which q_norm does *not* undo, unlike the projection scales #108 refitted.
- **Source:** new; #108's own diagnosis ("q_norm/k_norm absorb per-head uniform scales"). **Fit:** per-level; **zero engine work** (a container edit of bf16 norm weights).
- **Test:** 0.8B one-pass fit on bf16 inputs (attention-output KL), then a 27B fixed-code pair by editing the norm weights in an existing mix (≈1 card-hour). **Pass:** ≤0.99× (attention is 16 layers).
- **Gain:** −0.3 to −1%; P 0.2. **Conflicts:** #108 (different parameter).

**C3 · GDN gated-norm weight refit on frozen codes.** The 128-dim `linear_attn.norm.weight` (shared by the 48 heads of a layer) sits between the GDN core and silu(z)·out_proj. A one-pass LS fit against bf16 post-gate outputs gives a per-dimension correction shared across heads. Container edit only.
- **Why:** C9 wanting α ≈ 4–4.75 on gdn_out (#112) says the GDN branch's systematic shrink is the largest anywhere.
- **Test:** 0.8B fixed codes, 30 min. **Gain:** 0 to −1%; P 0.15. **Conflicts:** #108 (z-gate refits), #112 (row gains on out_proj; this is upstream of the gate).

**C4 · Depth-wise gain schedule fitted end to end.** One scalar per layer per branch (attention or GDN, MLP), i.e. 128 scalars on the writers' svh, fitted by KL on the qad pool with frozen codes. It compensates the coherent attenuation of each layer's contribution, which per-tensor closed forms cannot see because it is a property of the whole stack.
- **Source:** new framing of C9/swj; 2609.11716 (residual error is mostly an angular rotation; length changes small but coherent).
- **Fit:** final pass (per mix); 128 parameters, gradient on frozen codes, minutes.
- **Test:** 0.8B, compare with C9 and swj already applied. **Pass:** ≤0.98×.
- **Gain:** −0.5 to −3%; **P 0.2.** **Conflicts:** healing (#89) is the full-parameter version; this is its 128-parameter core and may be acceptable before the healing phase.

**C5 · Predicted-noise Wiener filter at layer boundaries.** Per level, store the low-rank (r = 32) error covariance each writer injects, $(W-\hat W)H(W-\hat W)^\top$. A mix's predicted residual-noise covariance at each boundary is the sum over its tensors. Take the residual signal covariance from one bf16 pass, and apply $h\leftarrow h-U(V^\top h)$, the low-rank Wiener gain, at 8 boundaries.
- **Source:** old F1 (fitted by regression on the quantized mix); new: closed form from per-level statistics, so it needs no quantized-model pass.
- **Fit:** per-level statistics + per-mix closed-form solve. Runtime +0.5% prefill (rank-32 GEMMs).
- **Test:** 0.8B, 3 seeds. **Gain:** −1 to −3%; P 0.15. **Conflicts:** old F1, #88 (that changed the quantization target; this is a runtime filter).

**C6 · Inner-product-unbiased refit inside the quantizer.** Replace exllamav3's least-squares `refit_scales` (which shrinks rows whose code carries noise) by $c=\|w\|_H^2/\langle\hat q,w\rangle_H$ per row (SoloQ's unbiased-projection scale), which is >1 by construction. Closed form, every tensor, every level.
- **Source:** 2610.07121. **Why:** the brief's mechanism. LS shrink is the coherent error; C9 at the grid edge suggests even the norm-restoring α = 1 is too small.
- **Fit:** per-level (quantizer). **Test:** 0.8B fixed codes (scales only), 3 K. **Gain:** 0 to −2% *over C9*; **P 0.15.** **Conflicts:** #85 / #112 (C9 = α-power grid; this is a parameter-free member of the same family); #48.

**C7 · Rate–distortion–perception decoding (principled default gains).** RDP theory: at fixed rate, matching the output *distribution* (what a KL score rewards) needs reconstructions with the source's variance, not the shrunken MMSE ones. Use a variance-matched reconstruction per row, $c=\sigma_w/\sigma_{\hat q}$, as the default gain for every tensor, and test seeded noise-injection decoding against it.
- **Source:** 2607.17232 (RDP tutorial); the theory behind C9/swj. **Fit:** per-level. **Test:** 0.8B fake-quant, 1 h. **Gain:** ≤C6; P 0.1 (noise injection P 0.03). **Conflicts:** C9.

**C8 · Subtractive-dither trellis (crazy).** The encoder codes $w-d$ with $d$ a seeded per-tile dither; the decoder adds $d$ back (2–4 ALU ops). The error becomes signal-independent: no shrink, no coherent part, at a small MSE cost.
- **Test:** 0.8B emulation in the CPU trellis plus fake-quant, 2 h. **Gain:** 0 to −1% (likely negative); P 0.05.

**C9 · Oracle bound for theme C (diagnostic).** On the 0.8B, fit per-row gains *and* biases on the verify set itself (oracle). The gap between oracle and swj+C9 bounds all of C1–C7. **Cost:** 2 h on the 4070. Decides whether theme C deserves more ideas.

**C10 · (Rejected) Anti-correlated rotation seeds across layers so coherent errors cancel.** The coherent component (gain shrink) has the same sign for every seed.

---

## D. Objectives, data and adaptivity

**D1 · Category-conditional correction banks with per-document routing.** Fit swj (and later heal) scale vectors per calib category (code, chat, agentic, math, multilingual) and store 5 banks (≈0.003 bpw). The plugin classifies each document from the layer-8 hidden-state mean of its first 256 tokens (a 5-way linear probe) and switches banks between prefill chunks. The bytes stay fixed; only the correction adapts.
- **Source:** new. Old G1 rejected adaptive *precision*; this adapts tiny *corrections*.
- **Fit:** per-level banks + plugin routing. **Rules:** deterministic function of the input; ask the challenge author first.
- **Test:** 0.8B fixed codes: per-category swj vs pooled, scored on each category's verify docs. **Pass:** ≥3% mean reduction vs pooled.
- **Gain:** −0.5 to −2% (more if healing banks follow); **P 0.2.** **Conflicts:** #91/#101 (statistics blends; this specialises, it doesn't blend).

**D2 · Position-aware sensitivity on long traces.** Our sensitivities come from 2k/8k-capped slices, while the private corpus scores up to 250k-token traces (19% of calib tokens). Measure per-family KL by position bin (attention vs GDN vs MLP groups up/down one K) on the 32k turn-aligned trace chunks, and re-price if attention's share grows with position.
- **Test:** 27B, ~6 evals on gpubox. **Gain:** 0 to −2%; P 0.1. **Conflicts:** #52 (ratios flat by position, but between whole mixes, never between families).

**D3 · Healing-aware allocation.** If healing recovers coherent error mostly in some families (MLP) and not others (attention, embedding), the knapsack should price post-heal sensitivity.
- **Test:** 0.8B. Per-family sensitivities before and after a 64-step scale heal; Spearman; rerun the knapsack; compare final healed KL. 4070, 1 day.
- **Gain:** −1 to −4% after healing; P 0.25. **Conflicts:** v3 note "healing-aware reallocation is a V4 question" (this only measures now); 2609.22238 saw rankings change after LSQ+ recovery.

**D4 · Pipeline-inversion study.** On the 0.8B at matched bytes: (a) knapsack mix, no heal; (b) uniform K + full per-mix heal; (c) knapsack + heal. If (b) ≈ (c), the per-level elaborations matter less than per-mix healing and effort should shift.
- **Test:** 4070, ~1 day. Strategy value only; P (changes the plan) 0.15.

**D5 · Tail-targeted healing data.** Rank qad documents by the mean of their top-25% token KL (quantized vs bf16) and heal on the top-k. FidelityQ (2610.14934) measured 1.45× enrichment and ~2% PPL; the worst decile held 45% of KL.
- **Fit:** healing phase. **Test:** 0.8B paired vs random draw. **Gain:** −0.5 to −2% of healed KL; P 0.25.

**D6 · Dynamic forward/reverse KL for healing.** Per vocabulary candidate, $\alpha_v=p_v/(p_v+q_v)$ (OnQ 2610.29378: on-policy +2 points over forward KL at 2 bits). The score is forward KL, so expect little at 2.5–4 bpw. **P 0.1.**

**D7 · Learned allocation by a differentiable relaxation (0.8B check of the knapsack).** Gumbel blend over the per-level decoded tensors with an exact budget (RCO 2605.00649 / GAMMA 2605.18475). It tests whether interactions the additive kfit2 misses are large.
- **Test:** 0.8B, 4070, 1 day. **Gain:** 0 to −3%; P 0.1. **Conflicts:** #17 (dropped at 27B for cost; a 0.8B check is cheap).

**D8 · Pairwise interaction terms in knap4** (adjacent-tensor chain DP, TWLA/LoopQuant). Composable-lens territory; listed for completeness. P 0.15.

**D9 · Per-document runtime precision. REJECTED** (old G1: bytes fixed).

**D10 · Test-time adaptation on the prompt. REJECTED** (old G2: it moves toward the data, not toward bf16).

**D11 · On-policy (self-generated) healing data. REJECTED** (#91: calib +10% with self-generated healing text).

---

## E. Codecs beyond the current trellis

**E1 · Stronger long-block codes for the embedding.** Rows are decoded by a gather (≈0 cost), so the embedding can afford exact L18–L20 Viterbi or Leech-lattice rows.
- **Why:** #113's ideal-table bound (0.944–0.975) on 4.7% of bytes ≈ 0.001–0.002 bpw. **Test:** CPU embtrellis2 at L18. **Gain:** −0.2 to −0.4%; P 0.3. Small. **Conflicts:** #113, #35.

**E2 · Variable-rate embedding rows** (offset table, rate by a frequency-agnostic importance such as the head Fisher of the same token). P 0.1. **Conflicts:** #32 (needs the multilingual guard slice).

**E3 · Entropy-coded TCQ with rANS decode inside the two-pass prefill pass and the GEMV.** This is the old C5 recast under the VRAM rule (no load-time expansion). The pre-pass already decodes once per step, so a tile-wise rANS stage there costs prefill time, not bytes; the GEMV pays per token.
- **Why:** ~0.03 bit of shaping plus no tail-biting loss (4–6%, #57). **Test:** CPU ECTCQ vs our trellis on real LDLQ tiles (2 days); then a decode-speed probe.
- **Gain:** −2 to −4% if built; **P 0.08** (decode speed). **Conflicts:** #57, #102.

**E4 · Polar / multilevel source codes with XOR-butterfly reconstruction (crazy).** Reconstruction is a butterfly like the Hadamard; encoding is successive-cancellation list decoding offline.
- **Test:** CPU, Gaussian source at N = 256 / 1024, 2 bits, against the trellis's 0.069 MSE (2 h). **P 0.03.**

**E5 · Per-tile codebook choice bit** (mul1 vs a second hash, chosen by H-weighted error; the bit hides in a scale sign as in AAAC 2605.08692). P 0.05. **Conflicts:** #100, #78-04.

**E6 · Soft-decision trellis rounding (BCJR posteriors + VAMP).** MPQ (2610.32127) at 2-bit E8P: LDLQ 15.01 → MPQ 11.63 vs coordinate descent 12.76 (Llama-3-8B). For trellis codes: forward–backward over states replaces codeword enumeration.
- **Fit:** per-level; quantization time ×10–30. **Test:** 0.8B, one family, K2, 3 seeds. **Gain:** −1 to −3%; P 0.12. **Conflicts:** #77 (adopted tile CD; small on the real quantizer).

**E7 · Curvature-weighted Viterbi branch metric** (1/d_i², CurveTQ 2610.09212). That paper reports it inert after RHT (within-block anisotropy 0.038 → 0.00013 bit). P 0.03.

**E8 · Learned 2^16-entry LUT codebook per family** (QTIP-HYB-like, 64 KB). Weights are Gaussian (M6) and the hash codebook lost (#100). P 0.05.

**E9 · Free start state per tile. KILLED by arithmetic:** 16 bits / 256 weights = 0.0625 bit to recover ~0.04 bit (#57).

**E10 · Bigger trellis. KILLED** (#113).

**E11 · Residual/superposed trellis ladders. KILLED:** GateRQ (2610.35985) measured 0.3–0.9 dB loss versus a direct code at the same rate.

---

## F. Model-level restructuring: recorded kills and what would revive them

| Idea | Status | Evidence | Revive if |
|---|---|---|---|
| F1 Cross-layer delta / motion-compensated coding (2605.24754) | dead | M1 (best-match \|cos\| 0.10–0.16 vs 0.057 random) | never on this model |
| F2 Layer drop + linear adapter | dead | E4; #90: even K0.125 pays only below 1.3 bpw | the low-bpw research ladder <1.3 |
| F3 MLP width pruning + higher K | dead | E10 null (no unit below 1e-2×median; bottom 10% hold ≤6.7%) | — |
| F4 SliceGPT residual truncation | dead | B6 (no coordinate earns 0 bits at 2.5) | ≤1.3 bpw |
| F5 Looped/recursive conversion | dead | M1 + training cost | — |
| F6 MoE-ification for bit allocation | dead | #96 (0.036 bit ceiling) | — |
| F7 Within-layer neuron VQ / merging | **dead today** | M10 (gate top-8 0.04–0.10 bit) | — |
| F8 Tied embedding/head, head←embedding map | **dead today** | M12 | — |
| F9 INR / hypernetwork weight generator | dead | M6 (Gaussian, ≈0.006 bit) | — |
| F10 Distil into another architecture of equal bytes | rejected | needs QAD-scale compute; KL vs the 27B large | — |
| F11 Mixture with a small bf16 model | rejected | a 0.8B bf16 model costs 0.48 bpw | — |
| F12 Delta from a public base checkpoint | rejected | the base's bytes count | — |
| F13 Fuse linear chains (v→o) | rejected | swish output gate and gated norm sit in between | — |
| F14 Zero-runtime sibling prediction (FA-A6/A7/A4-gate) | **dead today** | M9 | — |
| F15 Sparse + low-rank side branches (LQER/QERA/BiLoCo) | dead | quantization error is flat after RHT; B7/B8 costs | — |

---

## G. Engine-side structure

**G1 · Intra-tensor row DPCM via a 2-pass GEMM (q_proj, GDN z).** Root rows first; child rows' epilogue adds $\sum_k b_k y_{parent_k}$.
- **Why:** M10: q_proj 0.12–0.32 bit, z 0.07–0.20, minus 0.045 overhead, on 9.3% of bytes ≈ 0.005–0.015 bpw.
- **Fit:** per-level, format + kernel (one extra launch per tensor, ~+3% prefill on these tensors). **Test:** CPU real-trellis proxy on L11/L31 q_proj, L30 z. **Pass:** ≥0.12 bit net.
- **Gain:** −0.5 to −1.5%; P 0.15. **Conflicts:** #81-04 (extra-GEMM objection; this is an extra launch over the same bytes, not an extra GEMM).

**G2 · Side dot products for row exceptions** (the implementation of A1/A3/A4): one fused epilogue that overwrites up to r rows of y with bf16 dot products. Shared by A-theme ideas; specify once.

**G3 · Logit post-pass** (the implementation of B1/B4/B5): one elementwise/gather kernel after the head GEMV/GEMM; also used in KL mode at full M.

---

## H. Cross-field transplants

| # | Transplant | Our analogue | Verdict |
|---|---|---|---|
| H1 | Audio noise filling (PNS) | dither instead of zero at sub-1 K (old K2) | P 0.05, low-bpw ladder only |
| H2 | Simulated annealing / spin-glass ground states | stochastic LDLQ ensembles (old H3) | P 0.05; #41 seed spread 0.5% |
| H3 | Rate–distortion–perception | C7 | in §C |
| H4 | Turbo-iterative joint decoding of siblings | — | dead (M9) |
| H5 | Wyner–Ziv (decoder side information) | sibling outputs as side info | dead (M9) |
| H6 | Video in-loop / adaptive loop filter | C5 Wiener boundary filter | in §C |
| H7 | B-frames (bidirectional reference) | layer l from l−1 and l+1 | dead (M1) |
| H8 | Joint stereo (mid/side) | gate/up M/S coding | dead (M7, M9) |
| H9 | Compressed sensing | sparse weights | dead (Gaussian, M6) |
| H10 | Renormalisation group / coarse-graining | merge layers | dead (M1) |
| H11 | Error diffusion through depth (σΔ along the residual stream) | writers quantized against the accumulated upstream residual error | dead: #88-02 (all tensors 1.53× worse), #111 (per-mix only) |
| H12 | Intra prediction (image codecs) | **B1 head token tree** | alive (§B) |
| H13 | Super-weight / "hot pixel" exceptions (defect maps in sensors) | **A1/A2** | alive (§A) |

---

## I. Measurement and strategy

**I1 · KL source decomposition on the 27B K2.5 base.** Fixed-code swaps via the decode–patch–bf16 emulation:
- (a) massive channel exact (A3);
- (b) head bf16 (head share);
- (c) each family's real error replaced by Gaussian noise of equal tr(EHEᵀ) (coherent vs incoherent share);
- (d) swj+C9 vs oracle gains (C9).

Six evals on gpubox. It tells which of §A–§C to build, and it is the 27B analogue of the old "error-shape test".

**I2 · Size scaling of coherent fixes.** swj and C9 at the 2B and the 9B (staged, on the 4070) give a three-point trend of gain vs model size. It decides whether a further coherent-fix family is worth porting before the 27B check. 1–2 days.

**I3 · Methodology warning (M11).** Any "prediction / low-rank / shared-basis" gain measured in the readers' input-Hessian metric is inflated by the massive direction. Use the coder's basis (plain or suh-scaled) for rate claims. This affects any lens proposing H-metric low-rank side branches.

---

## Top 20 by expected gain per unit cost

| Rank | Idea | Gain @2.5 | P | First decision: cost |
|---|---|---|---|---|
| 1 | **A1** massive-row bf16 exception (L0–L3 writers) | −0.5 to −2% | 0.35 | 27B fixed-code pair via decode-patch-bf16, ~1 card-h |
| 2 | **B4** head Jensen logit bias | −0.2 to −1% | 0.30 | CPU headsim, minutes |
| 3 | **C2** attention temperature via q_norm/k_norm weights | −0.3 to −1% | 0.20 | 0.8B 1 h, then a 27B container edit |
| 4 | **C1** swj-b Jensen bias at the three gated products | −0.5 to −2% | 0.25 | 0.8B fixed codes, 1 h |
| 5 | **A3** exact massive channel end to end | −1 to −3% | 0.20 | extends #1's emulation, +1 card-h |
| 6 | **B1** head token-tree DPCM | −1 to −2% | 0.30 | colo CPU 1 day + headsim |
| 7 | **C4** depth-wise gain schedule (128 scalars, per mix) | −0.5 to −3% | 0.20 | 0.8B, 2 h |
| 8 | **C9** oracle-gain bound (decides theme C) | enabler | — | 0.8B, 2 h |
| 9 | **D1** category banks + routing | −0.5 to −2% | 0.20 | 0.8B, 4 h; author question |
| 10 | **A4** top-proxy row exceptions everywhere | 0 to −1% | 0.20 | colo census, minutes |
| 11 | **C6** inner-product-unbiased refit | 0 to −2% | 0.15 | 0.8B fixed codes, 1 h |
| 12 | **D3** healing-aware allocation | −1 to −4% (healed) | 0.25 | 0.8B, 1 day |
| 13 | **I1** 27B KL source decomposition | enabler | — | 6 evals |
| 14 | **C3** GDN gated-norm weight refit | 0 to −1% | 0.15 | 0.8B, 30 min |
| 15 | **C5** predicted-noise Wiener filter | −1 to −3% | 0.15 | 0.8B, 1 day |
| 16 | **G1** intra-tensor row DPCM (q_proj, z) | −0.5 to −1.5% | 0.15 | colo proxy, 1 day; kernel later |
| 17 | **A5** massive-channel re-anchor | −0.2 to −1% | 0.15 | 0.8B, 1 h |
| 18 | **D5** tail-targeted healing data | −0.5 to −2% (healed) | 0.25 | healing phase |
| 19 | **E6** soft BCJR + VAMP rounding | −1 to −3% | 0.12 | 0.8B, days |
| 20 | **C8** subtractive-dither trellis (crazy, cheap) | 0 to −1% | 0.05 | 0.8B emulation, 2 h |

## The 5 cheapest diagnostics that would most change this ranking

1. **27B decode–patch–bf16 pair for the massive row** (A1 arm 2, plus arm 1 = the 5 tensors fully bf16). About one card-hour, fixed codes, so there is no seed noise. ≥1.5% moves A1/A3/A4 to "build now"; arm 1 ≤0.5% kills theme A.
2. **headsim with the Jensen bias, and the token-tree CPU proxy** (B4, B1). Minutes for B4, one CPU day for B1. Decides whether the head (5–7% of KL) gets two cheap fixes.
3. **0.8B fixed-code trio: swj-b, q_norm temperature, oracle gains** (C1, C2, C9). One 4070 session. The oracle sets the ceiling for theme C; if swj+C9 already reach ≥80% of it, drop C1/C4/C6.
4. **Per-category swj on the 0.8B** (D1). Four hours. Below 3% kills per-document adaptivity cheaply, before anyone asks the challenge author.
5. **Top-4-row proxy share over all 496 tensors** (A4 census, extends today's `massrow.py`). Colo, minutes. It shows whether massive/super rows exist only in L0–L3 (A1 stays a 5-row fix) or recur at the stack's end (L63 down already shows 5.9%), which would widen theme A.

Scripts and results for today's measurements are in `/data/work/ideas-faraway/w2/`. Today I used only colo. I wrote nothing in the repo, and I left no jobs running.
