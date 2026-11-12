# Composable lens, round 2 (2026-11-08): ideas that must land before the freeze

This round covers only the per-level library, quantizer settings, codebooks, format and level menu, and knapsack inputs. Per-mix passes, healing/QAD and kernel work are excluded; where an idea needs runtime support, it uses an op the plugin already has.

All ideas build on the adopted stack:
- H_out at every level, on the block-diagonal path
- the #117 block order
- swj
- C9 on the writers
- tile CD on attn_qg
- σ 0.2 at K0.125
- the 55-level ladder

## 0. What I checked this round

All runs were read-only, on colo (r640). Scripts and results are in `/data/work/ideas-composable/`. Inputs:
- the bf16 27B
- the published calib H_in
- `/data/work/hout27/hout-bn` (bn1920 Σdy² diagonals)
- `headsens.npz` (w94 head Fisher)

The common model is the **high-rate Gaussian model**. With a Kronecker H_in ⊗ H_out and equal rounding noise per coordinate, LDLQ's expected error is proportional to (Σ input pivots) × (Σ output pivots). Pivots are taken from the LDL in exllamav3's order (last block first), after the sign-flipped 128-point Hadamard. The output factor is diag(h·sv²) with σ_out 0.01; sv² is the row mean square, because the quantizer normalises rows by sv before the Hadamard. A ratio of Σ pivots is therefore the predicted change in that tensor's two-sided objective. It ignores finite-K effects, and the 27B fraction that survives in the real quantizer is unknown; diagnostic 3 measures it.

| # | Check (script) | Result | Consequence |
|---|---|---|---|
| 1 | Anisotropy of the LDL metric within a tile (`gamma.py`: L10 qkv/z/out_proj, L11 q/o_proj, L3 k_proj) | **Input side:** diagonal spread inside a 16-column block after su·sign·RHT is **≈1e-4 bit** (8.7e-5…6.2e-4). The full 16×16 block's eigenvalue spread is 0.012–0.032 bit; spread between blocks is 0.014–0.10 bit. **Output side:** spread within a tile is 0.003–0.047 bit; between 128-row blocks it is **0.27–0.49 bit** for qkv/q/k but 0.002–0.005 for writers. The 17408-dim tensors (up, gate, down) were killed before finishing, probably on memory. | A D-weighted Viterbi metric (CurveTQ, 2610.09212) is inert after RHT. The one large gap left is **between output blocks**. |
| 2 | Allocation headroom at head level (`census.py`, all 400 tensors) | g128 = ½log₂(AM/GM) over 128-row blocks of mean(h·sv²): **ssm_qkv 0.282** (max 0.81 at L0), **attn_qg 0.169**, ssm_z 0.115, attn_v 0.081, attn_k 0.050, up 0.008, gate 0.006, writers ≤0.004. Parameter-weighted 0.048 bit. | Reproduces #96's per-head ceiling. Two-sided LDLQ cannot use it, because a diagonal H_out becomes block-diagonal after the 128-point output Hadamard and the blocks line up with the 128-row heads. |
| 3 | Segment split (`seg.py`) | A q\|k\|v split of in_proj_qkv captures **0.043 of 0.282 bit** (15%); a q\|gate split of attn_qg captures 0.062 of 0.169. | Per-segment K is not worth a format field: the spread is between heads, not between segments. |
| 4 | **Output-row interleave** (`perm.py`, `perm2.py`): Σ pivots with natural storage vs rows (or whole 16/32/64-row tile groups) dealt across the 128-row Hadamard blocks | **Row stride vs natural:** ssm_qkv **×0.717** (0.750→0.538; GM bound 0.502); attn_qg ×0.817; ssm_z ×0.895; attn_v ×0.919; attn_k ×0.951; up and gate ×0.995. A random permutation gives the same as the stride. **Tile-aligned groups:** 16-row qkv **×0.775**, z ×0.912, q ×0.857; 32-row ×0.827 / 0.947 / 0.887; 64-row ×0.893 / 0.977 / 0.998. For scale, H_out itself takes qkv from 1.01 to 0.75 under the same model. | **New lever, theme A.** It reaches about 93% of the full-dimension bound with no change to the Hadamard size or to rates. |
| 5 | Residual-stream **channel** permutation on the input side (`rperm.py`, 8 readers) | Σ pivots ×0.981–1.003; attention q_proj is best at ×0.981–0.984, gate/up ×0.998. Natural LDLQ is already within 1–3% of the GM bound. | Not worth a route-A conversion. The input side is done. |
| 6 | **Head: two-sided + interleave** (`head.py`: fisher_fit_all, centred lm_head rows, s5-3 slabs) | Top slab (60,928 rows, 98.3% of F·‖w‖²): one-sided ≈1.01, two-sided in sorted order 0.827, **two-sided with in-slab stride 0.329 (×0.40, 0.665 bit)**, GM 0.206. Slabs 2 and 3: ×0.91. | **Theme B.** The slab head already stores a row permutation plus a gather, so the interleave costs nothing in the format. |

**Operational blockers from the freeze review are not repeated here:** the `expandable_segments` setting on the rqah ladder branch, and scoring one full-stack level.

---

## 1. Ideas by theme

Each idea lists:
- **Mechanism**
- **Source** (paper ids or "new")
- **Why** it helps
- **Fit**
- **Test**: cheapest test, GPU-hours, machine
- **Gain/risk**: KL % at 2.5 bpw unless stated
- **Conflicts** with INDEX experiments

Machine names:
- **card 0**: gpubox R9700
- **4070**: the 0.8B proxy box
- **colo**: cpu1 / r640
- **Thor**: cpu2

### A. Output-side interleaving for two-sided LDLQ (new; must be decided before the ladder)

**A1 · Tile-interleaved output rows for GDN `in_proj_qkv`**
- *Mechanism:* Store the 10,240 output rows permuted in whole 16-row tiles, dealing tiles round-robin across the 80 Hadamard blocks, so every 128-row block holds tiles from 8 different heads. Within each mixed block, two-sided LDLQ's output feedback can then move error from high-h rows to low-h rows. Today each block is one head with nearly uniform h, so there is little to move. h and svh move with the rows. At inference, one gather after the GEMM (or folded into the depthwise conv1d's channel indexing) restores natural order.
- *Source:* New (check 4). Cross-area:
  - OSCAR's bit-reversal interleave of channels into quantization groups: removing it costs 70.01 → 68.00 (2605.17757).
  - NOVA-KV grouping on RULER-128K: sorted 37.2, random 74.4, equal-volume 75.4 (2608.04074).
  - Interleavers in channel coding, which spread bursts across codewords.
  - Here the "group" is the output Hadamard block of a two-sided LDLQ, not a scalar quantization group.
- *Why:* The between-head spread of h·sv² (0.28 bit on qkv) is invisible to block-diagonal feedback. Interleaving gives Σ pivots ×0.775 (tile) or ×0.717 (row), i.e. a 22–28% lower two-sided objective at high rate. That is about as much again as H_out already gives this tensor.
- *Fit:* Format change (per-tensor permutation field plus a graph gather, see G1) and a quantizer setting. Ladder levels for this family must be built with it.
- *Test:*
  1. Run M2: realized f2 on card 0, about 20 min.
  2. Then a 27B K2.5 KL pair through the dense-splice harness (M1): dequantize the permuted levels, un-permute, splice bf16. About 1.5 card-h, on golden-search27b two parts, two seeds.
  3. Pass: geo ≤ 0.995 with both parts < 1.
- *Gain/risk:* −0.3…−0.8% KL (central −0.5%; A1 and A2 together −0.4…−1.2%); P(≥0.3%) ≈ 0.4.
  - The central figure assumes about half the high-rate gain survives and KL elasticity ≈0.2 to two-sided error, as H_out: f2 −6…−14% at K4 gave KL −3.6%.
  - Risk: proxy gains can fail to reach KL (see #106). The gather adds about 2–4% to each GDN layer's prefill time if not fused.
- *Conflicts:*
  - **#96** (row-slab K, ceiling 0.036 bit): A1 changes no rates and reaches 0.24 of the 0.28-bit per-head-K ceiling at fixed K. That makes BACKLOG's per-head K moot.
  - **#106** (o_proj head-permutation act order, input side, proxy −9% with no KLD gain): different side and objective, but the same warning, hence the KL pair as gate.
  - The **512-point Hadamard** was closed without tests; A1 gets about 93% of the full-dimension bound without it.

**A2 · Same interleave for attention `q_proj` (attn_qg)**
- *Mechanism:* As A1, over the 12,288 q+gate rows. Mixed blocks contain q-half and gate-half tiles of different heads. The gather goes before the per-head reshape for q_norm.
- *Source:* New (check 4); combines A1 with the per-head q/gate h asymmetry (segment means 1.32 vs 0.68).
- *Why:* Σ pivots ×0.857 (tile) or ×0.817 (row). This tensor is 4.1% of parameters and carries much of the attention-input share of the H_out gain (13% on the 0.8B, #82).
- *Fit:* Format plus setting. Compose with tile CD on attn_qg (A9).
- *Test:* Run arms in the A1 pair (same splice), plus M2 on L11 q_proj.
- *Gain/risk:* −0.1…−0.3%. Same risks as A1.
- *Conflicts:* #77 (tile CD on attn_qg) composes; #96; #106 (o_proj side).

**A3 · Interleave for `in_proj_z`, attn `v_proj` and `k_proj`**
- *Mechanism:* Same mechanism on the remaining readers whose output blocks follow heads.
- *Source:* New (check 4).
- *Why:* Σ pivots: z ×0.912 (tile, 6.2% of parameters); v ×0.919 and k ×0.951 (row level; tiny tensors).
- *Fit:* Format plus setting.
- *Test:* An arm of A1's pair. Only include if A1 passes.
- *Gain/risk:* −0.05…−0.2%. The gather costs about the same as on qkv for a third of the gain, so this is marginal for z.
- *Conflicts:* None beyond A1.

**A4 · Choose the interleave granularity: row, 16, 32 or 64 rows**
- *Mechanism:* Decide how fine the dealing is. 16-row groups are whole EXL3 output tiles, so each stored tile maps to one contiguous 16-channel run and the gather is tile-granular. Coarser groups keep less of the gain.
- *Source:* New (check 4, `perm2.py`).
- *Why:* qkv keeps 0.775 / 0.827 / 0.893 of natural at 16 / 32 / 64 rows vs 0.717 per row. That is 80% / 61% / 38% of the row-level gain. The q_proj 64-row arm keeps nothing (0.998).
- *Fit:* Format decision.
- *Test:* None beyond the numbers above; take 16-row unless G1's gather is free at row level.
- *Gain/risk:* Preserves the A1/A2 gain; 32-row loses about 40% of it.
- *Conflicts:* —

**A5 · Balanced ("equal-volume") assignment instead of a plain stride**
- *Mechanism:* Sort the tiles by mean h·sv² and deal them in snake order (bit-reversal or zig-zag) so every block gets one tile from each quantile. A greedy pass then swaps tiles to equalise the blocks' AM/GM.
- *Source:* NOVA-KV equal-volume grouping (2608.04074: 74.4 random → 75.4 equal-volume); LIRQuant's zigzag permutation (2610.25675). Here it is applied to Hadamard blocks under two-sided feedback.
- *Why:* Row stride 0.538 vs GM bound 0.502 on qkv leaves about 7%; the gap at tile level is larger.
- *Fit:* Setting, the same field as A1.
- *Test:* colo CPU, 15 min. Extend `perm2.py` with a balanced arm. Pass: ≥3% lower Σ pivots than the stride at 16-row granularity.
- *Gain/risk:* An extra 0…−0.1% KL. Risk: overfitting h; gate with A6.
- *Conflicts:* —

**A6 · Out-of-sample check of the head-level structure in h (gate for A1–A5)**
- *Mechanism:* Interleaving trusts the between-head ratios of h much more than today's block-diagonal path does. Choose the permutation and LDLQ factor from one estimate of h (bn1664 snapshot, or the plain Σdy² diagonal). Evaluate the realized Σ pivots in the metric of another estimate (bn1920 or block halves).
- *Source:* New; split-half stability as in #33-17 and #33-27.
- *Why:* If early-layer head ratios are noise (#33-27: two 128-sequence blocks hold 4–27× the median energy in early layers), the gain partly disappears out of sample.
- *Fit:* Diagnostic.
- *Test:* CPU, 15 min, after copying two diagonal sets (2 × 0.67 GiB) from gpubox to colo. Pass: the out-of-sample ratio keeps ≥75% of the in-sample gain.
- *Gain/risk:* Gate only.
- *Conflicts:* #33-29 (estimators equal within 1%) suggests stability, but it was measured at tensor level, not head level.

**A7 · Tempered output factor h^p**
- *Mechanism:* Use h^p with trace renormalisation and p ∈ {0.5, 0.75, 1} in prepare_H_out_blocks. Possibly make p depend on K, with a smaller p at low K where errors are far outside the quadratic regime.
- *Source:*
  - ShamAN-Q tempering, p=½ best: 1.7B PPL 17.04 → 16.72 (2609.38521).
  - KBBQ's brake: full whitening λ=1 is worse than λ=0.75; NVFP4 RTN 91.3 vs 94.0 (2609.08135).
  - Prior-round C2/T7 proposed tempering in general; untested. New here: interleave raises its stakes.
- *Why:* A noisy h is overtrusted most where feedback has the most leverage, i.e. interleaved blocks and the head.
- *Fit:* Quantizer setting, all families.
- *Test:*
  1. w87's held-out proxy on colo (quantize on H_A, score on H_B), two-sided, 6 K, 3 values of p, about 3 CPU-h.
  2. If a cell gains ≥1%, run a 27B K2.5 fixed-ladder pair (1.5 card-h).
- *Gain/risk:* 0…−0.4%. Low risk.
- *Conflicts:* #114-03 swept σ_in with H_out on, not the shape of the output factor; #33-29.

**A8 · Retune σ_out for interleaved families**
- *Mechanism:* After mixing, each block's rotated H_out is far more anisotropic, so the fixed σ_out = 0.01 is a stronger relative prior. Sweep 0.003 / 0.01 / 0.03 on the interleaved tensors.
- *Source:* New.
- *Why:* σ_out was set for the 1,920-sequence collection (#33-24) on natural blocks and never swept.
- *Fit:* Setting, per family.
- *Test:* An arm in M2 (card 0, +10 min) or w87's proxy on colo.
- *Gain/risk:* 0…−0.1%.
- *Conflicts:* #114-03 (σ_in only).

**A9 · Interleave plus two-sided tile CD on qkv/z/qg**
- *Mechanism:* With mixed blocks, the output metric within a tile becomes more anisotropic, already 0.033–0.047 bit on natural qkv/q (check 1). Tile coordinate descent on the two-sided objective then has more to win. Run cd3 on the interleaved families.
- *Source:* #77, plus check 1, plus A1.
- *Why:* Tile CD with H_out on natural attn_qg gave only a 0.9989 proxy at K4. Interleave changes the metric CD sees.
- *Fit:* Setting. Costs ×3.7 quant time on those families, about +10 min per level.
- *Test:* An M2 arm (f2 for CD vs no CD on interleaved L10 qkv), card 0, 10 min. Pass: ≤0.99.
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* #77; the cdhout27 pass (no harm only).

### B. LM head (head donors are levels; must be decided before they are built)

**B1 · Two-sided head LDLQ with a Fisher-diagonal H_out, rows interleaved within each slab**
- *Mechanism:* Quantize each slab of the slab head with `ldlq_2hess_blocks`. The 1-D output factor is the per-row Fisher F_v = E[p(1−p)] times the centred row energy. Store the rows of each slab in stride (or balanced) order instead of sorted-by-Fisher order. The slab head already carries a row permutation and gather, so only the table changes.
- *Source:*
  - New numbers (check 6). Combines #34's slab head and Fisher, w81's 1-D block path, and A1's interleave.
  - The two-sided head was a merged Tier A item, and the prior round's H4 mentioned "strided order". Neither was quantified or tested.
- *Why:*
  - The head's KL is to second order exactly ½ Σ F_v Δz_v², with Δz = Δw·h: a Kronecker diag(F) ⊗ E[hhᵀ], if p and h are treated as independent.
  - Sorted rows are the worst case for block-diagonal feedback.
  - Top slab: one-sided 1.01 → two-sided sorted 0.83 → two-sided interleaved **0.33 (×0.40)** at high rate.
  - The head is about 5% of calib-part KL at 4.0 (#34-02) and probably 4–9% at 2.5, where it sits around K3–4.
- *Fit:* Per-level (head donors), with zero format change.
- *Test:*
  1. card 0. Head quant for 3 arms at the s5-3 and s6-3 budgets (one-sided today, two-sided sorted, two-sided interleaved), about 1.5 card-h; head quants need about 22 GB of host RAM.
  2. Score head-only KL with headsim on the 32,768 held-out head inputs, reporting the rare-token guard KL.
  3. Pass: head KL ≤0.8× at equal bytes and guard KL ≤1.1×.
  4. Then one whole-model pair on the calib part.
- *Gain/risk:* −0.7…−2.5% on calib-like text at 2.5; relatively larger at 4–5 bpw. P ≈ 0.5.
  - Risks: the diagonal Fisher ignores −ppᵀ; Fisher comes from top-512 truncated goldens; the high-rate model is weaker at K3.
- *Conflicts:*
  - #34 / #34-02 (slab head BUILD, corrected rowerr): B1 composes with it.
  - #24 (heads above K6 have no KL gain) concerns rate, not allocation.
  - #119 (embedding at its floor) is unrelated to the head.

**B2 · Two-sided head in natural vocabulary order (no permutation at all)**
- *Mechanism:* For a uniform head, or if the slab gather slips, quantize two-sided in vocab-id order. BPE ids roughly follow merge frequency, so blocks are only partly sorted, and feedback may already gain a lot.
- *Source:* New; a degenerate case of B1.
- *Why:* It needs neither the gather nor slab support.
- *Fit:* Per-level, no format change.
- *Test:* colo CPU, 5 min (diagnostic 1): Σ pivots in vocab order vs sorted vs interleaved. Run headsim only if ×≤0.8.
- *Gain/risk:* −0.2…−1%. Unknown until diagnostic 1.
- *Conflicts:* #34.

**B3 · Re-plan slabs on two-sided row errors**
- *Mechanism:* `headslab.py plan` and knap5's headgrid price rows with one-sided rowerr. With B1, equal head KL needs fewer K7 rows, so rebuild rowerr from two-sided donors and rerun the planner and headgrid.
- *Source:* #34 planner; w88 headgrid; B1.
- *Why:* The plan was chosen on errors B1 changes by up to ×0.40.
- *Fit:* Per-level plus knapsack input.
- *Test:* CPU minutes once B1's donors exist.
- *Gain/risk:* An extra 0…−0.5% via freed head bytes. 0.66 bit on the top slab ≈ 0.008 bpw.
- *Conflicts:* #34-02's rowerr fix is required first (done).

**B4 · Fisher floor and pooling for the interleaved head (rare-token guard)**
- *Mechanism:* Define the H_out of B1 as F_v + ε·quantile(F), pooled over the fit, qad and search pools (never verify). Rows with noisy, tiny Fisher must not get feedback that wrecks them.
- *Source:* Prior-round E2 (ε-smoothing for ordering); #32's lesson that rare rows matter more than frequency predicts.
- *Why:* Interleave makes feedback act on the full dynamic range of F.
- *Fit:* Setting of B1.
- *Test:* headsim arm with ε ∈ {0, 1e-3 quantile, median×1e-2}, judged on guard KL.
- *Gain/risk:* Protective; −0…−0.3%.
- *Conflicts:* #32.

**B5 · σ_in / σ_out sweep for the two-sided head**
- *Mechanism:* The head's damping was tuned one-sided with mean-row (K3 best 0.005, K5 default). Re-sweep with B1.
- *Source:* #34 mean-row damping note.
- *Why:* Two-sided feedback changes which regularisation is best.
- *Fit:* Setting.
- *Test:* 4 extra headsim arms, card 0, about 0.5 card-h.
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* —

**B6 · Low-rank correction for the hottest rows (−ppᵀ term)**
- *Mechanism:* For the top 1–4k Fisher rows, use a dense output factor block E[diag(p) − ppᵀ] restricted to them, plus diagonal elsewhere. Run dense ldlq_2hess on that sub-block.
- *Source:* Prior-round H4 (−ppᵀ); SoftWater (2608.12026) drops cross-class terms under a dither argument.
- *Why:* The hottest rows' −p_k p_l terms are largest.
- *Fit:* Setting of B1.
- *Test:* headsim, one arm.
- *Gain/risk:* 0…−0.2%; low prior.
- *Conflicts:* —

**B7 · Fisher from generated positions as the head's H_out**
- *Mechanism:* headsens has `fisher_*_gen` (scored positions only) beside `*_all`. The score uses generated positions, so use the gen Fisher (pooled over pools) for B1's factor and ordering.
- *Source:* #83 (gen-masked loss helped the H_out collection: K3 0.971, K2 0.950).
- *Why:* Prompt positions are not scored but shape today's Fisher.
- *Fit:* Setting of B1.
- *Test:* headsim arm at no extra quant cost beyond B1 (one more donor).
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #83, #83-02.

### C. Two-sided LDLQ calibration (H_out on at every level since 11-08)

**C1 · Two-sided `ldlq_drift` table**
- *Mechanism:* The global scale search multiplies its tile-MSE optimum by `LDLQ_DRIFT` (1.08 at K1 … 1.004 at K3). That table was measured one-sided. Measure the realized RMS(feedback-adjusted target)/RMS(W) under two-sided feedback per K and refit the table.
- *Source:* Prior-round T6/H64 named it; INDEX has no test.
- *Why:* Output feedback adds variance to the targets the trellis sees, so a one-sided drift under-scales.
- *Fit:* Setting.
- *Test:* colo CPU with the rxq CPU stand-in (cpukern), 10 tensors × K {1, 1.5, 2, 2.5, 3}, about 1 h. Act if the ratio moves ≥0.5%.
- *Gain/risk:* 0…−0.3%, larger at ≤K1.5.
- *Conflicts:* #107 (interpolated vs exact drift, null): that test was one-sided.

**C2 · Global scale search weighted by H_out**
- *Mechanism:* `g_scale_search_batch` scores each candidate by plain mean squared error over sampled tiles. Weight each tile by its 128-row output block's mean h (and sv²) so the shared scale fits the hot blocks. With interleave (A1) the weights become flat, so C2 matters for families that stay natural.
- *Source:* H_Z-PTQ (2610.40345): Fisher-weighted scale search matched the best tuned Lp without tuning.
- *Why:* qkv's spread between blocks is 0.47 bit (check 1); the shared scale is set by the average block.
- *Fit:* Setting.
- *Test:* M2-style f2 on 9 tensors (card 0, 15 min) or the CPU stand-in. Pass: ≤0.995 f2.
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* #39 (gain flat, but by plain MSE).

**C3 · Two-sided `su` refit**
- *Mechanism:* `refit_scales` alternates per-row svh and per-column su least squares in the H_in metric. With a diagonal H_out, the svh step is unchanged, because each row's scale is independent of its weight. The su step sums over rows, so weight the rows by h.
- *Source:* New; follows from the Kronecker objective.
- *Why:* su is shared by all rows, and today hot and cold rows count equally.
- *Fit:* Setting (refit stage), free at inference.
- *Test:* CPU, 1 h, w81's f2calc on 9 tensors' saved wq. Pass: f2 ≤0.997 with one-sided error not worse than ×1.01.
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* —

**C4 · σ_out per family × K (no interleave)**
- *Mechanism:* Sweep σ_out ∈ {0.003, 0.01, 0.03, 0.1} on w87's held-out two-sided proxy (factor from half A, scored on half B).
- *Source:* #114-03 machinery.
- *Why:* σ_out has never been swept since the 27B collection; σ_in was rechecked.
- *Fit:* Setting.
- *Test:* colo, about 3 CPU-h, same 27 tensors × 2 samples. Pass: a group cell ≥1% held out, ≥⅔ agreeing.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #114-03 (σ_in null with H_out).

**C5 · Two-sided tile CD on GDN `in_proj_qkv` / `in_proj_z` without interleave**
- *Mechanism:* Extend `tile_cd_families` to ssm_qkv / ssm_z. Their output metric within a tile has an eigenvalue spread of 0.033–0.047 bit, as large as attn_qg's 0.032 (check 1).
- *Source:* #77; check 1.
- *Why:* CD captures the anisotropy within a tile that Viterbi ignores.
- *Fit:* Setting. Costs about +10 min per level (qkv is 10% of parameters).
- *Test:* f2 on L10/L40 qkv/z at K2.5/K4, card 0, 15 min. Pass: ≤0.995.
- *Gain/risk:* 0…−0.15%.
- *Conflicts:* #77; CD×H_out (0.9989 proxy on attn_qg at K4).

**C6 · Two-sided band for donor scoring in rqah**
- *Mechanism:* The server checks the plain H_in error, and an honest H_out level sits about 1.076× above the one-sided minimum (#118). A worker that skips H_out therefore passes with a better score. Score in the two-sided metric instead: ship the bn h with each job and check proxy_err_2s.
- *Source:* Merged B1 (11-08).
- *Why:* Integrity of the frozen library.
- *Fit:* Ladder verification.
- *Test:* Replay 20 accepted dry-run jobs plus 20 synthetic "skip H_out" jobs; all synthetic ones must be flagged.
- *Gain/risk:* Protective.
- *Conflicts:* —

**C7 · Certify Hessians and H_out at job load**
- *Mechanism:* Check finiteness, positive diagonals and Cauchy–Schwarz on all 257 H_in files and the h vectors before any job runs.
- *Source:* Merged B2; LOG 11-06 (corrupt hinX for L43 o_proj).
- *Why:* A single corrupt file silently ruins one tensor in every level.
- *Fit:* Ladder hygiene.
- *Test:* CPU minutes.
- *Gain/risk:* Protective.
- *Conflicts:* —

**C8 · Log two-sided and per-row errors in every ladder job**
- *Mechanism:* Persist proxy_err_2s, the per-row h-weighted error and the pre-refit scales into each level's stamp and jsonl.
- *Source:* Prior-round C3/P1, extended to two-sided.
- *Why:* E2, E20, C6 and A-theme repricing need these numbers. Recomputing them later costs card time.
- *Fit:* Enabler.
- *Test:* check_ladder.
- *Gain/risk:* Enabler.
- *Conflicts:* —

### D. Trellis, codebook and scale search (per-level)

**D1 · Global scale by LDLQ replay on sampled panels**
- *Mechanism:* Replace the fine stage of the scale search (independent tiles × drift factor) with real LDLQ, including two-sided feedback, on 2 sampled 128-column panels per candidate scale. Score each candidate by the two-sided proxy of the panels.
- *Source:* Schur Replay (2609.36654: Euclidean score instead of Schur loses 90 questions); Proxy-static (2609.37416); prior-round C11. New angle: with H_out everywhere the drift factor is doubly stale (C1).
- *Why:* The scale curve is flat, but the drift approximation is worst at ≤K2.
- *Fit:* Setting. About +3% quant time.
- *Test:* f2 on 9 tensors at K1.5/K2.5 (card 0, 20 min). Pass: ≤0.995.
- *Gain/risk:* 0…−0.3%, mostly at ≤2 bpw sizes.
- *Conflicts:* #39 (MSE-flat gain); #107.

**D2 · One re-LDLQ round after the scale refit, at core levels only**
- *Mechanism:* For K2.0–3.25 levels: quantize, run refit_scales, then re-run LDLQ and the trellis on the refitted su/sv grid, then refit again.
- *Source:* CoS-VQ (2610.37336: joint index+scale same-start 21.0 → 13.7 Wiki2 at 2 bit); JARQ (2609.38599: independent per-group fit leaves a 21–25% objective gap); prior-round C8.
- *Why:* Codes chosen for the pre-refit grid are suboptimal on the post-refit grid (JARQ Prop. 3.1).
- *Fit:* Setting for about 10 levels. Doubles their quant time (+6 card-h).
- *Test:* f2 and one-sided proxy on 9 tensors at K2.5 (card 0, 30 min). Pass: f2 ≤0.99. Then a 27B pair.
- *Gain/risk:* 0…−0.8%. Uncertain: refit_scales already captures most of the scale coupling.
- *Conflicts:* #85 (refits on frozen codes); no INDEX test of re-quantization.

**D3 · Search wrap candidates at the ≤K1.5 levels**
- *Mechanism:* For sub-1 and K1–1.5 levels only, try 4–16 tail-biting wrap candidates per tile and keep the best.
- *Source:* #57: 16 candidates recover 12% / 8% / 4% of the gap at K1 / 2 / 3, i.e. 0.8% / 0.4% / 0.16% error.
- *Why:* The 1.0/1.25/1.5/1.75 sizes are now release sizes and the sub-1 levels are built; the gap is largest there.
- *Fit:* Setting (encoder only, no format change). Costs 4–16× trellis time on those quick levels.
- *Test:* CPU stand-in or settle64 at K0.5/K1/K1.5, colo, 2 h. Pass: ≥0.5% proxy.
- *Gain/risk:* −0.2…−0.5% at 1.0–1.25 bpw; 0 at 2.5.
- *Conflicts:* #57 (declined, as "not worth a kernel change"; this is encoder-only).

**D4 · K0.125–0.375: guarantee proxy ≤ 1 and add a zero-row fallback**
- *Mechanism:* #114 saw the GPU search at K0.125 worse than all-zero (err 1.014). After refit, set svh = 0 for any row whose error exceeds its energy, so the decoded row is exactly 0. This is legal in the format.
- *Source:* #114; MMSE shrinkage.
- *Why:* Bytes that add error are worse than nothing, and the knapsack picks sub-1 only at ≤1.3 bpw (#90).
- *Fit:* Setting (refit stage).
- *Test:* CPU check over the existing i4 K0.125 rows from w77/w79.
- *Gain/risk:* 0…−0.3% at 1.0 bpw.
- *Conflicts:* #114, #114-02.

**D5 · Tile CD for all families at sub-1 levels**
- *Mechanism:* Sub-1 levels are cheap to quantize, so cd3 there costs little. At low K the coupling between tiles is strongest.
- *Source:* Prior-round C6; #77.
- *Why:* Strengthens the 1.0–1.25 sizes.
- *Fit:* Setting.
- *Test:* settle64 at K0.25/0.5 (colo, 1 h). Pass: ≥1% proxy.
- *Gain/risk:* −0.1…−0.3% at 1.0 bpw.
- *Conflicts:* #77 (CD on attn_k closed for V3 at normal K).

**D6 · Shape check of the canonical fractional mask at 2.25–2.75**
- *Mechanism:* Compare the even-spread masks (0x8888 / 0xAAAA / 0xEEEE) against 2–3 alternative phases at the same rate on settle64.
- *Source:* Prior-round K3.
- *Why:* The masks around 2.5 are the most used, and the even spread was chosen by convention.
- *Fit:* Format (a mask is implied by width; a new mask would need a field).
- *Test:* colo, 1 h. Pass: an alternative ≥1% better at ≥2 fractional K.
- *Gain/risk:* 0…−0.2%; low prior.
- *Conflicts:* #36 (on-curve at the canonical mask).

### E. Knapsack inputs and sensitivity measurement

**E1 · Price with private-corpus category weights (calib part), not the pooled two-part slice**
- *Mechanism:* Every pricing measurement taken from now on (head/embedding options, stack factors, slopes) should be read per part. Combine the parts with the private mix's weights, which are calib-like.
- *Source:* #34-02 LOG: bf16 head −0.3% on xcorp vs −5.3% on the calib part.
- *Why:* Pooled ~65%-xcorp slices understate head and embedding value about 2.5×, so the knapsack and any search accept rule would sell head bytes to the trunk. The same bias likely affects category-sensitive trunk families (agentic text gains most, d1110).
- *Fit:* Knapsack input; also the accept rule.
- *Test:* CPU minutes. Re-read existing per-part rows (M6) and recompute knap5 head/embedding prices at calib weights. Act if the head budget at 2.5 moves ≥⅛ bit.
- *Gain/risk:* Protects 0.5…2% of calib-like KL (0 if all prices are already calib-based: headsim is calib-A).
- *Conflicts:* eval-data.md's two-part rule (it concerns the *sign* of decisions, not pricing).

**E2 · Two-sided level positions in kfit / knap5**
- *Mechanism:* Fit per-tensor error positions from proxy_err_2s (C8) rather than the one-sided proxy. One-sided proxies cannot see any output-side gain (A, B, C2–C4).
- *Source:* Prior-round G1/P2; #118 item 2(c) not done.
- *Why:* #118: H_out levels sit ×1.076 one-sided above plain at both K. Interleave will move qkv/qg/z further, family-dependently.
- *Fit:* Knapsack input.
- *Test:* CPU. Refit on the V3 levels' logs, cross-validated against the existing 27B pairs.
- *Gain/risk:* −0.1…−0.5% (needed to cash in A1–A3).
- *Conflicts:* #118.

**E3 · Apply #118's per-family offsets**
- *Mechanism:* knap5 overprices attention errors: attn_k 0.73–0.74, attn_v 0.80–0.83, attn_qg 0.92–0.93, ssm_qkv 0.95. Use today's levels as anchors or apply per-family factors.
- *Source:* #118.
- *Why:* Attention q/k/v currently get too many bits.
- *Fit:* Knapsack input.
- *Test:* knap5 rerun (minutes) plus 2 evals at 2.5 (0.5 card-h). Pass: predicted ≥0.2% better and measured not worse.
- *Gain/risk:* −0.1…−0.3%.
- *Conflicts:* #118 (lead decision pending).

**E4 · Stack factor as a function of K, from 2–3 full-stack levels**
- *Mechanism:* knap5 prices swj with a K curve but not H_out (0.98 at K2.5 vs 0.964 at K4), C9 (0.9925), #117 (0.997) or tile CD. Score the w92 full-stack K2.5 level plus K3.25 and K4 against plain twins and interpolate a combined factor by K.
- *Source:* New; #104 (swj shrinks with K) and #33-28 (H_out grows with K).
- *Why:* Mixed-K mixes trade K2.5 against K4 tensors with the wrong relative prices.
- *Fit:* Knapsack input.
- *Test:* 2 extra full-stack levels (built anyway in ladder order) plus 4 evals (about 1 card-h).
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #118.

**E5 · Family-resolved stack factors from fixed-code splices**
- *Mechanism:* Splice one family of an H_out level into a plain level (and swj / C9 similarly) for 5 family groups. This yields KL per unit of f2 per family (M5 uses existing arms first).
- *Source:* #82 (0.8B attribution: gdn_in 39%, mlp_in 44%, attn_in 13%, writers 3%).
- *Why:* The 27B shrink of H_out (8% → 2%) may be family-specific, and it sets how much A1 is worth.
- *Fit:* Knapsack input plus A-theme go/no-go.
- *Test:* 10 evals on card 0 (about 2 card-h).
- *Gain/risk:* Enabler. −0…−0.3% via pricing.
- *Conflicts:* #33-28, #82.

**E6 · Composition calibration β**
- *Mechanism:* Regress measured mix KL on the knapsack's additive prediction over existing V2/V3 mixes. If β<1 (compressive, 2610.09227: 0.26–0.77), rescale marginal prices near the operating point.
- *Source:* Quantize-by-Drift (2610.09227); prior-round P4.
- *Why:* Systematic non-additivity biases low-bit-heavy plans.
- *Fit:* Knapsack input.
- *Test:* CPU, minutes, on existing measurements.
- *Gain/risk:* 0…−0.5%.
- *Conflicts:* #11 (symmetric compounding replaced by kfit2's directional term). Check β *on top of* kfit2.

**E7 · Pair interaction terms within a layer in the layer DP**
- *Mechanism:* Measure 10 pair swaps (gate/up with down, ssm_qkv with out_proj, q with o, all at the same depth) to get β_ij = f(i,j) − f(i) − f(j). Add them to knap5's per-layer DP, which is exact for pairs inside a layer.
- *Source:* TWLA ILA-AMP (2606.13054: a pairwise chain DP gave 8.31 vs 11.92 PPL); LoopQuant's Shapley pairs (2610.16368).
- *Why:* Siblings sharing an input, or writers fed by noisy readers, interact.
- *Fit:* Knapsack input.
- *Test:* About 20 evals (about 2 card-h) on golden-search27b.
- *Gain/risk:* 0…−0.5%.
- *Conflicts:* #14 (single swaps too noisy; use group-level pairs); #16 (Shapley skipped for cost; this is 10 pairs).

**E8 · In-context slopes by noise on the dequantized weights**
- *Mechanism:* At the 2.5 mix, add Gaussian noise of known energy to the dequantized weights of one family × depth band (dense splice) at 3 levels. Fit α by least squares (Q-PACE), and use 4^−b scaling across K.
- *Source:* Q-PACE (2610.09183: forward-only α fit; analytic error curves no worse than measured; 80% of the FP4→FP8 gap vs SNIP's 54%).
- *Why:* It gives marginal prices at the operating point, not from bf16.
- *Fit:* Knapsack input.
- *Test:* 8 bands × 5 families × 3 levels = 120 short evals (about 4 card-h).
- *Gain/risk:* −0.2…−0.8%.
- *Conflicts:* #15 (noise injection rejected because it needed a bf16 forward on one card). New: noise goes on the quantized mix's dequantized weights through the splice harness, so no bf16 model is needed.

**E9 · Antithetic swaps around the operating point**
- *Mechanism:* Price each group by ½[f(+1 step) − f(−1 step)] instead of one-sided swaps, which cancels first-order curvature mismatch.
- *Source:* TWIG symmetric-difference costs (2610.31931); prior-round G4.
- *Why:* One-sided swaps confound slope and curvature at low K.
- *Fit:* Knapsack input (measurement design).
- *Test:* Applies to any new swap batch; same cost.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #14.

**E10 · GDN decay half-life as a kfit covariate**
- *Mechanism:* Add the per-layer mean head half-life log 2 / (−E log α) (from A_log / dt_bias) to the GDN-family log-sensitivity terms.
- *Source:* STEPQuant (2609.38169: half-life vs accumulated error ρ≈0.80 over 2,304 Qwen3.8 heads); prior-round F1.
- *Why:* Long-memory layers carry projection errors further.
- *Fit:* Knapsack input.
- *Test:* CPU refit on V2 data. Pass: CV log-RMSE improves ≥2%.
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* #19 (edge offsets null).

**E11 · Separate factors for sub-1 and i8 K1.125–1.875 levels**
- *Mechanism:* Run 6–8 targeted group swaps at 1.0–1.5 bpw to give each sub-2 level class its own factor.
- *Source:* #44 (K1 factor 1.35 vs 1.09; V2 under-predicted K1-heavy swaps by 20–36%); #90.
- *Why:* The sub-1 levels now ship at 1.0–1.75 bpw.
- *Fit:* Knapsack input.
- *Test:* 8 evals (about 1 card-h).
- *Gain/risk:* −0.5…−2% at 1.0–1.25 bpw; 0 at 2.5.
- *Conflicts:* #44, #90.

**E12 · Uncertainty penalty from #116's noise model**
- *Mechanism:* Add a penalty proportional to bootstrap disagreement on each option's price (BACKLOG B11), with the document-error term from #116.
- *Source:* #116; BACKLOG.
- *Why:* Stops extrapolated picks; #116 found document error as large as seed noise.
- *Fit:* Knapsack input.
- *Test:* CPU, V2 data.
- *Gain/risk:* Robustness.
- *Conflicts:* #116.

**E13 · Tempering exponent on error cost**
- *Mechanism:* Price each option as s·err^p with p swept {0.75, 1, 1.25}, judged on a search slice.
- *Source:* NatScale (2610.20353: removing the bit-decay factor raised KLD 0.194 → 0.292; τ choice changed rankings); prior-round G9.
- *Why:* A one-parameter fix for systematic curvature misfit.
- *Fit:* Knapsack input.
- *Test:* knap5 reruns plus 3 evals.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* —

**E14 · CUPED estimator for pricing evals**
- *Mechanism:* Regress each candidate's per-document KL on the incumbent's per-document KL, then compare the residuals.
- *Source:* Prior-round H7; standard A/B variance reduction.
- *Why:* Cuts the evals needed per price by about 30–60%, given high per-document correlation.
- *Fit:* Measurement design.
- *Test:* Re-analyse existing paired rows (CPU).
- *Gain/risk:* Cost saving.
- *Conflicts:* #116.

**E15 · Neyman-stratified sensitivity slices**
- *Mechanism:* Allocate documents per category in proportion to category weight × per-document KL standard deviation (from existing rows).
- *Source:* Brief's open candidate; survey sampling.
- *Why:* The same precision for fewer card-hours.
- *Fit:* Measurement design.
- *Test:* CPU, using existing per-document scores.
- *Gain/risk:* Cost saving.
- *Conflicts:* —

**E16 · Sensitivity at long positions**
- *Mechanism:* From existing long-trace chunk runs, compare per-family swap effects at positions >8k vs ≤2k.
- *Source:* #52 (ratios flat by position, but that was whole mixes).
- *Why:* Search slices cap documents at 2k; agentic traces are about 25% of scored positions.
- *Fit:* Knapsack input check.
- *Test:* CPU over stored rows. Act if a family differs ≥20%.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #52.

**E17 · Filter useless K0.125 options**
- *Mechanism:* Drop any sub-1 option whose post-refit proxy is ≥0.98, i.e. bytes that remove nothing.
- *Source:* #114 (err 1.014 at K0.125).
- *Why:* Keeps the knapsack from spending bytes on noise at 1.0 bpw.
- *Fit:* Knapsack input.
- *Test:* CPU.
- *Gain/risk:* 0…−0.2% at 1.0 bpw.
- *Conflicts:* #90.

**E18 · In-context head and embedding slopes at 2.5, read on the calib part**
- *Mechanism:* Score the main 2.5 mix with head ±½ bit and embedding t2 → t2.5 (4 evals). Read per part (E1).
- *Source:* Prior-round D6.
- *Why:* It directly prices the head's marginal bit where it matters.
- *Fit:* Knapsack input.
- *Test:* 0.5 card-h.
- *Gain/risk:* 0…−1%.
- *Conflicts:* #34.

**E19 · Partial pooling of family × depth factors (hierarchical kfit)**
- *Mechanism:* Shrink each factor toward its family mean unless the data support it (BACKLOG).
- *Source:* BACKLOG "partial pooling".
- *Why:* New V3 terms start from a prior rather than a free fit.
- *Fit:* Knapsack input.
- *Test:* CPU cross-validation on V2 data.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #20.

**E20 · Monotonicity audit of per-tensor level curves**
- *Mechanism:* For each tensor, flag any level whose two-sided error is not below that of the next-cheaper level, which would mean a bad job or a donor anomaly. Requantize or drop it.
- *Source:* New (needs C8).
- *Why:* One bad level in a 55-level library silently distorts the hull.
- *Fit:* Library hygiene.
- *Test:* CPU per finished level.
- *Gain/risk:* Protective.
- *Conflicts:* —

**E21 · Real-KL embedding swaps t0.5 / t1 / t1.25 for the low sizes**
- *Mechanism:* Price the embedding options at 1.0–1.5 bpw by measured swaps, read on the calib part.
- *Source:* #119's next step (3).
- *Why:* #119 concluded that pricing these bytes matters more than any representation change at ≤1.5 bpw.
- *Fit:* Knapsack input.
- *Test:* 3 evals plus the multilingual guard (0.5 card-h).
- *Gain/risk:* −0.3…−1% at 1.0–1.5 bpw.
- *Conflicts:* #119, #97.

**E22 · Low-budget slab heads (2.125–3.5) built two-sided**
- *Mechanism:* Offer low-budget slab heads in knap5 for the 1.0–2.5 sizes, using B1's two-sided interleaved donors.
- *Source:* #119 (2); w88 headgrid (−2.4…−3.8% predicted at 2.0–3.0 with the old donors).
- *Why:* Combines the headgrid gain with B1.
- *Fit:* Knapsack input plus level menu.
- *Test:* headsim at 4 budgets (about 1 card-h).
- *Gain/risk:* −1…−3% at 2.0 (overlaps B1 and B3).
- *Conflicts:* #34; the rare-token guard (headgrid guard 1.0 cut the gain to −1.6% at 2.0).

**E23 · Calibrate headsim's head-only KL against whole-model results**
- *Mechanism:* Fit the factor between headsim's predicted head-only KL change and the measured whole-model change from #34-02's pairs. s5-3 recovered about 65% of the u5 → bf16 gap while headsim predicted −86% of head KL.
- *Source:* #34-02.
- *Why:* knap5 prices head options in headsim units.
- *Fit:* Knapsack input.
- *Test:* CPU minutes.
- *Gain/risk:* 0…−0.5%.
- *Conflicts:* #34-02.

**E24 · Saturation check at 1.0–1.75 bpw**
- *Mechanism:* Refit kfit2 with a saturation term on the low-bpw mixes (r9 files on gpubox) plus the new sub-1 measurements.
- *Source:* BACKLOG; the ModelOpt saturation check (null at ≥1.6, with only 13 mixes).
- *Why:* kfit2 plus saturation fitted C≈1.8, which bites near the 1.0 sizes.
- *Fit:* Knapsack input.
- *Test:* CPU.
- *Gain/risk:* −0…−2% at 1.0–1.25.
- *Conflicts:* The 11-03 saturation null.

**E25 · Compounding term specific to layer type**
- *Mechanism:* Give GDN and full-attention layers separate κ knots in kfit2's directional compounding.
- *Source:* New; the hybrid architecture.
- *Why:* Error propagation through a recurrent state and through softmax attention plausibly differ.
- *Fit:* Knapsack input.
- *Test:* CPU cross-validation on V2's ~1,250 measurements. Pass: CV log-RMSE ≥2% better.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #11, #19.

**E26 · Weight the kfit fit toward the 2.0–3.0 operating region**
- *Mechanism:* Use weighted least squares, emphasising measurements whose mixes lie at 2.0–3.0 bpw.
- *Source:* New.
- *Why:* 2.5 is the key product; a global fit trades its accuracy against 6–9 bpw points.
- *Fit:* Knapsack input.
- *Test:* CPU, CV restricted to the 2.0–3.0 region.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* —

### F. Statistics and damping

**F1 · Partial activation denoising (ρ < λ)**
- *Mechanism:* Feed LDLQ the target WF and metric G from one noisy forward pass (2610.07522), with partial denoising ρ<λ.
- *Source:* 2610.07522: closes 18–31% of the parallel-to-sequential gap at W3, more at W2.
- *Why:* EXL3 quantizes from clean-input Hessians (parallel).
- *Fit:* Setting. Needs C and N matrices per linear (a noisy calibration pass of 1–2 card-h, TP2 or layer-streamed).
- *Test:* 0.8B on the 4070 first (3 seeds, K2/K3). Pass: ≤0.98.
- *Gain/risk:* 0…−0.5%; low prior.
- *Conflicts:* **#88** (full denoising target with level-matched noise: K3 1.031, K2 1.10). New here: partial ρ<λ, which was the paper's best setting, and propagated noise rather than level-matched noise.

**F2 · Gen-weighted H_in**
- *Mechanism:* Recollect H_in with loss-position weighting (gen-masked).
- *Source:* #86.
- *Why:* #86 gave K2 0.976 (every seed, z −4…−5) but missed its 0.97 bar.
- *Fit:* Setting. About 12 card-h to recollect.
- *Test:* 2 more K2 seeds on the 0.8B (4070, 1 h) before any 27B collection.
- *Gain/risk:* 0…−0.5%; low.
- *Conflicts:* #86, #91, #101.

**F3 · Shrink h at head level (James–Stein)**
- *Mechanism:* Bootstrap h over 128-sequence blocks, then shrink each head's mean h toward the tensor mean by its estimated noise.
- *Source:* New; complements A6/A7.
- *Why:* With interleave, noisy head means become overfit error allocation.
- *Fit:* Setting.
- *Test:* CPU, needs per-block sums (gpubox snapshots).
- *Gain/risk:* Protective; −0…−0.1%.
- *Conflicts:* #33-27 (bn smoothing between blocks, not between heads).

**F4 · Block-normalised Fisher for the head**
- *Mechanism:* Equalise each 128-document block's contribution to F_v before using it as B1's output factor, as #33-27 did for Σdy².
- *Source:* #33-27 (bn beat plain, 0.921 vs 0.936).
- *Why:* The head Fisher may be dominated by a few documents.
- *Fit:* Setting of B1.
- *Test:* headsim arm.
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* #33-27.

**F5 · Analytic SwiGLU output metric as a shrinkage target for gate/up h**
- *Mechanism:* Compute ‖W_d[:,r]‖²·E[(SiLU′(g_r)u_r)²] (gate) and ‖W_d[:,r]‖²·E[SiLU(g_r)²] (up) from one forward pass. Shrink the measured Σdy² toward them.
- *Source:* SACQ (2610.41408: with limited calibration, structure-aware 84.15 vs layer-local 74.55); prior-round T7.
- *Why:* gate/up hold 44% of H_out's gain on the 0.8B and are the largest tensors.
- *Fit:* Setting.
- *Test:* CPU on colo: cosine between analytic and measured, then w87's proxy. Pass: ≥1%.
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* #84 (an analytic SwiGLU diagonal failed its cosine gate, median 0.86, as a replacement); this uses it only as a shrinkage target.

### G. Format items that existing ops can absorb

**G1 · Per-tensor output permutation field plus a graph gather (enabler for A1–A3)**
- *Mechanism:* Store a u16 tile-id table (A4) in the exl3i encoding. The total is about 1.3 MB for 48 qkv + 48 z + 16 q tensors. The qwen35_exl3 graph applies the same gather op the slab head uses, after the GEMM (or folded into conv1d / q_norm / gated-norm reads later).
- *Source:* New; reuses #34-02's gather.
- *Why:* Needs no kernel change for V3, while a fused remap in the store epilogue is later work.
- *Fit:* Format change.
- *Test:* Selftest cases (census, round trip, decode M1/M16/M2048) plus a prefill timing of the gather on card 0. Budget: ≤3% of GDN layer time.
- *Gain/risk:* Enabler for −0.4…−1.2%. Risk: prefill time; the board scores gemm_us, and the gather is not a GEMM but is real time.
- *Conflicts:* 512-point Hadamard and per-head K were both closed without tests; G1 replaces both.

**G2 · GDN `in_proj_a` / `in_proj_b` to int8**
- *Mechanism:* The 48 layers × 2 × (48×5120) bf16 tensors (0.014 bpw) become per-row int8 (0.007 bpw), with the trellis impossible at N=48.
- *Source:* Prior-round B1 (re-estimated here).
- *Why:* Bytes that count in bpw and buy nothing. 2609.04098 showed a/b are the least sensitive GDN projections (a 2.1% / b 2.6% output error at NVFP4).
- *Fit:* Format.
- *Test:* Dense-splice int8-dequant arm (one 27B pair, 0.5 card-h). Pass: r ≤ 1.002.
- *Gain/risk:* Saves 0.007 bpw ≈ −0.1…−0.2% at 2.5.
- *Conflicts:* The design keeps a/b bf16 for speed (closed for D6 timing, not for bytes).

### H. Ladder and level menu

**H1 · Census of level usage; drop never-picked levels and reinvest the card time**
- *Mechanism:* Run knap5 on V3 prices at all release sizes (1.0–7.0, both variants). Drop levels used by <0.1% of parameters everywhere, and spend the saved time on D2, A9 or C5 at the core levels.
- *Source:* w58 repricing (the 17 i4 levels were never picked; mul1 K6/6.25 dropped).
- *Why:* At about 37 min per level, every dropped level funds roughly one extra setting at the 2.5 levels.
- *Fit:* Level menu.
- *Test:* CPU minutes.
- *Gain/risk:* Saves 3–8 card-h; 0 direct KL.
- *Conflicts:* —

**H2 · Build the planned head donors two-sided and interleaved**
- *Mechanism:* The 17-donor 1/16-budget head set (w88) should be built with B1's settings rather than one-sided.
- *Source:* w88; B1.
- *Why:* Avoids rebuilding the donors after the freeze.
- *Fit:* Level menu (head).
- *Test:* Gated by B1's headsim.
- *Gain/risk:* Carries B1's gain.
- *Conflicts:* #34.

### I. Fixed-code fits inside the ladder (part of the level pass)

**I1 · swj with h-weighted channels**
- *Mechanism:* Weight swj's joint gate/up refit per intermediate channel by the down-input gradient energy diag(W_dᵀ diag(h_down) W_d), approximated from down's bn h.
- *Source:* New; #85 / #104.
- *Why:* swj weights all channels equally today, while their downstream value differs.
- *Fit:* Per-level fit.
- *Test:* 0.8B fixed-code (4070, 1 h), then one 27B fixed-code pair (1 card-h). Pass: ≤0.995.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #104 (swj adopted), #85-02.

**I2 · Re-fit C9 α on the full-stack level**
- *Mechanism:* α was tuned on the 0.8B (plain and asym writers). Re-grid gdn_out {2, 3, 4} and attn_o {−1, 0} with fixed codes on the full-stack 27B K2.5 level (H_out + #117).
- *Source:* #112, #112-01.
- *Why:* H_out changes the coherent shrink C9 fixes.
- *Fit:* Per-level fit.
- *Test:* 4 evals (1 card-h).
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #112.

**I3 · C9 on readers (gdn_in, attn_in)**
- *Mechanism:* Per-family α on the input projections, the most H_out-sensitive families.
- *Source:* #85 (C9 over all families: K3 0.973, K2 0.911 on 0.8B); prior-round A2.
- *Why:* C9 at 27B was ported for writers only.
- *Fit:* Per-level fit.
- *Test:* 0.8B on the full stack (4070, 2 h). Pass: ≤0.99 at K2 and K3.
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* **#108** (joint GDN refits overlapped H_out/asym in the full stack: K3 0.999). C9 is per-row α, a different, simpler fit, but expect overlap.

**I4 · Gen-masked tokens for swj and C9 fits**
- *Mechanism:* Fit on scored (generated) positions only.
- *Source:* #83 / #83-02; prior-round A8.
- *Why:* Unscored prompt positions dilute the fit.
- *Fit:* Per-level fit.
- *Test:* 0.8B fixed-code (4070, 1 h).
- *Gain/risk:* 0…−0.2%.
- *Conflicts:* #83.

**I5 · Per-slab head α and temperature by headsim, frozen codes**
- *Mechanism:* After quantizing the head, fit one scale per slab (and a global logit temperature) to minimise exact head-only KL.
- *Source:* Prior-round A11.
- *Why:* A cheap coherent fix on the 5–9% head share.
- *Fit:* Per-level (head).
- *Test:* headsim (CPU or GPU, minutes).
- *Gain/risk:* 0…−0.4%.
- *Conflicts:* #34.

**I6 · Embedding row-energy restoration**
- *Mechanism:* Rescale each quantized embedding row to its original norm. The MSE-optimal scale shrinks rows, and the embedding feeds the residual stream directly as well as through norms.
- *Source:* Prior-round A13 / A4.
- *Why:* Coherent shrink is about 1.8% at K2.5 (prior A3); matters at low embedding K.
- *Fit:* Per-level (embedding).
- *Test:* One 27B pair at the t2 embedding, read on the calib part (0.5 card-h).
- *Gain/risk:* 0…−0.3%.
- *Conflicts:* #119 (the representation is at its floor; this is a scale fit, not a representation).

### M. Diagnostics and tools that unblock the above

**M1 · Dense-splice KL harness for output-side changes**
- *Mechanism:* Quantize with a permutation, dequantize, un-permute and store bf16 dense, then splice into the stored mix with `rad-convert --reuse --changed` as in #120-02.
- *Source:* #120-02.
- *Why:* Tests A1–A3 and G2 on 27B KL with no format or kernel work.
- *Fit:* Tool.
- *Test:* Reproduce one #120-02 control (r ≈ 1).
- *Gain/risk:* Enabler.
- *Conflicts:* —

**M2 · Realized f2 of interleave on the real quantizer**
- *Mechanism:* Run rxq on card 0 for L10 qkv/z and L11/L39 q_proj at K2.5 and K4, natural vs 16-row interleave (W rows and h permuted in the job). Report w81's f2calc ratio.
- *Source:* w81's harness.
- *Why:* Measures how much of the high-rate ×0.775 / ×0.857 survives.
- *Fit:* Diagnostic.
- *Test:* About 20 card-min. Pass: qkv f2 ≤0.90.
- *Gain/risk:* Decides A1–A3.
- *Conflicts:* —

**M3 · Head pivots in natural vocab order and with gen Fisher**
- *Mechanism:* Extend `head.py` with vocab-order and gen-Fisher arms, at all three slabs and for a uniform head.
- *Source:* Check 6.
- *Why:* Decides B2 vs B1 and sizes the head prize.
- *Fit:* Diagnostic.
- *Test:* colo, 5 min.
- *Gain/risk:* —
- *Conflicts:* —

**M4 · Three-arm headsim for B1**
- *Mechanism:* The one-sided / two-sided sorted / two-sided interleaved head quants at s5-3, scored by exact head-only KL and guard KL.
- *Source:* B1.
- *Why:* An exact head KL test before any whole-model eval.
- *Fit:* Diagnostic.
- *Test:* About 1.5 card-h.
- *Gain/risk:* —
- *Conflicts:* —

**M5 · KL elasticity per family from existing arms**
- *Mechanism:* Join existing 27B arms (base / bn / swj / C9 at K2.5 and K4) with their levels' f2 logs to get KL change per unit f2.
- *Source:* #33-28, #104, #112.
- *Why:* Converts A-theme f2 gains into KL predictions before spending card time.
- *Fit:* Diagnostic.
- *Test:* CPU, needs the levels' proxy logs.
- *Gain/risk:* —
- *Conflicts:* —

**M6 · Per-part re-read of existing pricing rows**
- *Mechanism:* Split stored rows of head/embedding swaps and recent pairs by part (xcorp / calib) and recompute the prices.
- *Source:* E1.
- *Why:* The cheapest way to see whether the knapsack is mispricing the head.
- *Fit:* Diagnostic.
- *Test:* CPU minutes.
- *Gain/risk:* —
- *Conflicts:* —

**M7 · 0.8B KL test of interleave**
- *Mechanism:* On the 0.8B with its edy H_out, quantize the GDN qkv/z with and without tile interleave at K2/K3, 3 seeds.
- *Source:* A1.
- *Why:* A fast KL sanity check of the mechanism before the 27B splice. 0.8B effects run about 3× those at 27B.
- *Fit:* Diagnostic.
- *Test:* 4070, about 1 h. Pass: ≤0.98 at K3.
- *Gain/risk:* —
- *Conflicts:* —

### X. Cross-area combinations

**X1 · Fused sibling interleave**
- *Mechanism:* Encode siblings that share an input (in_proj_qkv with in_proj_z; attention q with k and v) as one tensor with rows interleaved across them, giving the feedback a larger and more varied pool.
- *Source:* Interleaving across codewords (channel coding); A1.
- *Why:* z's and qkv's h differ (check 2), so more mixing gets closer to the GM bound.
- *Fit:* Format. Forces a K tie between siblings.
- *Test:* Extend `perm.py` to the concatenated qkv+z (colo, 10 min).
- *Gain/risk:* 0…−0.1% beyond A1, minus the cost of the K tie.
- *Conflicts:* The design note "fused GEMMs force K ties".

**X2 · Anti-pivoting rule for any stored row permutation**
- *Mechanism:* Wherever the format already allows a row permutation (head slabs; G1 tensors), store rows *anti-sorted* by output weight, never sorted.
- *Source:* Check 4/6; the opposite of pivoted Cholesky. Prior-round T9 rejected output pivoting as sorting.
- *Why:* Sorted order is the worst case for block-diagonal feedback; random or interleaved is near the bound.
- *Fit:* Design rule.
- *Test:* —
- *Gain/risk:* Carried by A1 and B1.
- *Conflicts:* T9 / #106 (pivoting by sort), the opposite direction.

---

## 2. Killed or demoted by this round's checks (not counted)

| Item | Reason |
|---|---|
| D-weighted Viterbi metric (CurveTQ 2610.09212) | After su·sign·RHT the diagonal spread within a tile is ≈1e-4 bit (check 1), so the weight is inert. |
| Residual-stream channel permutation (route A) | Σ pivots ×0.981–1.003 (check 5). LDLQ is already within 1–3% of GM on the input side. |
| Per-segment K (q\|k\|v, q\|gate) | 15% / 37% of the headroom between heads (check 3). |
| Interleave for MLP gate/up and writers | ×0.995; the spread between blocks is ≤0.008 bit (checks 2, 4). Writers' outputs are the residual stream anyway. |
| Per-head K (BACKLOG) and 512-point Hadamard | A1 at fixed K gets 0.24 of the 0.28-bit qkv ceiling (row level) and 93% of the full-dimension bound. |
| Best-of-N sign seeds for the spike readers | #41 (0.5% spread, not reproducible) and #120-02 (bf16-restore oracle R 1.0000). |

## 3. Ranked top 20 by expected gain per unit cost

| Rank | Idea | Expected gain (2.5 bpw) | Cost | Why this rank |
|---|---|---|---|---|
| 1 | **B1** two-sided head + in-slab interleave | −0.7…−2.5% (calib-like) | ~2 card-h; zero format | The largest predicted error cut (×0.40 on 98% of head weight); the permutation is free. |
| 2 | **B7** gen-position Fisher for the head | 0…−0.3% | +1 donor in B1 | Rides on B1. |
| 3 | **E1** private-mix weights in pricing | protects 0.5–2% | CPU minutes | The head is 15× more valuable on calib than xcorp. |
| 4 | **E3** #118 per-family offsets | −0.1…−0.3% | minutes + 0.5 card-h | Already measured, just apply. |
| 5 | **A1** tile interleave, GDN qkv | −0.3…−0.8% | G1 dev 1–2 days + ~2 card-h | New lever; ×0.775 objective at high rate. |
| 6 | **A2** interleave, attn_qg | −0.1…−0.3% | bundled with A1 | ×0.857. |
| 7 | **E23** headsim → whole-model factor | 0…−0.5% | CPU | Fixes head pricing units. |
| 8 | **A7** tempered h^p | 0…−0.4% | 3 CPU-h + 1.5 card-h | Cheap; also protects A1/B1. |
| 9 | **C4** σ_out sweep | 0…−0.3% | 3 CPU-h | Never swept. |
| 10 | **E2** two-sided level positions | −0.1…−0.5% | CPU (needs C8) | Needed to cash in A and B. |
| 11 | **C3** two-sided su refit | 0…−0.2% | 1 CPU-h | Free at inference. |
| 12 | **C1** two-sided drift table | 0…−0.3% | 1 CPU-h | Stale constant. |
| 13 | **B3** slab replan, two-sided | 0…−0.5% | CPU after B1 | Frees head bytes. |
| 14 | **A3** interleave z / v / k | −0.05…−0.2% | bundled | Smaller gain, same machinery. |
| 15 | **H1** level-usage pruning | saves 3–8 card-h | CPU | Funds D2 / A9. |
| 16 | **E17** K0.125 option filter | 0…−0.2% @1.0 bpw | CPU | Trivial. |
| 17 | **G2** GDN a/b int8 | −0.1…−0.2% | 0.5 card-h + format | Pure bytes. |
| 18 | **E7** pair interactions in the layer DP | 0…−0.5% | ~2 card-h | Exact in the current DP. |
| 19 | **E4** stack factor by K | 0…−0.3% | ~1 card-h | Relative pricing across K. |
| 20 | **I2** C9 α re-fit on the full stack | 0…−0.3% | 1 card-h | Cheap fixed-code check. |

## 4. The five cheapest diagnostics that would most change this ranking

1. **Head pivots in natural vocab order and with gen Fisher (M3).** colo, 5 min.
   - If vocab order already gives ≤0.5 of the one-sided value, B2 (no permutation) takes rank 1.
   - If the gen Fisher concentrates further, B1's prize grows.
2. **Out-of-sample stability of h between heads (A6).** CPU, 15 min, after copying two diagonal sets from gpubox.
   - If out-of-sample keeps <75% of the in-sample gain, A1–A3 drop below rank 15 and A7/F3 become prerequisites.
3. **Realized f2 of 16-row interleave on the real quantizer (M2).** card 0, about 20 min.
   - qkv f2 ≤0.85 moves A1/A2 into the top 3.
   - f2 ≥0.95 kills theme A.
4. **Per-part re-read of existing head and embedding pricing rows (M6).** CPU minutes.
   - Shows whether E1 is a 0% or a 1–2% item, and whether B1's whole-model test must be read on the calib part.
5. **Three-arm headsim (M4).** card 0, about 1.5 card-h.
   - The exact head-only KL settles B1, B3, B7 and H2 together, before any whole-model eval.
