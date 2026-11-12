The report is below.

# Codes lens, round 2: pre-freeze ideas for the weight format, per-level library and knapsack inputs

## Summary

- **Three new diagnostics ran on 27B tensors, about 10 CPU-minutes each on colo. Two of them close idea families that keep coming back:**
  - After the 128-point Hadamard, the LDL diagonal inside a 16-column tile is flat to ≤4·10⁻⁴ bit. So these have essentially nothing to gain:
    - D-weighted Viterbi (CurveTQ, XOR-Trellis);
    - fractional-K mask alignment;
    - H-weighted scale search.
  - Rate imbalance across input-side blocks is ≤0.09 bit, so column-split K is also closed.
- **The largest unexploited spread is across 128-row output groups:**
  - in_proj_qkv: about 0.5 bit, measured as an AM/GM ceiling with H_out × row energy, at layers 2 / 10 / 30 / 50 / 62.
  - q_proj (attn_qg): about 0.5 bit; at L3 the gate half holds 95–98% of the weighted energy.
  - Summed over the trunk, the ceiling is about 0.07 bpw.
  - The slab kernel being built for the head (#34-02) is the same machinery. **But the ladder can only be mixed slab-wise afterwards if `suh` is shared across levels.** Today `refit_scales` re-fits `suh` at every level. This is the most freeze-sensitive point in the report (idea 2.2).
- **About 0.026 bpw sits outside the trellis.** `linear_attn.in_proj_a` / `in_proj_b` are stored as bf16 (`v3/yaqa/collect.py:338` notes "not quantized"; about 43M parameters). The only paper on the same model (2609.04098) finds them the *least* sensitive GDN projections. This is the cheapest gain found in this round (idea 1.1, about −2…−3% KL at 2.5 bpw). It was listed as a TODO in the 11-02 ideas round (`measure.md` H1) and never run.
- **At sub-1 rates, a low-rank fp16 code beats the trellis at equal bits on attention and GDN input tensors:**
  - K0.125: −26% (in_proj_qkv), −58% (in_proj_z), ≥50× on L3 q_proj.
  - The attn_qg trellis at K0.125 is *worse than zeroing the tensor*.
  - This matters only for the ≤1.3 bpw research ladder.
- **Unit conversions used throughout:**
  - KL vs bpw at 2.0–2.5 on the board: ln(0.164/0.0666)/0.5 ≈ 1.8 per bpw, so **0.01 bpw ≈ 1.8% KL at 2.5** (≈1.6% at 3.0).
  - Proxy → KL: KL% ≈ proxy% × the affected tensors' KL share (V2 fit: KLD ∝ proxy^1.004).
  - 0.8B effects are divided by about 3 for the 27B, per the brief.

---

## 0. New measurements this round

Scripts and outputs are on colo in `/data/work/ideas-codes/r2/` (`gam.py` → `out.jsonl`, `lr.py` → `out2.jsonl`, `slab.py` → `out3.jsonl`). Inputs:
- the published calib H_in, with the damping v3.toml uses (0.005 for inputs, 0.025 otherwise);
- the block-normalized Σdy² H_out (`/data/work/hout27/hout-bn`);
- bf16 weights.

All values are in bits per weight. Each is ½·log₂(AM/GM): a high-rate ceiling, not a realized gain.

### D-1. Where the metric is anisotropic, before and after the 128-point randomized Hadamard

Column definitions:
- **g_diag** is the AM/GM of the conditional 16×16 block diagonal, i.e. the within-tile D diagonal.
- **g_eig** is the same over its eigenvalues. It is the ceiling for a metric-aware tile search, which is what tile CD attacks.
- **across-16** is the AM/GM of 16-column block traces. It is the ceiling for column-split K.
- **H_out** columns: the raw diagonal spread, and the spread across 128-row groups after the output Hadamard.

| tensor | g_diag raw → rot | g_eig rot | across-16 rot | within-128 rot | H_out diag raw | H_out across-128 | H_out g_eig |
|---|---|---|---|---|---|---|---|
| L3 q_proj | 0.964 → 1e-4 | **0.112** | 0.042 | 0.018 | 0.693 | **0.285** | 0.074 |
| L3 o_proj | 0.320 → 3e-4 | 0.047 | 0.072 | 0.025 | 0.024 | 0.001 | 0.002 |
| L10 in_proj_qkv | 0.205 → 1e-4 | 0.019 | 0.014 | 0.004 | 0.695 | **0.417** | 0.035 |
| L10 out_proj | 0.167 → 4e-4 | 0.023 | 0.036 | 0.008 | 0.030 | 0.002 | 0.003 |
| L10 gate (post-attn H) | 0.027 → 7e-5 | 0.003 | 0.008 | 2e-4 | 0.084 | 0.008 | 0.009 |
| L10 down | 0.051 → 1e-5 | 0.007 | 0.012 | 6e-4 | 0.032 | 0.002 | 0.003 |
| L39 q_proj | 0.196 → 7e-5 | 0.019 | 0.013 | 0.004 | 0.226 | 0.069 | 0.020 |
| L39 up | 0.026 → 7e-5 | 0.003 | 0.005 | 2e-4 | 0.025 | 8e-4 | 0.003 |
| L50 in_proj_z | 0.146 → 1e-4 | 0.014 | 0.012 | 0.003 | 0.366 | 0.098 | 0.034 |
| L63 down | 0.790 → 1.5e-4 | **0.128** | 0.089 | 0.055 | 0.013 | 0.003 | 0.001 |

The conditional blocks are computed in natural order. exllamav3 runs last-to-first, so read these numbers as indicative.

### D-2. Row-group (slab) ceilings

The group cost is c_g = Σ_{r∈g} h_r·‖w_r L‖². "2s" includes H_out; "1s" sets h = 1.

| tensor | 128-row groups 2s (1s) | sections 2s (1s) | notes |
|---|---|---|---|
| in_proj_qkv L2 / L10 / L30 / L50 / L62 | 0.54 / 0.47 / 0.33 / 0.24 / 0.90 (0.06–0.15) | q/k/v split: 0.05 / 0.16 / 0.04 / 0.001 / 0.14 | mean 0.50 per group, 0.08 per section |
| q_proj L3 / L19 / L39 / L63 | 1.06 / 0.20 / 0.23 / 0.43 | q vs gate: 0.59 / 0.05 / 0.03 / 0.22 (L3 1s: **0.86**) | share of the q half (2s): 0.05 / 0.32 / 0.64 / 0.16; per head (512 rows): 0.18 / 0.07 / 0.08 / 0.05 |
| in_proj_z L10 / L50 | 0.13 / 0.09 | — | group = head |
| k_proj + v_proj L3 | 0.62 | k vs v: 0.10 | |
| gate + up, L10 / L43 | 0.04 / 0.04 | gate vs up: 0.005 / 0.036 | gate holds 0.56 / 0.65–0.68 of the weighted energy |

Parameter-weighted ceiling:
- in_proj_qkv: 2.5B params × 0.50 bit;
- q_proj: 1.0B × 0.48;
- in_proj_z: 1.5B × 0.11;
- k_proj / v_proj: small.

Together that is ≈ **0.07 bpw**, of which the sections alone are ≈0.016 bpw. For comparison, #96 found ssm_qkv 0.33 and attn_qg 0.27, using H_out-only row importance.

### D-3. Sub-1 rates: low-rank vs trellis at equal bits

- Low-rank = best rank-r approximation in the metric ‖(W−Ŵ)L‖, with fp16 factors, r = K·k·n/(16(k+n)).
- Trellis rel = #90 family factor × the tensor's K1 rel. For L50 in_proj_z, L39 up and L3 q_proj the K1 rel is approximated.

| tensor | K0.125 LR / trellis | K0.25 | K0.5 |
|---|---|---|---|
| L10 in_proj_qkv | **0.37** / 0.50 | 0.31 / 0.32 | 0.26 / **0.17** |
| L50 in_proj_z | **0.14** / 0.33 | **0.12** / 0.19 | 0.10 / 0.09 |
| L3 q_proj | **0.020** / ≈1.7 (worse than zero) | **0.017** / ≈0.93 | **0.014** / ≈0.38 |
| L10 gate | **0.70** / 0.76 | 0.64 / **0.55** | 0.56 / **0.33** |
| L39 up | 0.88 / **0.57** | 0.83 / **0.40** | 0.77 / **0.24** |
| L10 down | 0.79 / **0.74** | 0.75 / **0.56** | 0.70 / **0.35** |

Keeping the top 1/8 of row groups at K1 and dropping the rest leaves 0.76–0.88 at K0.125: worse than low-rank everywhere.

---

## 1. Bytes outside the trellis

### 1.1 Quantize the GDN `in_proj_a` / `in_proj_b` projections
- **Mechanism:** These two small linears (176 rows × 5120 per GDN layer, about 43M parameters in total) are stored bf16. That is 0.0257 bpw of every mix spent at 16 bits. Store them as fp8-e4m3 with per-row scales (radiance's `qwen35_fp8` already has the op), int8-GPTQ, or EXL3 levels without an output rotation (48 rows is not a multiple of 128). Then let the knapsack price them like any other tensor.
- **Source:** "new" for us. 2609.04098 quantizes them to NVFP4 on this exact model and finds them the least sensitive GDN projections: GDN output error 2.1% / 2.6%, vs qkv 10.4%, z 9.9%, out 12.7%. Listed as TODO H1 in the 11-02 ideas round (`docs/ideas/2026-11-02/measure.md`) and never run.
- **Why:** Bytes freed: 8 bits → 0.0128 bpw; K6 → 0.016; K4 → 0.019. The gates are log-parameterized, so a GEMM error of 11% gave only 3.6% state error in that paper. Errors from different projections add in quadrature, so a/b at 4 bits add about 3% of a GDN layer's 4-bit error energy.
- **Fit:** Format change (small plugin op) plus per-level library (a/b levels) plus knapsack input.
- **Test:**
  1. 0.8B on the 4070: swap a/b to fp8 and to int8-GPTQ at fixed trunk codes, 3 seeds, KL vs base. Pass if r ≤ 1.003. About 0.5 h.
  2. 27B on card 0: re-solve knap5 with the freed bytes (CPU), then one pair at equal bytes against the K2.5 base. About 1.5 card-h.
  3. Census the container first (diagnostic §12.1).
- **Gain / risk:** −2…−3% KL at 2.5 (±1.5%). The risk is that their sensitivity per parameter is about 10× qkv's, so going below 6 bits may cost more than it frees. Let the knapsack decide.
- **Conflicts:** none in the INDEX. D6 (11-01) closed only the *compute* merge of a/b (speed), not their storage format.

### 1.2 Price GDN conv1d (and other 16-bit leftovers) as knapsack items
- **Mechanism:** The conv1d kernels (4 taps × 10240 channels × 48 layers, about 2M parameters) and the norm, A_log and dt_bias vectors are stored at 16 bits. Store conv1d as fp8 or int8 per channel.
- **Source:** new.
- **Why:** About 0.0012 bpw at bf16; half of it can be freed.
- **Fit:** Format change.
- **Test:** 0.8B swap, 15 min on the 4070; pass if r ≤ 1.002.
- **Gain / risk:** −0.1% KL. The risk is conv1d's sensitivity (it is in the recurrent input path).
- **Conflicts:** none.

---

## 2. Rate allocation inside a tensor

### 2.1 Trunk row-slab K on 128-row output groups (in_proj_qkv, q_proj, in_proj_z, k_proj / v_proj)
- **Mechanism:** Each 128-row output group of a tensor is an independent decode unit: the output Hadamard is block-diagonal by 128, and the two-sided H_out is block-diagonal by #115. So a per-group K can be assembled from the tensor's existing ladder levels. The knapsack then picks a slab plan per tensor from a small menu of plans per budget (as the head's headgrid does). The kernel is the slab-head kernel already planned as #34-02's follow-up.
- **Source:** extends #96 and BACKLOG "per-head K"; IQR-V4 / SoftWater-style class rates applied to the trunk. Different from #96 because:
  - (a) D-2 includes the W row-energy × H_out product;
  - (b) it targets only the three wide-spread families instead of a model-wide median;
  - (c) the kernel now exists for another reason.
- **Why:** The D-2 ceiling is ≈0.07 bpw overall, mostly from in_proj_qkv groups (0.24–0.90 bit) and q_proj (0.2–1.06 bit). The H_out LDLQ only exploits spread *within* a 128-row group, not *across* groups, so the two do not overlap.
- **Fit:** Format change (per-group K, shared with the head) plus knapsack input (slab plans). It needs 2.2 or 2.3.
- **Test:**
  1. Colo, 4 CPU-h: assemble groups from real K2 / K2.5 / K3 settle64 quants of 10 tensors, two-sided proxy at equal bytes vs the best per-tensor K. Pass if ≥0.10 bit-equivalent on the three families.
  2. Check split-half stability (§12.3).
  3. 27B, card 0, 2 card-h: a fixed-code KL pair with the slab tensors as bf16-dequantized rows (≈5 GB extra, fits beside a 2.6 bpw rest) vs per-tensor K at equal bytes, on golden-search27b, both parts < 1.
- **Gain / risk:** −1…−6% KL at 2.5; the realizable share of the ceiling is about 30–50%. Risks:
  - Σdy² noise inflates the spread;
  - KL does not follow Σdy² row weights;
  - the user closed per-head K on 11-08 without a test.
- **Conflicts:** #96 (negative by its model-wide rule); BACKLOG per-head K (closed without test). It might work this time: 2× the ceiling on the targeted families, and the kernel is coming anyway.

### 2.2 Make the ladder slab-mixable: share `suh` across all levels of a tensor
- **Mechanism:** `refit_scales` (`quantize.py:1730`) alternately re-fits the input-side `suh` and the output-side `svh` at every level. Rows taken from two levels therefore carry two different input scalings and cannot share one activation prep. Quantize every level of a tensor with one `suh`, for example the regularize() value or the refit from a reference level such as K3, and refit only `svh` per level. Also check that sign seeds and g_scale folding are level-independent (g_scale can fold into `svh` per row).
- **Source:** new; this is the constraint that makes 2.1 possible after the freeze.
- **Why:** Without it, any later trunk slab K, and any re-assembly of levels by row group, needs either a requant per mix or a per-slab-`suh` kernel (2.3). The decision cannot be changed after the ladder is quantized.
- **Fit:** Per-level quantizer setting (pre-freeze).
- **Test:** colo settle64, about 2 CPU-h: proxy cost of "svh-only refit with a fixed `suh`" vs full refit at K1.5 / 2.5 / 4 on 27 tensors (§12.2). Adopt if the cost is ≤0.3%.
- **Gain / risk:** Enabling only: 0% alone; it unlocks 2.1 and 2.3-free mixing. The risk is that the refit gain on `suh` is larger than expected (unmeasured).
- **Conflicts:** none.

### 2.3 Per-slab `suh` field in the format (alternative to 2.2)
- **Mechanism:** Allow each row slab its own `suh` plus input prep. This is trivial for the head (one decode row, M = 1). In trunk prefill it costs one extra k×M prep per slab (a few % of a GEMM for 2–3 slabs).
- **Source:** new.
- **Why:** It keeps the per-level `suh` refit if 2.2 shows the refit matters.
- **Fit:** Format change (exl3i field) plus kernel.
- **Test:** none until 2.2's diagnostic. If the fixed-`suh` cost is >0.5%, time a 2-slab prep on card 0.
- **Gain / risk:** Same as 2.1, minus prefill time.
- **Conflicts:** none.

### 2.4 Split `q_proj` into separate q and gate tensors
- **Mechanism:** Qwen3.8's q_proj produces [q(256) | gate(256)] per head. The weighted energy splits very unevenly and differently by layer: the q half holds 5% (L3), 32% (L19), 64% (L39), 16% (L63) of the two-sided energy. Store two EXL3 tensors with independent K and two launches (or one row-slab launch). No new kernel field is needed.
- **Source:** new.
- **Why:** The D-2 section ceiling is 0.59 / 0.05 / 0.03 / 0.22 bit (mean 0.22) on 3.9% of parameters ≈ 0.0086 bpw.
- **Fit:** Format (tensor split) plus per-level library (the levels split along with it) plus knapsack.
- **Test:** colo settle64 two-sided proxy at equal bytes, 1 CPU-h. Then a 27B fixed-code pair using rows of two existing levels (bf16 dequant), 1 card-h.
- **Gain / risk:** −0.3…−1%. The risk is the extra launch cost in prefill: 16 layers, small.
- **Conflicts:** #106 / #106-01 (head-permutation act order) dropped — a different mechanism (ordering, not rate).

### 2.5 Split `in_proj_qkv` into q / k / v tensors
- **Mechanism:** Same as 2.4 for the GDN q (2048), k (2048) and v (6144) sections.
- **Source:** new.
- **Why:** The section ceiling is 0.001–0.16 bit (mean 0.08) on 9.3% of parameters ≈ 0.0075 bpw. Most of qkv's spread is between heads, not sections, so 2.1 dominates.
- **Fit:** Format (split).
- **Test:** As 2.4.
- **Gain / risk:** −0.1…−0.6%. The risk is that the gain is layer-dependent (L50 ≈ 0).
- **Conflicts:** none.

### 2.6 Sensitivity-sorted output-row permutation for slab tensors
- **Mechanism:** Sort rows by c_r before forming 128-row groups, as the head is sorted by Fisher. Keep a row index map (gather in the epilogue), or restrict to permutations inside heads where downstream ops are invariant (rows within a GDN v-head and within the q/gate halves).
- **Source:** the head's `headslab.py perm`.
- **Why:** Sorting raises the across-group spread (the slab gain) and lowers the within-group spread (which H_out LDLQ exploits). The net effect needs measuring.
- **Fit:** Format (row map) plus per-level quantizer (the permutation precedes quantization), so it is pre-freeze.
- **Test:** colo, recompute D-2 with sorted rows (minutes), then a settle64 proxy.
- **Gain / risk:** Adds 0–40% to 2.1's ceiling. The risk is a scatter in the epilogue.
- **Conflicts:** #106 (head permutation for act order, dropped) — a different objective.

### 2.7 Per-head K on the inputs of o_proj / out_proj (column = head)
- **Mechanism:** Column-split K at head granularity on writer inputs.
- **Source:** extends #103 / #106 (same tensors, rate instead of order).
- **Why:** The D-1 across-16 spread is 0.036 (L10 out_proj) and 0.072 (L3 o_proj) bit, on about 7.6% of parameters.
- **Fit:** Format change (column split ⇒ split-K GEMM).
- **Test:** None recommended unless 2.1 passes.
- **Gain / risk:** ≤0.3%. A kernel split along K is costly.
- **Conflicts:** #103 and #106 (negative on ordering).

### 2.8 Column-split K on input-side 128-column blocks — closed by D-1
- **Mechanism:** A per-input-block K inside a tensor.
- **Source:** E3c, #96.
- **Why not:** After the Hadamard, the across-16-block spread is 0.005–0.09 bit and mostly far below 0.04.
- **Fit:** Format.
- **Test:** done (D-1).
- **Gain / risk:** <0.2%.
- **Conflicts:** E3c (#29-02): ceiling 0.05 bit, consistent.

### 2.9 Fractional-K mask alignment (put the extra bits on heavy columns or rows) — closed by D-1
- **Mechanism:** Replace the canonical even-spread masks (0x8888 / 0xAAAA / 0xEEEE) with masks aligned to tile columns or rows of larger D.
- **Source:** new.
- **Why not:** After the Hadamard, the D diagonal inside a 16-column tile is flat to ≤4·10⁻⁴ bit, and H_out's diagonal inside a 128-row group is flat by construction. There is nothing to align to.
- **Fit:** Format.
- **Test:** done.
- **Gain / risk:** 0.
- **Conflicts:** #36 (fractional K on the curve).

### 2.10 Two-K MLP halves (gate vs up, or Fisher-sorted halves of down) — closed by D-2
- **Mechanism:** Slab K for MLP tensors.
- **Source:** extends 2.1.
- **Why not:** 128-row ceilings of 0.04 bit; gate vs up 0.005–0.036 bit (but see 9.3: price gate and up separately).
- **Fit:** Format.
- **Test:** done.
- **Gain / risk:** <0.3%.
- **Conflicts:** #96.

### 2.11 Per-head K for k_proj / v_proj (4 KV heads × 256)
- **Mechanism:** Slabs of 128 rows on the small k and v projections of the 16 attention layers.
- **Source:** extends 2.1.
- **Why:** The L3 k+v two-sided group ceiling is 0.62 bit on about 0.6% of parameters ≈ 0.004 bpw. #118 also finds attn_k / attn_v priced 19–27% too high.
- **Fit:** Format (with 2.1).
- **Test:** In 2.1's CPU proxy.
- **Gain / risk:** −0.2…−0.7%. Mid-layer spread is unknown (only L3 measured).
- **Conflicts:** none.

### 2.12 Head-decay half-life as a prior for GDN per-head K
- **Mechanism:** Order or weight GDN heads by their decay half-life τ = log 2/(−E[log α]) from A_log and dt_bias. STEPQuant reports ρ≈0.80 between half-life and accumulated state error on Qwen3.8-27B. Use it as a second importance signal next to Σdy² for 2.1's slab plans.
- **Source:** 2609.38169, 2609.30950.
- **Why:** It needs no collection and has no estimator noise; a cross-check on the Σdy² spread.
- **Fit:** Knapsack input (slab plan prior).
- **Test:** Colo, minutes: rank correlation of τ with D-2's per-head c_g on in_proj_z and in_proj_qkv.
- **Gain / risk:** Improves the realizable share of 2.1 (maybe +10–20% of it). The risk is that it is a state-cache result, not weights.
- **Conflicts:** none.

---

## 3. Outliers: the massive channel seen by the readers

### 3.1 Reader-side bf16 restore oracle for channel 3994 (diagnostic)
- **Mechanism:** Copy #120-02, but for **input column 3994 of the residual readers** in L0–L11: q/k/v, in_proj_qkv / z / a / b, gate / up. Restore that column in the dense bf16 decode of the K2.5 codes and score the pair.
- **Source:** #120-02 covered writers only.
- **Why:** For readers the spike is on the input side, where the one-sided metric sees it directly.
  - D-1: the L3 q_proj g_eig of 0.112 bit comes almost entirely from the heavy 16-column block.
  - D-3: rank-282 (fp16) leaves 2.0% of L3 q_proj's weighted energy, while K2's whole error is 2.7%. So the K2 error on that tensor is concentrated in a few directions.
- **Fit:** Diagnostic for 3.2 / 3.3 / 3.5.
- **Test:** 27B card 0, about 30 min with the #120 tooling: fixed codes, two seeds; R ≤ 0.995 keeps the theme alive.
- **Gain / risk:** Ceiling unknown, −0…−2%. Caution: on the BOS token, q matters little (softmax over one key), while k / v / gate do matter.
- **Conflicts:** #120-02 (writers, DROP). The reader side was not tested.

### 3.2 Spike-column extraction: store reader column 3994 at fp16 plus a rank-1 epilogue
- **Mechanism:** Quantize W with column 3994 removed, with H_in conditioned on that column. Add x₃₉₉₄·w_col in the epilogue.
- **Source:** SpQR / OWQ outlier columns; #120 sparse-rows note.
- **Why:** The cost is 16 bits per output row = 0.003 bit per weight per tensor; across all readers in L1–L50, about 0.002 bpw. It is worth this only if 3.1 is positive.
- **Fit:** Format change plus a per-level quantizer setting.
- **Test:** Only after 3.1. Colo settle64 proxy, then a 27B pair.
- **Gain / risk:** 0…−1.5%.
- **Conflicts:** #120-02 / #120 note (no exl3i field reserved). A field would be needed, which is pre-freeze.

### 3.3 Rank-r side term plus trellis residual for H-spiked readers
- **Mechanism:** Ŵ = UVᵀ (rank r ≈ 16–64, int8 factors) from the top singular directions of W·L, plus the trellis on the residual at K − δ.
- **Source:** LQER / QERA / SpectrumQuant; D-3.
- **Why:** On L3 q_proj the weighted energy is low-rank (rank 282 holds 98%).
- **Fit:** Format change (a side GEMM, as the BACKLOG low-rank adapter note) plus per-level.
- **Test:** Colo, 2 CPU-h: settle64 proxy at equal bytes on L1–L11 readers.
- **Gain / risk:** −0…−1%. The risk is that the gain sits on sink tokens that matter little for KL.
- **Conflicts:** BACKLOG low-rank adapter (V4, per-mix). This variant is per-level and limited to early readers.

### 3.4 Sink-aware H_in for q_proj
- **Mechanism:** Re-collect q_proj's H_in on calib, down-weighting position-0 (sink) tokens, whose query does not affect output (single-key softmax). Use it for the query half only.
- **Source:** new; 2610.18139's Jacobian-weighted H_in in its narrowest form.
- **Why:** The massive-activation tokens dominate H_in's trace at L1–L11 (0.8–0.93 of trace, #120-01), but a large part of that contributes nothing to the loss for the query half.
- **Fit:** Per-level (statistics).
- **Test:** First, colo: the share of H_in trace from position 0 on 50 calib docs (needs one forward pass of the 0.8B or a 27B dump). Then a 0.8B KL arm.
- **Gain / risk:** −0…−0.5%. Risk of repeating #33-07 (gradient-weighted H_in, +31…+44%).
- **Conflicts:** #33-07 and #33-10 (full gradient-weighted H_in, worse). Here only one token class is down-weighted, on one half of one tensor.

### 3.5 Tile CD on the heavy 128-column block only, on every residual reader
- **Mechanism:** Run cd3 only on the tiles of the 128-column input block that holds channel 3994's spike after rotation (1/40 of the tiles), for all readers in L1–L50.
- **Source:** #77 / #117 / D-1.
- **Why:** g_eig concentrates there, and tile CD on attn_qg already gains 0.5–1.2%. The cost is ≈1/40 of full CD on those tensors.
- **Fit:** Per-level setting.
- **Test:** Colo settle64 (order-fixed) proxy on 9 readers, 1 CPU-h; adopt on a no-harm 27B pair.
- **Gain / risk:** −0.05…−0.3%.
- **Conflicts:** #77 (adopted on attn_qg); "tile CD on attn_k" closed without a test. This is a cheaper, targeted form.

### 3.6 Higher K for the heavy 16-column tile block only
- **Mechanism:** A single-block column split: the last-processed 16-column block of the heavy 128-block gets K+2.
- **Source:** new.
- **Why:** Bounded by the heavy block's share of g_eig. The cost is 2/320 bit per weight on that tensor.
- **Fit:** Format change.
- **Test:** After 3.1.
- **Gain / risk:** ≤0.3%. It needs a K field per column block.
- **Conflicts:** 2.8 (column split, closed in general). This is the one exception.

---

## 4. Trellis search and scales

### 4.1 Gate tile CD per tensor by g_eig
- **Mechanism:** Replace "cd3 on attn_qg only" with "cd3 wherever g_eig ≥ 0.03 bit". D-1 g_eig is cheap from H_in: 0.11 (L3 q), 0.13 (L63 down), 0.05 (L3 o_proj); 0.003–0.02 elsewhere.
- **Source:** #77; MPQ (2610.32127) block CD.
- **Why:** The #77 GPU family sweep averages away the layer outliers: "others 0.996–0.998" at K2.5.
- **Fit:** Per-level setting.
- **Test:** Colo: compute g_eig for all 400 tensors (about 1 CPU-h), then GPU cd3 on the top 5% by g_eig in a no-harm pair (card 0, 1 h).
- **Gain / risk:** −0.05…−0.3%; +quant time on large down_proj.
- **Conflicts:** #77 (adopted on attn_qg).

### 4.2 Second LDLQ pass after `refit_scales`
- **Mechanism:** After refit moves `suh` / `svh`, the codes were chosen for the old scales. Re-run LDLQ and Viterbi with the refit scales and keep the result if the proxy improves.
- **Source:** JARQ (2609.38599): +0–1% at GPTQ hosts at 3–4 bits. CoS-VQ: scale–code coupling.
- **Why:** Scale and codes are optimized one after the other only once.
- **Fit:** Per-level (×2 quant time).
- **Test:** Colo settle64 at K2 / 2.5 / 4, 27 tensors × 2 samples, 2 CPU-h. Rule: ≤0.99.
- **Gain / risk:** −0.2…−0.6%. The risk is ladder time ×2.
- **Conflicts:** none.

### 4.3 Choose g_scale by the actual two-sided proxy
- **Mechanism:** `g_scale_search_batch` picks the scale by tile MSE on a 1/3 subsample of tiles (wrapped diagonal plus extreme-RMS tiles). Add a 3-point scan (×0.95 / 1 / 1.05) scored by the full LDLQ + two-sided proxy.
- **Source:** ScaleSweep, H-Scale, Schur Replay (all show MSE-optimal ≠ output-optimal scales).
- **Why:** D-1 says H-weighting *within* tiles changes nothing, so any gain comes from LDLQ feedback and drift interaction, not the metric.
- **Fit:** Per-level (×3 on the trellis step).
- **Test:** Colo settle64, 2 CPU-h.
- **Gain / risk:** −0.1…−0.5%. Risk of overfitting to H (use held-out H_B, as w77).
- **Conflicts:** #39 (gain flat by MSE); merged B3 (KL recheck of the global gain). This one is per tensor and proxy-driven.

### 4.4 Measured per-tensor drift instead of the LDLQ_DRIFT table
- **Mechanism:** A first LDLQ pass measures the RMS of the feedback-adjusted targets per tensor, and the scale is then set from it. The current table is {1: 1.08, … 3: 1.004}, interpolated log-linearly, and 1.0 from K4 up.
- **Source:** new.
- **Why:** Drift depends on the H spectrum (spike readers vs MLP), not only on K.
- **Fit:** Per-level.
- **Test:** Colo, log measured drift vs table for 27 tensors (CPU minutes), then 4.3's proxy.
- **Gain / risk:** −0…−0.3%.
- **Conflicts:** #107 (interpolated vs exact-K table: null). This replaces the table with a measurement.

### 4.5 Null-space shrink before rotation
- **Mechanism:** Subtract from W its component along the near-null eigenvectors of H (bottom γ of the trace). The proxy loss is unchanged, ‖W‖ shrinks, and so does the trellis scale.
- **Source:** OSAQ (2605.04738): gains of 1–2% PPL on top of QuIP.
- **Why:** The trellis error scales with the RMS of the rotated weight. Energy in directions H does not see is wasted.
- **Fit:** Per-level.
- **Test:** Colo: the fraction of ‖W‖² in the bottom-5% eigenspace of the damped H, per tensor (minutes). Go on only if >3%.
- **Gain / risk:** −0…−0.5%. Damping limits the "null" space.
- **Conflicts:** none.

### 4.6 Two-path lookahead Viterbi
- **Mechanism:** Keep the best two trellis paths per tile and choose between them by the next block's LDLQ residual (one-block lookahead).
- **Source:** MPQ (soft commitment).
- **Why:** Euclidean ties are broken blind to feedback.
- **Fit:** Per-level (×2–3 quant time).
- **Test:** Colo, trellis2 on 16 tensors.
- **Gain / risk:** −0…−0.4%.
- **Conflicts:** #113 (beam search on wider windows: negative). This is a different axis.

### 4.7 Soft-decision LDLQ (BCJR-expected error feedback)
- **Mechanism:** Feed back the posterior-mean error from forward–backward over the trellis instead of the hard Viterbi error, then decide hard at the end.
- **Source:** BCJR-QAT (2605.10655), MPQ VAMP.
- **Why:** LDLQ commits greedily.
- **Fit:** Per-level (heavy).
- **Test:** Colo, small tensors only, 4 CPU-h.
- **Gain / risk:** Speculative, −0…−1%. Encode ×20.
- **Conflicts:** none.

### 4.8 Accept tile-CD moves on a held-out Hessian half
- **Mechanism:** Score tile CD's accept/reject on H_B (w77's split halves) instead of the H it optimizes.
- **Source:** DRQ (2610.11226): reconstruction-loss refinement overfits its activation sample.
- **Why:** It protects the adopted cd3.
- **Fit:** Per-level.
- **Test:** Colo w77 proxy, 1 CPU-h: held-out gain of cd3 with in-sample vs held-out acceptance.
- **Gain / risk:** ±0.1%; a safety check.
- **Conflicts:** #77, cdhout27.

### 4.9 Refit scales under the two-sided objective
- **Mechanism:** `refit_scales` solves ((QQᵀ)∘H)r = rowsum(Q∘(HW)) with H_in only. With H_out on, weight the rows by the block-diagonal H_out (row diagonal) when refitting `suh`.
- **Source:** new.
- **Why:** `suh` is shared by all rows, so its LS fit should favor high-H_out rows.
- **Fit:** Per-level.
- **Test:** Colo, w87 two-sided proxy, 1 CPU-h.
- **Gain / risk:** −0.1…−0.4%.
- **Conflicts:** none.

### 4.10 D-weighted Viterbi metric — closed by D-1
- **Mechanism:** Weight each step's error by D_ii.
- **Source:** CurveTQ (2610.09212), XOR-Trellis.
- **Why not:** Both papers find it inert under RHT. D-1: the within-tile D diagonal is flat to ≤4·10⁻⁴ bit.
- **Fit / test:** done.
- **Gain:** 0.
- **Conflicts:** —.

### 4.11 Stored start state per tile — closed by arithmetic
- **Mechanism:** Store (16 − K) start bits per 256-weight tile to remove the tail-biting constraint.
- **Source:** CurveTQ, #102.
- **Why not:** The free-vs-TB gap is 5.6% at K2 (#102), worth ≈0.039 bit. The bits cost 14/256 = 0.055 bpw.
- **Fit / test:** —.
- **Gain:** negative.
- **Conflicts:** #57, #102.

### 4.12 Wider window (L17 / L18) with the mul1 hash — closed
- **Why not:** V18 exact gives 0.994; beams lose (#113).
- **Conflicts:** #113.

### 4.13 Tile-CD sweep count per K
- **Mechanism:** cd1 at K ≥ 4 (to save ladder time); cd5 at K ≤ 2.
- **Source:** #77's per-K table (cd1 / cd2 / cd3: −0.5 / −0.7 / −0.8…−1.1%).
- **Fit:** Per-level.
- **Test:** In 4.1's run.
- **Gain / risk:** ±0.05%; it saves about 2 min per high-K level.
- **Conflicts:** #77.

---

## 5. Codebooks

### 5.1 Gaussian (cubic) warp only on the K ≥ 5 levels
- **Mechanism:** A codebook flag per level: mul1i8c at K5–7, mul1i8 below.
- **Source:** E3b's Gaussian int8 table: ×0.975 at K5 and ×0.969 at K6 vs the int8-rounded mul1; i8c (#78).
- **Why:** The bytesum code is platykurtic, which matters more at high K.
- **Fit:** Per-level codebook plus a decode flag (2 FMAs).
- **Test:** Colo settle64 at K5 / 5.5 / 6 with i8c vs i8, 2 CPU-h.
- **Gain / risk:** 0% at 2.5; about −1…−2% at 5–6 bpw.
- **Conflicts:** #78-04 (null on H_out at K2 / K3). K ≥ 5 untested on H_out.

### 5.2 Re-check E3b's K4 Gaussian-table anomaly (diagnostic)
- **Mechanism:** The Gaussian int8 table was ×1.008 at K2–3, **×1.108 at K4**, ×0.97 at K5–6. That is non-monotone, which looks like a table-scale or search bug.
- **Source:** E3b.
- **Why:** If K4 is really ≈0.98, the codebook-shape question reopens at the focus sizes.
- **Fit:** Diagnostic.
- **Test:** Colo, trellis.py rerun at K3.5–4.5 with a scale grid, 1 CPU-h.
- **Gain / risk:** It changes the ranking of 5.1 / 5.3.
- **Conflicts:** —.

### 5.3 Codebook-variant twins at equal K as knapsack options
- **Mechanism:** At K ≥ 5, offer i8 and i8c of the same tensor; the knapsack picks per tensor at equal bytes.
- **Source:** SAGE ABS (equal-payload code choice).
- **Why:** The gain is tensor-dependent (E3b range 0.99–1.01).
- **Fit:** Per-level library plus knapsack.
- **Test:** After 5.1.
- **Gain / risk:** Small; it adds ladder runs.
- **Conflicts:** #78.

### 5.4 Tuned 16-value i4 table for the sub-1 levels
- **Mechanism:** Replace the uniform int4 grid at K ≤ 0.875 with a Lloyd-fitted 16-value table (register-resident) for the low-rate trellis output distribution.
- **Source:** AAAC / Lloyd tables.
- **Why:** At sub-1 the effective symbol distribution is far from uniform.
- **Fit:** Per-level codebook (sub-1 only).
- **Test:** Colo, #90 rig, 2 CPU-h.
- **Gain / risk:** 0 at 2.5; −1…−3% proxy at K ≤ 0.5.
- **Conflicts:** #90-02 (i4 = i8 at K ≤ 1.5).

### 5.5 Trained 2^16 table — closed (user 11-07)
### 5.6 Hash (IQR) codebook — closed (#100)

---

## 6. Rotations

### 6.1 Learned 128×128 rotation for the highest-g_eig tensors only
- **Mechanism:** Replace the FWHT with a learned orthogonal 128-block (HARP-style, initialized at RHT) for about 20 tensors with g_eig > 0.05.
- **Source:** HARP (2605.29843): QTIP 3-bit +0.7% PPL.
- **Why:** g_eig is the anisotropy left after RHT.
- **Fit:** Format (dense 128×128 prep for those tensors).
- **Test:** Colo, proxy on L3 q_proj and L63 down, 2 CPU-h.
- **Gain / risk:** −0…−0.2%.
- **Conflicts:** "512-point Hadamard closed"; this is targeted.

### 6.2 Best-of-N sign seeds — closed (#41)
- Spread 0.5%, not reproducible.

### 6.3 Cross-block Kronecker mixing (128 ⊗ 40) — closed
- With fixed-rate codes and per-channel `suh`, block imbalance costs nothing; D-1 across-16 ≤0.09 bit.

### 6.4 KLT-then-Hadamard for the readers
- **Mechanism:** Rotate into the H eigenbasis, then apply the Hadamard.
- **Source:** SpectrumQuant, WUSH.
- **Why:** SpectrumQuant shows no gain at 70B, and 2609.37416 shows whitening hurts GPTQ-style feedback.
- **Fit:** Format.
- **Test:** None recommended.
- **Gain / risk:** ≈0.
- **Conflicts:** #47 (R1 negative).

### 6.5 64-point output Hadamard for slab tensors
- **Mechanism:** A finer slab unit (64 rows).
- **Source:** extends 2.1.
- **Why:** It captures more within-128 spread for the slab plans.
- **Fit:** Format (an output transform per tensor).
- **Test:** Recompute D-2 at 64-row groups (colo, minutes).
- **Gain / risk:** +0–20% of 2.1's ceiling; a kernel variant.
- **Conflicts:** —.

---

## 7. Hessian and metric inputs to the library

### 7.1 Temper H_out: use h^p with p ∈ {0.5, 0.75}
- **Mechanism:** Raise the block-normalized Σdy² diagonal to a power before the two-sided LDLQ.
- **Source:** ShamAN-Q (2609.38521): p = ½ beat p = 1 when error is large; KBBQ (λ < 1).
- **Why:** H_out's raw spread is 0.4–0.7 bit on q / qkv / z, and it is noisy. Tempering hedges noise in the estimate.
- **Fit:** Per-level setting.
- **Test:** Colo w87 held-out two-sided proxy, p × {K0.5, 2.5, 4}, 2 CPU-h. Then a 27B pair if ≥1%.
- **Gain / risk:** −0.2…−0.8%.
- **Conflicts:** #33-28 (H_out adopted), #114-03 (σ_in with H_out). p is untested.

### 7.2 Winsorize the top 1% of H_out rows
- **Mechanism:** Clip extreme rows of Σdy² before LDLQ.
- **Source:** #33-27 (heavy tail; block normalization helped).
- **Fit:** Per-level.
- **Test:** In 7.1's run.
- **Gain / risk:** ±0.2%.
- **Conflicts:** #33-27.

### 7.3 Sweep σ_out with the block-diagonal path
- **Mechanism:** σ_out ∈ {0.003, 0.01, 0.03, 0.1}. It is fixed at 0.01, a value taken from the paper.
- **Source:** Gate 2b: more output regularization helped full H_out.
- **Fit:** Per-level.
- **Test:** Colo w87 proxy, 1 CPU-h.
- **Gain / risk:** −0…−0.5%.
- **Conflicts:** #114-03 swept σ_in only.

### 7.4 Partial denoising pretransform (λ > ρ)
- **Mechanism:** Quantize WF in the metric G built from injected, propagated noise (2610.07522), with ρ < λ.
- **Source:** 2610.07522: closes 18–31% of the gap between parallel and sequential quantization.
- **Why:** It is mix-independent, so it avoids #111's failure mode.
- **Fit:** Per-level.
- **Test:** 0.8B KL arm, 1 h.
- **Gain / risk:** −0…−0.5% at 27B; the sequential gap itself vanished there (#105).
- **Conflicts:** #88 (level-matched denoising, worse). This one is partial, with a different noise model.

### 7.5 Small-α gradient-weighted H_in blend
- **Mechanism:** H_in + α·s·Σ‖J_t‖²x_txᵀ_t, with α ≤ 0.2, chosen per tensor on held-out data.
- **Source:** 2610.18139.
- **Fit:** Per-level.
- **Test:** 0.8B, 1 h.
- **Gain / risk:** ±0.5%. Repeats a failed direction.
- **Conflicts:** #33-07 and #33-10 (full replacement, +20…+44%).

---

## 8. Sub-1 and low-rank levels

### 8.1 Low-rank levels (LR-r) as sub-1 options for attention and GDN input tensors
- **Mechanism:** A level type "rank r, int8 or fp16 factors" at the sub-1 byte points, used where it beats the trellis (D-3).
- **Source:** D-3; SVD-LLM family.
- **Why:** At K0.125: −26% (qkv), −58% (z), ≫ on q_proj. At K0.25: −38% on z.
- **Fit:** Format (side GEMM) plus per-level library plus knapsack.
- **Test:** First a colo proxy over 49 tensors (about 2 CPU-h), then knap4 at 1.0 / 1.25 / 1.5 bpw.
- **Gain / risk:** 0% at 2.5; −2…−6% at 1.25 bpw (estimate).
- **Conflicts:** #90 (sub-1 pays only < 1.3 bpw).

### 8.2 Re-admit a K0 (zero) level
- **Mechanism:** Zero bytes, the tensor dropped (a bias-free output).
- **Source:** #90 / #114.
- **Why:** The GPU search at K0.125 is worse than all-zero (err 1.014). attn_qg's K0.125 rel is 13.6× its K1.
- **Fit:** Knapsack menu.
- **Test:** CPU knapsack re-solve at 1.0–1.5.
- **Gain / risk:** 0 at 2.5; small at ≤1.3.
- **Conflicts:** K0 dropped 11-01; its premise ("K0.125 removes 16%") is falsified by #114.

### 8.3 Low-rank plus K0.125 hybrid
- **Mechanism:** Rank-r plus a trellis residual at sub-1.
- **Source:** 3.3, D-3.
- **Fit:** Format.
- **Test:** Colo proxy.
- **Gain / risk:** As 8.1.
- **Conflicts:** —.

### 8.4 Repair the sub-1 search: 2-tile chains or a free start only at K ≤ 0.375
- **Mechanism:** At K0.125 a tile carries 32 bits against a 16-bit state window; tail-biting dominates.
- **Source:** #57, #102.
- **Fit:** Format (sub-1 decode only).
- **Test:** Colo: free/TB at K0.125–0.5 (§12.5).
- **Gain / risk:** 0 at 2.5.
- **Conflicts:** #102.

### 8.5 Row-drop levels (top groups at K1, rest zero) — negative by D-3
- **Why not:** 0.76–0.88 rel at K0.125, worse than low-rank.

---

## 9. Knapsack inputs and level menu

### 9.1 Per-family price anchors from the V3 levels
- **Mechanism:** knap5 overprices attn_k (0.73), attn_v (0.81) and attn_qg (0.92) by 8–28% (#118). Use today's ladder proxies as anchors, or per-family factors.
- **Source:** #118 2(b).
- **Fit:** Knapsack input.
- **Test:** CPU re-solve at 2.0–3.0 (minutes); one 27B pair of the changed mixes.
- **Gain / risk:** −0.3…−1%.
- **Conflicts:** none (flagged to the lead, not done).

### 9.2 Log the two-sided proxy per tensor at every ladder level
- **Mechanism:** rxq writes proxy_err_2s (#115 added the field). Price H_out levels in kfit2 from it instead of from one-sided proxies, which sit ×1.076 high.
- **Source:** #118 2(c) (not done, too expensive on V2). On V3 levels it costs nothing.
- **Fit:** Knapsack input.
- **Test:** kfit2 CV once 5+ levels exist.
- **Gain / risk:** −0.2…−0.5%.
- **Conflicts:** #118.

### 9.3 Give gate and up separate sensitivities
- **Mechanism:** knap5 inherits V2's fused gate_up entry. D-2: gate holds 0.56–0.68 of the pair's weighted energy (1.3–2.1× up).
- **Source:** D-2, #118.
- **Fit:** Knapsack input (and the §1d K-tie decision).
- **Test:** CPU; one 27B pair.
- **Gain / risk:** −0.2…−0.8% (MLP is 42% of parameters).
- **Conflicts:** none.

### 9.4 Check knapsack bucket resolution at 2.5
- **Mechanism:** 1 MiB buckets cost about 2% at 1.5 bpw (#90; 256 KiB fixes it). Check 2.0–3.0.
- **Source:** #90.
- **Fit:** Knapsack.
- **Test:** CPU, minutes.
- **Gain / risk:** −0…−0.5%.
- **Conflicts:** —.

### 9.5 Slab plans as knapsack items
- **Mechanism:** For each 2.1 tensor, a hull of slab plans per byte budget, as `KN_HEADGRID`.
- **Source:** w88 headgrid.
- **Fit:** Knapsack.
- **Test:** With 2.1.
- **Gain / risk:** 2.1's.
- **Conflicts:** —.

### 9.6 Pairwise interaction terms for coupled tensors
- **Mechanism:** Measure joint-minus-sum KL for about 10 adjacent pairs (gate/up→down, qkv→out) at 27B. Add a penalty where |interaction| > noise.
- **Source:** TWLA ILA-AMP, CASA, LoopQuant.
- **Fit:** Knapsack input.
- **Test:** 27B, 3 card-h.
- **Gain / risk:** −0…−1%.
- **Conflicts:** #14 (single-tensor swaps are noise-limited); 2609.01587 (additivity 0.91–1.10).

### 9.7 Price the sub-1 levels from GPU-measured curves
- **Mechanism:** The CPU proxy is about 13% optimistic at K0.125.
- **Source:** #114.
- **Fit:** Knapsack.
- **Test:** From ladder logs.
- **Gain / risk:** Low-bpw sizes only.
- **Conflicts:** —.

### 9.8 Anisotropy covariates in kfit2
- **Mechanism:** Add g_eig and the slab ceiling per tensor as features predicting deviation from the family factor.
- **Fit:** Knapsack.
- **Test:** CV on V3 data.
- **Gain / risk:** ±0.3%.
- **Conflicts:** #19 / #20 (re-ranker null).

### 9.9 Build pass 1 first and freeze the menu by its usage
- **Mechanism:** Card-time saving.
- **Source:** w58 (pass 1 within 0.2–0.8% of all).
- **Fit:** Menu.
- **Gain / risk:** Cost only.
- **Conflicts:** —.

### 9.10 Decide the shared-input K ties with data
- **Mechanism:** D-2 shows that tying gate and up costs ≤0.005–0.036 bit, and k vs v 0.10 bit (L3). Price untied launches (§1d open decision).
- **Fit:** Knapsack and format.
- **Test:** CPU.
- **Gain / risk:** −0…−0.5%.
- **Conflicts:** §1d.

---

## 10. Embedding trellis

### 10.1 Embedding row-slab K by measured per-row sensitivity, with floors
- **Mechanism:** The embedding gather decodes one row at a time, so a K per 128-row block is only an offset table: no GEMM issue. Sort rows by the measured gradient energy at h0 (from #119's planned Fisher collection), and floor rows that never appear in calib at t2.
- **Source:** #119 IDEAS #2; the IQR-V4 floored head rule.
- **Fit:** Format (gather offsets) plus a per-level embedding library (t-levels already exist).
- **Test:** 27B GPU 1–2 card-h for the Fisher; CPU plan; KL swap with the multilingual guard.
- **Gain / risk:** −0.3…−1% on calib-like text. #32's frequency trap is guarded by the floors.
- **Conflicts:** #32 (vocab tiering dropped); #119.

### 10.2 Embedding LDLQ in the true Fisher metric
- **Source / Fit / Test / Conflicts:** #119's next step (5120² Fisher at h0, then `embldlq.py`).
- **Gain:** −0.3…−4% at 1–2 bpw; ≤0.5% at 2.5.

### 10.3 Embedding slabs ordered by the head's Fisher
- **Mechanism:** Use headsens' per-vocab Fisher as a proxy for embedding row importance, with no new collection.
- **Fit:** Format.
- **Test:** CPU rank correlation with gradient energy once 10.1's data exists.
- **Gain / risk:** As 10.1 if correlated. The risk is that head and embedding sensitivities differ (#119: the head predicts the embedding poorly).
- **Conflicts:** #119.

---

## 11. LM head

### 11.1 Two-sided head LDLQ with the Fisher diagonal as H_out
- **Mechanism:** Quantize the head donors with `--hout` set to the existing Σ_t p(1−p) Fisher diagonal (headsens, qad pool, in permuted order), σ_out 0.01, block-diagonal path.
- **Source:** IQR-V4's vocabulary-weighted head; SoftWater (2608.12026: −6.5…−8× head KL at 2 bits).
- **Why:** The Fisher is far more concentrated than the trunk's Σdy²: about 400 rows hold 50%, and the top block holds 39%. Slabs handle the across-group spread; two-sided LDLQ handles the within-group spread.
- **Fit:** Per-level (head donors, which are being remade anyway).
- **Test:** One K4 donor with and without H_out (card 0, about 1 h each, ~22 GB host RAM). headsim exact head-only KL on 32,768 held-out inputs, block bootstrap. Pass if ≤0.95.
- **Gain / risk:** Head KL −10…−30% → −0.3…−1.5% on calib-like text (head share ~5%). The risk is that Fisher top-512 truncation biases the tail rows.
- **Conflicts:** Design: "vocabulary-weighted head vs row-slab head: open". This combines the two.

### 11.2 Sweep the softmax shift t
- **Mechanism:** Replace mean-row correction (t = 1) by t ∈ {0.5, 1, 1.5, 2, 4}.
- **Source:** 2609.31291 (KL-optimal t ≠ 1).
- **Fit:** Per-level (head).
- **Test:** headsim, about 1 card-h per t at K4.
- **Gain / risk:** Head KL −0…−10%, about ≤0.3% total.
- **Conflicts:** #34 (mean-row adopted).

### 11.3 Fisher-weighted shift vector
- **Mechanism:** Subtract a = Σ_v F_v w_v / Σ F_v (Fisher-weighted mean row) instead of the plain mean.
- **Source:** generalizes 11.2.
- **Fit / Test:** As 11.2.
- **Gain / risk:** As 11.2.
- **Conflicts:** —.

### 11.4 Low-rank head residual in the whitened metric
- **Mechanism:** Rank-6 int8 correction of (W − Ŵ)·C^{0.75}.
- **Source:** ARCHead (2608.02703).
- **Fit:** Format (extra GEMV at decode).
- **Test:** headsim.
- **Gain / risk:** Head KL −5…−15% at ≤4 bits; less with slabs.
- **Conflicts:** —.

---

## 12. Diagnostics that feed the ranking

### 12.1 Container census of non-trellis bytes
- **Mechanism:** Sum the payload by dtype per tensor class in a V2 3.00 container (a/b, conv1d, norms).
- **Test:** CPU, minutes.
- **Decides:** 1.1 / 1.2.

### 12.2 `refit_scales` gain split by `suh` and `svh`, per K
- **Mechanism:** settle64 with (i) no refit, (ii) `svh` only, (iii) both.
- **Test:** Colo, 2 CPU-h.
- **Decides:** 2.2 vs 2.3.

### 12.3 Split-half stability of the slab ceilings
- **Mechanism:** Recompute D-2 from the two halves of the 1,920-sequence Σdy² snapshot; correlate per-group costs.
- **Test:** Colo, minutes, if halves exist on disk.
- **Decides:** how much of 2.1's 0.07 bpw ceiling is noise.

### 12.4 Heavy-block share of a reader's trellis error
- **Mechanism:** For 9 readers at K2.5, the share of the H-weighted error in the spike's 16-column block.
- **Test:** Colo settle64, 1 CPU-h.
- **Decides:** 3.2 / 3.5 / 3.6 / 4.1.

### 12.5 Free-start vs tail-biting at sub-1 (and recalibrating settle64)
- **Test:** Colo, 1 CPU-h.
- **Decides:** 8.4 and the sub-1 pricing (9.7).

### 12.6 gate-vs-up KL ratio from existing swap data
- **Mechanism:** Fit gate and up separately on the V2 data where they were swapped separately, if any.
- **Test:** CPU.
- **Decides:** 9.3.

---

## Ranked top 20 by expected gain per unit cost

Gains are KL changes at 2.5 bpw.

| # | idea | gain | cost | why it ranks here |
|---|---|---|---|---|
| 1 | 1.1 Quantize GDN in_proj_a / b | −2…−3% | 0.5 h 4070 + 1.5 card-h + small op | 0.026 bpw of bf16; least-sensitive tensors on this model |
| 2 | 9.1 Per-family price anchors | −0.3…−1% | CPU minutes + 1 pair | measured mispricing of 8–28% |
| 3 | 9.3 Separate gate / up sensitivity | −0.2…−0.8% | CPU + 1 pair | 1.3–2.1× energy ratio on 42% of parameters |
| 4 | 11.1 Two-sided head LDLQ (Fisher H_out) | −0.3…−1.5% (calib) | ~2 card-h, headsim | statistic exists; donors are being remade |
| 5 | 2.2 Shared `suh` across levels | enables 2.1 | 2 CPU-h | irreversible after the ladder |
| 6 | 9.4 Bucket resolution | −0…−0.5% | CPU minutes | known ~2% loss at 1.5 |
| 7 | 2.4 Split q_proj into q and gate | −0.3…−1% | CPU + 1 card-h + launch | 0.22-bit section ceiling |
| 8 | 2.1 Trunk row-slab K | −1…−6% | CPU proxy 4 h + 2 card-h + kernel (shared with head) | largest ceiling (0.07 bpw) |
| 9 | 7.1 H_out tempering | −0.2…−0.8% | 2 CPU-h + 1 pair | raw spread 0.4–0.7 bit is noisy |
| 10 | 3.1 Reader column-3994 oracle | 0…−2% (diagnostic) | 0.5 card-h | decides a whole family |
| 11 | 9.2 Two-sided proxy logged at every level | −0.2…−0.5% | ~0 | free with rxq |
| 12 | 7.3 σ_out sweep | −0…−0.5% | 1 CPU-h | never swept at 27B |
| 13 | 4.2 Re-LDLQ after refit | −0.2…−0.6% | 2 CPU-h; ×2 ladder time | |
| 14 | 9.10 Data-driven K ties | −0…−0.5% | CPU | §1d decision |
| 15 | 2.5 Split in_proj_qkv into q / k / v | −0.1…−0.6% | CPU + pair | |
| 16 | 4.3 g_scale by the two-sided proxy | −0.1…−0.5% | 2 CPU-h | |
| 17 | 4.9 Two-sided `suh` refit | −0.1…−0.4% | 1 CPU-h | |
| 18 | 10.1 Embedding row-slab K | −0.3…−1% (calib) | 2 card-h + gather offsets | |
| 19 | 3.5 Heavy-block-only tile CD | −0.05…−0.3% | 1 CPU-h + no-harm pair | |
| 20 | 4.1 Tile CD gated by g_eig | −0.05…−0.3% | 1 CPU-h + pair | |

The low-bpw ladder (1.0–1.5) has its own leaders, which are worth ≈0 at 2.5: 8.1 (low-rank levels), 8.2 (K0) and 5.4 (tuned i4 table).

## The 5 cheapest diagnostics that would most change the ranking

1. **Container census plus a 0.8B a/b fp8 swap** (§12.1, 1.1). About 15 min CPU + 30 min 4070. It decides whether #1 is real. If a/b turn out already 8-bit or sensitive, the top drops to the pricing fixes.
2. **refit gain split by `suh` / `svh`** (§12.2). 2 CPU-h on colo. It decides whether the ladder can be made slab-mixable for free (2.2), and so whether 2.1 can move from #8 toward #3.
3. **Split-half slab ceilings** (§12.3). Minutes on colo. If most of the 0.5-bit in_proj_qkv spread is Σdy² noise, 2.1 / 2.5 / 2.6 / 2.11 / 9.5 all fall out of the ranking.
4. **Reader column-3994 restore oracle** (3.1). About 30 min of card 0. It decides 3.2 / 3.3 / 3.5 / 3.6 (and partly 4.1) in one read.
5. **Head two-sided headsim at K4** (11.1). About 2 card-h. headsim is exact for the head, so it gives the head-side gain without a full pair. If head KL drops by ≥20%, the item moves to #2.
