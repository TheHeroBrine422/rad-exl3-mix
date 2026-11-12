# Codes lens — 2026-11-07 ideas pass

Weight-format ideas: trellis, codebook, rotation, outliers, K granularity, embedding and head codes.

**How to read this report.**
- The 11-02 codes lens (`docs/ideas/2026-11-02/codes.md`) already listed about 60 format ideas. This pass re-prices the open ones against what changed since:
  - H_out is adopted only at K ≥ 4.
  - The settle proxy's LDLQ order bug is fixed.
  - #109–#113 are closed.
  - swj and tile CD (cd3 on attn_qg) are adopted.
- It also adds new ideas, three of them grounded in CPU diagnostics run during this session.
- Labels: **(new)** = not in the 11-02 lists; **(re-priced X)** = 11-02 idea X with new evidence.
- Exchange rate (faraway): at 2.5 bpw, d ln KL / d bpw ≈ −1.7, so **0.01 bpw ≈ 1.7% KL**.
- 1% less weight error ≈ 0.007 bit. A uniform x% error cut on a family that carries a KL share f is worth about −x·f% KL.
- Parameter shares: gate+up 42%, down 21%, GDN qkv 9.4%, GDN z 5.6%, GDN out 5.6%, attention q 3.7%, o 1.9%, embedding 4.7%, head 4.7%.

---

## 0. Measured in this session

Scripts: `/data/work/ideas-codes/{blkdiag,aspen}.py` on colo (outputs `j*.json`, `as-*.json`) and `/tmp/ideas-codes/menu.py` on the VM.
- Inputs: the calib H_in and the bf16 weights.
- Setup: exllamav3's scaled Hessian (su = column RMS of the row-normalised W), adopted damping (inputs 0.005, rest 0.025), a sign-randomised 128-point Hadamard, last-to-first LDLQ order. The error weight of rotated coordinate j is var(x_j | x_<j) = diag(chol(H_r))².
- These are iid-error (Euclidean) proxies. Treat every number as a hypothesis until the real trellis in exllamav3 order (settle64) confirms it: the 11-02 lens saw an iid ordering proxy reverse sign on the real quantizer.

**M1. One channel dominates the inputs of every attention/GDN reader, at every depth.** Top-1 diagonal share of tr(H):

| tensor | L3 q | L11 q | L2 GDN qkv | L10 GDN qkv | L30 GDN qkv | L43 MLP in | L30 down | L30 ssm_out | L43 o_proj |
|---|---|---|---|---|---|---|---|---|---|
| share | 0.93 | 0.88 | 0.80 | 0.67 | 0.63 | 0.29 | 0.0075 | 0.0044 | 0.0047 |

On the scaled diagonal, the top channel is 2,000–4,700× the mean on attention/GDN inputs and 1,373× on the MLP input. Writers are flat.

**M2. Column-split K is dead, even with channel sorting.** Water-filling gain of a per-128-block K:

| tensor | natural order | sorted by scaled diag, desc. | ΣD ratio (sorted / natural) | net of sorting |
|---|---|---|---|---|
| L3 q | 0.009 bit | 0.359 | 1.634 | −0.004 |
| L11 q | 0.012 | 0.528 | 2.052 | −0.002 |
| L10 GDN qkv | 0.010 | 0.217 | 1.331 | +0.001 |
| L43 gate | 0.006 | 0.067 | 1.090 | −0.001 |
| L30 down | 0.004 | 0.045 | 1.059 | 0.000 |
| L30 ssm_out | **0.044** | 0.177 | 1.195 | +0.005 |
| L43 o_proj | 0.025 | 0.176 | 1.235 | −0.001 |

Sorting moves energy into a few blocks, so allocation gains, but the 128-Hadamard can no longer average and ΣD rises by the same amount. Per-coordinate water-filling is invariant, as theory says. This retires 11-02 F1/D7 for readers and leaves ≤0.044 bit on writers.

**M3. The massive-channel cost sits inside 16-column LDLQ blocks, not in the pivots.**
- Under scalar pivots, extracting the top-1/2/4 channels gives ΣD ×0.994–0.998: nothing.
- The block-LDL penalty A/S = Σ‖L_bb‖²_F / Σ L_jj² (in-block coupling that Euclidean Viterbi cannot see) tells a different story:

| tensor | A/S r128 | A share in the worst 1% of blocks | extract top-1 (A ratio) | extract top-2 | 256-RHT | 512-RHT |
|---|---|---|---|---|---|---|
| L3 q_proj | **1.118** | **10.7%** | 0.932 | **0.910** | 0.953 | 0.940 |
| L11 q_proj | **1.142** | **11.4%** | 0.946 | **0.900** | 0.938 | 0.937 |
| L43 q_proj | 1.030 | 2.3% | 0.997 | 0.990 | 0.984 | 0.978 |
| L3 k_proj | 1.044 | 3.7% | 0.995 | 0.978 | 0.978 | 0.979 |
| L2 GDN qkv | 1.029 | 2.0% | 0.996 | 0.996 | 0.986 | 0.982 |
| L10 GDN qkv / z | 1.024 / 1.033 | 2–3% | 0.995 / 0.990 | 0.992 / 0.983 | 0.988 / 0.984 | 0.984 / 0.980 |
| L30 / L50 GDN qkv | 1.020 / 1.017 | ~2% | 0.997 / 0.998 | 0.991 / 0.996 | 0.990 / 0.991 | 0.985 / 0.988 |
| L43 gate | 1.007 | 1.4% | 0.998 | 0.996 | 0.997 | 0.995 |

- In early attention, ~11% of the block-LDLQ error sits in 3 of 320 sixteen-column blocks. Extracting two channels removes most of it.
- This is the first mechanism that explains part of faraway M5's 1.74× (L11 attn_qg) and 1.35× (L10 ssm_qkv) gaps.

**M4. mul1's successor menus are already better than random but far from stratified.**

| K | strata covered by all 2^K successors: mul1 / iid table | one-step greedy D: mul1 / iid table |
|---|---|---|
| 2 | 0.17 / 0.10 | 0.306 / 0.390 |
| 3 | 0.022 / 0.002 | 0.122 / 0.162 |
| 4 | 0 / 0 | 0.053 / 0.069 |

A perfectly stratified K2 menu has coverage 1.0. Together with the 11-02 result (mul1 is 4% / 6.3% above D(R) at K2 / K3, Gaussian tiles, free start), a stratified code has at most a few percent headroom.

---

## A. Codebook values (computed code; decode cost bounded)

**A1 · Revive i8c for the levels that run without H_out (K ≤ 3.5)** (re-priced D3 / #78)
- **Mechanism:** the cubic 0.02 warp of the mul1 bytesum before the int8 round (2 FMAs in decode) Gaussianises the codebook marginal.
- **Why it was dropped, and why that no longer holds:** #78-03 / #78-04 found no gain on top of H_out, at a time when H_out was planned for every level. Since 11-07, H_out applies at K ≥ 4 only. The K ≤ 3.5 levels, which include the 2.5 key product, are plain LDLQ — exactly the regime where i8c passed.
- **Source:** #78, #78-02, #79, #80 (and 2605.08692's tuned 4-bit tables as the general idea).
- **Why it should help:** 0.8B KL, plain quants: K3 0.983 (6 seeds, z −3.7), K2 0.973 (5 seeds), K4 0.977. 27B CPU proxy (old order): K2 0.991, K2.5 0.987, K3 0.982 (param-weighted 0.994 / 0.989 / 0.986). It stacks with CD (#79: combined 0.979 at K2.5).
- **Fit:** per-level codebook flag (`cb=i8c`); no ladder cost if decided before the freeze. The HIP decode needs the i8c path (BACKLOG item) in GEMV, the A8 plane and the f16 plane.
- **Test:**
  - CPU: settle64 (exllamav3 order), w43's 27-tensor set, K2/2.5/3, mul1i8 vs i8c. Pass: geomean ≤0.99 at K2.5.
  - 4070: 0.8B, i8c on top of the adopted swj, 3 seeds × K2/K3. Pass: every seed <1.
  - About 1 CPU-h plus 1.5 card-h. Do not try a 27B pair: a ~1% effect sits inside the 2.5%-per-pair seed floor.
- **Expected:** −0.7 to −1.5% KL at 2.5. Risk: decode +2 FMAs (~+3–5% GEMV ALU); the #100 lesson that codebook proxies under-price real KL (here the 0.8B KL exists, which mitigates it).
- **Conflicts:** #78-04, which only tested on top of H_out.

**A2 · i8c / mul1i8 twin levels at the H_out boundary (K3.25–4.5)** (new)
- **Mechanism:** where H_out starts (K4), offer both codebooks at a few levels and let knap4 choose per tensor. This resolves whether i8c helps on H_out tensors without a global decision.
- **Fit:** per-level library (×2 for ~6 levels).
- **Test:** CPU proxy on settle64 with H_out enabled at K4: if i8c is ≤0.995 on two-sided LDLQ, keep only i8c.
- **Expected:** 0 to −0.5% at 3.5–4.5; ~0 at 2.5. Low priority.

**A3 · Per-K warp strength a(K)** (re-priced B2)
- **Mechanism:** one cubic coefficient per level instead of a global 0.02. At K4 the CPU proxy had cubic0.05 0.977 vs cubic0.02 0.979; low K may want less.
- **Fit:** per-level constant in the encoding; zero runtime cost.
- **Test:** CPU sweep a ∈ {0, 0.01, 0.02, 0.035, 0.05} × K ∈ {1.5, 2, 2.5, 3, 4, 5, 6} on settle64. Pass: ≥0.3% over a = 0.02 at any K that matters.
- **Expected:** 0 to −0.3% on top of A1.

**A4 · Exact Gaussian-rank remap LUT (1,021 bytesum bins → value)** (re-priced)
- **Mechanism:** replace the cubic with the exact rank-Gaussianisation as a 1K-entry table (11-02 Gaussian tiles: ×0.996 / ×0.992 at K2/K3, about the same as cubic).
- **Fit:** codebook flag; free in the prefill pre-pass; one LDS gather in GEMV.
- **Test:** only if A1 passes, CPU arm against A1. Pass: ≥0.3% better.
- **Expected:** ~0 over A1. Risk: GEMV slowdown (IQR saw table gathers bound GEMV).

**A5 · Lloyd-refit remap on real compensated targets** (new)
- **Mechanism:** after a Viterbi pass, set each bytesum bin's value to the H-weighted centroid of the LDLQ targets assigned to it, then re-Viterbi (2–3 rounds), per K (global, not per tensor).
- **Source:** BARQ (2610.04490) / AAAC (2605.08692) refit idea, but at trellis-bin level and global per K.
- **Why:** captures any non-Gaussianity of compensated targets (drift adds tails), which M6 (raw weights Gaussian) does not rule out.
- **Fit:** per-level table (1K values).
- **Test:** CPU settle64 on 27 tensors, fit on sample 0, score on sample 1. Pass: ≥0.5% held-out at K2.5.
- **Expected:** 0 to −0.5%. Risk: overfitting a 1K-value table is low.

**A6 · Family/depth-specific tail warp for early attention/GDN inputs** (new)
- **Mechanism:** a heavier-tail warp (or A7's tail extension) only for tensors where M3 shows massive-pivot coupling (L3/L7/L11 q/k, L2/L10 GDN). Large rank-1 feedback there may push compensated targets beyond the codebook range.
- **Fit:** per-tensor codebook flag (2 variants).
- **Test:** G2 census first; then CPU arm on those 8 tensors.
- **Expected:** 0 to −0.3%.
- **Conflicts:** M6 argues the raw weights are Gaussian; this targets compensated targets only.

**A7 · Tail-extended codebook** (re-priced B7)
- **Mechanism:** map the extreme ~0.1% of bytesum values to ±4–4.5σ so overloaded compensated targets are not clipped.
- **New evidence:** M3 shows rank-1 pivots up to 37× the block mean in early attention, a plausible overload source. The 11-02 ordering reversal (×3.4) was also attributed to drift/overload.
- **Test:** G2 census, then a CPU sweep. Pass: ≥0.5% at K2.
- **Expected:** 0 to −0.5%. Risk: costs bulk density.

**A8 · Stratified ("XOR-palette") int8 computed code** (new adaptation)
- **Mechanism:** value = P[parity-hash(state)][branch bits], with 2^p int8 palettes × 2^K values, each palette rank-interleaved so every state's successor menu covers all strata. This is Ungerboeck-style set partitioning on a 16-bit bitshift trellis.
- **Source:** XOR-Trellis (2610.00432: 8.10 vs QTIP-3INST 8.48 PPL at 2.25 bpw with Hadamard, but confounded by its group scales); M4 (mul1 menus cover all strata only 17% / 2% of the time).
- **Why:** better menus could close part of mul1's 4% / 6.3% gap to D(R).
- **Fit:** codebook format change before the freeze. Decode is cheaper than mul1 (XOR parities + an LDS palette of ≤256 B).
- **Test:**
  - CPU: Gaussian tiles at K2/2.5/3 with tail-biting, palettes Lloyd-fitted. Pass: ≤0.985× mul1i8 at K2.5.
  - Then real tensors, then a 0.8B KL arm with 3 seeds, because the hash codebook (#100) showed proxy under-pricing ×3–4.
- **Expected:** 0 to −1.5%, P 0.15. Risk: high; a late format change.

**A9 · Partial stratification of mul1** (new, cheap variant of A8)
- **Mechanism:** keep mul1's hash but XOR the top branch bit with one hash parity bit before the window update, so a state's successors pair up across the median. One extra XOR in decode.
- **Test:** CPU Gaussian tiles. Pass: ≥0.5% at K2–3.
- **Expected:** 0 to −0.5%, P 0.15.

**A10 · Per-K multiplier / bytesum check at high K (K5–7)** (new)
- **Mechanism:** 11-02 B4 found mul1's constant good at K2–3. At high K the measured error per bit is only ×3.67–3.81 (ideal ×4.0), so the code may lose efficiency where the window covers only 2–3 steps. Search the multiplier and the byte count per high K on iid Gaussian tiles.
- **Fit:** per-level constant (decode unchanged, except a constant in the hash).
- **Test:** CPU minutes. Pass: ≥2% at K6–7.
- **Expected:** 0 at 2.5; helps 6.0/7.0 only.

**A11 · "Uncoded-LSB" (parallel-transition) code for the mul1 K6.5–8 levels** (new)
- **Mechanism:**
  - Classical high-rate TCQ: only the top K−2 bits go through the trellis; 2 fine bits per step are uncoded (rounded), on a uniform offset grid.
  - Viterbi complexity falls by 4×, which matters because qtf2 runs only ~25–35k tiles/s at KA 6–7.
  - Quality may improve where L16 memory is only ~2.5 steps.
- **Fit:** format change for f16-path levels only.
- **Test:** CPU Gaussian iid at K6–8 vs mul1. Pass: equal error at ≥3× encode speed, or ≥3% lower error.
- **Expected:** 0 at 2.5; ladder time and 6–7 bpw quality. Low priority.

---

## B. Per-tile side information

**B1 · Per-tile XOR salt, best-of-2 / best-of-4** (re-priced D1; still untested on real tiles)
- **Mechanism:** each 256-weight tile stores 1–2 bits naming an XOR salt applied to the state before the hash. The encoder runs N Viterbis and keeps the best.
- **Source:** 11-02 measurement: Gaussian tiles best-of-2 ×0.985/0.988, best-of-4 ×0.974/0.975 at K2/K3; per-tile error correlation 0.93–0.95. AAAC (2605.08692) did group-level table selection.
- **Why:** selection buys more than spending the same 0.004–0.008 bpw on rate: net −1.0 to −1.5% error at equal bits.
- **Fit:** format change (side plane 0.004–0.008 bpw, packed outside the 16K-word tile); quantizer ×N on the Viterbi share (~80% of level time).
- **Test:** settle64, real LDLQ tiles in exllamav3 order, 27 tensors × 2 samples at K2.5, best-of-2 and best-of-4, net of bits. Pass: net ≤0.985. Then a 0.8B arm (3 seeds), about 4 CPU-h plus 2 card-h.
- **Expected:** −0.7 to −1.5% KL. Risk: ladder time ×1.8–3.4; every decoder (GEMV, A8 / f16 / A4 planes) reads the plane.
- **Conflicts:** #41 (seed selection) died because tensor-level averaging removes the spread; a tile does not average.

**B2 · Salt as a per-tensor knapsack option** (re-priced B6)
- **Mechanism:** salted levels exist only for K ≤ 3.5 and are offered as twins; knap4 buys the 0.008 bpw where sensitivity is high.
- **Fit:** per-level library (twin levels).
- **Expected:** keeps about 70% of B1's gain at a fraction of the ladder time.

**B3 · Hot-tile salts only** (new)
- **Mechanism:** best-of-16 salts only on the tiles of the ~1% hot 16-column blocks (M3), chosen by the full D_b metric e^T D_b e, which penalises the error's projection on the massive direction. Side info is a sparse (tile, salt) list.
- **Why:** M3 puts 10–11% of the block-LDLQ error of early attention q in those 3 blocks, and with 16 alternatives a low-projection path is likely.
- **Fit:** format change (sparse list), but the encoder cost is ~16× on 1% of tiles, so negligible.
- **Test:** settle64 on L3/L7/L11/L15 q and k at K2.5 with the D_b-metric selection. Pass: ≥3% tensor error.
- **Expected:** −0.1 to −0.5% KL. Low risk.

**B4 · Per-tile 1-bit codebook-scale select (×1 / ×1.15)** (new)
- **Mechanism:** the variant set is two scales instead of two salts. It attacks tile-level overload and energy mismatch (no per-tile scale exists).
- **Fit:** same plane as B1; can combine 1 salt bit + 1 scale bit.
- **Test:** CPU Gaussian and real tiles: variant correlation and best-of-2. Pass: better than salt best-of-2, or additive with it.
- **Expected:** −0.2 to −0.8%.
- **Conflicts:** #40 (8-bit group scales lose at equal bits); a 1-bit selector costs 32× less.

**B5 · Salt selection by the full block metric** (re-priced B5b)
- **Mechanism:** choose the salt by e^T D_b e instead of tile MSE. Free once B1 exists; matters where A/S > 1.02.
- **Expected:** −0.1 to −0.3% on top of B1.

**B6 · Salts first on H_out levels (K ≥ 4)** (new)
- **Mechanism:** two-sided feedback is 72% of a gate_proj's quant time there, so ×N Viterbi costs relatively less.
- **Expected:** a K4 gain similar to B1. Ladder-time argument only.

**B7 · Early-abort multi-salt Viterbi** (new)
- **Mechanism:** run the N salted Viterbis in lockstep and drop a salt when its best partial path cost exceeds the leader's by a margin after half the tile. Cuts B1's encoder cost toward ~1.5×.
- **Test:** CPU measurement of the selection-quality loss vs time.
- **Expected:** an enabler for B1.

**B8 · Salt-variant decorrelation diagnostic on real tiles** (new)
- **Mechanism:** measure the variant error correlation on LDLQ-compensated targets (it was 0.93–0.95 on Gaussian tiles). If >0.98, B1 dies cheaply.
- **Cost:** 30 CPU-min.

**B9 · Per-tile scan-order bit (row- vs column-major path)** (new)
- **Mechanism:** a decorrelated variant with different trellis/LDLQ interaction.
- **Expected:** ≤ B1; more kernel complexity. Low.

---

## C. Trellis structure

**C1 · Canonical fractional-mask shape check** (new)
- **Mechanism:** the even-spread masks (0x8888 / 0xAAAA / 0xEEEE) are a convention. Test clustered masks (0xFF00), seam-aligned masks (extra bits at tile start/end, where tail-biting constrains the path), and alternating phases.
- **Fit:** format convention; must be settled before the freeze; zero runtime cost.
- **Test:** CPU settle64 at K2.25/2.5/2.75 on 27 tensors; minutes. Pass: ≥0.5%.
- **Expected:** 0 to −0.3%.

**C2 · Seam placement** (re-priced A10)
- **Mechanism:** choose the tail-biting roll offset so the wrap-constrained steps land on the least sensitive positions (rows with the lowest svh²·h).
- **Fit:** encoder only.
- **Expected:** 0 to −0.2%.

**C3 · Multi-candidate wrap search for top-sensitivity tensors only** (new scope of #57)
- **Mechanism:** 16 wrap candidates recover 12/8/4% of the ~5% tail-biting gap at K1/2/3. Apply it only to the ~30 tensors with the highest knapsack sensitivity (L63 gate_up, early attention).
- **Fit:** encoder only.
- **Expected:** −0.05 to −0.15%.

**C4 · Mask phase rotated per tile index** (new)
- **Mechanism:** decorrelates fractional-step positions across tiles. TCQ theory says ≈0; bundle with C1.

**C5 · L18 only for K ≥ 5** (new; wait for w71's K5/K6 readout)
- **Mechanism:** the high-K window covers ~3 steps. If V18 gains ≥2% at K5–6 (unknown; 0.6% at K2–3), a per-K window could pay for the 6.0/7.0 sizes.
- **Cost:** decode +10% (radiance's packed 16-bit hash breaks).
- **Expected:** 0 at 2.5.

**C6 · Edge-emitting values from a (16+K)-bit window** (re-priced D5)
- **Mechanism:** more value diversity with 2^16 survivors. Decode costs the same as L18 in radiance's kernels (+10–28%).
- **Expected:** for ≤0.6% (#113's V18), not worth it. Kept only as a C5 companion at high K.

---

## D. Rotation and incoherence (grounded by M3)

**D1 · 512-point input Hadamard for attention/GDN input families** (re-priced D2)
- **Mechanism:** a 512-point FWHT replaces the 128-point one on attention/GDN inputs (5120, 6144 and 17408 are all multiples of 512).
- **Why:**
  - M3: A ×0.940 (L3 q), ×0.937 (L11 q), ×0.978–0.988 elsewhere on attention/GDN inputs; MLP ×0.995.
  - ~20% of params gain 1.5–2%, early attention 6%.
- **Fit:** per-tensor rotation-size field (format) plus a prep-pass FWHT512; decode unchanged.
- **A8 interplay:** the massive channel spreads over 4 g128 groups instead of 1, so A8 error on those groups rises. Pair with E1, or price both modes.
- **Test:**
  - settle64, real trellis, on the M3 tensor set at K2.5. Pass: ≥1.5% geomean on the family.
  - aqproxy on E1c dumps for the A8 cost.
  - 0.8B KL arm with 3 seeds.
- **Expected:** −0.3 to −0.6% KL. Risk: prep kernel work; fast-variant A8 cost.

**D2 · 256-point input Hadamard** (new)
- **Mechanism:** the middle option. M3: ×0.984–0.991 broadly, ×0.94–0.95 at early q. It halves the A8 spreading of D1.
- **Expected:** −0.2 to −0.5%.

**D3 · Variable Hadamard size map: 512 only on the block holding the massive channel** (re-priced D4)
- **Mechanism:** keeps 128 elsewhere, so A8 groups stay clean except one 512-span.
- **Fit:** prep-pass size map.
- **Test:** CPU A/S variant of aspen.py.
- **Expected:** most of D1 on early attention with half its A8 cost.

**D4 · Hadamard-block stride interleave, f16-only tensors** (re-priced C5)
- **Mechanism:** each 16-column LDLQ block takes one coordinate from each of 16 Hadamard blocks, which removes in-block rank-1 coupling (the M3 penalty) entirely.
- **Constraint:** g128 A8 groups in K order would each carry one massive coordinate, R1-like (#47: 6.6× worse A8). So it is valid only for tensors that always run on the f16 path (accurate variant), which knap4 must respect.
- **Expected:** A/S → ~1.00, i.e. −2 to −10% error on early attention; −0.2 to −0.5% KL for the accurate variant only.

**D5 · Two-level H128∘stride∘H128** (re-priced D1)
- **Mechanism:** 11-02 Euclidean proxy 0.925–0.998. Same A8 caveat as D4. Superseded by D1/E1 unless real-trellis tests disagree.

**D6 · Place the massive channel with an exact residual-stream permutation** (new)
- **Mechanism:**
  - A permutation of the 5120 residual channels is exact and free: it commutes with RMSNorm gains (permute them too) and folds into the embedding, the head and every reader's and writer's weights.
  - Unlike R1 (#47), it needs no gain folding.
  - Use it to choose (a) which 127 channels share the massive channels' Hadamard block (the least important ones, so the A8 coarse step and the in-block coupling hit low-value coordinates) and (b) the block's LDLQ processing position (first or last). The real quantizer is very order-sensitive (11-02: block-ascending ×3.4 worse on L3 q).
- **Fit:** conversion-time model permutation; per level; zero runtime cost. One permutation for all layers, which works if the massive index is the same across layers (E8).
- **Test:** settle64 on L3/L11 q and k and L2/L10 GDN qkv: column swaps of H and W placing the massive block at positions {0, 39} and with chosen neighbours. Pass: ≥3% on early attention without loss elsewhere.
- **Expected:** 0 to −1%, P 0.2.

**D7 · Free per-layer permutations of writer inputs** (new; extends #106)
- **Mechanism:** v-dimension permutations inside heads (GDN out_proj, o_proj) plus head order. M2 says rate allocation is a wash, but #106 shows real-order effects (o_proj −3%).
- **Test:** add within-head orders to #106's order-fixed proxy.
- **Expected:** 0 to −0.3%.

**D8 · 256-point output Hadamard for attention q/k/v (head_dim 256)** (new)
- **Mechanism:** today's 128-point output Hadamard splits each 256-dimensional head in two. Aligning with heads matters if F2 (per-head K) is used; otherwise neutral.
- **Test:** only together with F2.

**D9 · Per-tensor rotation size as twin levels {128, 512}** (new)
- **Mechanism:** if D1 is mixed across layers (early helps, late neutral), offer both at the K ≤ 3.5 levels for attention/GDN inputs and let the knapsack (with A8 costs) pick.
- **Fit:** per-level library ×2 for ~20% of params.

---

## E. Massive channels and hot blocks (grounded by M1 and M3)

**E1 · Top-2 massive-channel extraction ("bypass") on attention q/k/v** (re-priced E1, now with block-LDL evidence)
- **Mechanism:**
  - The prep pass reads x_c for the 2 dominant channels of the shared input, zeroes them before su and the rotation, and adds W[:, c]·x_c in the epilogue (rank-2, bf16 or int8 columns).
  - LDLQ runs on H with those rows and columns removed.
- **Why:**
  - M3: the A/S penalty falls from 1.12–1.14 to 1.03–1.04 on L3/L11 q; error ×0.910 / ×0.900 (top-2) vs ×0.932 / ×0.946 (top-1).
  - L3 k ×0.978; later q ×0.990.
  - It also removes the massive coordinate from the A8 g128 groups (E3).
- **Bytes:** ~20–25 KB per channel per tensor, so ≈0.0007 bpw for all 16 attention layers. Compute: 2·N FMAs per token (+0.04%).
- **Fit:** format change (per-tensor channel list and columns) plus small prep and epilogue changes in the exl3i plugin; per level.
- **Test:** settle64 real-trellis arms (base / x1 / x2) for the 16 attention q/k/v at K2/2.5/3. Pass: ≥5% geomean on L3–L15 q, no tensor worse. Then a 0.8B arm (3 seeds), about 2 CPU-h plus 2 card-h.
- **Expected:** −0.2 to −1% KL; more if early layers' KL share exceeds their parameter share (kfit2's compounding term says it does). Low risk: exact algebra.
- **Conflicts:** #47 (R1 spreads the spike; this removes it); faraway FA-C2.

**E2 · Extraction for early GDN qkv/z** (new scope)
- **Mechanism:** M3 shows L2/L10 at ×0.983–0.996: marginal.
- **Fit:** only as a knapsack twin once E1's machinery exists.
- **Expected:** −0.05 to −0.2%.

**E3 · A8 value of extraction** (new)
- **Mechanism:** with 63–93% of input energy in one channel, the g128 block holding it is quantised far more coarsely on spike tokens.
- **Test:** CPU aqproxy on the E1c dumps (A8-g128 rel_out with and without the top-2 channels zeroed), per family and depth. Pass: ≥15% lower A8 error on attention/GDN inputs.
- **Expected:** fast variant −0.1 to −0.5% KL. This is the ingredient that makes D1 safe for A8.

**E4 · Hot-block list-Viterbi rescoring by the full D_b metric** (re-priced C8, localised)
- **Mechanism:** keep the top-16 paths per tile only in the ~3 hot 16-column blocks per tensor (10–11% of A in early q) and pick the path with the smallest e^T D_b e.
- **Fit:** encoder only; no format change.
- **Test:** settle64 on early attention.
- **Expected:** recovers part of the 12–14% penalty on 1% of tiles: −2 to −5% on L3/L11 q, −0.05 to −0.3% KL.

**E5 · Hot-block iterative target shift** (new)
- **Mechanism:**
  - For a hot block D_b ≈ aI + β u uᵀ, with u the flat massive direction.
  - Run Viterbi, compute each row's error projection c_r = uᵀ e_r, shift the targets by −γ c_r u / 16, then re-Viterbi.
  - Do 2–3 fixed-point rounds and accept on the true metric. This linearises the coupled metric.
- **Fit:** encoder only.
- **Expected:** like E4 at lower cost. Compare both on CPU.

**E6 · Hot-block K bump** (new; F1-lite)
- **Mechanism:** a sparse list of ≤3 16-column blocks per tensor coded at K+1. Metadata is tiny; kernel tile word counts vary only on listed blocks.
- **Use:** only if E1, E4 and E5 all fail on real tiles.

**E7 · Undamped or block-damped requant of hot blocks** (re-priced FA-C2)
- **Mechanism:** damping is added as δ·mean(diag)·I, which is huge compared with normal channels when the massive channel inflates mean(diag) by ~100×. Try damping relative to the median diagonal.
- **Test:** CPU damping sweep on early attention only.
- **Expected:** 0 to −0.5% on those tensors.
- **Conflicts:** #38 (damping kept per family, tested at layers 11/43/63 without this split).

**E8 · Massive-index stability census** (new diagnostic)
- **Mechanism:** the top-2 channel indices per layer for all readers. Same indices across layers lets D6 (one permutation) and E1 (one gather list per layer type) work.
- **Cost:** 15 CPU-min.

**E9 · Per-tensor extraction count from the A/S census** (new)
- **Mechanism:** choose 0/1/2 channels by A/S reduction ≥2% (M3 style), over all 64 layers × input families.
- **Fit:** per-level policy.
- **Enabler:** run aspen.py across all layers (diagnostic #4 below).

---

## F. Rate granularity

**F1 · Section-split K for fused projections, built from the existing library** (new)
- **Mechanism:**
  - Slice GDN in_proj_qkv into q (2048) | k (2048) | v (6144) rows, attention q_proj into q (6144) | output gate (6144), and gate_up into gate | up (F9).
  - Each section can take a different ladder level.
  - One-sided LDLQ codes rows independently. Section boundaries are multiples of 128, so the output Hadamard does not cross them. su differs slightly per level after refit_scales, so each section needs its own prep (as siblings already do, #42).
  - No requant is needed for K ≤ 3.5. Levels with H_out coupled rows during quantization, so a cross-level section mix needs a check.
- **Why:**
  - #96's spread ceiling on ssm_qkv (0.33 bit) and attn_qg (0.27 bit) sits between heads, and q/k vs v is the coarsest version of that spread.
  - GDN q and k are L2-normalised and attention q/k pass through q_norm/k_norm (#108: per-head scale errors are absorbed), so q/k sections are plausibly cheaper than v and the gate.
- **Fit:** per-level library unchanged; graph/kernel change (segmented launch or 2–3 GEMMs); knap4 needs section sensitivities.
- **Test:**
  - CPU: section-level water-filling ceiling from the 27B Σdy² diagonal × per-section LDLQ proxy error (diagnostic #5).
  - Then 0.8B: section swaps at K±0.5, 3 seeds, fixed codes (slicing needs no requant).
  - Pass: a predicted ≥0.005 bpw at 2.5 (≈ −0.8% KL).
- **Expected:** 0 to −2%, P 0.3. Risk: launch overhead on small sections.

**F2 · Per-head K by exact head permutation and a two-segment split** (new route for #96's remainder)
- **Mechanism:**
  - Permute GDN heads consistently across q/k/v/z/a/b/conv/out_proj, or attention KV groups with their q heads, o_proj columns and q/k norms. This is exact.
  - Then split each tensor into a high-K and a low-K segment of contiguous heads. No row-slab format is needed.
- **Why:** #96's ceiling ≈0.04 bpw overall (≈ −6% KL at the exchange rate). It rests on Σdy² row importance, which proved weaker at 27B than on the 0.8B, so it is discounted heavily.
- **Fit:** conversion-time permutation plus F1's segmented launch.
- **Test:** 0.8B with heads ranked by Σdy² vs by GDN decay half-life (F3); hi/lo split at equal bytes, 3 seeds. Pass: ≤0.98.
- **Expected:** −0.5 to −2%, P 0.25.

**F3 · Analytic section and head priors** (new)
- **Mechanism:**
  - The normalisation invariance of q/k (only direction errors matter: weight the error by the projection orthogonal to q).
  - GDN decay half-life for v heads (STEPQuant 2609.38169: half-life vs error persistence ρ ≈ 0.8, for state error).
  - Use these as kfit2 features or as the F2 ranking.
- **Test:** correlate with measured section KL from F1's swaps.

**F4 · Natural per-block K for writers only** (new, from M2)
- **Mechanism:** M2 gives ssm_out 0.044 bit and o_proj 0.025, so ≈0.003 bpw model-wide ≈ −0.5% KL. It needs the per-k-block K format.
- **Verdict:** only if E6's kernel support exists anyway. Low.

**F5 · MLP neuron groups at separate K** (new)
- **Mechanism:** the MLP is a sum over neurons, so split it exactly into two sub-MLPs (gate/up rows and down columns) via the free intermediate permutation, sorted by neuron importance, each with its own K.
- **Constraint:** M2 says down's input-side allocation is a wash. The gain must come from gate/up output-side importance, where #96 bounded the FFN at 0.031 bit.
- **Expected:** 0 to −0.5%. Low.

**F6 · Section split in the head** (new)
- **Mechanism:** the slab head already sorts rows by Fisher; nothing to add except D8-like alignment.
- **Verdict:** skip unless E1/F1 kernels make it free.

---

## G. Encoder and search changes inside the fixed format

**G1 · Extend cd3 to attn_k and to early-layer (L ≤ 15) attention/GDN inputs** (re-priced #77)
- **Mechanism:** #77's GPU family sweep gave attn_k 0.991/0.994 at ~2 s a tensor.
- **Why early layers:** M3 puts in-block structure there, which re-Viterbi against the full residual partly addresses.
- **Fit:** quantizer config (pipeline/config/v3.toml); about +1–3 min a level.
- **Test:** add arms to the CD27 harness. Pass: per-tensor proxy ≤0.995 and no-op gates bit-exact.
- **Expected:** −0.03 to −0.15% KL. Nearly free.

**G2 · Overload census in exllamav3 order** (new diagnostic)
- **Mechanism:** fraction of compensated targets beyond the codebook range (±3.46·scale) per tensor, K and block position, from settle64.
- **Decides:** A6, A7 and G3.
- **Cost:** 1 CPU-h.

**G3 · Saturating LDLQ feedback in hot blocks** (new)
- **Mechanism:** clip the error propagated from a block when it exceeds c·σ, as sigma-delta overload stabilisation. It targets the 37× pivots of M3.
- **Test:** CPU settle64 on early attention. Pass: ≥2% on those tensors.
- **Expected:** 0 to −0.3% KL.
- **Conflicts:** 11-02 C9 (QTEA uniform decay, expected ≈0); this clips only on overload.

**G4 · g_scale search scored by the LDLQ Schur metric** (re-priced C8 / G3)
- **Mechanism:** score scale candidates by the δᵀ S_{G|P} δ metric (2609.36654) instead of raw tile MSE. Matters most where pivots are uneven (early attention).
- **Test:** CPU on 9 tensors.
- **Expected:** 0 to −0.5%.

**G5 · Re-run the g_scale search per codebook variant** (implementation note for A1/B1/B4)
- **Mechanism:** i8c and salts change the optimal scale; the pipeline must not reuse mul1i8 scales.
- **Test:** a single check in settle64.

**G6 · Scale ↔ re-Viterbi alternation** (re-priced C7)
- **Mechanism:** JARQ (2609.38599) gained ≤1% on strong hosts. With swj and LS-shrink in mind, test it with a norm-preserving refit (H1), not LS.
- **Expected:** 0 to −0.5%.

---

## H. Scales coupled to the code (the refit_scales junction)

**H1 · Refit form for the input scales (suh) on fixed codes** (new)
- **Mechanism:**
  - exllamav3's refit_scales LS-refits both suh and svh after the trellis.
  - The swj finding (LS shrinks the gain of rows carrying noise uncorrelated with W) applies to input channels too, and no experiment has isolated suh.
  - Fixed-code arms on suh: {LS (today), no refit, H-norm-preserving, inner-product-preserving c = ‖w‖²_H / ⟨q,w⟩_H (SoloQ 2610.07121)}.
- **Fit:** per level; zero runtime cost; one fit pass.
- **Test:** 0.8B, fixed codes at K2/K3 × 2 seeds, ~1 card-h (no rounding noise within a pair). Pass: ≤0.985.
- **Expected:** 0 to −1%.
- **Conflicts:** #85 / #112 (svh α on writers), #108 (joint refits beyond the MLP failed in the full stack); none tested the suh side alone.

**H2 · C9 α for attention/GDN readers after swj** (re-priced)
- **Mechanism:** #112 found the gain almost entirely on gdn_out (α at the grid edge); readers untested in that form.
- **Test:** 0.8B, fixed codes.
- **Expected:** 0 to −0.5%. Low priority given #108.

**H3 · Per-section scale refit with F1** (new)
- **Mechanism:** if sections take different levels, refit each section's svh jointly (fixed codes) so the q/k/v outputs stay consistent through GDN's normalisation.
- **Expected:** a small companion to F1.

---

## I. Embedding codes (a lookup: decode cost is irrelevant)

**I1 · Best-of-16 salts per embedding tile** (new)
- **Mechanism:** 4 bits per 256 values = 0.016 bpw on the embedding = 0.00075 bpw model-wide, for an expected ×0.95–0.96 embedding error.
- **Fit:** embedding-only format (the row-gather kernel reads the salt); no GEMM impact.
- **Test:** embtrellis2.py arm. Pass: net ≥3% at equal bits.
- **Expected:** −0.1 to −0.2% KL.

**I2 · Exact larger-state code for the embedding (L18 with an iid int8 table)** (new, from #113's synthetic bounds)
- **Mechanism:** an iid table at L20 reached 0.975 / 0.962 / 0.944 of mul1 L16 (K2/3/4) under exact search. The embedding decodes one row at a time, so table size and window width cost nothing at runtime. The encoder runs once (colo, about a day with V18 ×3.6).
- **Expected:** −0.05 to −0.15% KL.

**I3 · i8c-style warp for the embedding** (re-priced I2)
- **Expected:** ~ −0.01%. Free once A1 exists.

**I4 · Row-Hadamard size 512 / 5120 for the embedding** (re-priced D7)
- **Expected:** ≤0.1%. Only bundled with I1.

---

## J. Head codes

**J1 · bf16 option for the top Fisher slab(s)** (new)
- **Mechanism:** ~400 rows hold 50% of the head's Fisher mass and one 128-row block holds 39%. A bf16 top slab costs ~1.3 MB (≈0.0004 bpw); add it to headsim's donor set.
- **Fit:** head slab plan option; no new kernel if the slab head already supports mixed donors.
- **Test:** headsim. Pass: ≥3% head KL at equal bytes.
- **Expected:** 0 to −0.3% KL.

**J2 · A/S and massive-channel check on lm_head.input** (new diagnostic)
- **Mechanism:** the final-norm output may also carry the massive channel. If A/S > 1.05, E1-style extraction for the head (rank-2 bypass, decode cost 2·V FMAs per token).
- **Cost:** 2 CPU-min.

**J3 · Salts and i8c for head slabs** (new)
- **Mechanism:** inherits A1/B1. Head decode reads the plane per token: XOR cost only.
- **Expected:** −0.05 to −0.1% KL.

**J4 · Softmax shift t-grid** (ranked E5, still open)
- **Mechanism:** t ∈ {0.5, …, 4} beyond the mean-row t = 1.
- **Expected:** 2609.31291 saw ≈0 on a Qwen3.5 GPTQ head. Run only in the same headsim batch as J1.

**J5 · 64-row granularity for the top slab** (new)
- **Expected:** 0 to −0.1%. Only with J1.

---

## K. A4 / i4, the low-K research ladder, and the high-K end

**K1 · Stratified int4-native palette code for the i4 levels** (new; only if #46-03 ships A4)
- **Mechanism:** replace "mul1 rounded to the int4 grid" (×1.007 K2, ×1.02 K2.5, ×1.06 K3) with palettes of int4 values in an XOR-Trellis-style code. 2610.00432 designed exactly this for FP4 hardware, and iu4 takes the values directly.
- **Fit:** i4 codebook format.
- **Test:** CPU at K2.5/3. Pass: recovers ≥50% of the i4 penalty.
- **Expected:** fast/A4 variant at ≤3 bpw: −1 to −3% on A4 tensors.

**K2 · Per-rate int4 factor and i4c warp** (new)
- **Mechanism:** the clipped-i8 idea (#109 negative for int8 at K7) may work for int4 at K2.5–3, where 16 levels, not the code, are the bottleneck.
- **Test:** CPU sweep, minutes.
- **Expected:** −1 to −3% on i4 tensors, only if A4 ships.

**K3 · Palette code for sub-1 levels** (new)
- **Mechanism:** the salted 0-bit read repeats values. A 2-value palette per state class may code K0.5–0.875 better.
- **Scope:** research ladder only (below 1.3 bpw).

**K4 · Dual-plane int8 (W_hi + 2⁻⁷W_lo) for the top tensors of the fast 7.0** (new)
- **Mechanism:** two iu8 GEMMs sharing the int8 activation plane: 2/255 vs f16's 1/116 TOPS⁻¹, about 10% faster than the f16 path for the same tensor.
- **Why:** the int8 grid floors at about mul1 K6.8 (#94/#109); no code on a uniform int8 grid with ±4σ range beats it.
- **Fit:** knap4 compute mode.
- **Expected:** 7.0 fast variant only.

**K5 · Record the int8-grid floor in knap4** (new note)
- **Mechanism:** Δ²/12 on a ±4σ / 255-level grid ≈ 2^{−2·6.8}. Any i8 level above K6.75 should be priced from measured curves (as v3.md says) and never offered as "K7 at int8".
- **Effect:** process only.

---

## L. Format and process

**L1 · Reserve encoding fields now** (new)
- **Mechanism:** reserve salt-plane offset, extraction list, rotation size and warp coefficient in the exl3i encoding (empty by default), with refusal tests. Kernels can then add them without a format bump.
- **Caveat:** the ladder still has to be requantised to use them, so the freeze decision stands. Cheap insurance for V3 extras.

**L2 · Decide sub-2% code effects by proxy plus 0.8B, not by 27B pairs** (new)
- **Mechanism:** the 27B seed floor (~2.5% a pair; σ = 0.018 at K2.5) cannot resolve A1/B1/D1-sized effects. Write pass rules on settle64 (real order) and 0.8B seed triplets, with a 27B no-harm pair as the only 27B gate (as for #77 CD27).
- **Fit:** methodology.

**L3 · Spend twin-level budget only at K2.25–3.5** (new)
- **Mechanism:** where a code variant ties (A2, B2, D9), build twins only in the key-product range and keep single levels elsewhere. This bounds ladder hours.

---

## X. Recorded dead or near-dead (evidence, so nobody re-runs them)

| ID | idea | why dead |
|---|---|---|
| X1 | Column-split K with sorted channels | M2: net ±0.005 bit; sorting trades Hadamard averaging for allocation one for one |
| X2 | Column-split K for readers, natural order | 0.004–0.021 bit (M2, faraway M4) |
| X3 | Massive-channel extraction "for the pivots" | scalar ΣD ×0.994–0.998; the gain is all in-block (M3). E1 stands, on M3's evidence |
| X4 | L18 / L20 windows | #113: V18 0.994; B20-4096 worse at K2; decode +10–28% |
| X5 | Edge-emitting windows at K ≤ 4 | decode cost as L18 for ≤0.6% |
| X6 | Tile rings, chained trellis, partial start bits | #102: free-start value is the 16−K stored start bits; chaining creates no information |
| X7 | Sign-seed selection per 128-band (bits "free" in su/sv) | band averages ~5,000 tiles: spread ~0.13%, best-of-4 ≈ −0.15% for N× quant time |
| X8 | Per-tile multi-bit gain scales | #40: groups lose 8–20% at equal bits; only B4's 1-bit selector survives |
| X9 | Learned butterfly rotation (HARP 2605.29843) | QTIP 3-bit +0.04 PPL for +0.05 bpw |
| X10 | Own-basis KLT for embedding/head | faraway M8 gain ≈0.0042 bpw vs a 52 MB basis ≈0.0155 bpw |
| X11 | R1 residual rotation for the accurate variant only | needs a second ladder (rotated and unrotated tensors cannot mix); A8 6.6× worse (#47) |
| X12 | Antipodal and position-salted codebooks | ≈0 by TCQ theory (11-02 B8/B9) |
| X13 | Within-tile diagonal-weighted Viterbi (CurveTQ) globally | Γ_b 0.0001–0.0007 bit after the RHT. The hot blocks are rank-1 coupled, not diagonal, so E4/E5 instead |
| X14 | Byte-count or multiplier variants at K2–3 | 11-02: all ×1.003–1.16 worse |

---

## Ranked top 20 (expected gain per unit cost)

| # | idea | expected KL at 2.5 | cost | before freeze? |
|---|---|---|---|---|
| 1 | **A1** i8c for the non-H_out levels (≤K3.5) | −0.7 … −1.5% | 1 CPU-h + 1.5 card-h (4070) + HIP i8c decode | yes |
| 2 | **G1** cd3 on attn_k + early attention/GDN inputs | −0.03 … −0.15% | config; +1–3 min a level | yes |
| 3 | **C1** canonical fractional-mask shape check | 0 … −0.3% | CPU minutes | yes |
| 4 | **E4/E5** hot-block rescoring / target shift (encoder only) | −0.05 … −0.3% | CPU test; encoder code | no (encoder) |
| 5 | **E1** top-2 massive-channel bypass on attention q/k/v | −0.2 … −1% (more for the fast variant via E3) | 2 CPU-h + 2 card-h; prep/epilogue change | yes |
| 6 | **A3** per-K warp strength | 0 … −0.3% (on top of A1) | CPU hours | yes |
| 7 | **B1** per-tile salt best-of-2/4 | −0.7 … −1.5% | ~4 CPU-h + 2 card-h; ladder ×1.8–3.4 on Viterbi | yes |
| 8 | **F1** section-split K from the existing library | 0 … −2% (P 0.3) | CPU ceiling minutes; 0.8B fixed-code swaps; segmented launch | no requant needed |
| 9 | **D1/D2** 512/256-RHT on attention/GDN inputs | −0.3 … −0.6% | settle64 + aqproxy; prep FWHT change | yes |
| 10 | **J1** bf16 top head slab | 0 … −0.3% | headsim | no |
| 11 | **B3** hot-tile salts (sparse) | −0.1 … −0.5% | CPU; sparse side list | yes |
| 12 | **H1** suh refit form on fixed codes | 0 … −1% | ~1 card-h 0.8B | refit is per level (before the ladder) |
| 13 | **F2** per-head K via exact permutation | −0.5 … −2% (P 0.25) | 0.8B 3 seeds; conversion permutation | permutation yes |
| 14 | **G2 + A7** overload census → tail-extended codebook | 0 … −0.5% | 1 CPU-h, then sweep | yes |
| 15 | **D6** residual permutation placing the massive block | 0 … −1% (P 0.2) | CPU settle64 arms | yes |
| 16 | **B4** per-tile scale-select bit | −0.2 … −0.8% | CPU (shares B1's plane) | yes |
| 17 | **E3** A8 value of extraction | fast variant −0.1 … −0.5% | aqproxy on E1c dumps, CPU | with E1 |
| 18 | **I1** embedding best-of-16 salts | −0.1 … −0.2% | embtrellis arm | embedding format |
| 19 | **A8** stratified int8 palette code | 0 … −1.5% (P 0.15) | CPU Gaussian, real tiles, 0.8B KL | yes (late, risky) |
| 20 | **K1/K2** i4 palette / factor (A4 path) | −1 … −3% on A4 tensors if A4 ships | CPU minutes | if #46-03 passes |

Stacking: A1, B1, E1, D1 and G1 attack different error sources (codebook marginal, per-tile selection variance, in-block rank-1 coupling, incoherence span, tile re-solve). They should mostly add, which puts the plausible combined code-side gain at **−2 to −4% KL at 2.5**. i8c × CD already stacked on the CPU (#79). Salt × i8c and E1 × D1 overlap partly: on early attention, D1 and E1 attack the same penalty, so choose by the real-trellis test.

---

## The 5 cheapest diagnostics that would most change this ranking

1. **Real-trellis check of M3 (settle64, exllamav3 order).** Arms base / top-1 / top-2 extraction / 256- and 512-RHT / E5 target shift on L3, L7, L11, L15 q and k plus L2 and L10 GDN qkv/z at K2/2.5/3, both samples.
   - Decides E1 vs D1 vs E4/E5, and whether the block-LDLQ proxy's 9–10% (and part of faraway's 1.74×) is real.
   - ~1–2 CPU-h on colo or Thor (colo is at load ~24 from w71 until evening).
2. **Order-fixed i8c re-proxy plus 0.8B i8c × swj stack.** 27 tensors × 2 samples at K2/2.5/3 on settle64, then 3 seeds × K2/K3 on the 4070.
   - Decides A1 (the top item) and A3's sweep range.
   - ~1 CPU-h + 1.5 card-h.
3. **Salt variant correlation and best-of-2/4 on real LDLQ tiles (B8 → B1).** The same 27-tensor set at K2.5, net of side bits.
   - If correlation is >0.98 on compensated targets, B1/B3/B4/I1 all fall in the ranking.
   - ~3 CPU-h.
4. **Full A/S + hot-block + massive-index census.** `aspen.py` as written, over all 64 layers × {q, k, v, GDN qkv, z, gate} plus lm_head.input.
   - Decides E1/E9's scope, D6/E8 (one permutation for all layers?), J2, and whether D1 pays outside early attention.
   - ~250 one-minute jobs, about 15 min wall time on colo.
5. **Section-level water-filling ceiling for F1/F2.** Per-section (q|k|v, q|gate, gate|up) mean Σdy² from the published 27B diagonal × per-section LDLQ proxy error from the levels already on disk.
   - Below ~0.003 bpw predicted at 2.5, F1/F2 drop out of the top 20; above ~0.01 they move to the top 5.
   - CPU minutes.

No GPU box was used. CPU work ran on colo at nice 10 with one thread a job (one job with 4 threads). Scratch is `/data/work/ideas-codes/`. Nothing was written in the repo.
