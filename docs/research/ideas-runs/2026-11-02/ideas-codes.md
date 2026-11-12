I've stopped experimenting; more harness debugging wouldn't change the ranking. The report follows.

# Research ideas, lens **codes**: EXL3-V3 weight format (2026-11-02)

The weight format is EXL3/QTIP: a bitshift trellis with a 16-bit state over 256-weight tiles (16 input × 16 output), the mul1 codebook (a 4-byte bytesum of `state·0x83DCD12D`), 128-point randomized Hadamard on both sides, rank-1 scales (one input scale vector, one output scale vector, one global scale), and block LDLQ over 16-column blocks with a Euclidean tile search. The ideas below change that format or how the search fills it. Before writing them I ran a set of CPU diagnostics on the local 27B Hessians and weights (§0). One of those runs overturned a tempting headline, so read §0 first.

**Conventions.**
- "Proxy" is the relative Hessian-weighted error tr(ΔW H ΔWᵀ)/tr(W H Wᵀ).
- 1% proxy error on all tensors is worth about 0.007 bpw, or roughly 1% KL (kfit2's exponent is ≈1.0).
- At 2–2.5 bpw, 0.1 bpw is worth about 17–20% KL (board numbers: 0.164 → 0.0666 per 0.5 bpw).
- Assumed KLD shares at 2.5 bpw: gate/up ~40%, down ~25%, attention and GDN input/output projections ~25%, head ~5–7%, embedding ~2–3%. These are rough, from kfit2 and turboderp's S.

---

## 0. Measured in this session (CPU, local calib Hessians and bf16 weights)

| check | result |
|---|---|
| Γ_b: gain available to a diagonal-weighted Viterbi inside 16-column LDLQ blocks, after our 128-RHT | **0.0001–0.0007 bit** on 7 tensors; 0.017–0.30 bit in the native basis. CurveTQ/XOR-Trellis branch weighting is dead for us. |
| Within-block correlation penalty A/S = Σtr(D_b)/Σ(scalar-LDL d_i), r128, natural order | MLP in 1.003; L10 GDN in 1.044; L43 attention in 1.056; L3 attention in **1.155**; ssm_out 1.033; o_proj 1.019. With 512-RHT it is 1.001–1.016; with Haar 1.001–1.013. |
| Euclidean block-LDLQ proxy A relative to r128 (natural order) | 512-RHT 0.955–0.998; Haar 0.942–0.998; two-level H128∘stride∘H128 0.925–0.998. Gains sit on attention/GDN/out-projection inputs; MLP inputs are flat. |
| LDLQ order by rotated-H diagonal, iid-error proxy | ascending (low energy first) 0.884–0.992; GPTQ-style descending 1.004–**2.57** |
| **Same, real LDLQ + real trellis, L3 q_proj, K2, input scales included** | natural rel **0.00202**; block-ascending **0.00678 (×3.4 worse)**. The iid proxy is wrong about ordering (likely compensation drift/overload). Sorted-coordinate and descending runs did not finish (harness hang, below). |
| Massive input channel (H_jj / mean diag) | 2,809–4,765 for attention/GDN inputs; 114–739 for MLP inputs |
| Exact extraction of the top-1 input channel (proxy, natural order) | L3 attention 0.864; L43 0.965; L10 GDN 0.965; MLP 0.997–0.999; out-projections 0.998 (for +0.003 bpw) |
| Γ_cross: water-filling gain of a per-k-block K, natural order | 0.019–0.074 bit at 128-row blocks on attention/GDN/out; 0.007–0.009 on MLP |
| Codebook variants, Gaussian tiles, free start (paired SE ≈0.2%) | mul1 K2 MSE 0.0650 vs D(R) 0.0625 (4% above); K3 6.3% above. Gaussian-rank remap ×0.996 / ×0.992 (K2/K3); 3-byte sum ×1.013 / ×1.022; 8-byte ×1.003 / ×1.007; 2-byte ×1.11 / ×1.16; other multipliers ×1.016–1.092 (mul1's constant is good). |
| Per-tile codebook selection (XOR salt) | best-of-2 ×0.985 / ×0.988; best-of-4 ×0.974 / ×0.975; per-tile error correlation between variants 0.93–0.95 |

**Caveats.**
- The H-only proxies assume equal, independent rounding error per coordinate. They used damping 0.025·mean diag and did not include the input scale vector (it is included in the real-trellis run).
- All CPU Viterbi runs are free-start, so they are 4–6% optimistic in absolute terms (#57); ratios between arms are fine.
- The real-trellis harness kept hanging: torch's and the Viterbi library's OpenMP runtimes fight, and some numpy runs stalled for reasons I did not resolve. Fixing it is diagnostic #1.
- **Rule from the ordering reversal:** every H-only proxy number above (rotation size, outlier extraction, Γ_cross) is a hypothesis until the real-quantizer test confirms it.

---

## A. Trellis termination and structure

**A1. Two- and four-tile tail-biting rings.** Join 2 or 4 output-adjacent tiles of the same LDLQ block into one tail-biting ring of 512 or 1,024 weights. Tile A's start window comes from tile B's tail, and the reverse. The Viterbi still runs on independent rings (QTIP Alg. 4 on the longer ring). The decoder reads one extra 32-bit word from the partner tile to set its start state.
- *Source:* QTIP Alg. 4 (2406.11235); CurveTQ's tail-biting length sweep (2610.09212: T = 64 → 128 → 256 recovers 0.98 and 2.22 of 4.10 ppl). New here: rings across tiles that are already searched together.
- *Why:* #57 measured the 256-length tail-biting cost at 4–6% proxy (free/TB 0.948 at K2, 0.962 at K4). If the boundary cost scales ~1/T, a 512-ring halves it at 0 bpw.
- *Fit:* format change (decode plan, quantizer ring length); then per-level.
- *Test:* CPU: extend `viterbi_tiles_tb` to multi-tile rings; real LDLQ tiles of the 33-tensor set at K2/2.5/3. Pass: ≥1.5% proxy at K2.5. Then 0.8B whole-trunk K2.5 arm vs base, paired KL (4070, ~10 min). Kernel: exl3i plan + selftest, ~2 days.
- *Expected:* −1.5 to −2.5% KL at 2.5 bpw. Risk: the gap may not scale as 1/T; GEMV lane-plan complexity.
- *Conflicts:* #57 closed a better wrap *search*, not a longer ring.

**A2. Partial stored start state (s bits per tile).** Store s of the (16−K) wrap bits explicitly and keep the rest tail-biting. This gives a continuous knob from pure tail-biting (s=0) to free start (s=16−K) at s/256 bpw.
- *Source:* CurveTQ's stored start state (+1–3 downstream points at 2 bits for QTIP; ≈0 at 3–4 bits). New: partial bits.
- *Why:* a full free start at K2 costs 0.055 bpw, about 7.6% of rate-equivalent error, to buy ~5%: a loss in proxy terms. Only partial bits can pay.
- *Fit:* format change.
- *Test:* CPU curve s = 0, 2, 4, 8, 14 on real tiles. Pass: some s beats the rate-equivalent line (1%/0.0072 bpw).
- *Expected:* 0 to −0.5% KL. Risk: probably dominated by A1.
- *Conflicts:* #57.

**A3. Free start as a per-tensor knapsack option.** Offer "free-start tiles" as an option per tensor at +0.055 bpw, and let the knapsack buy it only where KL sensitivity exceeds what the proxy shows.
- *Source:* CurveTQ; new as an allocation option.
- *Why:* CurveTQ says the start state matters more on generative metrics than per-layer error suggests.
- *Fit:* per-level option.
- *Test:* 0.8B: free-start vs tail-biting quant at K2, paired KL; pass if the KL gain exceeds the proxy gain by >1.5×.
- *Expected:* 0–0.5%. Risk: just a worse A1.
- *Conflicts:* #57.

**A4. Longer state (L = 18 or 20).** Widen the bitshift window. The value stays a hash of the window, so decode ALU cost is essentially unchanged; Viterbi states grow 4–16×.
- *Source:* QTIP's L ablation (Table 10: L12 → L16 took W2 from 6.97 to 6.83 at 2 bits).
- *Why:* the free-start Gaussian gap to D(R) is 4% (K2) and 6.3% (K3); a larger L closes part of it.
- *Fit:* format change and quantizer change.
- *Test:* CPU Viterbi with 2^18 states on Gaussian tiles (4× time), then real tiles. Pass: ≥1% at K2–3.
- *Expected:* −0.5 to −1.5% KL. Risk: quantizer time ×4–16 for every level requant (~3 h a level today), though the trellis is only part of that time.

**A5. Edge-emitting values from a (16+K)-bit window.** Keep 65,536 survivor states but let each transition emit `cb[hash of the 16+K-bit window]`. Value diversity rises 2^K× at the same state count.
- *Source:* CurveTQ "edge-indexed emission" (1.27 ppl at 2 bits on Qwen3-4B, 0.17 at 3, ~0 at 4).
- *Why:* more distinct reproduction values per path at the same memory.
- *Fit:* format change (decoder reads K more bits per window); quantizer distance evaluations ×2^K.
- *Test:* CPU Viterbi variant on Gaussian, then real tiles; K2/K2.5/K3.
- *Expected:* −0.5 to −1% KL at 2.5 bpw. Risk: the gain shrinks fast with K.

**A6. Trellis path order inside a tile.** Test a k-major path order against the tensor-core permuted order. It matters only where errors are correlated within the block (A/S > 1).
- *Source:* new.
- *Fit:* format change (decode permutation).
- *Test:* CPU, two orders on the L3/L43 attention tiles.
- *Expected:* ≈0%. Low priority.

**A7. V = 2 (two weights per step).** Rejected: QTIP Table 11 shows LUT L16 V2 at 6.79 vs V1 6.83, within noise, while kernel complexity rises.

**A8. Row-chained trellis across k-blocks.** Rejected: CurveTQ measured chained ≈ tail-biting at T=256 (13.59 vs 13.38), and chaining makes decode serial across tiles.

**A9. 8k × 32n tile shape.** Halving the k extent of an LDLQ block halves within-block correlation. Expected gain is the A/S excess (0.3–15% on single tensors), but C6/D2 get it without a kernel rewrite. Low priority.

**A10. Seam placement.** Put the tail-biting seam at the least sensitive row of the tile (with diag H_out) by choosing the ring's rotation offset. Free, since the offset is implied by the layout. Expected ≈0–0.3%. Low priority.

## B. Codebook values

**B1. Computed Gaussianization of mul1.** Replace the bytesum value v by g(v) = v + αv³ (or a 2-segment piecewise-linear map), with α fitted so the marginal matches Gaussian quantiles. The structure is preserved (a monotone map), and it costs 1–2 extra FMAs in decode.
- *Source:* this session's Gaussian-rank remap (×0.996 K2, ×0.992 K3); E3b's Gaussian int8 table (×0.975 K5, ×0.969 K6, but ×1.108 at K4, because that table re-ranked the hash and broke the structure).
- *Why:* the Irwin–Hall(4) marginal is platykurtic and clipped at ±3.46σ.
- *Fit:* new codebook flag (`cb=mul1g`); per-level requant.
- *Test:* CPU trellis.py codebook sweep K2–6 on the 33 tensors with the cubic map, int8-rounded. Pass: ≤0.995 at K2.5–3 and no loss at K4.
- *Expected:* −0.3 to −0.8% KL at 2.5 bpw; larger at ≥5 bpw. Risk: int8 rounding may eat it at K≥6.
- *Conflicts:* #29-01 / #43 (it keeps the i8 grid).

**B2. Per-K codebook selection.** Gaussian-shaped variant at K ≥ 5, mul1 below. Supported by E3b's ×0.97 at K5–6; matters only for sizes ≥ 4 bpw. Per-level; CPU re-check, then a 0.8B K5 arm. Expected ≈0% at 2.5 bpw and −1 to −2% at 5–6 bpw.

**B3. Byte-count variants.** Closed by measurement: 2-, 3-, 6- and 8-byte sums are all equal or worse (×1.003–1.16). mul1's 4-byte sum is the right choice.

**B4. Multiplier search.** Other multipliers measured 1.6–9% worse, so 0x83DCD12D is already good. A brute-force search over ~10⁴ odd constants (64 Gaussian tiles each, K2, CPU-days) might find 0.3–0.8%. Zero inference cost (a constant behind a codebook flag). Low expected gain.

**B5. Per-tile codebook selection (salt bits).** Each tile stores 1–2 bits choosing an XOR salt applied to the state before the hash. The quantizer runs N Viterbis per tile and keeps the best. Decode cost is one XOR plus a side-plane read.
- *Source:* this session (best-of-2 ×0.985, best-of-4 ×0.974 at K2/K3, Gaussian tiles); AAAC (2605.08692) used hidden-bit table selection at group level. New: selection at the trellis-tile level.
- *Why:* per-tile error spread is 9–10% (correlation 0.93–0.95 between variants). Selection buys 2.4–2.8× more than spending the same bits on rate: at equal bpw, −1.0% net (2 variants) / −1.5% net (4).
- *Fit:* format change (salt plane: 2 bits per 256 weights = 0.008 bpw); quantizer ×N time.
- *Test:* CPU on real LDLQ tiles (does the spread survive LDLQ targets?). Pass: best-of-4 ≤ 0.98 at K2.5. Then 0.8B arm.
- *Expected:* −1 to −1.5% KL at 2.5 bpw. Risk: 4× trellis time per level; the side plane breaks the 16K-word tile alignment unless packed separately.
- *Conflicts:* #41 (rotation seeds) is dead because tensor-level averaging removes the spread; this works at tile level.

**B5b. Salt selection by the true block metric.** Pick the salt minimizing ηᵀD_bη (the full 16×16 within-block metric), or the H_out-weighted error, instead of Euclidean MSE. This captures the within-block correlation (A/S up to 1.155) at no extra cost beyond B5. Expected on top of B5: −0.2 to −0.5% on attention/GDN. Same test with the metric swapped.

**B6. Salt bits only where they pay.** Offer B5 as a per-tensor knapsack option (+0.008 bpw). Per-level, cheap once B5 exists. Expected: B5's gain at a fraction of its bytes.

**B7. Tail-extended codebook for LDLQ overload.** Map the extreme ~0.1% of bytesum values to ±4–4.5σ, so compensated targets beyond the codebook range (max|x| reached 4.1 in this session's run) are not clipped.
- *Source:* new; motivated by the drift hypothesis behind the ordering reversal.
- *Fit:* codebook flag.
- *Test:* CPU: measure the fraction of targets beyond ±3.46·scale per block, then sweep the tail mapping. Pass: ≥0.5% at K2.
- *Expected:* 0 to −0.8% KL. Risk: it costs codebook density in the bulk.

**B8. Antipodal (sign-symmetric) codebook**, cb(s ⊕ m) = −cb(s). Low priority, expected ≈0%. CPU Gaussian test only.

**B9. Position-salted (time-varying) code**: salt the hash by position-in-tile. TCQ theory says ≈0 on Gaussian sources; one CPU run settles it.

**B10. Small trained LUT (HYB-style, 4K int8 entries plus a computed part).** QTIP shows RPTC (random LUT) ≈ 1MAD (0.068 vs 0.069), so training buys nothing on Gaussianized weights. Rejected unless D-theme work leaves non-Gaussian residuals.

**B11. int8-native K7/K8 code** (IQR-V3-style 6-bit-byte sums, exact int8). This is a speed item: it would let K7/K8 run on the int8 path (today ×1.25 / ×4.5 error after rounding). Not a 2.5 bpw item; V3 stops below 7 bpw.

**B12. Per-row codebook choice via the output-scale sign bit.** Rejected by the variance argument: a row spans ~320 tiles, so the min-of-2 gain is ~1/√320 of B5's.

## C. LDLQ order and within-block structure

**C1. Real-quantizer LDLQ block-order test.** Process 128-blocks in natural, energy-descending (GPTQ act-order), energy-ascending and random orders. This changes only the quantizer's LDL, not the format.
- *Source:* XOR-Trellis energy ordering (8.06 → 8.00); GPTQ act-order; this session (the proxy predicted ascending −4..−12%; the real trellis on L3 q_proj gave ascending ×3.4 worse).
- *Why:* the 128-RHT leaves 800–4,800× diagonal spread across blocks, so order matters a lot either way. Descending may be the real winner.
- *Fit:* per-level; zero format change. Keep the best order per tensor by *measured* proxy, so it can never be worse.
- *Test:* fix the harness (L1), then trellis.py with a permutation option over 4 orders × 16 tensors × K2/3, including the g_scale search. Pass: ≥2% proxy on any family. Then 0.8B arm.
- *Expected:* −0.5 to −2% KL (sign unknown until tested). Risk: order interacts with the scale search.
- *Conflicts:* "column order" is in the BACKLOG quantizer knobs, never run.

**C2. Drift-aware ordering proxy.** Predict per-block error as D(R)·Var(compensated target_b)·tr(D_b) instead of tr(D_b). Target variance comes from L and the feedback recursion. This lets orders, rotations and extraction be ranked without running Viterbi.
- *Source:* new; fixes the failure in §0.
- *Fit:* tool.
- *Test:* calibrate against the C1 real runs; pass if rank correlation ≥0.8 across orders and tensors.
- *Expected:* enables C1/C4/D/E screening at near-zero cost.

**C3. Per-tensor best-of-orders as library variants.** Quantize each tensor in 2–3 orders and keep the lowest proxy, recording the order in provenance only (no format effect). Per-level; ×2–3 quant time on attention/GDN/out tensors only (~25% of bytes). Expected: C1's gain without choosing globally.

**C4. Coordinate permutation in the rotated domain.** Store a per-tensor permutation of the k coordinates after the Hadamard. The activation prep pass writes the rotated activations in that order; siblings share it because it is derived from the shared H.
- *Source:* this session (coordinate sort proxy: down L62 0.928, ssm_out 0.961, where the 128-block order gave only 0.993–0.985); new as a format feature.
- *Why:* regroups coordinates across Hadamard blocks, kills the within-block rank-1 coherence (A/S → ~1.00), and allows any order.
- *Fit:* format change (prep-pass scatter, 2 bytes per input channel).
- *Test:* after C2/L1, real trellis on down/ssm_out. Pass: ≥2%.
- *Expected:* −0.5 to −1.5% KL. Risk: A8-g128 group scales must be taken over the permuted order (grouping similar-energy coordinates may even help A8); same drift caveat.

**C5. Hadamard-block interleave.** A fixed stride permutation, so each LDLQ block takes one coordinate from each of 16 Hadamard blocks.
- *Source:* this session (proxy 0.966–0.986 on attention/GDN/out; A/S → 1.00–1.06).
- *Fit:* same mechanism as C4 with a fixed permutation (no index plane).
- *Test:* real trellis, natural order otherwise.
- *Expected:* −0.3 to −1%. Risk: it mixes energies inside A8 groups, which is bad for int8 activations, so only for f16-mode tensors.

**C6. Intra-block re-Viterbi.** After a block's tile Viterbi, re-run each tile once with targets corrected by the off-diagonal part of D_b, using the other positions' decisions. That is a coordinate-descent pass on the full 16×16 metric.
- *Source:* MPQ's block CD (2610.32127).
- *Why:* A/S 1.02–1.155 on attention/GDN/out inputs is within-block correlation that the Euclidean search ignores.
- *Fit:* per-level; quantizer ×2.
- *Test:* CPU on L3/L43/L10 tensors; pass ≥1% proxy.
- *Expected:* −0.3 to −1% KL. Risk: may converge to the same path.

**C7. Post-LDLQ tile coordinate descent (whole tensor).** After LDLQ, sweep the tiles 1–3 times. Each tile is re-Viterbi'd against the exact target given all other tiles' current errors, W_b + (E_¬b H_¬b,b)H_bb⁻¹, and accepted only if the true proxy drops.
- *Source:* MPQ (LDLQ+CD on Llama-3-8B, E8P 2-bit: 15.01 → 12.76 ppl); JARQ (accept-if-better refinement).
- *Why:* LDLQ is greedy; CD fixes early decisions with later information.
- *Fit:* per-level; quantizer +1–3 passes (~1.5–2× time).
- *Test:* CPU real tensors K2/K2.5, 1 and 3 sweeps. Pass: ≥2% proxy at K2.5. Then a 0.8B arm.
- *Expected:* −0.5 to −2.5% KL. Risk: trellis tiles have less freedom than E8 blocks; gains shrink at K3+.

**C8. List-Viterbi plus full-metric rescoring.** Keep the top-8 paths per tile and choose the combination minimizing the full D_b and H_out metric. A cheaper relative of C6/C7. Expected −0.2 to −0.8%.

**C9. Error-feedback decay** (QTEA's γ_i). QTEA gained only +0.46 avg on Qwen3-14B ternary and was noisy on Llama. One CPU λ sweep, expected ≈0. Low priority.

**C10. CurveTQ weighted branch metric.** Closed by this session's Γ_b = 0.0001–0.0007 bit after the 128-RHT. Revisit only with H2 (output side, native basis).

## D. Rotation and incoherence

**D1. Two-level Kronecker mixing (128-RHT, then cross-block mixing).** Apply a second orthogonal mixing across corresponding coordinates of the 40–136 Hadamard blocks: a small dense Haar/Hadamard over blocks, Q_k ⊗ I_128. Weights are stored in the new basis; the prep pass applies it.
- *Source:* SoloQ K-RPBH (2610.07121: block energy imbalance 67.9% → 22.1%, KL-to-Gaussian 0.114 → 0.0041); the RHT theory paper (2605.06014: two RHTs give Gaussian marginals, a third gives block decorrelation); this session (two-level proxy 0.925–0.973 on attention/GDN/out).
- *Why:* the rank-1 coherence of the massive channel inside one 128-block (A/S 1.04–1.155).
- *Fit:* format change (prep pass + k-side basis; ~40–136 FMAs per activation element, shared by siblings).
- *Test:* real trellis, natural order, then 0.8B.
- *Expected:* −0.5 to −1.5% KL. Risk: prep cost at decode M=1 (small); the iid-proxy caveat.

**D2. 512-point input Hadamard.** k = 5120, 6144 and 17408 are all multiples of 512.
- *Source:* this session (proxy 0.955–0.968 on attention/GDN/ssm_out, 0.981 o_proj, 0.996–0.998 MLP).
- *Why:* same mechanism as D1, simpler (4 more butterfly stages).
- *Fit:* format change (prep transform size per tensor; A8 groups can stay 128).
- *Test:* real trellis on 16 tensors at K2.5; 0.8B arm (dims 1024/2048 allow it).
- *Expected:* −0.4 to −1% KL. Risk: little stacking with C-ordering (r512+sort only 0–2% beyond sort in the proxy).

**D3. Householder spike-spreading.** One reflection P = I − 2uuᵀ maps the massive channel e_j onto a flat vector across *all* k coordinates before the block Hadamard. Cost: one dot product and one axpy per token in prep (2k FLOPs); r reflections handle r outlier channels.
- *Source:* new (it targets exactly the coherence §0 found; cheaper than D1/D2).
- *Why:* the outlier holds 2,800–4,800× mean diag in attention/GDN inputs; spreading it over 5120 instead of 128 coordinates removes the within-block rank-1 structure.
- *Fit:* format change (prep pass; per-tensor u is just j plus a sign vector; siblings share it).
- *Test:* H-proxy A first (minutes; compare with E1's 0.86–0.97), then real trellis.
- *Expected:* −0.5 to −1.5% KL (attention/GDN inputs ~20% of KLD). Risk: same iid caveat; activation range for A8 after spreading must be checked (R1's lesson, #47).
- *Conflicts:* #47 (R1) failed because norm gains had to be folded in; here nothing is folded and the transform is local to the GEMM input.

**D4. Variable Hadamard size by block.** Use 512/1024 only for the block(s) containing the outlier channel and 128 elsewhere. It is a prep-pass size map: proxy-test after D3; expected ≤ D3's gain at lower prep cost.

**D5. Output-side rotation size with H_out.** Test 128 vs 512 vs no output Hadamard, now that diag H_out is in the quantizer.
- *Source:* #33-12; CurveTQ's substitution argument.
- *Fit:* format change.
- *Test:* 0.8B: three arms on bdiag settings.
- *Expected:* 0 to −1%.
- *Conflicts:* #48 (auto output scales off was worse); here scales stay on.

**D6. Learned butterfly rotation (HARP).** On QTIP 2-bit Llama-2-7B it gave 6.87 → 6.63 ppl for +0.05 bpw; 3-bit 5.41 → 5.37. Costs 9.7 h per 7B. Format change, training-like. Expected −0.5 to −1% at 2 bpw, ~0 at 3. Low priority on cost.

**D7. Embedding row rotation size (128 vs 512 within a row).** CPU `embtrellis.py` arm. Expected ≤0.1% KL.

**D8. Null-space shrink before the trellis.** Remove (or ridge-shrink) the component of W in the bottom-γ input-Hessian eigenspace before RMS scaling, so codebook range is not spent on directions that do not matter.
- *Source:* OSAQ (2605.04738; small gain on QuIP: 6.50 → 6.37); new for trellis plus RMS scaling.
- *Fit:* per-level; zero inference cost.
- *Test:* CPU diagnostic: fraction of row energy in the bottom eigenspace; if >3%, run a trellis A/B.
- *Expected:* 0 to −0.8%. Risk: the FP16 change is not exactly loss-free.

## E. Outlier handling

**E1. Massive input-channel extraction.** Store the top 1–4 input channels as dense int8/fp16 columns outside the trellis. The prep pass zeroes x_j before the Hadamard; the epilogue adds W_j·x_j.
- *Source:* TORQUE pre-rotation retention (2609.36032), with a Hessian criterion; this session (proxy 0.864 L3, 0.965 L43/L10, ≈1.0 MLP/out-projections, +0.003 bpw per channel).
- *Why:* the same coherence as D3, removed exactly.
- *Fit:* format change (one rank-1 epilogue FMA, an extraction in prep; siblings share j).
- *Test:* real trellis on attention/GDN tensors. Pass: ≥2% after paying 0.003 bpw.
- *Expected:* −0.4 to −1% KL. Risk: overlaps D3; pick the cheaper one.

**E2. Outlier channel splitting.** Duplicate x_j into m positions in different Hadamard blocks, with the weight column divided by m. A sparse variant of D3; the k dimension grows by m−1. Proxy test only if D3 fails.

**E3. Outlier output-row extraction for residual writers.** Code rows with extreme diag H_out (the massive residual channel written by down/o_proj/ssm_out) separately.
- *Source:* new; uses #33's diagonals.
- *Test:* 0.8B diag ranking; proxy with H_out.
- *Expected:* 0 to −0.5%.

**E4. Outlier block at higher K.** Place the outlier's Hadamard block in a dedicated 16-row k-block at a higher fractional K. Needs F1. Expected ≈ E1, without the dense side path.

**E5. Sparse residual on hot tiles** (MosaicQuant-style second low-K pass). Rejected: residual ladders lose 0.3–0.9 dB against a direct code (GateRQ 2610.35985; RRQ 2608.04048).

## F. Rate granularity inside a tensor

**F1. Column-split K (per k-block K).** 128-row k-blocks get their own fractional K by water-filling over tr(D_b).
- *Source:* this session's Γ_cross: natural order 0.074 bit (L3 attention), 0.072 (o_proj), 0.029 (ssm_out), 0.019–0.022 (GDN/L43), 0.007–0.009 (MLP). That is a 3–10% proxy on attention/out tensors.
- *Fit:* format change (per-k-block word count and shift pattern; the head-slab offset machinery transposed to the k side); per-level variants.
- *Test:* CPU real trellis with 2-level K plans (e.g. K2.25/K3 split at equal bytes). Pass: ≥2% on attention/out.
- *Expected:* −0.5 to −1.5% KL. Risk: water-filling with feedback interacts with drift (§0); a sorted order would remove the gain (Γ_cross → 0.002–0.018), so it is a substitute for C1.
- *Conflicts:* #29-02 (E3c's row-slab ceiling) was output-side and input-blind; this is input-side.

**F2. Row-slab K for the trunk with diag H_out (revisit E3c).** 128-row output slabs aligned with output Hadamard blocks get K by water-filling over slab-mean h.
- *Source:* #33-09 (diag H_out CV 1.5–4 for input projections, ~0.2 for residual writers); the head-slab infrastructure (#34).
- *Fit:* format change (exists for the head); per-level.
- *Test:* CPU from 0.8B diagonals: ½log2(AM/GM) per tensor (minutes). If ≥0.05 bit, run a 0.8B arm.
- *Expected:* 0 to −1.5% KL. Risk: output Hadamard averaging may flatten slab means.
- *Conflicts:* "row-slab K dropped for the trunk" (design §6b) because H_out hurt then; the diagonal now helps (#33-12).

**F3. Joint row × column slab plans.** Only if both F1 and F2 pass; bookkeeping-heavy.

**F4. K finer than quarter steps.** Closed by #45 (≤0.8%).

**F5. Per-tile K map.** Too costly in metadata and kernel for an expected ≤ F1 gain. Rejected.

## G. Scales and side information

**G1. Rotated-domain per-block input scales chosen inside LDLQ (drift compensation).** For each 16-row k-block, pick a scale for the compensated targets (3–5 candidates by proxy) and store it as a rotated-domain input scale, k fp16 values per tensor (10 KB). The activation prep multiplies by it for free.
- *Source:* new. EXL3 has one global scale plus a global `ldlq_drift` factor; the §0 reversal points to per-block overload.
- *Why:* blocks quantized late carry inflated targets; one global scale cannot fit all blocks.
- *Fit:* format change in prep only (extra per-coordinate scale); per-level.
- *Test:* CPU real trellis natural vs G1, K2/K2.5. Pass: ≥1%. Also re-run C1 with G1 on (it may unlock ascending orders).
- *Expected:* −0.5 to −2% KL. Risk: interacts with the input-scale refit.

**G2. Alternating scale refit ↔ re-Viterbi (2–3 rounds).** Refit already exists (11% of quant time); add rounds.
- *Source:* JARQ (the joint scale fit leaves a 21–25% gap when groups are refit independently); CoS-VQ (same-start joint index+scale: Qwen3-4B Wiki2 21.0 → 13.7).
- *Fit:* per-level.
- *Test:* CPU two rounds; pass ≥0.5%.
- *Expected:* −0.2 to −0.8%. Risk: EXL3's rank-1 scales leave less to gain than group scales.

**G3. Hessian-weighted global-scale scoring.** Score scale candidates by the LDLQ-weighted Schur metric δᵀS_{G|P}δ (Schur Replay 2609.36654) or the full proxy, instead of the plain MSE on sampled tiles that `g_scale_gss` appears to use. Per-level, CPU A/B. Expected −0.1 to −0.5%.

**G4. Per-tile gain scales.** Dead (#40: raw gain 1.3–1.8%, groups lose 8–20% at equal bits).

**G5. K-specific `ldlq_drift` factor tuning.** Sweep the global drift factor per K and fractional pattern. Per-level, cheap CPU. Expected 0 to −0.3%.
- *Conflicts:* #39 (gain vs K, never logged on GPU).

## H. Output side (H_out) inside the code

**H1. Output-side LDLQ ordering for two-sided quantization.** With the rotated H_out = Had·diag(h)·Had, the output-side LDL order matters as on the input side (same drift caveat).
- *Source:* this session's input-side finding; #33-12.
- *Fit:* per-level; zero format change.
- *Test:* 0.8B: bdiag arms with natural/ascending/descending output order, paired KL (~5 min per arm).
- *Expected:* 0 to −1%.

**H2. Native output basis plus a diag-H_out weighted Viterbi.** For input projections, drop the output Hadamard and weight the Viterbi per output row by h_r. This is CurveTQ's "weighting substitutes for rotation", applied where anisotropy is large: CV 1.5–4 means a native-basis Γ of ~0.3–0.6 bit.
- *Source:* CurveTQ (2610.09212), XOR-Trellis (2610.00432); new on the output axis.
- *Fit:* format change (per-tensor output-rotation flag); per-level.
- *Test:* 0.8B: compute output Γ_b from the diagonals (minutes), then arms vs bdiag.
- *Expected:* −0.5 to −2% KL, high variance. Risk: rows lose incoherence. Diag-H_out output scales already failed (#33-06, gate 2d +4–15%); this is rounding weights, not scales.

**H3. Output-side A/S diagnostic.** Does the output 128-Hadamard create within-block correlation in H_out' as the input side does? Minutes on 0.8B diagonals; decides H1/H2/D5.

## I. Embedding codes

**I1. Per-tile salt on the trellis embedding.** B5 applied to embedding rows: tiles decode independently, and the salt plane is gathered with the row. Expected ≈ B5's −1% of embedding error, about −0.05% KL. Do only if B5 ships.

**I2. Gaussianized codebook for the embedding** (B1). Free once B1 exists; ~−0.01% KL.

**I3. Embedding row Hadamard size** (see D7).

## J. Head codes

**J1. Softmax-shift search beyond the mean row.** Mean-row subtraction (t=1) already gave −30% head KL (#34). Search t ∈ {0.5, 1, 1.5, 2, 4} by headsim on held-out head inputs.
- *Source:* 2609.31291 (Phi picked t=4; Qwen3.5 with GPTQ ≈0).
- *Fit:* per-level head variant; free (the head is linear).
- *Test:* headsim, minutes.
- *Expected:* 0 to −0.3% total KL.

**J2. Group-wise shift per slab.** Coordinate search per 128-row slab, as in that paper's appendix; headsim. Expected 0 to −0.2%.

**J3. λ̄-weighted rounding inside head slabs.** The head's output Hessian is exactly diag(p(1−p)) (SoftWater 2608.12026), so two-sided LDLQ with that diagonal needs no backward pass. Within Fisher-sorted slabs the spread is small. Per-level; headsim. Expected −5..−15% head KL, i.e. −0.3 to −0.8% total.

**J4. Low-rank residual for the head (ARCHead).** Rank ~6 in the C^0.75 metric. Against a Hessian-aware INT4 head it gave 1.0058 vs 1.0104 relative ppl. Format change (a rank-r branch on the head GEMM, negligible in prefill since only the last token runs it). Expected −0.2 to −0.5% total. Risk: decode bytes.

## K. QAD-compatible code features

**K1. Rotated-domain scale healing.** Freeze trellis indices and train per-coordinate rotated-domain input scales (G1's vector) plus output scales by KL to bf16. That is k+n parameters per tensor, folded into prep and epilogue for free.
- *Source:* RoleQ scale distillation (2610.21808); ThinkBit (KL 0.641 → 0.282 from scales alone); Tetra (+1.7–3.3 MMLU, 0 bytes); SQuAT (gain LR ≈1000× the shape LR).
- *Fit:* training (V5), final pass on shipped mixes.
- *Test:* 0.8B: freeze K2.5 codes, 50–100M tokens of qad-pool KL, paired KL on search slices.
- *Expected:* −3 to −8% KL at 2.5 bpw (plan-level number, not code-specific). Risk: overfitting the qad pool.

**K2. STE latent QAD with periodic Viterbi refresh** (TVQ 2610.41822; E8-QAT 2610.27699). TVQ: ≤0.2 ppl over QTIP at 8× cost. Check the drift budget (BCJR-QAT 2605.10655: lr × steps must exceed the Voronoi radius, or codes never move). Training. Expected −1 to −3% after K1. Risk: high cost.

**K3. Discrete healing moves.** Re-Viterbi selected tiles with KL-gradient-adjusted targets, or re-choose B5 salts, and accept on held-out KL (TWIG 2610.31931). TWIG's own ablation found continuous scales better (KL 0.325 vs 0.371). Do after K1. Expected 0 to −1%.

**K4. Gain/shape STE split per tile.** Separate STEs for the tile gain and the direction (SQuAT 2610.31412: 0.1–0.4 ppl over a joint STE). Only matters if K2 runs.

## L. Adjacent quantizer ideas that sit on the code

**L-a. Activation-denoising target and metric (2610.07522).** Feed LDLQ the pretransformed WF and metric G. On W2 QuIP# E8 it closed 18–38% of the parallel→sequential gap. It needs one noisy forward per layer (C, N statistics). Per-level. Test on 0.8B arms. Expected −1 to −3% at 2.5 bpw.

**L-b. Jacobian-weighted input Hessian.** H = Σ‖J_t‖²x_t x_tᵀ from one logit-probe backward (2610.18139). Choose α per tensor by proxy only (a fixed α is unsafe).
- *Conflicts:* #33-07 (gradient-weighted H_in +31..+44%); the difference is a blend with α chosen per tensor, so do it last.

---

## Ranked top 20 by expected gain per unit cost

| rank | idea | expected KL at 2.5 bpw | cost |
|---|---|---|---|
| 1 | **C1 LDLQ block-order test** (incl. descending) | −0.5..−2%, sign unknown | CPU day; zero format |
| 2 | **A1 multi-tile tail-biting rings** | −1.5..−2.5% | CPU day + ~2 kernel days |
| 3 | **G1 per-block rotated-domain scales in LDLQ** | −0.5..−2% | CPU day; prep multiply |
| 4 | **C7 post-LDLQ tile coordinate descent** | −0.5..−2.5% | CPU day; +1–2× quant time |
| 5 | **B5 per-tile salt selection** (+B5b metric) | −1..−1.5% net | CPU day; small kernel; ×4 trellis time |
| 6 | **D3 Householder spike-spreading** (or E1) | −0.5..−1.5% | CPU hours; prep axpy |
| 7 | **H3→H2 output-side native weighting** | −0.5..−2% (high variance) | 0.8B, hours |
| 8 | **F2 trunk row-slab K from diag H_out** | 0..−1.5% | CPU minutes, then 0.8B |
| 9 | **D2 512-point input Hadamard** | −0.4..−1% | CPU + prep change |
| 10 | **B1 Gaussianized mul1** | −0.3..−0.8% | CPU + 2 FMAs decode |
| 11 | **F1 column-split K** | −0.5..−1.5% | format change |
| 12 | **C6 intra-block re-Viterbi** | −0.3..−1% | quantizer ×2 |
| 13 | **H1 output-side ordering** | 0..−1% | 0.8B arms |
| 14 | **J3 λ̄-weighted head rounding** | −0.3..−0.8% | headsim |
| 15 | **A5 edge-emitting window** | −0.5..−1% | CPU + decode change |
| 16 | **J1 head shift search** | 0..−0.3% | minutes |
| 17 | **D1 two-level Kronecker mixing** | −0.5..−1.5% | prep change |
| 18 | **A4 L=18/20** | −0.5..−1.5% | quant time ×4–16 |
| 19 | **G2 refit↔re-Viterbi rounds** | −0.2..−0.8% | quant time |
| 20 | **D8 null-space shrink** | 0..−0.8% | CPU |

Four groups substitute for each other: C1/C4/C5, D1/D2/D3/E1, F1 vs ordering, and B5 vs A2. Expect the realized total of the top 20 to be well below their sum, likely −3..−6% KL at 2.5 bpw before K1 (healing).

## The 5 cheapest diagnostics that would most change this ranking

1. **Fix the real-trellis harness, then run orders × rotations × extraction on ~16 tensors.** The hang is the torch/Viterbi OpenMP clash and the stalled numpy runs. Use the same 16 tensors for all arms: natural / ascending / descending / random block order; r128 / r512 / Householder; top-1 extraction; K2/K2.5 with the scale search. This decides C1–C5, D1–D4, E1 and F1, and checks the one real data point (ascending ×3.4 worse). CPU boxes, ~1 day.
2. **Drift and overload profile.** Per-block RMS of compensated targets and the fraction beyond the codebook range, natural vs sorted, K2–3. It explains the proxy failure and decides G1, B7 and C2. CPU, an hour.
3. **Per-tile salt spread on real LDLQ tiles.** Does best-of-4 ×0.974 survive LDLQ targets and int8 rounding? Decides B5/B6/I1. CPU, an hour.
4. **Tail-biting length sweep on real tiles** (256 / 512 / 1,024 rings, plus the partial-start-bits curve). Decides A1–A3. CPU, an hour.
5. **Output-side structure from the 0.8B diag H_out.** Per-tile and per-slab AM/GM of h, output A/S after the output Hadamard, and native Γ_b. Decides H1–H3, F2 and D5. Minutes on the 4070's saved diagonals.
