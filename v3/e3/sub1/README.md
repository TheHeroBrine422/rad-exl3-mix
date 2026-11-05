# #90 · Sub-1 K levels (K0.125–0.875) on real 27B tensors, CPU trellis proxy

Decision (user, 2026-11-04): sub-1 K is in V3. This measures the proxy error curve the knapsack needs.

Setup: `trellis2.py` (block LDLQ + bitshift trellis, salted 0-bit read for KA = 0), codebook cubic0.02_i8 (i8c, the
adopted codebook), K 0.125 … 0.875 at 1/8 steps plus 1, 1.25, 1.5, 2 as anchors to the existing menu, 128 sampled
rows, rotation seed 0, global scale factors {0.5, 0.7, 0.85, 1, 1.15} (below K1 the best reconstruction scale
shrinks, so the default {1, 1.15, 1.3} would overstate the error). Tensors: GDN layers 2 / 10 / 30 / 50 / 62 × ssm_qkv, ssm_z, ssm_out, ffn_gate_up, ffn_down;
attention layers 3 / 19 / 43 / 63 × attn_qg, attn_k, attn_v, attn_output, ffn_gate_up, ffn_down (49 tensors). cpu2
and cpu1, half each (`run.sh`), one thread a job.

Pass rules (written 2026-11-04 ~3:05 PM CST, before data):
- The curve is usable for pricing if rel(K) falls monotonically at every 1/8 step for every tensor (noise allowance:
  a step may be flat within 1%), and the K1 anchor agrees with the existing fracsweep / ladder K1 within 10%.
- Sub-1 is worth its kernel work at a size if knap4 (sub-1 options priced as rel(K)/rel(K1) × its K1 price, family
  ratios from here) gives ≥ 3% lower predicted KLD than the same menu without sub-1 at that size. Expected: yes at
  1.25–1.75, unclear at 2.0–2.5. (Decision already made by the user; this sizes the gain per bpw.)
- Runtime cost: decode of a 0-bit step = one salted codebook read (P+D, small); measured with the kernel pass.

## #90-02 · int4-grid codebook cost by K (2026-11-04 ~4:00 PM CST, rule before data)
Question (user): does V3 need an i4 ladder K0.125–3.5 next to the i8 one? `i4run.sh`: 10 tensors, int4grid vs mul1_i8
at K0.125 … 3.5, paired. Rule: below the highest K where i4 / i8 ≤ 1.01 on the parameter-weighted mean (and no tensor
> 1.03), the i4 ladder **replaces** the i8 one (one quant serves f16, A8 and A4); between that K and K3 both are
built (only if d1106 passes); above K3 i8 only. (mul1-based grids: the adopted cubic i8c codebook has no int4 twin yet;
the ratio is what transfers.)

## #94 · int8-rounded codebook above K7 (i8 K7.25 / K7.5 / K7.75?) (2026-11-05 ~9:00 PM CST, before data)
Question (user): the int8-rounded codebook costs ×1.02 at K7 and ×1.35 at K8 (GPU smoke, whole K only). Are i8 levels
at K7.25 / 7.5 / 7.75 worth adding to the ladder (the fast 7.0 / 8.0 keep K > 7 tensors on the int8 path)?
- Setup: `i8hi.sh`, `trellis2.py` mul1 vs mul1_i8, K 6.5, 7, 7.25, 7.5, 7.75, 8, the 10 tensors of #90-02, 128 rows,
  same seed / factors; ratio = proxy error mul1_i8 / mul1 per tensor, median over tensors.
- Rule: an i8 level K7.25 / 7.5 / 7.75 joins the ladder if its median ratio is **≤ 1.10** and its 90th percentile
  ≤ 1.20. Sanity: K7 and K8 must reproduce the GPU smoke within ±0.05 (else the CPU proxy is not trusted here).

## #95 · IQR-V4 ideas on the CPU proxy: act order; bf16 (row, 128) scales from K5 (2026-11-05 ~9:10 PM CST, before data)
- **95-01 act order:** inside each 128-column Hadamard block the 8 16-column sub-blocks quantized in descending
  order of their H_in diagonal mass (EXL3-compatible: a permutation at 16-column granularity, applied to the stored
  codes' column order and undone at load, no runtime cost) vs natural order. Same tensors as #90-02 at K2, 2.5, 3, 4,
  5; proxy error ratio per tensor. **Rule:** a 0.8B KL test (then 27B) if the median ratio ≤ 0.99 at ≥ 2 of the 5 K
  and > 1.005 at none.
- **95-02 bf16 (row, 128) scales** on attn_output / ssm_out (and, reported only, down_proj): one bf16 scale per row
  and 128-column block on top of EXL3's row × column scales, refit after the trellis (closed form per block), at K5,
  K6 (+ K4 reported), cost 0.125 bit per weight on those tensors (16 bits / 128). **Rule:** worth a 0.8B KL test if
  the median proxy error ratio, net of its bits (compare at equal total bits against the next quarter step), is
  ≤ 0.97 on attn_output and ssm_out at K5 or K6.

## #96 · Trunk row-slab K with output-side row importance (2026-11-05 ~9:20 PM CST, before data)
E3c (#29-02) put the row-slab ceiling at ~0.05 bit from input-side row norms alone; output-side importance (the 27B
Σdy² diagonal, block-normalized 1,920-sequence collection) was missing. Slabs of 128 rows (= EXL3's output Hadamard
blocks, so a slab's Σdy² mass is rotation-invariant), also 256 / 1024 reported.
- **Stage 1 (analytic, `rowslab2.py`, every trunk tensor):** per slab weight w_s = Σ_{r∈s} h_r · ref_r (ref_r = W_r H_in
  W_rᵀ), continuous-bit ceiling gain = ½ log₂(AM(w)/GM(w)) bits, vs E3c's input-only number, per family.
  **Rule:** stage 2 if the parameter-weighted median ceiling at S = 128 is **≥ 0.10 bit**.
- **Stage 2 (real trellis, quarter-step K per slab, K2–4):** the gain at equal bits under Σdy²-weighted proxy error.
  **Rule:** a format / kernel design note (K per slab) if ≥ 0.06 bit (≈ ≥ 4% KLD at 2–4 bpw) on the big families.

## #97 · Embedding menu at both ends (2026-11-05, before data)
7.0 is a V3 size and the solver already sits on t6 (the menu top) at 7.0; sub-1 sizes may sit on the menu floor.
CPU sweep as `v3/e3/emb/` (trellis-coded embedding, one scale per row, H = I): t6.25 … t8 in quarter steps and t0.25 /
0.5 / 0.75; reconstruction error per level and the error-per-bit slope. **Rule:** a level joins the menu if it lies on
the error-vs-bits convex hull with the existing levels and Lloyd e1–e8 (else it is dominated and skipped).

## #90-03 · Firmer sub-1 / i4 curves: second seed and new depths (2026-11-05 ~12:00 AM CST, before data)
Replication for pricing (no new decision): (a) the 49 #90 tensors at rotation seed 1 + row seed 1, K0.125–2 as #90;
(b) i4 vs i8 at K0.125–3.5 on 10 new tensors (layers 6 / 26 / 46 / 58 / 39, GDN and attention families). Output
`sub1-s1.jsonl`, `i4b.jsonl`. Reported: per family, the seed-to-seed spread of rel(K)/rel(K1) and whether #90 /
#90-02's verdicts change when the data are pooled (if a verdict flips, the pooled one is used and the flip is logged).

## #100 · IQR-style hash codebook for EXL3 trellis codes (2026-11-05 ~2:15 AM CST, before data; decide before the ladder)
Why (user): switching codebooks after the ladder costs a ~200 GPU-hour requant, and decode speed matters. IQR-V4's
computed value: `__umul24(state, A) + B`, mask to 6 bits a byte, one `sudot4` = byte sum + bias (`libr4d/r4d_iqr_dec.h`,
~3 VALU a value incl. the hash, int8-native), vs EXL3 mul1i8 (QTIP 1MAD: full 32-bit multiply, 4-byte sum, scale,
int8 round; ~13 VALU today, ~4–5 with the closed form).
- **Quality (CPU proxy, `trellis2.py` + a new codebook option, colo):** the 10 #90-02 tensors + 10 #90-03 ones, K1–7
  (1/2 steps) and K0.25 / 0.5, the same rows / seeds / scale grid (to 1.5), ratio = proxy error hash / mul1i8.
  **Rule:** adopt for V3 if the parameter-weighted ratio is ≤ 1.005 at every K in 1–6 and no tensor > 1.02; reject if
  > 1.01 at any K in 2–5; in between: a 0.8B KL check (K2 / K3 / K4, 2 seeds) decides at ≤ 1.01.
- **Speed (before adopting):** ops a value in our decode pass / GEMV, and a microbench on the first free card.
- Also needed if adopted: the quantizer side (exllamav3 codebook compile option as for i8 / i4, exl3cb), a new
  encoding tag (`cb=iqrh`), the plugin decode paths. Runtime cost: decode and prefill (faster); quality per the rule.
- **#100 amendment (2026-11-05 ~2:30 AM CST, before data; user):** a quality cost above +1% may still be worth a large
  decode gain (e.g. +30%). So: ≤ 1.005 everywhere in K1–6 → adopt; otherwise **no automatic reject** below ×1.05: the
  decision waits for the measured speed (GEMV at M = 1 / 4 / 16 and the prefill decode pass, on a card) and is the
  user's, from a trade-off table: KL cost per K as an equal-quality bpw cost (V2's KL-vs-size slope) next to the
  measured decode and prefill speedups. Above ×1.05 at any K in 2–5: rejected.
