# cb208 results — codebook decode cost on gfx1201 (w208, 2026-11-20)

Card: **dad1, RX 9070 XT** (gfx1201, power cap 330 W, clocks as found), ROCm 7.16 / AMD clang 24 (`/opt/rocm/core-10.1`).
Confirmed on **box2 card 2, R9700** (power cap 210 W): same ranking, factors a little higher (section "box2").
Method, candidate definitions, exactness and timing protocol: `README.md`. Tables below are `report.py results.jsonl`.

## Findings

Expert = one gate_up (5120 → 4608) + one down (2304 → 5120) call. Factor = t(candidate) / t(base), base = the same
kernel with a shift-and-mask "codebook". M = 1 factors, K = 1 / 2 / 3 / 4 (w8a8, best placement of the row's M):

| candidate | table | extra VALU / weight | K=1 | K=2 | K=3 | K=4 | M=8, K=1 / 2 |
|---|---|---|---|---|---|---|---|
@F1@

1. **Where decode shows.** The baseline GEMV is ALU-bound at K ≤ 2 and bandwidth-bound at K = 4 (base expert
   @BASE@ µs at K = 1 / 2 / 3 / 4, M = 1; K = 4 equals `raw` and moves ~500 GB/s). So every codebook costs most at
   K = 1 (mad 1.53×) and almost hides at K = 4 (mad 1.05×). Fractional K costs the same decode instructions a
   weight as integer K (asm table: mad +3.12 at K = 2 and at K = 2.5) and follows the same trend: mad 1.53 / 1.42 /
   1.24 / 1.27 / 1.14 / 1.05× at K = 1 / 1.5 / 2 / 2.5 / 3 / 4. More tokens dilute the decode (it is paid once a
   weight): mad 1.53× at M = 1 → 1.23× at M = 8 (K = 1).
2. **Ranking (cheap → dear) at M = 1, the same on both cards (neighbours can swap as M grows):** lc1 (one 24-bit multiply + 64 B table) <
   mad ≈ mad4 < cptcmin < cptc5 = cptc6-stored-in-full ≈ cptcst10 < mul1_i8 ≈ qtip_1mad < int4grid < cptcst12 <
   qtip_3inst ≈ cptc6 with its mirror fold done per weight.
3. **mad stays the cheap computed code.** Three instructions a weight (24-bit mad, and, dot-as-bytesum). The
   multiply-hash codes with 32-bit constants (mul1, QTIP 1MAD / 3INST) pay a full `v_mul_lo_u32`, which costs about
   as much as three simple instructions on this card (see the rule below): 2.1–2.4× at K = 1 against mad's 1.5×.
   **mad4** (proposed int4 form, README) costs exactly what mad costs, in both modes.
4. **CPTC, two multiplies:** cptc5 (63 entries, table in LDS) 1.84× at K = 1, 1.59× at K = 2, 1.40× at K = 3, 1.10× at
   K = 4 (M = 1; from global 1.77× at K = 1); 1.41× at M = 8. That is +0.2 to +0.35 over mad at K ≤ 3 — per weight 5 ALU instructions + one table load against
   mad's 3. On the paper's scale (0.48–0.62× on H100, 0.98× on L40S as speed ratios) the 9070 XT gives 0.54× at K = 1,
   0.63× at K = 2, 0.71× at K = 3, 0.91× at K = 4 at M = 1: H100-like at low K, as predicted.
   - **Mirror fold:** done per weight (Algorithm 1 as written) it costs 5.5 more instructions a weight (cptc6 2.44×
     at K = 1). It need not be paid: the 64 stored entries expand to a 127-entry table at kernel start (LDS) or at
     load (global), which is what `cptc6full` times — identical to cptc5. Size axis stays 64 B.
   - **Eq. (3) shift-accumulate products: no help on RDNA4.** With correct 24-bit multiplies the direct form is
     faster (cptc6 lds 1.93 vs lds+eq3 2.24 geomean): the recurrence adds a shift-add and a code extract a product
     and saves nothing, a 16 × 16 multiply is one `v_mul_u32_u24`. (It only looked better while the direct form was
     miscompiled to `v_mul_lo_u32`, see 6.)
   - **cptcmin** (Q = ⌈K⌉ + 1, no table): 1.71× at K = 1, cheaper than cptc5 by the load; w4a4 the same.
5. **Tables: placement and size.**
   - LDS vs global: equal within ~3% for the 64-byte tables in the GEMV (lds slightly ahead on geomean). Registers
     (cptc5, 8 byte-permutes + 3-level select): 2.1×, clearly worse than a load. → **LDS (or global) for every table.**
   - **The state table is not the cheap end on this card.** cptcst10 has zero decode ALU and one load a weight, yet
     costs 1.71× (glb) / 2.13× (lds) at K = 1, M = 1, more than lc1 (1.29×), which does a multiply *and* a load from
     64 bytes. cptcst12 (4 KB) 2.29×. The load's cost grows with the table: @LOC@
     So the paper's H100 result (state table well ahead of two multiplies) does not carry over: at M = 1 cptcst10
     from global is only a little ahead of cptc5 (1.71 vs 1.84 at K = 1, 1.26 vs 1.40 at K = 3), from M = 4 cptc5 is
     ahead (M = 8, K = 1: 1.41 vs 1.49), and lc1 beats both everywhere.
   - Not tried: the paper's per-lane replicated state table (conflict-free gather). It needs 32× the table in LDS
     (32 KB for L = 10), copied in by every workgroup in this kernel structure, so it would need long-lived workgroups.
   - LDS tables are copied in once a workgroup (inside every number; negligible for 64 B, part of cptcst12's cost).
6. **A codegen trap that matters for the plugin.** CPTC's products must be written so that LLVM keeps
   `v_mul_u32_u24` (README "CPTC multiplies"). The natural spellings give `v_mul_lo_u32`, and the GEMV is then
   @V1@ The same holds for the byte-sum → grid maps of mul1_i8 / int4grid / qtip_1mad (fixed with an
   explicit `& 1023`, worth ~8%).
7. **mad folded into the dot** (`madgemv`'s form) wins only at M = 1 (1.41 vs 1.53 at K = 1, 1.06 vs 1.14 at K = 3);
   at M = 2 it already loses (1.48 vs 1.34) and at M = 8 it is 2.3× (one dot a weight a token, 4× the activation
   bytes). mad4 folded: same picture.
8. **w4a4.** The int4 kernel's base equals the int8 base at M = 1 and is cheaper from M = 4 (expert @W4@ µs at
   M = 8, K = 1 / 2 / 3 / 4 against @W8@): half the dots and activation loads; the decode is unchanged, so its
   *factor* is slightly higher at M = 8 (mad4 1.33× vs 1.21× at K = 1) while absolute time is lower.
   int4grid (mul1 hash + clamp) is the dearest int4 code (2.2× at K = 1); cptcmin 1.68×; mad4 1.55×.
9. **Prefill (M = 256).** Two measurements, neither shows a codebook problem:
   - *Decode pass* (decode once to int8 / int4, then a codebook-free GEMM — the T2 structure): expert decode
     @DEC@
     The pass is write-bound: `raw` = `base`, and mad adds ≤ 3%; cptc5 +16% at K = 1 falling to +3% at K = 4.
   - *Fused* (decode inside the WMMA kernel, column M=256 below): factors 1.0–1.35, but this straightforward GEMM is
     governed by activation-fragment traffic and a configuration moves up to 12% between passes; table-from-global variants even
     read below 1.0. Read it as "decode is ≤ ~20–30% of an unoptimised fused prefill at low K", not as a ranking.

**For the frontier (ALU axis):** @RULE@

## Caveats

- The scan (row-major tiles) is the bench's own; production's `tensor_core_perm` runs pay the same decode ALU.
- Sums are int32 over the whole row with one scale a token; production's 128-column float groups would add the same
  epilogue work to every candidate (factors would shrink slightly).
- CPTC tables are Eq. (2) start tables (no Lloyd); lc1's multiplier and the state tables' cell order are placeholders.
  Time does not depend on them; quality is the lead's axis.
- The fused M = 256 GEMM is not a tuned prefill kernel (see 9).
- The decode pass reuses one output buffer; its write rate (~1 TB/s) says the writes are absorbed by the cache
  hierarchy, as they would be for a prefill scratch buffer of this size.
- Launch overhead (an empty launch is ~2.2 µs in a back-to-back batch) is inside every number, two launches an expert.

