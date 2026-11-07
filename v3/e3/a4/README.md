# #46-03 · A4 on i4 weights as a per-tensor knapsack option (fast variant, ≤ 3 bpw)

Decision (user, 2026-11-05): W4A4 (int4-grid weights from the i4 ladder, K ≤ 3, × int4 activations g64 on the iu4
WMMA) becomes a per-tensor compute mode next to A16 and A8 for the fast variant at ≤ 3 bpw. #46-02 (d1106) failed
only as a global setting (A4 on every MLP tensor, K2: ×1.14 unhealed / ×1.16 healed, 0.8B); the knapsack places it
per tensor. Shared-input groups (q/k/v, gate/up, GDN qkv/z) share one mode. Only i4-codebook tensors can take A4.

Steps (gpubox card 0 when free; written 2026-11-05 ~1:00 AM CST, before data):
1. **Per-tensor costs (27B):** E1b fake-quant hook (`v3/e1b/aqproxy.py`, `EVALQ_AQ`-style), A4-g64 on one
   (family × 16-layer depth band) at a time, on the K2 and K3 i4 ladder levels, paired KLD on `golden-search27`
   (search pool). Families: gate_up, down, attn qkv, attn_out, gdn_in, ssm_out. Plus one combined arm (the knapsack's
   pick from step 3) to check additivity: combined ΔKLD within ±25% of the sum of its parts, else the knapsack gets
   an interaction factor before step 4. ~2–4 card-hours.
2. **Speed price:** iu4 tile time from E2 (~490 vs ~255 TOPS for iu8, g64 group-scale cost included) plus the
   two-pass decode to an int4 plane, as a per-shape time table for knap4. Replaced by a real kernel timing if built.
3. **knap4** with modes {A16, A8-g128, A4-g64} at 2.0 / 2.5 / 3.0, fast variant under its speed budget, vs the same
   knapsack without A4.
4. **Rule:** the int4-plane kernel (two-pass decode to int4, g64 activation quantizer, iu4 tile with the fold) is
   built if at one or more of 2.0 / 2.5 / 3.0 the A4 knapsack reaches **≥ 10% lower predicted `gemm_us` at equal
   predicted KLD**, or **≥ 0.03 bpw smaller at equal predicted KLD and `gemm_us`**, than the A8-only fast variant.
   Otherwise A4 stays out of V3 and is recorded as measured. If built: end-to-end KLD and `gemm_us` on the chosen
   mix decide whether it ships (the real numbers, not the predictions), published either way.
- **Runtime cost:** this is a speed option (the point is lower `gemm_us`); KLD cost priced per tensor.

## Amendment 1 (w72, 2026-11-07 3:35 AM CST, before any data): K1, the real kernel, additivity and K-interaction

Written before the first #46-03 score; it adds to steps 1 and 3 and keeps step 4's rule as it is.
- **Levels:** i4 **K1**, K2, K3 (user 11-07: K1 too), all 64 trunk layers, the ladder's settings (`--scaled-hessian
  --sigma-reg 0.025 --sigma-reg-in 0.005`, seed offset 0, no H_out; swj / asym are post-quant and stay off, as in the
  swap27 arms). `--codebook i4` without `--qtf2` (ourq refuses qtf2 + i4 until w53's patch is merged; at integer K the
  exl3cb i4 kernel gives the same codes, w53 `qtf2i4check`). Packed `exl3i codebook=mul1i4 rate=K` (exl3quant 0.3.0) and
  mixed into **V2 3.00** for every K: head, embedding and every non-trunk tensor are the same bytes in every arm (the
  scripts check the non-trunk tensor hashes of every mix against the base's).
- **Measurement: the real A4 kernel, not the E1b hook.** The hook lives in exl3r4d and never sees an exl3i / mul1i4
  weight; the A4 path (radiance-hero `qc-exl3-v3` 42960d5: actq4 + int4 plane + iu4 tile) is byte-exact against its C
  emulation (w34 test_a4, w53 section R on real 27B i4 tensors), so the arms measure the arithmetic that would ship.
  Modes are per-weight `exl3.act.<w>` metadata (radmix `act` lines). **Baseline per K: all 336 trunk weights a8g128**
  (the fast variant's A8 mode). **Arm:** one family × 16-layer band at a4g64 (a4g128 second), the rest a8g128. A8 / A4
  run at launches of ≥ 512 rows (the plugin's default threshold, as served); shorter launches take the f16 path in
  both arms. One f16 control at K2 (no act keys) ties A8's own cost to E1b / E1c.
- **Families / bands:** gate_up, down, attn_in (attn_qg / k / v, one mode), attn_out, gdn_in (ssm_qkv / ssm_z, one
  mode), ssm_out × L0-15, 16-31, 32-47, 48-63. Golden `golden-search27` (search pool, 103 docs). Readout per arm: paired
  ΔKLD and ratio vs its K's baseline, per-document bootstrap SE (`paired.py`), p99 and top-1 of both.
- **Additivity (step 1's combined arm):** gate_up + down in all four bands at a4g64 (the set w34's speed numbers need:
  ≈ −17% of the layer), at K2 (K1 / K3 if the card allows). Pass: measured ΔKLD within ±25% of the sum of its eight
  parts; else knap5 gets the measured / summed ratio as an interaction factor on A4 costs before step 4.
- **K-interaction (user 11-07 design note):** per (family, band), Δ(K1) / Δ(K3) and Δ(K2) / Δ(K3). "Additive and
  K-independent" if the ΔKLD-weighted geometric mean of each ratio is within ±25% of 1 and no family with Δ > 2 SE differs
  by more than ×1.5; otherwise the A4 price depends on K, and the table carries it (it is per K either way).
- **knap5 input:** A4 cost of tensor t at i4-K = its E1c A8 cost + the measured Δ(family, band, K) shared over the
  band's tensors in proportion to E1c's A4 rel_out (keeps the within-band shape); fractional K log-linear between the
  measured integers, clamped. An arm whose Δ is below 0.5 SE is priced at 0.5 SE (no free A4). A4 g128 is offered
  only where measured. The step-3 / step-4 run is w58's fast-variant knap5 at 2.0 / 2.5 / 3.0 with and without A4
  (`KN_NOA4=1`), compared at equal predicted KLD by interpolating along each λ curve.
