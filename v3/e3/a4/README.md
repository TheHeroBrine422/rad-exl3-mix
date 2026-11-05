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
