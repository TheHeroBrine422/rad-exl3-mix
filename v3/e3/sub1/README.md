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
