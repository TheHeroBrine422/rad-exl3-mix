# #169 rule: the H_out prior's held-out test on the pass-1 calibration swaps (w129)

Written 2026-11-12 10:40 PM CST, **before any calibration-swap data exists** (no slice cut, no mix scored; committed
alone). Design: `search/refit/README.md` "Pass-1 calibration swaps" step 3 + validation (d).

## Question
Does the per-tensor H_out prior (log s_w = A θ + β p_w, p_w = log h̄_w centred per group, h̄ = E2 / E1 from the
#139 diag files) predict per-tensor-varied swaps better than the block-only model (β = 0)?

## Data (fixed now)
- **Heads / embedding:** every mix is V2 3.00's container (`V2-3.00` alias on ds8) with its trunk replaced by pass-1
  ladder levels (codes 144dd6e1, final at the staged settings incl. the swj rerun); head and embedding stay V2 3.00's.
- **Fit slice `fit129`:** calib, merged fit + search pool, documents in least-used-first order (`slices.py`'s
  `draw_merged` ordering: ledger decision counts + no registry, ties shuffled with seed 169), taken in that order until
  ≥ 140,000 scored positions at a per-document cap of 8,192 tokens. Ledger role fit.
- **Held-out slice `s129`** (two-part): c = the next documents in the same order until ≥ 45,000 scored positions at
  cap 4,096; x = `pickx.py` 130 pieces of xcorp `xc8k-b2-c2`, seed 16930, heavy categories (agentic, tools, long,
  reasoning) cut to 2,048 scored positions, excluding the xcorp documents of golden-search27b…27f-x and golden-hd122-x.
  Ledger role check.
- **Mixes (92 + 2 anchors), all on `fit129`** unless stated:
  1. U: the 20 pass-1 levels uniform (K1.25–5 in quarters, K5.5–7 in halves).
  2. G: bases uniform K2.5 and K4; per base, each of the 6 groups (attn_in, attn_out, ssm_in, ssm_out, ffn_gu,
     ffn_down) × 2 depth halves (layers 0–31 / 32–63) × 2 donors (K2.5 → K1.5 / K4; K4 → K3 / K5.5) = 48.
  3. P (this rule's test): base uniform K2.5, donor K1.5, 8 families: attn_qg (layers 0–63), attn_output (0–63),
     ssm_qkv (0–31), ssm_z (32–63), ssm_out (0–31), ffn_gate_up (32–63), ffn_down (0–31), ffn_down (32–63). In each
     family the 8 tensors with the highest h̄ (arm hi) and the 8 with the lowest h̄ (arm lo) are swapped to K1.5, one
     mix per arm = 16. h̄ = E2 / E1 summed over the tensor's HF parts from the K2.5 level's diag files (`diagsum.py`).
  4. H: 8 of step 2's mixes rescored on `s129` + the two bases (uniform K2.5, K4) on `s129` as anchors:
     g2.5-ffn_down-lo-k1.5, g2.5-ffn_gu-hi-k1.5, g2.5-ssm_in-lo-k4, g2.5-attn_out-hi-k4, g4-ffn_down-hi-k3,
     g4-ssm_out-lo-k3, g4-ffn_gu-lo-k5.5, g4-attn_in-hi-k5.5 (lo = layers 0–31, hi = 32–63).
- **Model:** `search/v3/fit.py` (kfit3's form, SModel) on a menu of the 20 pass-1 levels (`mkmenu.py`, err = proxy ×
  R, h̄ from the K2.5 diag), default options (τ 0.02, φ sd 1, θ sd 4, group sd 0.3, smooth 0.5); per-row SE = paired
  per-document delta-method SE of log(KLD_row / KLD_anchor), anchor = uniform K2.5 of the same slice (SE 0).
- **Training rows for the test:** U + G on `fit129` (68). P and H are held out.

## Measures
d_i = [log P(m_i) − log P(base)] − [log M(m_i) − log M(base)] (P predicted, M measured, base = uniform K2.5 on
`fit129`): the error in the predicted swap ratio, what a search decision sees.
1. **Primary:** RMSE of d over the 16 P mixes, arm β fit vs arm β 0 (both fitted on the same 68 rows).
2. **Order:** per family, sign(log P(hi) − log P(lo)) == sign(log M(hi) − log M(lo)), arm β fit.
3. Reported, no decision: arm β 1; per family the measured log(hi / lo) with its paired SE; on H, the RMSE of the
   predicted vs measured log ratio to the same base on `s129` (both arms; the c and x parts separately too); fitted β.

## Decision
**PASS** (the prior is on in the refit the search starts from: β fit) iff (1) RMSE_d(β fit) < RMSE_d(β 0) **and**
(2) the order is right in ≥ 6 of the 8 families. Otherwise **FAIL**: the refit uses β = 0. A family whose two arms
cannot be scored (eval failure after one retry) counts as wrong. The final refit (either way) is fitted on every row
(U + G + P + H with their own slice labels).

## Ladder check (report-only flag, user ok 11-12 "swj integrity" item 3)
A uniform level is **flagged** (possible bad swj → clean swj redo before the search uses it) if its fit129 KLD is not
below the next lower level's, or if log KLD lies more than max(3 paired SE, 0.02) above the straight line in K between
its two neighbours' log KLD.
