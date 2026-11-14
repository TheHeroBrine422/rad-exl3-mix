# RULES-173-03b — interaction test: do pairs of moves add? (w158, #173-03 step 3b; runbook 3b, user 11-13)

Written 2026-11-14 ~9:10 AM CST and committed alone, before the test's slice is drawn or any of its mixes is built.
Question (user 11-13): the knapsack and the search price a mix as kfit3 / M1 does: per-tensor costs added within a layer and
compounded across layers (P_l = P_{l-1} (1 + kappa_l X_l) + X_l). Do real pairs of moves add like that, or do they interact
(same layer, same input, adjacent layers, depth)? The outcome sets M1's interaction terms / the reranker's features (BACKLOG).

## Slice (fresh, fit + search pool)
`golden-i173c`: calib documents of the merged fit + search pool in draw171.py's least-used-first order (ledger decision counts +
the w131 registry's; never a document of a ready / active slice, never one of fit129's), seed 17303, cap 4,096, ≥ 110,000 scored
positions; cut from the published calib golden with golden_pick.py on ds8. Calib only (the structure is a property of the
model; xcorp would add a second scale). Ledger role `fit` (use `inter173`). Never verify / public.

## Bases and moves
Bases B25 / B40 = RULES-173-03's M1m re-solve picks at sizes 2.5 and 4.0 (the wave-1 fit; the K ≥ 1 menu), A8, a/b T4.
A move = one trunk weight set to a lower level: at B25 to **K1**, at B40 to **K2** (large enough for a single weight's effect
to be measured on ~110k positions; the pick's own level of that weight is recorded). Pairs (layer numbers fixed now; L % 4 == 3
is full attention):

| id | category | weight a | weight b |
|---|---|---|---|
| p1 | same layer FFN pair, early | blk.4.ffn_gate_up | blk.4.ffn_down |
| p2 | same layer FFN pair, middle | blk.30.ffn_gate_up | blk.30.ffn_down |
| p3 | same layer FFN pair, late | blk.58.ffn_gate_up | blk.58.ffn_down |
| p4 | same input GDN (qkv + z), early | blk.5.ssm_qkv | blk.5.ssm_z |
| p5 | same input GDN (qkv + z), late | blk.53.ssm_qkv | blk.53.ssm_z |
| p6 | same input attention (q + k) | blk.7.attn_qg | blk.7.attn_k |
| p7 | same input attention (k + v) | blk.39.attn_k | blk.39.attn_v |
| p8 | adjacent layers, same type | blk.20.ffn_down | blk.21.ffn_down |
| p9 | adjacent layers, same type | blk.44.ssm_qkv | blk.45.ssm_qkv |
| p10 | far layers, same type | blk.20.ffn_down | blk.52.ffn_down |
| p11 | far layers, same type | blk.8.ssm_qkv | blk.44.ssm_qkv |
| p12 | mixer x FFN, same layer | blk.12.ssm_out | blk.12.ffn_down |
| p13 | attention out x next FFN (adjacent) | blk.27.attn_output | blk.28.ffn_gate_up |

B25: p1–p13 (13 pairs); B40: p1–p12 (12 pairs) → 25 pairs, each move alone and both together: 75 mixes + the 2 bases = **77
scores**. If a base already holds a weight at or below the move's level, that move is replaced by the same weight one layer
deeper of the same type (recorded); all mixes census-checked as RULES-173-03 (mad trunk + head, trellis embedding, A8, v4).

## Statistic
Per pair, on the slice's per-document KL sums (positions-weighted mean KL): r_a = log KL(base+a) − log KL(base), r_b, r_ab;
**interaction I = r_ab − r_a − r_b** and the relative interaction ρ = I / (r_a + r_b); 95 % CIs by a paired per-document bootstrap
(2,000 resamples, seed 1733; one resample for all arms of a size). Also: the M1m prediction of r_ab given its own r_a, r_b
(kfit3's compounding) vs measured: residual R = r_ab(measured) − r_ab(M1m), and M1m's predicted I.
Reported per pair, per category, per size, pooled (inverse-variance weighted mean of ρ over pairs).

## Reading (fixed now)
- **Additive within noise**: the pooled ρ's 95 % CI covers 0 and |pooled ρ| < 0.10 → M1 needs no interaction term; the
  reranker gets no pair features.
- **A category interacts**: its mean ρ over pairs (both sizes) has a 95 % CI excluding 0 and |ρ| ≥ 0.10 → recommend the matching
  M1 term: same-layer FFN / mixer × FFN → m133's `sib` (within-layer product); adjacent layers → `adj`; same input → a per-layer
  input-sharing term (new; BACKLOG). Sign recorded (sub-additive ρ < 0 means a pair costs less than the sum).
- **Depth**: p1 vs p2 vs p3 and p4 vs p5 reported (no decision).
- **M1m captures it** if the pooled |R| (inverse-variance weighted) is ≤ half the pooled |I| — then the compounding already
  carries the interaction and no term is added even when the additive reading fails.
Runtime class: none (a pricing question; no format or kernel change).
