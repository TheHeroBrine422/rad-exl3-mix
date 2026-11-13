# search/refit — the knapsack refit after ladder pass 1 (w120, 2026-11-12; ideas round 2 ranks 14 / 15)

Scaffolding only: tested on existing data (dry run below), decides nothing. The lead runs it once pass 1's levels and
the calibration swaps exist; one fixed fit per search run (lead note 4), refit before the search starts.

| file | what |
|---|---|
| `kfit3.py` | kfit2's model (search/kfit2.py) + (a) V2 swaps as a weak prior (`--v2-wt`, own slice offsets), (b) a per-tensor H_out prior `log s_w = A θ + β p_w`, p_w = log h̄_w (the tensor's error-weighted output sensitivity, ½Σh·e / Σe from #139's diag files), `--beta fit / 1 / 0`, (c) V3 rows (`rows27.json`) as their own families with per-group transfer factors and the swj factor on gate / up, (d) validation: `--holdout REGEX`, `--holdout-frac F --seed S` (random split of `--holdout-pool`), `--compare` (β 0 / fit / 1 on the same split), `--v3-maxkld` (rows above: predicted, not fitted) |
| `diagsum.py` | per-tensor sums of each ladder job's diag file (R, E1, E2, h̄ = E2 / E1, pre-refit too), one line per (level, tensor); CPU, numpy only |
| `rows27.json` | this week's 27B whole-trunk mixes with per-tensor proxies, flagged by stamp and slice: w92 (27c, 4 rows), w110 #145 (27d, 6), w114 #149 (27f, 4 provisional + 2 pending), w99 #129 (27c, K6 in V2 8.00: skipped, see below). w111 / w113 (P16) are not rows: row permutations, no proxy change, within noise |
| `../../data/refit/` | slim proxies (key, K, proxy_err) and diag sums of those levels; `v2fit-d.tar.xz` = kfit2's V2 input dir (hproxy, evq-results, mixes) |

```
python3 diagsum.py OUT.jsonl LEVELROOT...                      # on the host with the level dirs (ladder: ROOT/<level>/out)
python3 kfit3.py --v2 d --v3 rows27.json --diag DIAGSUM.jsonl --compare --holdout-frac 0.2 --seed 1 --report rep.json
python3 kfit3.py --v2 d --v3 rows27.json --diag DIAGSUM.jsonl --beta fit --out model.json   # the fit knap5 reads (after the swaps)
```
~1–3 min a fit, single-threaded (colo). `model.json` is kfit2dump-like (per-tensor s, eb, φ, κ, q, c0, emb, V3 family
factors); knap5 still reads `data/v3/kfit2-model-final.json` — swapping it in is the lead's step after the validation.

## Dry run (11-12, colo, existing data; LOG 11-12 w120)
- **H_out prior (b):** β fitted freely lands at 0.85–1.13 on every split (the prior's direction holds), but on random
  20% holdouts of V2's block swaps it does not predict better: log-RMSE β0 / fit / 1 = 0.170 / 0.161 / 0.159 (seed 1),
  0.197 / 0.199 / 0.199 (seed 2), 0.196 / 0.205 / 0.203 (seed 3). Expected: V2's swaps move whole group × 8-layer
  blocks, where per-tensor differences average out. The prior is for per-tensor picks; only per-tensor-varied swaps
  (below) can test it. Not decided.
- **V3 rows (c):** with rows of KLD ≤ 2 fitted, V3 in-sample within-slice RMSE 0.0075; held out: w92 fs1 +1.8%, w110
  K1 −1.1%, 27f U1 +29% (27f has no other fitted row: a held-out slice needs one kept anchor row, else its offset is
  unknown). φ(v3hout) ≈ φ(v3plain) ≈ 1.1–1.3 (V3 errors cost ~20% more KL than V2's per unit proxy at the same s_w).
- **Sub-1 rows are mis-predicted** by kfit2's compounding: predicted / measured +0.41…+0.71 log for the splices,
  +0.8…+1.3 at U0.75, +2.5…+3.4 at U0.5 (prediction 70–180 vs 5.7). Same finding as knap5's sub-1 anchors
  (`v3/levels/reprice/sub1plan.md`): the depth compounding has to weaken at large P before sub-1 rows can be fitted.
- **w99's K6 rows** (V2 8.00: bf16 embedding, K7–8 head, approximated by e6 / ours-K8) collapse φ3 to ~0.01 when fitted;
  skipped by default.

## Pass-1 calibration swaps (design; card time on box2 + card 0 via rxrun.multi eval units)
Purpose: re-anchor the level-dependent terms (per-level error → KL at V3 settings), the family transfer factors and the
depth compounding at V3's levels, and give the H_out prior a held-out test with per-tensor variation. All on a
**calib fit-pool slice** (fitting: AGENTS.md data rules), ~120k scored positions (fit pool, `calib_slice.py 140000
8192 SEED fit`, ledger role fit), plus one fresh search slice for the held-out check.
1. **Uniform levels (20 mixes):** every pass-1 level uniform in the V2 3.00 head / embedding (until the V3 head is
   decided). Re-anchors r(K) per level and gives the knapsack its level ladder at V3 settings.
2. **Group × depth swaps (48):** bases uniform K2.5 and K4 (the first-use levels); per base, each of the 6 groups
   (attn_in, attn_out, ssm_in, ssm_out, ffn_gu, ffn_down) × 2 depth halves (layers 0–31 / 32–63) × 2 donors
   (K−1 and K+1.5 from the base) = 24. Fits φ per group and κ's shape.
3. **Per-tensor-varied swaps (16) — the prior's test:** in base K2.5, swap the 8 tensors with the highest and the 8
   with the lowest predicted ½Σh·e per proxy unit (h̄ from the pass-1 diag sums) inside one family and depth band to
   K1.5, one family per mix (8 families × 2). The H_out prior predicts a large ratio between the two arms of a family;
   block-only kfit2 predicts ~1. Held out of the fit (validation (d)): pass = the prior's log-RMSE on these 16 is
   lower than β = 0's and the pairwise order inside each family is right in ≥ 6 / 8 (rule to commit before data).
4. **Held-out check (8):** 8 of step 2's mixes rescored on a fresh search slice (not fitted): the refit's predicted
   ratios vs measured, prior vs no prior.
Cost: 92 mixes. Packs: the 20 pass-1 level packs (~23 min each, ram.lock exclusive; box2 has the RAM for two at once)
≈ 8 h CPU-side, shared with the ladder's own packing; mixes by radmix ~1 min each; scores ~4.3 min a mix on a 280k
slice (card 0; box2 cards ~10% slower), 120k fit slice ≈ 2 min → 92 × 2 + 8 × 4.3 ≈ **3.6 card-h** (≈ 1 h wall on
box2's 4 cards + card 0). Swap mixes are eval units in `rxrun.multi` (mixes need their levels on one host: run them on
the host that built K1.5 / K2.5 / K4 / K5.5, or copy those 4 level dirs to one host).

**Run (w129, #169, 11-13):** `w129/` — done: prior FAIL (β 0), recommended refit `fit129c` (84 calib rows); held-out
xcorp ratios are not predicted by calib-fitted rows (0.14 log-RMSE vs 0.03 on calib); see `w129/README.md` and LOG 11-13 1:15 AM CST.
