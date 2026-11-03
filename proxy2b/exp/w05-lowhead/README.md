# w05-lowhead — low-budget row-slab LM heads for the 2.0 / 2.5 mixes (ranked.md E1)

**Question.** knap4 offers slab heads only at K5/5.5/6 (`KN_HEADSLAB`), so at 2.0–2.5 bpw the mix takes a uniform K3 head
(V2 2.0 and 2.5 ship `hd-K3`). Should slab plans at 2.25–3.75 bits/row join the menu, and what do they buy in the whole model?

**Mechanism.** These are #34's slab heads (vocabulary rows sorted by the qad gen-row Fisher, a K per 128-row block in at most
3 slabs, the tail floor chosen per budget by headsim), extended downward:
- donors K2–K4 by quarters, mean-row (−30% head KL, #34);
- floors ≥ K2;
- guard-checked.

`lcurve.json` (old, non-mean-row donors K2/3/3.5/4) already showed budget 3.0 at 0.0060 vs uniform K3 0.0079.

## Already measured on the CPU (`results/`)

`knap.sh` uses knap4 with `knap4-headcurve.patch` (`KN_HEADCURVE`) on the V3 table (quality only). The curve is
`pickcurve.py` applied to the existing `lcurve.json`: `results/headcurve-prelim.json`, guard ≤ 1.3× u_lo.

| target | KLD without → with the curve | head picked |
|---|---|---|
| 2.0 | 0.14796 → 0.13997 (**−5.4%**) | slab 2.375 bits/row (was K3) |
| 2.5 | 0.05827 → 0.05568 (**−4.5%**) | slab 3.125 |
| 3.0 | 0.02663 → 0.02595 (−2.6%) | slab 3.25 |

The regression check passes: the patched knap4 without `KN_HEADCURVE` equals the unpatched one byte for byte.

**Caveats.**
- knap4 prices the head **lower** than headsim measures (its uniform K2/K3/K4 head cost 0.0207/0.0051/0.0013 vs headsim
  0.0379/0.0079/0.0018). The curve enters as a ratio, so the predicted gains are probably conservative.
- The 2.0 pick has a **K2 floor**. Its rare-token guard KL is ~3.9× that of the K3 head it replaces (`lcurve`: s2.375-f2
  0.063 vs u3 0.016). pickcurve's guard compares with u2 and so lets it through. The golden-ml check below is the binding
  guard.

## Steps (`run.sh`, iceman; each skipped when done)

1. **Donors** K2, 2.25, …, 4 in sensitivity order, mean-row, σ_reg 0.005 (head-settings: K3's winner). They are taken from
   headremake.sh's `/data/yaqa/headfinal` when that exists.
2. **Plans** at every 1/8 budget 2.25–3.75 × every floor ≥ K2, plus uniform plans; headsim on the calib-A head dump.
3. **`pickcurve.py`** → `headcurve.json`, then `knap.sh` (CPU; also runnable on the VM or colo) at 2.0/2.5/3.0.
4. **Whole-model transfer check.** bf16 heads are spliced into the V2 2.50 (and 2.00) release, all through the same bf16 path:
   `u3` (uniform K3 donor), `slab3` (the budget-3.0 plan, equal bytes to u3), the knap4 2.5 pick, and the 2.0 pick. They are
   scored on golden-s2k (search pool, 583 docs) and golden-ml (multilingual guard), each paired against u3.

## Draft pass rule (for the lead to commit before data)

- **Curve:** per budget, the floor plan with the lowest headsim KL among those with guard KL ≤ 1.3× u_lo
  (`pickcurve.py`, as written).
- **Transfer (adopt `KN_HEADCURVE`):** on golden-s2k, `slab3` vs `u3` must reach relative ΔKLD ≤ **−1.0%** with z ≤ **−3**,
  and the measured Δ must be ≥ **0.5×** the headsim-predicted head-only Δ.
- **Guard:**
  - On golden-ml, every tested plan vs u3 must have ratio ≤ **1.02** (or z < 2).
  - A pick with floor < K3 that fails the guard removes all floor-K2 plans from the curve. pickcurve is then re-run with
    `HS_FLOOR=3` plans only, and knap.sh re-run.
- The 2.0/2.5 picks hold fewer head bytes than u3. Their whole-model Δ vs u3 is **reported** beside headsim's prediction (the
  knapsack spends the saved bytes on the trunk, which a head swap does not), and is not a pass gate.

## What the result changes

- **Pass:** merge `knap4-headcurve.patch`. knap4 at 2.0/2.5 then picks slab heads. That needs `hdslab-hcK<B>.rad`
  containers and the plugin's slab-head support (TODO: "Plugin: slab head"), which is now needed at low budgets too.
- **Fail on transfer:** headsim mis-prices low-budget heads. Keep uniform heads at ≤ 3 bpw and re-check the K5 slab
  (#34's whole-model check is still open).
- **E2 interplay:** if two-sided uniform heads win (w05-head2s), this curve is rebuilt from two-sided sorted-order donors or
  replaced. Run E1's steps 1–3 anyway: its uniform plans are E2's comparator.

**GPU time:**
- 9 donors: ~2–3 h, 0 if headfinal has them (each needs host RAM exclusively, ~22–26 GB).
- headsim: ~30 min.
- 10 whole-model evals: ~1–1.5 h.
- Card 0 total ~4–5 h.

**Run:** `CARD=0 ./run.sh` on iceman. Results go to `/data/yaqa/w05-lowhead/result.txt`.

## Patch

`knap4-headcurve.patch`: `KN_HEADCURVE=FILE` adds head options `ours-hcK<B>` = `ours-K<lo>` bytes + (B − lo)·V·H/8 + 1 MiB,
with error × (KL_B / KL_u<lo>)^(1/q). Containers are `hdslab-hcK<B>.rad`. It is a no-op when unset (`results/knap-prelim.txt`:
"REGRESSION OK").

## Runtime cost (user rule, 2026-11-03)
D (row-slab head: several head GEMMs + gather; the head runs on every decoded token, in prefill only on the last position, so `gemm_us` is unaffected).
