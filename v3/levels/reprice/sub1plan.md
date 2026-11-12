# Sub-1 pricing in knap5: code path and pass-3 measurement plan (w120, 2026-11-12)

The ladder's seven sub-1 levels (K0.125–0.875, i8, pass 3, built last) had no price: `[knap].sub1_ratios = ""`, and
knap5's old `KN_SUB1` path made `ours-K<k>` options that the V3R menu then dropped (it rebuilds the trunk menu from the
integer `ours-K*` options and pops the rest). This note is the plan; nothing here decides anything. Ideas round 2 rank
23 ("saturation fit and real anchors at ≤ 1.75 bpw", ~6 card-h after the sub-1 levels) is the measurement half.

## Code path (done, tested on CPU)

| piece | what |
|---|---|
| knap5 `KN_SUB1` + V3R | `i8-K<k>` options (a8 mode) for every `KN_I8_SUB1` level: error = base(K1) × R8(1) × ratio(tensor or family, k) × kl_scale(k); bytes linear below K1 as #90; ratios log-linear in k between the file's points (clamped). `KN_SUB1_TENSORS=1` uses the per-tensor table where a tensor has one. |
| `sub1ratios.py` | ratios from the 27B levels' own `level/proxy.jsonl` (one-sided `proxy_err` = kfit2's error measure) against the K1 level: per tensor and per family (geo), p10 / p90 spread, CPU #90 side by side. |
| knap5 `KN_SUB1_FIT` | whole-model KL anchors (`sub1-anchors.json`): uniform or layer-parity splice mixes with sub-1 levels vs a reference mix on the same slice → an extra scale per sub-1 K, solved jointly with the swj anchors (`KN_SWJ_ANCHORS`; swj is on in every measured T arm). `KN_SUB1_FIT_OUT` writes the fit. |
| knap5 `KN_SAT` / sat anchor | the saturation link KLD = C (1 − e^(−P/C)) (kfit2's `KFIT_SAT` form): a `"sat": true` anchor (a K1 mix vs a ≥ 2 bpw mix on one slice) fits C; the anchors are then solved in the saturated scale and every output line gets `kld_sat`. The DP still minimises P (monotone: the same picks at λ 0; at λ > 0 the time trade shifts slightly — reported, not used, until a validation says so). |
| rxq.knap env | sub-1 levels → `KN_I8_SUB1`; `KN_SUB1` = `[knap].sub1_ratios` (relative to `v3/nf`). Still unset in v3.toml: the lead sets it when the pass-3 prices exist (below). |
| check | `pipeline/tests/check_knap.py --part=v3`: 49-level library, sub-1 priced, every solved anchor reproduced to 1e-3. |

## What the data give today

- **Ratios (27B, stamp 159591b2, w110's K0.5 / K1 levels, `data/levels/reprice/sub1/sub1ratios-27b.json`):**
  err(K0.5) / err(K1) per family 2.16 (down) – 2.60 (attn_k), close to #90's CPU trellis (2.22–3.02; the 27B is lower
  on attn_qg 2.50 vs 3.02 and ssm_z 2.32 vs 2.75). Only K0.5 is measured; other sub-1 K extrapolate flat from it
  (clamped). w114 adds K0.25 / K0.75 (stamp 4f831984, today) → rerun `sub1ratios.py` on the five proxies.
- **KL anchor (#145, golden-search27d):** splice even K0.5 / odd K1.5 vs uniform K1 = 2.088 (T arms) → kl_scale(0.5)
  × **0.81** on top of the proxy ratio (the proxy over-prices K0.5 by ~19% relative to K1 in the model's KL).
- **swj at sub-1 (#145):** gate / up factor f(0.5) 0.894, f(1) 0.763, f(1.5) 0.828 (solved with K2.5 0.860, K4 0.973;
  `swj-anchors.json`). K0.125–0.375 take f(0.5) (flat; no data; #141-04's 0.8B doubt at K0.5 did not carry to 27B).
- **Saturation:** the uniform-K1 model KLD is 1.60 vs 0.057 at K2.5 (ratio ~28). Whether that ratio is real needs one
  slice with both: **#149's U1 + #150's K2.5 (both in V2 3.00, golden-search27f)** — pending in w114; fill
  `149-sat` (ratio = KL(U1) / KL(K2.5)) and the three `149-u*` uniform anchors (U0.25 / U0.5 / U0.75 vs U1), drop
  `skip`, and turn `145-splice` into a check (it is a second slice of the same K0.5 point).

## Update 11-12 ~11:00 AM CST (w120): #149's first T arms, and kfit2's compounding at sub-1

w114's #149 T arms on golden-search27f (read from its running queue, provisional until w114 reports; both parts,
position-weighted): U1 1.3980, U0.75 2.6611, U0.5 5.7323, splice even K0.5 / odd K1.5 2.6924 → U0.5 / U1 **4.10**,
U0.75 / U1 1.90, splice / U1 **1.93** (#145 on 27d with f32 swj: 2.09). The K0.25 / K0.75 proxies are in
`data/levels/reprice/sub1/` and `sub1ratios-27b.json` now has K0.25 / 0.5 / 0.75 (the 27B is close to #90's CPU
trellis except attn_qg / ssm_z / ssm_qkv at K0.25: 4.4 / 4.2 / 3.8 vs 7.4 / 5.6 / 4.6).

**The model cannot fit both uniform and splice mixes at sub-1.** Fitted to U0.5 (kl_scale), knap5 puts the splice at
0.97 × U1 (measured 1.93): kfit2's depth compounding (κ ≈ 30–45 at layer 0, ~3–4 in the middle; fitted on V2 mixes
with KLD ≤ ~0.4) turns a modest per-layer cost into U0.5's 4.1× by compounding, so half the layers at K0.5 barely
move the splice. The measurements are near additive instead (splice ≈ (U0.5 + U1.5) / 2 with U1.5 ≈ 0.4 × U1 gives
~2.25 vs 1.93: mildly sub-additive). A saturation link C does not help (fitted to the splice it caps U0.5 near 2×:
degenerate, kl_scale → bound). kfit3 shows the same: fitted on rows with KLD ≤ 2, it predicts the sub-1 rows with
log-RMSE ~1.8. So, as prep (no decision):
- `sub1-anchors.json` fits on **mixed-K mixes only** (149-splice fixes K0.5); uniform sub-1 mixes are checks. The
  knapsack's 1–1.5 bpw mixes are mixed-K, so a splice is the closer anchor. K0.25 / 0.75 take K0.5's scale (clamped)
  until their splices exist.
- Step 2 below: the two splices are **fit** anchors (K0.25 and K0.75), the uniform U-mixes checks.
- The real fix is in the model (compounding that weakens at large P, or per-K compounding): a kfit3 item after
  pass 1, with these rows as its test (the 11-12 dry run is in LOG).

## Pass-3 measurement plan (when the sub-1 levels exist; card time on any of our R9700s)

1. **Prices (CPU, minutes):** `sub1ratios.py` over the 7 sub-1 levels + K1 (pass 2) → `sub1ratios-ladder.json`;
   set `[knap].sub1_ratios` to it with `KN_SUB1_TENSORS = "1"` in `[knap].env`.
2. **Anchors (one fresh two-part slice, `eval/slices/mkslice27.sh`; all mixes in V2 3.00, T arms only):**
   uniform U0.125, U0.25, U0.5, U0.75 (4), U1 (the reference), U2.5 (the saturation reference; any K2.5 ladder level),
   and two splices: even K0.25 / odd K1.5 and even K0.75 / odd K1.25 (mixed-K inside the sub-1 band: the **fit**
   anchors of K0.25 / K0.75; the uniform U-mixes are checks of the compounding). 8 mixes: 6 level packs (~23 min each the first time, ram.lock
   exclusive) + 8 scores (~3 min each on ~280k positions) ≈ **3.5 card-h**.
3. **Family transfer at sub-1 (checks, not fits):** in a uniform K1.5 base, one family group at K0.5 at a time —
   attn_in (q/k/v), ssm_in (qkv/z), gate_up, writers (o / out / down): 4 mixes on the same slice (`--reuse` packs,
   ~10 min each) ≈ **0.7 card-h**. Pass: the model (after step 2's fit) predicts each ratio within ±10% (the slice's
   paired SE is ~2–3% at these KLs); a family outside → a per-family kl_scale (one more parameter each).
4. **swj at the bottom (one fixed-code pair):** U0.25 swj on / off (scale-only arm, `--planes svh` hygiene) →
   anchor f(0.25) (today flat at f(0.5)). ≈ **0.3 card-h** with the swj pass the level gets anyway.
5. **Validation (held out from the fit):** three provisional knap5 solves at 1.0 / 1.25 / 1.5 bpw (priced with steps
   1–4), packed and scored on a *second* fresh slice; predicted ratios between them vs measured, with and without the
   saturation link. **Rule to write before the data:** keep `KN_SAT` in the solves' λ trade iff it predicts the three
   pairwise ratios with lower log-RMSE than no saturation; the anchors stay either way. ≈ **1.5 card-h**.

Total ≈ **6 card-h** + CPU minutes (rank 23's estimate). Every mix is checked identical outside the arm (radmix
hygiene) before it is scored. Ledger rows for both slices (role select / fit-hessian as their use; line v3).

## Caveats
- Every anchor is a whole-model ratio, scored on a search-pool slice (fit-pool data would be the textbook place for a
  sensitivity fit, but the fit pool is small (551 docs) and these anchors fix ~1 parameter per K; record the use in
  the ledger as `fit`). Verify is never used here.
- kfit2's depth compounding (κ) was fitted on V2 mixes (KLD ≤ ~0.3); at sub-1 the model's KLD is 1–3. The sat link
  and the splice checks are what tell whether the compounding still holds there.
- Stamps: #145 is 159591b2 (f32 swj), #149 4f831984 (bf16 swj); the frozen ladder's stamp will differ again. The
  ratios move with the quantizer only through the proxy (step 1 re-measures it); the KL scale is re-fit in step 2.
