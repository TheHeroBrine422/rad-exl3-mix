# RULES-floor202 — top-band search model with an explicit A8 floor (#202)

Written 2026-11-16 ~12:05 PM CST before any fit (user: "do it"). Why: the search model (`search/v3/fit.py`, kfit3 form:
KLD = slice offset × P(allocation)) has no term for the activation (A8) error that no allocation removes. At 6–7 bpw
that floor is a large share of KLD, so the fit absorbs it into the weight costs: its local slope b is ~2× the measured
one at 6.5 / 7.0 (fit ~0.7 vs measured ~0.3), and 7.0 made 167 decisions since 11-16 2:25 AM CST with no confirmed move.

1. Model: `fitfloor` = fit.py with one added parameter per fit, an additive floor F ≥ 0 inside the offset:
   KLD = exp(soff) · (F + P). Prior: log F ~ N(log 4e-4, 1.0) (Deadcode's w8a8 floor estimate, wide). Everything else as
   fit.py (labels, anchors, weights, priors). Fitted on the b3 group's rows (4.5–7.0), as the driver's b3 fit input.
2. Gate (offline, decides use): time-ordered replay as RULES-187 / rank193.py (predict later evals of 5.5–7.0 from fits
   on earlier rows; metric S = RULES-187 §5, pair accuracy, Spearman per size), fitfloor vs fit.py on the same rows.
   Pass: dS 80 % CI wholly above 0 for the 6.0–7.0 rows. Report F with its posterior SE.
3. Live trial (whatever the gate says, as #192 inbox principle): knapsack solves from fitfloor at 5.5, 6.0, 6.5, 7.0
   (the driver's solve move logic on the fitfloor prediction; 2 variants each: the plain optimum and a Thompson draw),
   submitted as inbox `decide` files `w188-floor-<size>-<var>.json`, judged like any candidate (stage 1 + confirmation).
4. If the gate passes and ≥ 1 live candidate confirms: the lead may switch the driver's b3 fit to fitfloor (driver change
   with tests, separate step). The final picks include every confirmed fitfloor model.
Data: search rows only (calib search pool + xcorp parts); t183 report-only; never verify / final / public.
Runtime: none (allocation only).

## Amendment 1 — 2026-11-16 12:20 PM CST, before any low-band fit (user: "im expecting that the additive model is no longer accurate [below 2.5] and we need some kind of saturation model")
Low band (b1 rows, 1.5–2.75): model `fitsat` = fit.py with one added parameter S > 0: KLD = exp(soff) · P / (1 + P / S)
(saturating: linear in P while P << S, bounded by S). Prior log S ~ N(log 1.0, 1.5) (KLD in nats; wide). Same replay gate
as §2 on the 1.5–2.75 rows (pass: dS 80 % CI wholly above 0), same live trial as §3 at 1.5, 1.75, 2.0, 2.5, 2.75
(`w188-sat-<size>-<var>.json`). Report S ± SE and fitted vs measured slopes. The band split in the lead note of 12:15 PM
is withdrawn.

## Amendment 2 — 2026-11-16 12:30 PM CST, before any fit (user: "it might only be necessary 1.5 / 1.75 / 2.0, it might be necessary higher, idk")
Where each term matters is measured, not assumed:
1. fitsat is fitted on b1 AND b2 rows (1.5–4.0) separately, plus one joint fit on b1 + b2; fitfloor on b2 AND b3 rows
   (3.0–7.0) separately. Replay metrics are reported **per size** (S, pair accuracy, Spearman with 80 % CIs), not only
   pooled.
2. Per size, the share of the term at the size's incumbent: P / S (saturation; > ~0.1 = bending) and F / (F + P) (floor).
3. Use per band follows the per-size gate: a band switches only where its sizes pass; live solves (§3) at every size
   whose per-size dS is ≥ 0 in the band's fit.

## Amendment 3 — 2026-11-16 ~2:15 PM CST, before any sub-2 fit (user: "curious trying it only on the <2 bpw sizes")
Extra fitsat arm on the 1.5 + 1.75 rows only (own fit; reference = fit.py on the same rows), plus a second arm on
1.5 + 1.75 + 2.0 (2.0 is the edge case). Same per-size replay gate and S / (P/S) report. Live solves (optimum + Thompson)
at 1.5 and 1.75 from the sub-2 fit, `w188-sat2-<size>-<var>.json`, judged like any candidate.
