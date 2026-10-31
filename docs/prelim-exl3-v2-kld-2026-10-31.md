# Preliminary EXL3-V2 KLD, update (Qwen3.8-27B, 2026-10-31 ~02:30)

> **Superseded** by the EXL3-V2 release numbers (README, `data/v2/`): interim results on a different held-out slice.

Same format as `prelim-exl3-v2-kld.md` (2026-10-30, round 8 mixes). Since then: our K1–8 quants only (ours-only
stream), K1 fit fix, embedding fix, streaming search with Thompson and local-search mixes. Each row is the current
best measured **all-ours** mix (no turboderp tensors; this is what makes 6.0 / 7.0 worse than round 8, whose mixes
used turboderp's high-bit tensors) at that target (fit-slice KLD, slice-normalised; `finalsel.py`), re-measured on the **held-out**
ver1 slice (81 documents, 124k positions, docs capped at 8k tokens, bf16 KV). Real bpw = file bytes × 8 / 26.9B.

'vs leaderboard frontier': our ver1 KLD mapped to the leaderboard through all 16 EXL3-V1 containers measured on
both (ratio leaderboard / ver1, interpolated in bpw), divided by the Pareto front of all other entries at our size
(log-linear between points). 'vs unsloth' = the same against the UD frontier (the category score). Leaderboard
snapshot 2026-10-30 (1.05M private positions).

| real bpw | KLD | top-1 | nearest IQR-V3 below | nearest IQR-V3 above | vs leaderboard frontier | vs unsloth |
|---|---|---|---|---|---|---|
| 1.99 | 0.15957 | 83.9% | — | IQ2R-V3 (2.46, 0.06807) | 0.55× | 0.55× |
| 2.49 | 0.06005 | 89.9% | IQ2R-V3 (2.46, 0.06807) | IQ2R-L-V3 (2.79, 0.04241) | 1.05× | 0.53× |
| 3.00 | 0.02712 | 93.3% | IQ2R-L-V3 (2.79, 0.04241) | IQ3R-S-V3 (3.15, 0.02584) | 0.99× | 0.51× |
| 3.50 | 0.01362 | 95.3% | IQ3R-V3 (3.46, 0.01698) | IQ3R-L-V3 (3.80, 0.01129) | 1.06× | 0.51× |
| 4.00 | 0.00704 | 96.6% | IQ3R-L-V3 (3.80, 0.01129) | IQ4R-S-V3 (4.13, 0.00762) | 0.87× | 0.45× |
| 4.50 | 0.00371 | 97.6% | IQ4R-V3 (4.46, 0.00512) | IQ4R-L-V3 (4.78, 0.00342) | 0.89× | 0.42× |
| 5.00 | 0.00195 | 98.2% | IQ4R-L-V3 (4.78, 0.00342) | IQ5R-S-V3 (5.06, 0.00249) | 0.83× | 0.33× |
| 5.98 | 0.00112 | 98.8% | IQ5R-V3 (5.42, 0.00182) | IQ6R-S-V3 (6.05, 0.00106) | 1.23× | 0.48× |
| 7.00 | 0.00036 | 99.2% | IQ6R-L-V3 (6.94, 0.00072) | IQ7R-V3 (7.42, 0.00060) | 0.63× | 0.32× |
| 7.32 | 0.00028 | 99.3% | IQ6R-L-V3 (6.94, 0.00072) | IQ7R-V3 (7.42, 0.00060) | 0.54× | 0.29× |

Best mixes **with** turboderp's high-bit tensors at ≥ 6 bpw (round 8, same ver1 slice; the stream allows these again since
02:17 via KFIT_TD_MIN=5):

| real bpw | KLD | top-1 | nearest IQR-V3 below | nearest IQR-V3 above | vs leaderboard frontier | vs unsloth |
|---|---|---|---|---|---|---|
| 5.98 | 0.00072 | 98.9% | IQ5R-V3 (5.42, 0.00182) | IQ6R-S-V3 (6.05, 0.00106) | 0.80× | 0.31× |
| 7.00 | 0.00032 | 99.3% | IQ6R-L-V3 (6.94, 0.00072) | IQ7R-V3 (7.42, 0.00060) | 0.56× | 0.29× |
| 7.50 | 0.00029 | 99.3% | IQ7R-V3 (7.42, 0.00060) | IQ8R-V3 (8.04, 0.00047) | 0.62× | 0.32× |

Change vs round 8 on the same ver1 slice: 2.0 −8%, 2.5 −1%, 3.0 +1%, 3.5 −2%, 4.0 −2%, 4.5 −8%, 5.0 −16%,
**6.0 +56%** (worse), 7.0 +12%. Paired SE on this slice is ~±1.5%: 2.5–4.0 are unchanged within noise.
The 7.5 row is a 7.32 bpw mix (the best measured one at that target came in under size).

Selection effect: on the fit slices these picks looked 1–10% better than round 8 at 2.5–4.0; on ver1 it is 1–2%.
ver1 / fit is ~0.91 for the new picks vs ~0.88 for round 8, i.e. a ~3% winner's curse from picking the best of many
noisy measurements. The final re-measures on ver2, which removes it.

Caveats: our measurements on a calib slice, not leaderboard scores; the mixes are still being refined.
