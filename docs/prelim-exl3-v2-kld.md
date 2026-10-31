# Preliminary EXL3-V2 KLD (Qwen3.8-27B, 2026-10-30)

> **Superseded** by the EXL3-V2 release numbers (README, `data/v2/`): interim results on a different held-out slice.

EXL3-V2: mixed-K EXL3, our own K1–8 quants on the challenge calib Hessians plus turboderp's, the per-tensor mix picked
by a fitted sensitivity model and an exact knapsack solver. Same f16-activation kernels as EXL3-V1.

KLD vs bf16 on a **held-out** slice of calib (81 documents never used for fitting: 124k positions, docs capped at 8k
tokens, bf16 KV). Real bpw = file bytes × 8 / 26.9B. The comparison columns are from the leaderboard as of 2026-10-30,
where scoring is still in progress; the frontier is the Pareto front of all other entries, log-linear between points (so 'vs frontier' compares at exactly
our size; the neighbouring IQR-V3 entries are listed for a direct read).

| real bpw | KLD | top-1 | nearest IQR-V3 below | nearest IQR-V3 above | vs leaderboard frontier |
|---|---|---|---|---|---|
| 1.98 | 0.17269 | 83.1% | — | IQ2R-V3 (2.46, 0.06807) | 0.55× |
| 2.49 | 0.06060 | 90.2% | IQ2R-V3 (2.46, 0.06807) | IQ2R-L-V3 (2.79, 0.04241) | 0.93× |
| 3.00 | 0.02685 | 93.4% | IQ2R-L-V3 (2.79, 0.04241) | IQ3R-S-V3 (3.15, 0.02584) | 0.85× |
| 3.50 | 0.01383 | 95.3% | IQ3R-V3 (3.46, 0.01698) | IQ3R-L-V3 (3.80, 0.01129) | 0.85× |
| 4.00 | 0.00717 | 96.7% | IQ3R-L-V3 (3.80, 0.01129) | IQ4R-S-V3 (4.13, 0.00762) | 0.76× |
| 4.50 | 0.00403 | 97.4% | IQ4R-V3 (4.46, 0.00512) | IQ4R-L-V3 (4.78, 0.00342) | 0.83× |
| 5.00 | 0.00232 | 98.0% | IQ4R-L-V3 (4.78, 0.00342) | IQ5R-S-V3 (5.06, 0.00249) | 0.87× |
| 5.98 | 0.00072 | 98.9% | IQ5R-V3 (5.42, 0.00182) | IQ6R-S-V3 (6.05, 0.00106) | 0.64× |
| 7.00 | 0.00032 | 99.3% | IQ6R-L-V3 (6.94, 0.00072) | IQ7R-V3 (7.42, 0.00060) | 0.46× |
| 7.50 | 0.00029 | 99.3% | IQ7R-V3 (7.42, 0.00060) | IQ8R-V3 (8.04, 0.00047) | 0.51× |

Caveats: these are our own measurements on a calib slice, not leaderboard scores, and the mixes are still being
refined. The same turboderp containers measure 0.97–1.27× higher on the (partially scored) leaderboard than on this
slice, so treat the ratios as approximate.
