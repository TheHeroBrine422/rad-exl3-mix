# #200 final checks (w187, 11-16, box2): confounders and mechanism (raw: arms/*/served/census.txt, mix.txt; scale RMS below)

| pair | mix.txt (paths normalised) | census.txt | note |
|---|---|---|---|
| P20-4M vs A4M | identical (674 lines) | identical | same template, pack path, head / embedding / GDN lines |
| P50-4M vs A4M | identical | identical | |
| B4P20 vs B4 | identical | identical | |

Same data for all 4 M arms: plan sha1 2c3a906fcba5, 193 plan steps (4,000,406 rows), chunk list `chunks200.txt`.
The 1 M seeds: s2 47 steps (1,002,250 rows), s3 50 steps (1,015,239 rows) (their own permutations: seed 2 and 3 give different
orders, so the plans end at different steps; the decay is 20 % of each run's own plan, as #191).
The soup S4seed: members D0 (49 steps, 1,013,729 rows), A-s1 (49, 1,015,449), s2 (47, 1,002,250), s3 (50, 1,015,239): 4,046,667
rows in total (B4: 4,000,406; +1.2 %).

## Scale offsets (log multipliers of each arm's trained scales: 6,799,360 values, 400 files)

| arm | Adam updates | wd | RMS log | max abs log | values beyond the trust clamp (0.3) |
|---|---|---|---|---|---|
| D0 (1 M) | 49 | 0 | 0.0057 | 0.033 | 0 |
| A4M | 193 | 0 | 0.0110 | 0.097 | 0 |
| P20-4M | 193 | 20 | 0.0043 | 0.034 | 0 |
| P50-4M | 193 | 50 | 0.0023 | 0.016 | 0 |
| B4 | 49 (4 micro-batches each) | 0 | 0.0060 | 0.034 | 0 |
| B4P20 | 49 | 20 | 0.0042 | 0.024 | 0 |
| S4seed (mean of 4 members) | | 0 | 0.0034 | 0.031 | 0 |

Per-set KL (x181g / fe38): P20-4M vs A4M +5.5 % / +1.1 %; P50-4M vs A4M +11.4 % / +2.7 %; B4P20 vs B4 +1.1 % / -0.2 % (not the same sign: the pooled +0.82 % rests on x181g).

The pull-back does what it says: wd 0 → 20 → 50 takes the RMS from 0.011 to 0.0043 to 0.0023 (A4M to P-arms, 193 updates).
The back-of-envelope equilibrium of an Adam-sized step pulled at rate lr·wd is lr / sqrt(2 lr wd) ≈ 0.005 (wd 20) and 0.003
(wd 50); the measured values are of that order (a rough check, not a model).
