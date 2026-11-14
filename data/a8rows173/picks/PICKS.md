# M1m-in (wave-1 fit, K >= 1 menu) re-solve picks — RULES-173-03, w158 11-14

Assignments: `picks-M1m-in/pick158-<size>.json` (menu173m-v1 option names; a/b T4 at <= 5.0, T5 at 6.0 / 7.0). **Predicted KLs are over-optimistic** (A3: the 1.5 / 2.5 / 4.0 picks measured +80 / +58 / +27 % above prediction on fit129; vs the uniform start only -6 % [-18, +3] at 2.5 and -17 % [-23, -11] at 4.0, predicted -47 / -40 %): starting points for the search, not finals.

| size | bpw (container) | bpw (trunk + head) | head | embedding | pred KL (fit129, A8) | trunk |
|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | infeasible at K ≥ 1 (needs the sub-1 levels) |
| 1.25 | 1.2500 | 1.2373 | u2 | t1.5 | 0.28184 | K1–3, 321 tensors vs the uniform start (pred 0.55956) |
| 1.5 | 1.5000 | 1.4997 | u2.25 | t1.5 | 0.15511 | K1–3.375, 311 tensors vs the uniform start (pred 0.29737) |
| 1.75 | 1.7500 | 1.7622 | u2.375 | t1.5 | 0.09772 | K1–3.625, 310 tensors vs the uniform start (pred 0.17635) |
| 2 | 2.0000 | 2.0246 | u2.5 | t1.5 | 0.06543 | K1–3.875, 306 tensors vs the uniform start (pred 0.11156) |
| 2.5 | 2.5000 | 2.5432 | u2.875 | t1.625 | 0.03186 | K1–4.375, 309 tensors vs the uniform start (pred 0.05087) |
| 2.625 | 2.6250 | 2.6682 | u3 | t1.75 | 0.02686 | K1–4.5, 312 tensors vs the uniform start (pred 0.04246) |
| 2.75 | 2.7500 | 2.7932 | u3 | t1.875 | 0.02269 | K1–4.5, 313 tensors vs the uniform start (pred 0.03553) |
| 2.875 | 2.8750 | 2.9182 | u3.125 | t2 | 0.01920 | K1–4.625, 314 tensors vs the uniform start (pred 0.02985) |
| 3 | 3.0000 | 3.0432 | u3.125 | t2.125 | 0.01629 | K1–4.75, 317 tensors vs the uniform start (pred 0.02513) |
| 3.125 | 3.1250 | 3.1682 | u3.375 | t2.25 | 0.01382 | K1–4.875, 310 tensors vs the uniform start (pred 0.02119) |
| 3.25 | 3.2500 | 3.2932 | u3.375 | t2.375 | 0.01174 | K1–5, 315 tensors vs the uniform start (pred 0.01795) |
| 3.375 | 3.3750 | 3.4182 | u3.625 | t2.5 | 0.00999 | K1.125–5.25, 314 tensors vs the uniform start (pred 0.01522) |
| 3.5 | 3.5000 | 3.5432 | u3.625 | t2.625 | 0.00850 | K1.25–5.25, 316 tensors vs the uniform start (pred 0.01293) |
| 3.625 | 3.6250 | 3.6682 | u3.75 | t2.75 | 0.00725 | K1.25–5.5, 316 tensors vs the uniform start (pred 0.01098) |
| 3.75 | 3.7500 | 3.7932 | u3.75 | t2.875 | 0.00619 | K1.375–5.5, 316 tensors vs the uniform start (pred 0.00934) |
| 3.875 | 3.8750 | 3.9182 | u4 | t3 | 0.00529 | K1.5–5.75, 314 tensors vs the uniform start (pred 0.00796) |
| 4 | 4.0000 | 4.0432 | u4 | t3.125 | 0.00453 | K1.625–5.75, 319 tensors vs the uniform start (pred 0.00679) |
| 4.125 | 4.1250 | 4.1682 | u4.125 | t3.25 | 0.00388 | K1.625–6, 311 tensors vs the uniform start (pred 0.00579) |
| 4.25 | 4.2500 | 4.2932 | u4.125 | t3.375 | 0.00333 | K1.75–6, 319 tensors vs the uniform start (pred 0.00496) |
| 4.375 | 4.3750 | 4.4182 | u4.375 | t3.5 | 0.00287 | K1.875–6.25, 318 tensors vs the uniform start (pred 0.00425) |
| 4.5 | 4.5000 | 4.5432 | u4.5 | t3.625 | 0.00247 | K2–6.25, 313 tensors vs the uniform start (pred 0.00365) |
| 5 | 5.0000 | 5.0432 | u4.875 | t4.125 | 0.00140 | K2.375–6.75, 317 tensors vs the uniform start (pred 0.00202) |
| 6 | 6.0000 | 6.0494 | u6m | t5 | 0.00057 | K3.375–7, 304 tensors vs the uniform start (pred 0.00082) |
| 7 | 7.0000 | 6.9998 | u7.625m | t7m | 0.00040 | K5.25–7.5, 318 tensors vs the uniform start (pred 0.00043) |
