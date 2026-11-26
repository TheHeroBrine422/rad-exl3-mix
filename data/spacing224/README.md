# #224 K-grid spacing vs continuous allocation (model-predicted)
`search/v3/spacing/spacing224.py` on colo, srch183 fits (b1 fit0611, b2 fit0612, b3 fit0610), menu173m-v2. Predicted KL of the
knapsack optimum (moves.mckp, iters 8) with the trunk on each K grid, as a ratio to the 1/64 grid (≈ continuous). "menu" = the
measured menu we ship (1/8 to K5, 1/4 to K7, K7.5). Off-grid levels interpolated (bytes linear, log error linear in K).

| size | K 1 | 1/2 | 1/4 | 1/8 | 1/16 | menu (shipped) |
|---|---|---|---|---|---|---|
| 2.0 | 1.106 | 1.022 | 1.005 | 1.0012 | 1.0003 | 1.0012 |
| 2.5 | 1.074 | 1.021 | 1.005 | 1.0010 | 1.0003 | 1.0010 |
| 3.0 | 1.061 | 1.015 | 1.004 | 1.0010 | 1.0002 | 1.0010 |
| 3.5 | 1.067 | 1.015 | 1.003 | 1.0010 | 1.0002 | 1.0010 |
| 4.0 | 1.054 | 1.015 | 1.003 | 1.0009 | 1.0002 | 1.0009 |
| 4.5 | 1.055 | 1.014 | 1.003 | 1.0008 | 1.0002 | 1.0010 |
| 5.0 | 1.053 | 1.013 | 1.003 | 1.0007 | 1.0002 | 1.0016 |
| 6.0 | 1.046 | 1.012 | 1.002 | 1.0004 | 1.0001 | 1.0016 |
| 7.0 | 1.003 | 1.002 | 1.000 | 1.0000 | 1.0000 | 1.0000 |
