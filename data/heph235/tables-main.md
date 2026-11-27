# w235 tables (tag main; 36 matrices with HEPH results, 216 ours_metrics lines)

**Cross-check** ours_metrics rel vs w211 base (min over F3): 144 lines, max |rel diff| 0.00e+00

### HEPH std L=12 seed=1  (rate = K + Huffman exponents + 0.022)

| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.077 | 1.003 | 2.784 [2.546, 3.039] | **3.100 [2.832, 3.406]** | 3.376 [2.858, 4.023] | 2.882 [2.543, 3.307] | 3.063 [2.662, 3.601] | 4.015 [3.754, 4.289] | 2.394 [2.343, 2.449] | 0/36 | +0.75 |
| 2 | 2.077 | 2.003 | 2.707 [2.451, 3.005] | **2.992 [2.712, 3.299]** | 3.272 [2.710, 3.964] | 2.772 [2.387, 3.264] | 2.952 [2.504, 3.503] | 3.987 [3.692, 4.327] | 2.244 [2.186, 2.306] | 0/36 | +0.73 |
| 3 | 3.077 | 3.003 | 2.118 [1.898, 2.371] | **2.338 [2.083, 2.622]** | 2.529 [2.022, 3.149] | 2.170 [1.821, 2.607] | 2.328 [1.934, 2.806] | 3.232 [2.979, 3.507] | 1.691 [1.621, 1.754] | 0/36 | +0.61 |

### HEPH std L=12 without the 0.022 random-access offsets (Huffman exponents only)  (rate = K + Huffman exponents + 0.0)

| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.055 | 1.003 | 2.784 [2.540, 3.051] | **2.999 [2.752, 3.301]** | 3.266 [2.741, 3.845] | 2.788 [2.444, 3.185] | 2.963 [2.575, 3.459] | 3.881 [3.612, 4.148] | 2.318 [2.267, 2.372] | 0/36 | +0.73 |
| 2 | 2.055 | 2.003 | 2.707 [2.452, 2.990] | **2.901 [2.613, 3.218]** | 3.173 [2.647, 3.839] | 2.688 [2.333, 3.179] | 2.863 [2.429, 3.392] | 3.866 [3.580, 4.187] | 2.177 [2.120, 2.234] | 0/36 | +0.71 |
| 3 | 3.055 | 3.003 | 2.118 [1.896, 2.380] | **2.268 [2.024, 2.541]** | 2.454 [1.971, 3.063] | 2.106 [1.769, 2.558] | 2.259 [1.914, 2.717] | 3.135 [2.893, 3.403] | 1.641 [1.571, 1.697] | 0/36 | +0.59 |

### HEPH std L=16 (K = 2 only is run)  (rate = K + Huffman exponents + 0.022)

| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.077 | 2.003 | 2.799 [2.532, 3.113] | **3.092 [2.795, 3.424]** | 3.368 [2.792, 4.139] | 2.878 [2.501, 3.378] | 3.051 [2.594, 3.612] | 4.138 [3.838, 4.462] | 2.311 [2.251, 2.372] | 0/36 | +0.75 |

### HEPH std L=12, codebook seed 2  (rate = K + Huffman exponents + 0.022)

| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.077 | 2.003 | 2.690 [2.448, 2.976] | **2.973 [2.681, 3.300]** | 3.251 [2.651, 3.981] | 2.766 [2.397, 3.272] | 2.922 [2.473, 3.489] | 3.964 [3.658, 4.300] | 2.230 [2.173, 2.292] | 0/36 | +0.72 |
| 3 | 3.077 | 3.003 | 2.118 [1.890, 2.375] | **2.337 [2.082, 2.628]** | 2.528 [2.031, 3.141] | 2.169 [1.806, 2.606] | 2.329 [1.933, 2.838] | 3.245 [2.990, 3.522] | 1.684 [1.615, 1.742] | 0/36 | +0.61 |

### HEPH std L=12, fractional K (ours measured at K = 1.5 / 2.5, non-tail-biting legacy pattern search; interpolation uses all of ours' measured points)  (rate = K + Huffman exponents + 0.022)

| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 1.577 | 1.503 | 2.908 [2.648, 3.205] | **3.200 [2.903, 3.524]** | 3.484 [2.918, 4.164] | 2.983 [2.603, 3.446] | 3.154 [2.694, 3.756] | 4.216 [3.924, 4.522] | 2.429 [2.371, 2.490] | 0/36 | +0.71 |
| 2.5 | 2.577 | 2.503 | 2.646 [2.396, 2.949] | **2.903 [2.623, 3.228]** | 3.159 [2.593, 3.893] | 2.700 [2.321, 3.173] | 2.868 [2.413, 3.415] | 3.916 [3.621, 4.222] | 2.152 [2.088, 2.210] | 0/36 | +0.74 |

### HEPH sw (diag H x s_g^2 weights) L=12  (rate = K + Huffman exponents + 0.022)

| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 1.577 | 1.503 | 2.717 [2.475, 3.010] | **2.990 [2.714, 3.298]** | 3.272 [2.701, 3.965] | 2.777 [2.427, 3.218] | 2.943 [2.506, 3.477] | 3.952 [3.662, 4.265] | 2.263 [2.211, 2.319] | 0/36 | +0.67 |
| 2 | 2.077 | 2.003 | 2.550 [2.306, 2.817] | **2.838 [2.554, 3.163]** | 3.096 [2.554, 3.769] | 2.636 [2.260, 3.097] | 2.800 [2.336, 3.339] | 3.791 [3.504, 4.091] | 2.124 [2.069, 2.181] | 0/36 | +0.72 |
| 2.5 | 2.577 | 2.503 | 2.496 [2.258, 2.800] | **2.739 [2.459, 3.053]** | 2.988 [2.457, 3.646] | 2.541 [2.154, 3.005] | 2.706 [2.287, 3.259] | 3.709 [3.422, 4.003] | 2.022 [1.959, 2.078] | 0/36 | +0.70 |
| 3 | 3.077 | 3.003 | 1.989 [1.757, 2.238] | **2.195 [1.954, 2.471]** | 2.373 [1.908, 2.976] | 2.040 [1.699, 2.457] | 2.185 [1.813, 2.656] | 3.057 [2.833, 3.307] | 1.576 [1.513, 1.634] | 0/36 | +0.58 |

### HEPH h1 (no calibration, h = 1) L=12  (rate = K + Huffman exponents + 0.022)

| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3.077 | 3.003 | 2.588 [2.387, 2.805] | **2.856 [2.635, 3.097]** | 3.320 [2.940, 3.714] | 2.658 [2.384, 2.983] | 2.640 [2.286, 3.057] | 3.391 [3.104, 3.699] | 2.405 [2.217, 2.623] | 0/36 | +0.75 |

**Crossover (rate-matched geo ratio = 1, log-linear in K):** K1=3.100, K2=2.992, K3=2.338

### rel_diag: HEPH std L=12 / ours (same K; rate-matched needs ours K = 1..4)

| K | same-K ratio geo [CI] | rate-matched geo [CI] |
|---|---|---|
| 1 | 1.129 [1.105, 1.156] | 1.244 [1.216, 1.272] |
| 2 | 0.964 [0.939, 0.989] | 1.062 [1.036, 1.090] |
| 3 | 0.728 [0.708, 0.749] | 0.803 [0.780, 0.824] |

### rel_mse: HEPH std L=12 / ours (same K; rate-matched needs ours K = 1..4)

| K | same-K ratio geo [CI] | rate-matched geo [CI] |
|---|---|---|
| 1 | 1.243 [1.200, 1.286] | 1.367 [1.320, 1.419] |
| 2 | 1.068 [1.025, 1.113] | 1.176 [1.128, 1.223] |
| 3 | 0.847 [0.810, 0.885] | 0.934 [0.893, 0.977] |

### rel(full H) / rel(diag H) per method (geo mean over matrices; < 1: error sits in directions the full H weights less than its diagonal says)

| K | HEPH gate_up | ours gate_up | HEPH down | ours down |
|---|---|---|---|---|
| 1 | 0.659 | 0.198 | 0.962 | 0.527 |
| 2 | 0.634 | 0.160 | 0.959 | 0.482 |
| 3 | 0.623 | 0.150 | 0.958 | 0.469 |

### Exactness / on-grid / exponent cost / encode time (HEPH std L=12 seed 1)

| K | % bit-exact (mean, min..max) | on-grid all | infeasible seqs | enc ms per row gate_up / down |
|---|---|---|---|---|
| 1 | 21.5 (21.2..22.7) | True | 0 | 74.5 / 34.5 |
| 2 | 42.7 (42.3..44.1) | True | 0 | 28.7 / 14.0 |
| 3 | 72.9 (72.0..73.1) | True | 0 | 16.6 / 8.3 |

Exponent stream (all rows of each matrix; Huffman code per matrix; first byte of each row raw):

| kind | huff bpw mean [min..max] | entropy bits/delta | frac delta=0 | frac |delta|<=1 | first-byte bpw |
|---|---|---|---|---|---|
| gate_up | 0.0530 [0.0479..0.0551] | 1.424 | 0.564 | 0.9996 | 0.0016 |
| down | 0.0562 [0.0536..0.0574] | 1.474 | 0.527 | 0.9996 | 0.0035 |

### Per matrix (std L=12 seed 1; rel under the expert's full H; ours = w211 base best-of-3; ratio rate-matched)

| matrix | class | K | HEPH rel | ours rel (same K) | ours at HEPH rate | ratio |
|---|---|---|---|---|---|---|
| L5 e10 down | medium | 1 | 0.19615 | 0.10171 | 0.09135 | 2.147 |
| L5 e10 down | medium | 2 | 0.03986 | 0.02235 | 0.02024 | 1.969 |
| L5 e10 down | medium | 3 | 0.00650 | 0.00552 | 0.00501 | 1.298 |
| L5 e10 gate_up | medium | 1 | 0.27986 | 0.06686 | 0.05950 | 4.704 |
| L5 e10 gate_up | medium | 2 | 0.05983 | 0.01359 | 0.01224 | 4.888 |
| L5 e10 gate_up | medium | 3 | 0.01176 | 0.00327 | 0.00295 | 3.984 |
| L5 e12 down | medium | 1 | 0.33343 | 0.15546 | 0.13994 | 2.383 |
| L5 e12 down | medium | 2 | 0.07257 | 0.03573 | 0.03236 | 2.242 |
| L5 e12 down | medium | 3 | 0.01394 | 0.00896 | 0.00812 | 1.717 |
| L5 e12 gate_up | medium | 1 | 0.33058 | 0.09615 | 0.08576 | 3.855 |
| L5 e12 gate_up | medium | 2 | 0.07179 | 0.02048 | 0.01845 | 3.891 |
| L5 e12 gate_up | medium | 3 | 0.01461 | 0.00499 | 0.00451 | 3.241 |
| L5 e20 down | cold | 1 | 0.35591 | 0.16716 | 0.15062 | 2.363 |
| L5 e20 down | cold | 2 | 0.07820 | 0.03857 | 0.03497 | 2.236 |
| L5 e20 down | cold | 3 | 0.01529 | 0.00969 | 0.00879 | 1.740 |
| L5 e20 gate_up | cold | 1 | 0.25089 | 0.06566 | 0.05837 | 4.299 |
| L5 e20 gate_up | cold | 2 | 0.05266 | 0.01315 | 0.01185 | 4.445 |
| L5 e20 gate_up | cold | 3 | 0.01028 | 0.00315 | 0.00284 | 3.616 |
| L5 e40 down | hot | 1 | 0.25062 | 0.11371 | 0.10230 | 2.450 |
| L5 e40 down | hot | 2 | 0.05394 | 0.02563 | 0.02323 | 2.322 |
| L5 e40 down | hot | 3 | 0.00985 | 0.00641 | 0.00581 | 1.695 |
| L5 e40 gate_up | hot | 1 | 0.25546 | 0.05780 | 0.05135 | 4.975 |
| L5 e40 gate_up | hot | 2 | 0.05479 | 0.01146 | 0.01032 | 5.309 |
| L5 e40 gate_up | hot | 3 | 0.01109 | 0.00274 | 0.00247 | 4.486 |
| L5 e42 down | hot | 1 | 0.21497 | 0.10252 | 0.09195 | 2.338 |
| L5 e42 down | hot | 2 | 0.04388 | 0.02237 | 0.02024 | 2.168 |
| L5 e42 down | hot | 3 | 0.00744 | 0.00551 | 0.00500 | 1.489 |
| L5 e42 gate_up | hot | 1 | 0.27642 | 0.06745 | 0.05993 | 4.613 |
| L5 e42 gate_up | hot | 2 | 0.06182 | 0.01356 | 0.01220 | 5.066 |
| L5 e42 gate_up | hot | 3 | 0.01190 | 0.00325 | 0.00293 | 4.059 |
| L5 e126 down | cold | 1 | 0.36653 | 0.17535 | 0.15803 | 2.319 |
| L5 e126 down | cold | 2 | 0.08074 | 0.04075 | 0.03693 | 2.186 |
| L5 e126 down | cold | 3 | 0.01589 | 0.01024 | 0.00928 | 1.712 |
| L5 e126 gate_up | cold | 1 | 0.25215 | 0.06270 | 0.05566 | 4.530 |
| L5 e126 gate_up | cold | 2 | 0.05320 | 0.01233 | 0.01110 | 4.791 |
| L5 e126 gate_up | cold | 3 | 0.01049 | 0.00294 | 0.00266 | 3.949 |
| L20 e19 down | hot | 1 | 0.35180 | 0.14744 | 0.13342 | 2.637 |
| L20 e19 down | hot | 2 | 0.07813 | 0.03439 | 0.03129 | 2.497 |
| L20 e19 down | hot | 3 | 0.01532 | 0.00868 | 0.00790 | 1.939 |
| L20 e19 gate_up | hot | 1 | 0.30765 | 0.07890 | 0.07134 | 4.313 |
| L20 e19 gate_up | hot | 2 | 0.06590 | 0.01745 | 0.01589 | 4.146 |
| L20 e19 gate_up | hot | 3 | 0.01301 | 0.00432 | 0.00394 | 3.301 |
| L20 e59 down | medium | 1 | 0.36714 | 0.17724 | 0.16006 | 2.294 |
| L20 e59 down | medium | 2 | 0.08091 | 0.04171 | 0.03785 | 2.138 |
| L20 e59 down | medium | 3 | 0.01616 | 0.01054 | 0.00956 | 1.690 |
| L20 e59 gate_up | medium | 1 | 0.28375 | 0.09541 | 0.08558 | 3.316 |
| L20 e59 gate_up | medium | 2 | 0.06045 | 0.02082 | 0.01884 | 3.209 |
| L20 e59 gate_up | medium | 3 | 0.01234 | 0.00512 | 0.00464 | 2.660 |
| L20 e65 down | medium | 1 | 0.34966 | 0.16228 | 0.14617 | 2.392 |
| L20 e65 down | medium | 2 | 0.07677 | 0.03747 | 0.03395 | 2.261 |
| L20 e65 down | medium | 3 | 0.01491 | 0.00941 | 0.00853 | 1.747 |
| L20 e65 gate_up | medium | 1 | 0.32601 | 0.11150 | 0.09983 | 3.266 |
| L20 e65 gate_up | medium | 2 | 0.07098 | 0.02474 | 0.02233 | 3.179 |
| L20 e65 gate_up | medium | 3 | 0.01434 | 0.00612 | 0.00553 | 2.595 |
| L20 e133 down | hot | 1 | 0.32889 | 0.15229 | 0.13706 | 2.400 |
| L20 e133 down | hot | 2 | 0.07162 | 0.03554 | 0.03216 | 2.227 |
| L20 e133 down | hot | 3 | 0.01391 | 0.00893 | 0.00809 | 1.720 |
| L20 e133 gate_up | hot | 1 | 0.31364 | 0.08411 | 0.07509 | 4.177 |
| L20 e133 gate_up | hot | 2 | 0.06802 | 0.01807 | 0.01629 | 4.176 |
| L20 e133 gate_up | hot | 3 | 0.01381 | 0.00442 | 0.00399 | 3.460 |
| L20 e155 down | cold | 1 | 0.36920 | 0.17971 | 0.16202 | 2.279 |
| L20 e155 down | cold | 2 | 0.08065 | 0.04205 | 0.03811 | 2.116 |
| L20 e155 down | cold | 3 | 0.01577 | 0.01060 | 0.00961 | 1.641 |
| L20 e155 gate_up | cold | 1 | 0.26691 | 0.08528 | 0.07630 | 3.498 |
| L20 e155 gate_up | cold | 2 | 0.05664 | 0.01837 | 0.01659 | 3.414 |
| L20 e155 gate_up | cold | 3 | 0.01135 | 0.00450 | 0.00407 | 2.787 |
| L20 e188 down | cold | 1 | 0.38416 | 0.18047 | 0.16298 | 2.357 |
| L20 e188 down | cold | 2 | 0.08461 | 0.04251 | 0.03857 | 2.193 |
| L20 e188 down | cold | 3 | 0.01677 | 0.01072 | 0.00973 | 1.724 |
| L20 e188 gate_up | cold | 1 | 0.26016 | 0.08146 | 0.07310 | 3.559 |
| L20 e188 gate_up | cold | 2 | 0.05624 | 0.01758 | 0.01592 | 3.534 |
| L20 e188 gate_up | cold | 3 | 0.01114 | 0.00431 | 0.00391 | 2.852 |
| L35 e32 down | cold | 1 | 0.34636 | 0.15013 | 0.13512 | 2.563 |
| L35 e32 down | cold | 2 | 0.07552 | 0.03472 | 0.03144 | 2.402 |
| L35 e32 down | cold | 3 | 0.01413 | 0.00873 | 0.00791 | 1.786 |
| L35 e32 gate_up | cold | 1 | 0.25613 | 0.06457 | 0.05810 | 4.408 |
| L35 e32 gate_up | cold | 2 | 0.05533 | 0.01437 | 0.01303 | 4.246 |
| L35 e32 gate_up | cold | 3 | 0.01093 | 0.00356 | 0.00323 | 3.379 |
| L35 e34 down | hot | 1 | 0.34422 | 0.14816 | 0.13349 | 2.579 |
| L35 e34 down | hot | 2 | 0.07384 | 0.03369 | 0.03056 | 2.416 |
| L35 e34 down | hot | 3 | 0.01401 | 0.00843 | 0.00765 | 1.831 |
| L35 e34 gate_up | hot | 1 | 0.24001 | 0.05841 | 0.05260 | 4.563 |
| L35 e34 gate_up | hot | 2 | 0.05088 | 0.01294 | 0.01175 | 4.331 |
| L35 e34 gate_up | hot | 3 | 0.01006 | 0.00321 | 0.00292 | 3.451 |
| L35 e56 down | medium | 1 | 0.36612 | 0.16836 | 0.15137 | 2.419 |
| L35 e56 down | medium | 2 | 0.07988 | 0.03812 | 0.03451 | 2.315 |
| L35 e56 down | medium | 3 | 0.01539 | 0.00951 | 0.00862 | 1.786 |
| L35 e56 gate_up | medium | 1 | 0.28555 | 0.09913 | 0.08915 | 3.203 |
| L35 e56 gate_up | medium | 2 | 0.06185 | 0.02218 | 0.02009 | 3.079 |
| L35 e56 gate_up | medium | 3 | 0.01247 | 0.00549 | 0.00498 | 2.504 |
| L35 e104 down | hot | 1 | 0.34174 | 0.14800 | 0.13308 | 2.568 |
| L35 e104 down | hot | 2 | 0.07506 | 0.03411 | 0.03085 | 2.433 |
| L35 e104 down | hot | 3 | 0.01351 | 0.00855 | 0.00773 | 1.747 |
| L35 e104 gate_up | hot | 1 | 0.20702 | 0.04746 | 0.04271 | 4.848 |
| L35 e104 gate_up | hot | 2 | 0.04382 | 0.01067 | 0.00967 | 4.532 |
| L35 e104 gate_up | hot | 3 | 0.00854 | 0.00265 | 0.00241 | 3.544 |
| L35 e133 down | medium | 1 | 0.31814 | 0.15358 | 0.13804 | 2.305 |
| L35 e133 down | medium | 2 | 0.06752 | 0.03467 | 0.03139 | 2.151 |
| L35 e133 down | medium | 3 | 0.01240 | 0.00865 | 0.00784 | 1.583 |
| L35 e133 gate_up | medium | 1 | 0.30558 | 0.10200 | 0.09163 | 3.335 |
| L35 e133 gate_up | medium | 2 | 0.06793 | 0.02321 | 0.02099 | 3.237 |
| L35 e133 gate_up | medium | 3 | 0.01382 | 0.00580 | 0.00525 | 2.633 |
| L35 e143 down | cold | 1 | 0.40565 | 0.19066 | 0.17183 | 2.361 |
| L35 e143 down | cold | 2 | 0.08956 | 0.04511 | 0.04085 | 2.192 |
| L35 e143 down | cold | 3 | 0.01761 | 0.01141 | 0.01033 | 1.704 |
| L35 e143 gate_up | cold | 1 | 0.26750 | 0.08283 | 0.07467 | 3.583 |
| L35 e143 gate_up | cold | 2 | 0.05779 | 0.01893 | 0.01717 | 3.366 |
| L35 e143 gate_up | cold | 3 | 0.01111 | 0.00474 | 0.00430 | 2.583 |

### Interpolation check: ours measured (fractional K, non-TB) / log-linear interpolation of ours' integer K points

- K = 1.5: geo 0.920 [0.918, 0.923] over 36 matrices
- K = 2.5: geo 0.951 [0.950, 0.952] over 36 matrices

### HEPH on ALL rows vs the 288 / 320-row subset (std L=12 seed 1; all-rows rel / subset rel per matrix, geo [CI]); all-rows frac_exact, time

| K | n | rel all / subset | gate_up | down | % bit-exact (all rows) | enc s per matrix gate_up / down |
|---|---|---|---|---|---|---|
| 1 | 36 | 0.997 [0.989, 1.005] | 0.998 [0.982, 1.013] | 0.997 [0.993, 1.000] | 21.5 | 300 / 144 |
| 2 | 36 | 0.999 [0.990, 1.007] | 0.999 [0.981, 1.015] | 0.998 [0.993, 1.003] | 42.7 | 109 / 56 |
| 3 | 36 | 1.002 [0.993, 1.010] | 1.005 [0.986, 1.022] | 0.998 [0.993, 1.003] | 72.9 | 66 / 34 |

All-rows HEPH vs ours (ours on the subset; rows unpaired), rate-matched geo ratio [CI]:

- K = 1: 3.092 [2.827, 3.386] (n = 36)
- K = 2: 2.987 [2.696, 3.330] (n = 36)
- K = 3: 2.342 [2.089, 2.636] (n = 36)
