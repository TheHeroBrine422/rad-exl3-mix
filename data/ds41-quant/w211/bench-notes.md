# viterbi3 throughput on box2 (2026-11-20, vbench.py / vpar.sh; E3_V2=3 TB=1, mad, 48 random tiles per K, one process per logical cpu)
Per-tile ms (mean over processes): idle single process: K1 4.1, K2 3.5, K3 5.7 (noisy), K4 3.9.

| procs (cpus) | K1 | K2 | K3 | K4 |
|---|---|---|---|---|
| 12 (24-35, one per core) | 18.7 | 6.9 | 7.8 | 5.5 |
| 24 (24-47, one per core) | 22.5 | 11.1 | 9.7 | 5.7 |
| 48 (24-47 + SMT 72-95) | 43.0 | 23.0 | 18.4 | 11.0 |
Throughput (tiles/ms) at 24 procs: K1 1.07, K2 2.16, K3 2.5, K4 4.2; at 48: 1.1 / 2.1 / 2.6 / 4.4 (no SMT gain). K1/K2 are L3/memory bound
(cost history [256][2^(16-K)] floats = 33 / 16 MB per thread).
