# #222 per-size lr screens (box2 queue, 1-card trainer, w214 calib plan data, cap 8192)
Reads: j217 (results.tsv rows L30-b / L35-a), jc228 / jgap via qad226/jl226.py --golden. Paired per-doc bootstrap CI vs the unhealed release.

| arm | lr | j217 2M | jc228 2M | gap 2M | notes |
|---|---|---|---|---|---|
| L30-b (3.00) | flat 1e-4 | 0.953 [0.918, 0.983] (raw 1.5M 0.937) | 0.971 [0.948, 0.992], 559/642 better | 0.976 [0.881, 1.070]; w/o top 2 docs 0.937 | |
| L35-a (3.50) | flat 1e-4 | pending | 1.005 [0.941, 1.076], 473/642, median 0.984; w/o 2 docs 0.971 | 1.226 [1.060, 1.401], median 0.989; w/o 2 docs 1.117 | blow-ups: long-chat (WildChat) docs 172-184 in jgap, jc228 doc 452 (0.021 -> 0.202) |
| L40-a (4.00) | flat 1e-4 | mean 1.083 [0.980, 1.236]; w/o blow-ups 0.997; median doc 0.985 | 0.981 [0.944, 1.027] | 1.306 [1.131, 1.536] | mid-run blow-ups (raw 0.75M / 1M mean 2.7, w/o 1.08 / 1.12) |
| L40-b (4.00) | flat 5e-5 | mean 1.027 [0.980, 1.111]; w/o blow-ups 0.986; median doc 0.989 | 0.976 [0.959, 0.990] | 1.035 [0.983, 1.096] | milder (raw 1M mean 1.46, w/o 0.99) |
