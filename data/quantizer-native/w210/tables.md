### layer 3, plain settings (w209's): host pipeline (w209, `base`) vs the card pipeline (`plain`), seconds a stage

| tensor | n x k | K | build | total | prep | scale search | Hessian factor | tile search | LDLQ feedback | refit | pack | load (file) | record write | proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.self_attn.q_proj | 12288x5120 | 2 | host | 5.52 | 0.42 | 0.36 | 0.38 | 3.46 | 0.05 | 0.62 | 0.07 | 0.20 | 0.000 | 9.5224e-04 |
| 3.self_attn.q_proj | 12288x5120 | 2 | card | 4.47 | 0.06 | 0.30 | 0.04 | 3.48 | 0.09 | 0.47 | 0.03 | 0.43 | 0.010 | 9.5224e-04 |
| 3.self_attn.q_proj | 12288x5120 | 4 | host | 4.84 | 0.42 | 0.30 | 0.21 | 3.02 | 0.05 | 0.55 | 0.12 | 0.20 | 0.010 | 6.7043e-05 |
| 3.self_attn.q_proj | 12288x5120 | 4 | card | 4.00 | 0.04 | 0.26 | 0.04 | 3.03 | 0.09 | 0.47 | 0.06 | 0.43 | 0.018 | 6.7043e-05 |
| 3.self_attn.k_proj | 1024x5120 | 2 | host | 1.22 | 0.11 | 0.15 | 0.40 | 0.31 | 0.03 | 0.21 | 0.01 | 0.11 | 0.000 | 2.6460e-02 |
| 3.self_attn.k_proj | 1024x5120 | 2 | card | 0.59 | 0.01 | 0.12 | 0.04 | 0.31 | 0.03 | 0.08 | 0.00 | 0.28 | 0.001 | 2.6460e-02 |
| 3.self_attn.k_proj | 1024x5120 | 4 | host | 0.92 | 0.10 | 0.14 | 0.21 | 0.27 | 0.02 | 0.14 | 0.01 | 0.11 | 0.000 | 2.4625e-03 |
| 3.self_attn.k_proj | 1024x5120 | 4 | card | 0.56 | 0.00 | 0.14 | 0.04 | 0.27 | 0.03 | 0.08 | 0.00 | 0.28 | 0.002 | 2.4625e-03 |
| 3.self_attn.v_proj | 1024x5120 | 2 | host | 1.20 | 0.11 | 0.15 | 0.39 | 0.31 | 0.03 | 0.20 | 0.01 | 0.11 | 0.000 | 3.0864e-02 |
| 3.self_attn.v_proj | 1024x5120 | 2 | card | 0.64 | 0.01 | 0.12 | 0.04 | 0.31 | 0.06 | 0.10 | 0.00 | 0.36 | 0.002 | 3.0864e-02 |
| 3.self_attn.v_proj | 1024x5120 | 4 | host | 0.91 | 0.11 | 0.14 | 0.20 | 0.27 | 0.02 | 0.14 | 0.01 | 0.11 | 0.000 | 2.3343e-03 |
| 3.self_attn.v_proj | 1024x5120 | 4 | card | 0.62 | 0.01 | 0.14 | 0.04 | 0.27 | 0.05 | 0.10 | 0.00 | 0.36 | 0.003 | 2.3343e-03 |
| 3.self_attn.o_proj | 5120x6144 | 2 | host | 3.36 | 0.28 | 0.19 | 0.51 | 1.75 | 0.04 | 0.47 | 0.03 | 0.19 | 0.000 | 8.6447e-04 |
| 3.self_attn.o_proj | 5120x6144 | 2 | card | 2.48 | 0.03 | 0.15 | 0.06 | 1.74 | 0.17 | 0.31 | 0.01 | 0.42 | 0.008 | 8.6447e-04 |
| 3.self_attn.o_proj | 5120x6144 | 4 | host | 2.89 | 0.28 | 0.17 | 0.31 | 1.53 | 0.04 | 0.40 | 0.06 | 0.19 | 0.000 | 5.3979e-05 |
| 3.self_attn.o_proj | 5120x6144 | 4 | card | 2.33 | 0.02 | 0.16 | 0.06 | 1.51 | 0.19 | 0.32 | 0.05 | 0.42 | 0.015 | 5.3979e-05 |
| 3.mlp.gate_proj | 17408x5120 | 2 | host | 7.49 | 0.54 | 0.51 | 0.36 | 4.91 | 0.06 | 0.78 | 0.09 | 0.23 | 0.010 | 3.0514e-02 |
| 3.mlp.gate_proj | 17408x5120 | 2 | card | 6.45 | 0.09 | 0.44 | 0.04 | 4.93 | 0.20 | 0.68 | 0.07 | 0.52 | 0.020 | 3.0514e-02 |
| 3.mlp.gate_proj | 17408x5120 | 4 | host | 6.68 | 0.57 | 0.43 | 0.20 | 4.29 | 0.06 | 0.73 | 0.17 | 0.23 | 0.010 | 1.9292e-03 |
| 3.mlp.gate_proj | 17408x5120 | 4 | card | 5.78 | 0.07 | 0.38 | 0.04 | 4.29 | 0.19 | 0.67 | 0.12 | 0.52 | 0.028 | 1.9292e-03 |
| 3.mlp.up_proj | 17408x5120 | 2 | host | 7.53 | 0.56 | 0.52 | 0.37 | 4.93 | 0.06 | 0.78 | 0.09 | 0.24 | 0.010 | 3.9289e-02 |
| 3.mlp.up_proj | 17408x5120 | 2 | card | 6.39 | 0.09 | 0.44 | 0.04 | 4.97 | 0.13 | 0.66 | 0.05 | 0.51 | 0.014 | 3.9289e-02 |
| 3.mlp.up_proj | 17408x5120 | 4 | host | 6.68 | 0.56 | 0.43 | 0.20 | 4.30 | 0.06 | 0.73 | 0.17 | 0.24 | 0.010 | 2.4895e-03 |
| 3.mlp.up_proj | 17408x5120 | 4 | card | 5.68 | 0.07 | 0.38 | 0.04 | 4.33 | 0.11 | 0.66 | 0.09 | 0.51 | 0.025 | 2.4895e-03 |
| 3.mlp.down_proj | 5120x17408 | 2 | host | 13.41 | 1.36 | 0.51 | 2.83 | 5.04 | 0.19 | 3.15 | 0.09 | 1.16 | 0.010 | 3.7883e-02 |
| 3.mlp.down_proj | 5120x17408 | 2 | card | 8.86 | 0.08 | 0.44 | 0.47 | 5.09 | 0.30 | 2.43 | 0.05 | 2.10 | 0.013 | 3.7883e-02 |
| 3.mlp.down_proj | 5120x17408 | 4 | host | 12.61 | 1.38 | 0.43 | 2.67 | 4.42 | 0.19 | 3.10 | 0.17 | 1.16 | 0.010 | 2.4132e-03 |
| 3.mlp.down_proj | 5120x17408 | 4 | card | 8.14 | 0.05 | 0.38 | 0.47 | 4.44 | 0.28 | 2.42 | 0.09 | 2.10 | 0.038 | 2.4132e-03 |

| K | build | layer s | prep | scale search | Hessian factor | tile search | LDLQ feedback | refit | pack | not the tile search | load | write |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | card | 29.9 | 0.35 | 2.02 | 0.72 | 20.83 | 0.97 | 4.73 | 0.21 | 9.1 | 4.62 | 0.07 |
| 2 | host | 39.7 | 3.38 | 2.39 | 5.24 | 20.71 | 0.46 | 6.21 | 0.39 | 19.0 | 2.24 | 0.03 |
| 4 | card | 27.1 | 0.27 | 1.83 | 0.73 | 18.14 | 0.94 | 4.73 | 0.41 | 9.0 | 4.62 | 0.13 |
| 4 | host | 35.5 | 3.42 | 2.04 | 4.00 | 18.10 | 0.44 | 5.79 | 0.71 | 17.4 | 2.24 | 0.04 |

### card pipeline vs w209's host pipeline, layer 3, K2 / K4: 42 of 42 files byte-identical (record, suh, svh of 7 tensors x 2 rates; `raw/ident/ident.txt`)

### synthetic expert batches (random weights, random SPD Hessians), one card (card 2), `exl3v3_tool batch`

| run | matrix n x k | K | matrices | batch | Hessians | pipe | wall s | matrices/s | values/s | tiles/s | card s | tile search | scale search | refit | factor | prep | feedback | pack | reader s | writer s | proxy mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| down-B16-r | 5120x2304 | 2 | 128 | 16 | one a matrix | yes | 119.3 | 1.072 | 12.65M | 49.4k | 118.0 | 86.2 | 16.8 | 8.8 | 1.8 | 2.7 | 1.0 | 0.7 | 7.9 | 2.1 | 1.2102e-02 |
| down-B16-r | 5120x2304 | 4 | 128 | 16 | one a matrix | yes | 106.0 | 1.207 | 14.24M | 55.6k | 104.7 | 74.7 | 14.7 | 8.8 | 1.5 | 2.3 | 1.3 | 1.3 | 11.2 | 1.4 | 7.6565e-04 |
| down-B16 | 5120x2304 | 2 | 128 | 16 | one a matrix | yes | 130.4 | 0.981 | 11.58M | 45.2k | 128.1 | 85.3 | 16.9 | 11.9 | 4.1 | 5.4 | 3.1 | 1.3 | 13.2 | 4.0 | 1.2102e-02 |
| down-B16 | 5120x2304 | 4 | 128 | 16 | one a matrix | yes | 114.3 | 1.120 | 13.21M | 51.6k | 111.8 | 74.0 | 14.6 | 10.1 | 3.0 | 5.6 | 2.4 | 2.0 | 21.1 | 4.5 | 7.6565e-04 |
| down-B32 | 5120x2304 | 4 | 128 | 32 | one a matrix | yes | 113.1 | 1.132 | 13.35M | 52.1k | 109.6 | 74.2 | 14.6 | 11.0 | 2.8 | 3.6 | 1.5 | 1.8 | 10.7 | 2.8 | 7.6565e-04 |
| gu-B1 | 2304x5120 | 4 | 16 | 1 | one a matrix | no | 28.9 | 0.553 | 6.53M | 25.5k | 19.9 | 10.1 | 1.8 | 3.0 | 2.1 | 0.3 | 2.4 | 0.1 | 7.8 | 1.1 | 5.8616e-04 |
| gu-B16-nopipe | 2304x5120 | 4 | 64 | 16 | one a pair | no | 75.7 | 0.846 | 9.97M | 39.0k | 60.0 | 36.9 | 7.3 | 10.0 | 2.6 | 0.7 | 1.4 | 1.0 | 12.6 | 2.9 | 5.8389e-04 |
| gu-B16-own | 2304x5120 | 4 | 64 | 16 | one a matrix | yes | 76.2 | 0.840 | 9.91M | 38.7k | 70.0 | 36.3 | 7.3 | 12.0 | 5.5 | 3.8 | 4.0 | 1.1 | 17.9 | 4.3 | 5.8611e-04 |
| gu-B16-pair-r | 2304x5120 | 2 | 128 | 16 | one a pair | yes | 135.0 | 0.948 | 11.18M | 43.7k | 131.0 | 85.7 | 16.8 | 17.6 | 3.8 | 4.1 | 2.1 | 0.8 | 23.1 | 3.4 | 9.1312e-03 |
| gu-B16-pair-r | 2304x5120 | 4 | 128 | 16 | one a pair | yes | 117.3 | 1.091 | 12.87M | 50.3k | 115.1 | 74.6 | 14.6 | 17.4 | 2.6 | 2.9 | 1.7 | 1.2 | 13.8 | 2.2 | 5.8391e-04 |
| gu-B16-pair-ss4 | 2304x5120 | 4 | 64 | 16 | one a pair | yes | 58.7 | 1.091 | 12.87M | 50.3k | 56.5 | 36.7 | 1.8 | 10.5 | 2.4 | 1.9 | 2.0 | 1.1 | 6.8 | 1.6 | 5.8377e-04 |
| gu-B16-pair | 2304x5120 | 2 | 128 | 16 | one a pair | yes | 141.2 | 0.906 | 10.69M | 41.8k | 138.3 | 84.5 | 16.7 | 23.1 | 4.1 | 4.8 | 3.8 | 1.1 | 29.6 | 3.8 | 9.1312e-03 |
| gu-B16-pair | 2304x5120 | 4 | 128 | 16 | one a pair | yes | 118.1 | 1.084 | 12.79M | 50.0k | 116.4 | 74.5 | 14.7 | 17.4 | 2.9 | 3.6 | 2.0 | 1.2 | 11.2 | 2.9 | 5.8391e-04 |
| gu-B16-pooled | 2304x5120 | 4 | 64 | 16 | one pooled | yes | 61.8 | 1.035 | 12.21M | 47.7k | 59.7 | 37.2 | 7.3 | 10.6 | 1.0 | 1.4 | 1.0 | 1.1 | 6.6 | 0.9 | 5.8539e-04 |
| gu-B32-pair | 2304x5120 | 4 | 128 | 32 | one a pair | yes | 134.8 | 0.950 | 11.20M | 43.8k | 125.9 | 74.0 | 14.7 | 22.8 | 4.4 | 5.0 | 2.7 | 2.4 | 26.8 | 5.0 | 5.8391e-04 |
| gu-B4 | 2304x5120 | 4 | 32 | 4 | one a pair | yes | 34.0 | 0.941 | 11.10M | 43.4k | 32.9 | 18.4 | 3.6 | 4.9 | 1.9 | 1.8 | 2.0 | 0.3 | 7.3 | 1.5 | 5.8393e-04 |
| two-c2 | 2304x5120 | 4 | 128 | 16 | one a pair | yes | 117.1 | 1.093 | 12.90M | 50.4k | 113.9 | 74.0 | 14.4 | 17.2 | 2.8 | 2.8 | 1.5 | 1.2 | 22.5 | 2.3 | 5.8391e-04 |
| two-c3 | 2304x5120 | 4 | 128 | 16 | one a pair | yes | 118.9 | 1.076 | 12.70M | 49.6k | 116.7 | 73.6 | 14.4 | 18.4 | 3.2 | 3.0 | 2.4 | 1.6 | 13.2 | 2.4 | 5.8391e-04 |

### extrapolations

- 27B trunk (64 layers ~ layer 3), one level at K2, plain settings: card pipeline 32 min (w209's host pipeline, same session: 42 min)
- 27B trunk (64 layers ~ layer 3), one level at K4, plain settings: card pipeline 29 min (w209's host pipeline, same session: 38 min)
- DS4.1-F routed experts at K2 (15,360 x 3 matrices, 2.12B tiles): an expert (gate + up sharing a Hessian, down with its own) = 3.13 card-s, of which 2.01 s tile search (gate / up 0.93, down 1.03 matrices/s, the mean of two passes of 128, batches of 16, with the reader / writer threads) -> 13.4 card-h = **3.3 h on 4 cards**, factorisations, scale searches and refits included (the search alone at this rate: 2.1 h; w209's search-only line at the full-batch rate: 1.7-1.9 h). The ~7B trunk weights add ~0.04 h at layer 3's rate. Not in it: reading 290 GB of weights and the per-expert Hessians from disk (the synthetic reader copies from memory), more than one rate
- DS4.1-F routed experts at K4 (15,360 x 3 matrices, 2.12B tiles): an expert (gate + up sharing a Hessian, down with its own) = 2.70 card-s, of which 1.75 s tile search (gate / up 1.09, down 1.16 matrices/s, the mean of two passes of 128, batches of 16, with the reader / writer threads) -> 11.5 card-h = **2.9 h on 4 cards**, factorisations, scale searches and refits included (the search alone at this rate: 1.9 h; w209's search-only line at the full-batch rate: 1.7-1.9 h). The ~7B trunk weights add ~0.04 h at layer 3's rate. Not in it: reading 290 GB of weights and the per-expert Hessians from disk (the synthetic reader copies from memory), more than one rate

### layer 3 with the release's settings: new (`exl3v3_tool`) vs the release pipeline (rxq), same card

| settings | K | tensor | new proxy | rxq proxy | new / rxq - 1 | new 2-sided proxy | rxq 2-sided | new s | of which search | CD | rxq s | new / rxq time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 2 | 3.self_attn.q_proj | 9.5224e-04 | 9.5379e-04 | -0.16% | — | — | 4.5 | 3.5 | 0.0 | 5.8 | 0.77x |
| plain | 2 | 3.self_attn.k_proj | 2.6460e-02 | 2.7393e-02 | -3.41% | — | — | 0.6 | 0.3 | 0.0 | 1.6 | 0.36x |
| plain | 2 | 3.self_attn.v_proj | 3.0864e-02 | 3.1241e-02 | -1.21% | — | — | 0.6 | 0.3 | 0.0 | 1.6 | 0.40x |
| plain | 2 | 3.self_attn.o_proj | 8.6447e-04 | 8.6426e-04 | +0.02% | — | — | 2.5 | 1.7 | 0.0 | 3.8 | 0.65x |
| plain | 2 | 3.mlp.gate_proj | 3.0514e-02 | 3.0511e-02 | +0.01% | — | — | 6.5 | 4.9 | 0.0 | 7.9 | 0.81x |
| plain | 2 | 3.mlp.up_proj | 3.9289e-02 | 3.9286e-02 | +0.01% | — | — | 6.4 | 5.0 | 0.0 | 7.9 | 0.81x |
| plain | 2 | 3.mlp.down_proj | 3.7883e-02 | 3.7875e-02 | +0.02% | — | — | 8.9 | 5.1 | 0.0 | 16.2 | 0.55x |
| plain | 4 | 3.self_attn.q_proj | 6.7043e-05 | 6.6328e-05 | +1.08% | — | — | 4.0 | 3.0 | 0.0 | 4.8 | 0.84x |
| plain | 4 | 3.self_attn.k_proj | 2.4625e-03 | 2.5136e-03 | -2.03% | — | — | 0.6 | 0.3 | 0.0 | 0.9 | 0.59x |
| plain | 4 | 3.self_attn.v_proj | 2.3343e-03 | 2.3978e-03 | -2.65% | — | — | 0.6 | 0.3 | 0.0 | 0.9 | 0.66x |
| plain | 4 | 3.self_attn.o_proj | 5.3979e-05 | 5.3890e-05 | +0.17% | — | — | 2.3 | 1.5 | 0.0 | 2.9 | 0.81x |
| plain | 4 | 3.mlp.gate_proj | 1.9292e-03 | 1.9285e-03 | +0.04% | — | — | 5.8 | 4.3 | 0.0 | 6.7 | 0.87x |
| plain | 4 | 3.mlp.up_proj | 2.4895e-03 | 2.4901e-03 | -0.03% | — | — | 5.7 | 4.3 | 0.0 | 7.0 | 0.82x |
| plain | 4 | 3.mlp.down_proj | 2.4132e-03 | 2.4128e-03 | +0.01% | — | — | 8.1 | 4.4 | 0.0 | 14.7 | 0.55x |
| ladder | 2 | 3.self_attn.q_proj | 7.4016e-04 | 7.4459e-04 | -0.59% | — | — | 4.5 | 3.5 | 0.0 | 5.4 | 0.84x |
| ladder | 2 | 3.self_attn.k_proj | 1.1370e-02 | 1.1367e-02 | +0.03% | — | — | 0.6 | 0.3 | 0.0 | 1.0 | 0.57x |
| ladder | 2 | 3.self_attn.v_proj | 2.1739e-02 | 2.1744e-02 | -0.03% | — | — | 0.6 | 0.3 | 0.0 | 1.0 | 0.57x |
| ladder | 2 | 3.self_attn.o_proj | 8.5430e-04 | 8.5159e-04 | +0.32% | — | — | 2.4 | 1.8 | 0.0 | 3.3 | 0.74x |
| ladder | 2 | 3.mlp.gate_proj | 3.0391e-02 | 3.0389e-02 | +0.01% | — | — | 6.4 | 5.0 | 0.0 | 7.5 | 0.86x |
| ladder | 2 | 3.mlp.up_proj | 3.9038e-02 | 3.9041e-02 | -0.01% | — | — | 6.4 | 5.0 | 0.0 | 7.4 | 0.86x |
| ladder | 2 | 3.mlp.down_proj | 3.7829e-02 | 3.7830e-02 | -0.00% | — | — | 8.8 | 5.1 | 0.0 | 15.5 | 0.57x |
| ladder | 4 | 3.self_attn.q_proj | 4.6226e-05 | 4.6163e-05 | +0.14% | — | — | 4.1 | 3.1 | 0.0 | 5.1 | 0.80x |
| ladder | 4 | 3.self_attn.k_proj | 7.0889e-04 | 7.0838e-04 | +0.07% | — | — | 0.6 | 0.3 | 0.0 | 1.2 | 0.46x |
| ladder | 4 | 3.self_attn.v_proj | 1.3566e-03 | 1.3569e-03 | -0.02% | — | — | 0.6 | 0.3 | 0.0 | 1.2 | 0.46x |
| ladder | 4 | 3.self_attn.o_proj | 5.3217e-05 | 5.3039e-05 | +0.34% | — | — | 2.2 | 1.6 | 0.0 | 3.3 | 0.67x |
| ladder | 4 | 3.mlp.gate_proj | 1.9195e-03 | 1.9187e-03 | +0.04% | — | — | 5.7 | 4.4 | 0.0 | 6.9 | 0.82x |
| ladder | 4 | 3.mlp.up_proj | 2.4703e-03 | 2.4708e-03 | -0.02% | — | — | 5.7 | 4.4 | 0.0 | 6.9 | 0.82x |
| ladder | 4 | 3.mlp.down_proj | 2.4097e-03 | 2.4098e-03 | -0.00% | — | — | 8.1 | 4.5 | 0.0 | 16.6 | 0.49x |
| release | 2 | 3.self_attn.q_proj | 9.4661e-04 | 9.5073e-04 | -0.43% | 6.7314e-04 | 1.3044e-03 | 16.9 | 3.6 | 12.0 | 24.2 | 0.70x |
| release | 2 | 3.self_attn.k_proj | 1.1795e-02 | 1.1786e-02 | +0.07% | 4.4019e-03 | 2.6864e-03 | 0.7 | 0.3 | 0.0 | 1.4 | 0.46x |
| release | 2 | 3.self_attn.v_proj | 2.6145e-02 | 2.6206e-02 | -0.23% | 1.2452e-02 | 6.6022e-03 | 0.6 | 0.3 | 0.0 | 1.4 | 0.45x |
| release | 2 | 3.self_attn.o_proj | 8.2130e-04 | 8.2153e-04 | -0.03% | 2.0586e-03 | 6.0091e-04 | 2.7 | 1.8 | 0.0 | 3.7 | 0.73x |
| release | 2 | 3.mlp.gate_proj | 3.1939e-02 | 3.1937e-02 | +0.01% | 2.5638e-02 | 2.4895e-02 | 6.9 | 5.1 | 0.0 | 8.0 | 0.87x |
| release | 2 | 3.mlp.up_proj | 4.1697e-02 | 4.1701e-02 | -0.01% | 3.2388e-02 | 3.0376e-02 | 7.0 | 5.0 | 0.0 | 8.1 | 0.86x |
| release | 2 | 3.mlp.down_proj | 3.8468e-02 | 3.8467e-02 | +0.00% | 3.5648e-02 | 3.5833e-02 | 10.5 | 5.2 | 0.0 | 15.8 | 0.66x |
| release | 4 | 3.self_attn.q_proj | 5.9181e-05 | 5.9266e-05 | -0.14% | 4.0648e-05 | 7.9192e-05 | 15.0 | 3.1 | 10.6 | 19.8 | 0.76x |
| release | 4 | 3.self_attn.k_proj | 7.3323e-04 | 7.3287e-04 | +0.05% | 2.7063e-04 | 1.6003e-04 | 0.6 | 0.3 | 0.0 | 1.2 | 0.50x |
| release | 4 | 3.self_attn.v_proj | 1.6486e-03 | 1.6520e-03 | -0.21% | 7.5799e-04 | 3.5846e-04 | 0.6 | 0.3 | 0.0 | 1.1 | 0.55x |
| release | 4 | 3.self_attn.o_proj | 5.1234e-05 | 5.1164e-05 | +0.14% | 1.2863e-04 | 3.7326e-05 | 2.4 | 1.6 | 0.0 | 3.3 | 0.74x |
| release | 4 | 3.mlp.gate_proj | 2.0217e-03 | 2.0207e-03 | +0.05% | 1.6124e-03 | 1.5665e-03 | 6.2 | 4.4 | 0.0 | 7.3 | 0.86x |
| release | 4 | 3.mlp.up_proj | 2.6475e-03 | 2.6486e-03 | -0.04% | 2.0387e-03 | 1.9139e-03 | 6.2 | 4.4 | 0.0 | 7.2 | 0.87x |
| release | 4 | 3.mlp.down_proj | 2.4528e-03 | 2.4525e-03 | +0.01% | 2.2650e-03 | 2.2780e-03 | 10.0 | 4.5 | 0.0 | 16.1 | 0.62x |

| settings | K | tensors | new layer s | of which tile search | rxq layer s | new / rxq time | geo-mean proxy new / rxq |
|---|---|---|---|---|---|---|---|
| plain | 2 | 7 | 29.9 | 20.8 | 44.9 | 0.67x | -0.68 % |
| plain | 4 | 7 | 27.1 | 18.1 | 37.9 | 0.71x | -0.50 % |
| ladder | 2 | 7 | 29.7 | 21.2 | 41.2 | 0.72x | -0.04 % |
| ladder | 4 | 7 | 26.8 | 18.4 | 41.3 | 0.65x | +0.08 % |
| release | 2 | 7 | 45.3 | 21.4 | 62.7 | 0.72x | -0.09 % |
| release | 4 | 7 | 41.1 | 18.5 | 55.9 | 0.74x | -0.02 % |

### the extras, each against the release pipeline (rxq) on the same tensor (different sign draws)

| extra | tensor | K | new: off -> on | new on / off | rxq: off -> on | rxq on / off | new on / rxq on - 1 | note |
|---|---|---|---|---|---|---|---|---|
| tile CD 3 (no H_out) | 3.self_attn.q_proj | 4 | 4.6226e-05 -> 4.5655e-05 | 0.9876 | 4.6163e-05 -> 4.5598e-05 | 0.9878 | +0.13% | CD objective after / before: new 0.9882, rxq 0.9883; CD time new 10.6 s |
| tile CD 3 (no H_out) | 3.self_attn.k_proj | 4 | 7.0889e-04 -> 7.0151e-04 | 0.9896 | 7.0838e-04 -> 7.0137e-04 | 0.9901 | +0.02% | CD objective after / before: new 0.9915, rxq 0.9920; CD time new 1.2 s |
| H_out two-sided LDLQ: one-sided proxy | 3.self_attn.q_proj | 4 | 4.6226e-05 -> 5.9224e-05 | 1.2812 | 4.6163e-05 -> 5.9247e-05 | 1.2834 | -0.04% |  |
| H_out two-sided LDLQ: two-sided proxy | 3.self_attn.q_proj | 4 | 4.1115e-05 | — | 7.8966e-05 | — | not comparable | scaled Hessian: new reports tr(E^T H E M) in the metric LDLQ minimised (the scaled one), rxq in the unscaled rotated H; the next rows have one definition |
| H_out two-sided LDLQ: one-sided proxy | 3.self_attn.k_proj | 4 | 7.0889e-04 -> 7.3323e-04 | 1.0343 | 7.0838e-04 -> 7.3287e-04 | 1.0346 | +0.05% |  |
| H_out two-sided LDLQ: two-sided proxy | 3.self_attn.k_proj | 4 | 2.7063e-04 | — | 1.6003e-04 | — | not comparable | scaled Hessian: new reports tr(E^T H E M) in the metric LDLQ minimised (the scaled one), rxq in the unscaled rotated H; the next rows have one definition |
| H_out two-sided LDLQ: one-sided proxy | 3.self_attn.o_proj | 4 | 5.3217e-05 -> 5.1234e-05 | 0.9627 | 5.3039e-05 -> 5.1164e-05 | 0.9646 | +0.14% |  |
| H_out two-sided LDLQ: two-sided proxy | 3.self_attn.o_proj | 4 | 1.2863e-04 | — | 3.7326e-05 | — | not comparable | scaled Hessian: new reports tr(E^T H E M) in the metric LDLQ minimised (the scaled one), rxq in the unscaled rotated H; the next rows have one definition |
| H_out two-sided LDLQ: one-sided proxy | 3.mlp.gate_proj | 4 | 1.9195e-03 -> 2.0217e-03 | 1.0533 | 1.9187e-03 -> 2.0207e-03 | 1.0532 | +0.05% |  |
| H_out two-sided LDLQ: two-sided proxy | 3.mlp.gate_proj | 4 | 1.6124e-03 | — | 1.5665e-03 | — | not comparable | scaled Hessian: new reports tr(E^T H E M) in the metric LDLQ minimised (the scaled one), rxq in the unscaled rotated H; the next rows have one definition |
| H_out, no scaled Hessian: one-sided / two-sided proxy | 3.self_attn.k_proj | 4 | 2.6785e-03 / 9.3484e-05 | — | 2.7910e-03 / 9.4271e-05 | — | -4.03% / -0.84% | the two-sided proxy has one definition here |
| H_out, no scaled Hessian: one-sided / two-sided proxy | 3.self_attn.o_proj | 4 | 5.2041e-05 / 3.4607e-05 | — | 5.1983e-05 / 3.4581e-05 | — | +0.11% / +0.08% | the two-sided proxy has one definition here |
| H_out + tile CD 3 (the release's q) | 3.self_attn.q_proj | 4 | 5.9181e-05 | — | 5.9266e-05 | — | -0.14% | CD on the two-sided objective: after / before new 0.9886, rxq 0.9881; CD time new 10.6 s |
| scale clamp | 3.self_attn.k_proj | 0.125 | 4.1443e-01 -> 4.1443e-01 | 1.0000 | 4.1376e-01 -> 4.1376e-01 | 1.0000 | +0.16% | scales replaced: new 0 (rxq's log has no count) |
| scale clamp | 3.self_attn.v_proj | 0.125 | 6.2932e-01 -> 6.2932e-01 | 1.0000 | 6.3057e-01 -> 6.3057e-01 | 1.0000 | -0.20% | scales replaced: new 0 (rxq's log has no count) |
| scale clamp | 3.self_attn.o_proj | 0.125 | 2.5556e-02 -> 2.5561e-02 | 1.0002 | 2.5650e-02 -> 2.5657e-02 | 1.0003 | -0.37% | scales replaced: new 4 (rxq's log has no count) |

- R16 `r16-v`: proxy 1.3560e-03, the record decoded by the oracle (stored row order) 1.3561e-03
- R16 `nor16-v`: proxy 1.3566e-03, the record decoded by the oracle (stored row order) 1.3567e-03
- R16 `r16-o`: proxy 5.3186e-05, the record decoded by the oracle (stored row order) 5.3305e-05
- stacked gate_up `stack-gu` K2: the stacked record (suh.1) through `exl3i_decode_rows` vs the two halves decoded alone: max |diff| = 0; batch of 2 in 14.2 s (factor 0.85 s)
- stacked gate_up `stack-gu` K4: the stacked record (suh.1) through `exl3i_decode_rows` vs the two halves decoded alone: max |diff| = 0; batch of 2 in 11.9 s (factor 0.07 s)
- stacked gate_up `stack-gu-sameseed` K4: the stacked record (suh.1) through `exl3i_decode_rows` vs the two halves decoded alone: max |diff| = 0; batch of 2 in 13.0 s (factor 0.92 s)
- swj: `{"swj": "model.language_model.layers.3.mlp.gate_proj", "K": 2.5, "n": 17408, "k": 5120, "rows": 2048, "resid_before": 0.08307553, "resid_after": 0.08203232, "cg_mean": 1.00142, "cu_mean": 0.99019, "fit_sec": 0.26}`
- swj: `{"swj_ref": "model.language_model.layers.3.mlp.gate_proj", "channels": 17408, "rows": 2048, "cg_same": 0.8781594634056091, "cg_max_diff": 0.010000109672546387, "cu_max_diff": 0.02685987949371338, "cu_mean_diff": 0.0012493212707340717, "resid_before_ref": 0.08307554342675025, "resid_after_ref": 0.08203234248797349, "cg_mean_ref": 1.0014338493347168, "cu_mean_ref": 0.9901596307754517}`

### the scale search's g and the proxy at low rates, new vs rxq (3.k_proj, 3.v_proj; ladder settings, the release's sub-1 dampings)

| tensor | K | new g | rxq g | new / rxq - 1 | new proxy | rxq proxy | new / rxq - 1 | new g, f32 search | new g, another seed |
|---|---|---|---|---|---|---|---|---|---|
| 3.self_attn.k_proj | 0.125 | 1.7564 | 1.7176 | +2.26% | 4.1443e-01 | 4.1376e-01 | +0.16% | 1.7172 | 1.7161 |
| 3.self_attn.v_proj | 0.125 | 1.7604 | 1.7198 | +2.36% | 6.2932e-01 | 6.3057e-01 | -0.20% | 1.7712 | 1.7546 |
| 3.self_attn.k_proj | 0.25 | 1.1723 | 1.1785 | -0.52% | 2.5525e-01 | 2.5722e-01 | -0.76% | nan | nan |
| 3.self_attn.v_proj | 0.25 | 1.1688 | 1.1747 | -0.51% | 4.2253e-01 | 4.2357e-01 | -0.25% | nan | nan |
| 3.self_attn.k_proj | 0.5 | 0.9790 | 0.9816 | -0.27% | 1.3936e-01 | 1.3978e-01 | -0.30% | nan | nan |
| 3.self_attn.v_proj | 0.5 | 0.9784 | 0.9819 | -0.36% | 2.4898e-01 | 2.4927e-01 | -0.12% | nan | nan |
| 3.self_attn.k_proj | 1 | 0.8437 | 0.8431 | +0.07% | 5.2560e-02 | 5.2750e-02 | -0.36% | nan | nan |
| 3.self_attn.v_proj | 1 | 0.8436 | 0.8444 | -0.10% | 9.8065e-02 | 9.8168e-02 | -0.11% | nan | nan |
| 3.self_attn.k_proj | 2 | 0.7771 | 0.7762 | +0.12% | 1.1370e-02 | 1.1367e-02 | +0.03% | nan | nan |
| 3.self_attn.v_proj | 2 | 0.7779 | 0.7779 | +0.01% | 2.1739e-02 | 2.1744e-02 | -0.03% | nan | nan |
| 3.self_attn.k_proj | 4 | 0.7226 | 0.7232 | -0.08% | 7.0889e-04 | 7.0838e-04 | +0.07% | nan | nan |
| 3.self_attn.v_proj | 4 | 0.7215 | 0.7246 | -0.43% | 1.3566e-03 | 1.3569e-03 | -0.02% | nan | nan |

### `scale_sample 4` (the scale search on a quarter of its sample tiles) vs the default, layer 3, ladder settings

| tensor | K | proxy default | proxy sample 4 | sample 4 / default - 1 | g_scale default / sample 4 | scale search s default -> sample 4 | tensor s default -> sample 4 |
|---|---|---|---|---|---|---|---|
| 3.self_attn.q_proj | 2 | 7.4016e-04 | 7.4224e-04 | +0.280% | 0.7737 / 0.7743 | 0.31 -> 0.08 | 4.53 -> 4.33 |
| 3.self_attn.q_proj | 4 | 4.6226e-05 | 4.6179e-05 | -0.102% | 0.7134 / 0.7129 | 0.27 -> 0.12 | 4.11 -> 4.12 |
| 3.self_attn.k_proj | 2 | 1.1370e-02 | 1.1362e-02 | -0.072% | 0.7771 / 0.7779 | 0.12 -> 0.03 | 0.59 -> 0.51 |
| 3.self_attn.k_proj | 4 | 7.0889e-04 | 7.0824e-04 | -0.091% | 0.7226 / 0.7214 | 0.14 -> 0.07 | 0.57 -> 0.51 |
| 3.self_attn.v_proj | 2 | 2.1739e-02 | 2.1730e-02 | -0.040% | 0.7779 / 0.7772 | 0.12 -> 0.03 | 0.59 -> 0.51 |
| 3.self_attn.v_proj | 4 | 1.3566e-03 | 1.3561e-03 | -0.036% | 0.7215 / 0.7216 | 0.14 -> 0.07 | 0.57 -> 0.51 |
| 3.self_attn.o_proj | 2 | 8.5430e-04 | 8.5151e-04 | -0.327% | 0.7762 / 0.7757 | 0.15 -> 0.04 | 2.42 -> 2.35 |
| 3.self_attn.o_proj | 4 | 5.3217e-05 | 5.3096e-05 | -0.228% | 0.7143 / 0.7181 | 0.15 -> 0.09 | 2.18 -> 2.18 |
| 3.mlp.gate_proj | 2 | 3.0391e-02 | 3.0390e-02 | -0.003% | 0.7752 / 0.7760 | 0.44 -> 0.12 | 6.38 -> 6.25 |
| 3.mlp.gate_proj | 4 | 1.9195e-03 | 1.9189e-03 | -0.031% | 0.7168 / 0.7191 | 0.38 -> 0.14 | 5.65 -> 5.55 |

- the last commit's build against w209's host pipeline again (3.k_proj, 3.gate_proj, K2 / K4, `--exl3-drift` = w209's drift; the drift tables agree above K1 anyway): 12 of 12 files byte-identical (`raw/final/ident2.txt`)
