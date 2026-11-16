## Q1: KL-IQR's head / embedding in our trunk (t183, whole slice, gen rows, bf16 KV)

Bytes at known cost (container bytes of the head and the embedding):

| base | ours head | ours head bytes | their head | their head bytes | ours emb | ours emb bytes | their emb | their emb bytes |
|---|---|---|---|---|---|---|---|---|
| 5.5 | u5.375 | 854,976,000 | small | 797,604,085 | t4.625 | 735,782,400 | small | 804,561,879 |
| 4.0 | u3.75 | 596,723,200 | tiny | 638,681,907 | t3.375 | 537,126,400 | tiny | 486,707,036 |
| 2.5 | u2.75 | 437,798,400 | tiny | 638,681,907 | t1.875 | 298,739,200 | tiny | 486,707,036 |

| base | arm | head in the serve | embedding in the serve | t183 KLD | rel vs (a) | 95 % CI on rel |
|---|---|---|---|---|---|---|
| 5.5 | a | u5.375 | t4.625 | 0.001305 | — | — |
| 5.5 | b | true bf16 | t4.625 | 0.001260 | -3.45 % | ±1.01 % |
| 5.5 | c | their small head (bf16 donor) | t4.625 | 0.001342 | +2.82 % | ±0.85 % |
| 5.5 | d | u5.375 | true bf16 | 0.001310 | +0.37 % | ±7.18 % |
| 5.5 | e | u5.375 | their small emb (bf16 donor) | 0.001299 | -0.46 % | ±2.08 % |
| 4.0 | a | u3.75 | t3.375 | 0.007080 | — | — |
| 4.0 | b | true bf16 | t3.375 | 0.006653 | -6.04 % | ±1.66 % |
| 4.0 | c | their tiny head (bf16 donor) | t3.375 | 0.007033 | -0.67 % | ±1.02 % |
| 4.0 | d | u3.75 | true bf16 | 0.006865 | -3.03 % | ±3.23 % |
| 4.0 | e | u3.75 | their tiny emb (bf16 donor) | 0.007221 | +1.99 % | ±1.91 % |
| 2.5 | a | u2.75 | t1.875 | 0.050391 | — | — |
| 2.5 | b | true bf16 | t1.875 | 0.048529 | -3.70 % | ±1.24 % |
| 2.5 | c | their tiny head (bf16 donor) | t1.875 | 0.048858 | -3.04 % | ±1.12 % |
| 2.5 | d | u2.75 | true bf16 | 0.047701 | -5.34 % | ±1.78 % |
| 2.5 | e | u2.75 | their tiny emb (bf16 donor) | 0.048246 | -4.26 % | ±1.46 % |

Bytes of the bf16 donors are 2,542,796,800 each (plain bf16, the core's own encoding); the base's own bf16 head / embedding are byte-identical to the checkpoint (rel_fro 0). Their head = the sum of output.kB weights, ids and vrow.

## Q2: bf16 embedding off the card (host RAM), quality and VRAM

VRAM: the card bytes of each serve (menu accounting: trunk + head + fixed [+ embedding on the card]), and what the engine measured after load (card used = total - the startup report's 'settled free').

| S | card bytes now (trunk+head+fixed+emb) | card bytes, bf16 emb on host (same trunk) | engine used now (GiB) | engine used, bf16 emb on host (GiB) | t183 now (a) | t183 bf16-emb at S (d) | S' (label, bf16-emb on host) | t183 at S' (same card bytes as now) |
|---|---|---|---|---|---|---|---|---|
| 2.5 | 8,404,987,904 (2.5000 bpw) | 8,106,248,704 (2.4111 bpw) | 11.40 | 11.12 | 0.050391 | 0.047701 | 3.2563 (card 2.5000 bpw) | 0.043685 (-13.31 % vs a) |
| 4.0 | 13,447,983,104 (4.0000 bpw) | 12,910,856,704 (3.8402 bpw) | 16.09 | 15.59 | 0.007080 | 0.006865 | 4.7563 (card 4.0000 bpw) | 0.005571 (-21.32 % vs a) |
| 5.5 | 18,490,937,344 (5.5000 bpw) | 17,755,154,944 (5.2811 bpw) | 20.79 | 20.11 | 0.001305 | 0.001310 | 6.2563 (card 5.5000 bpw) | 0.000968 (-25.81 % vs a) |

S' = the size at which the trunk + head bytes equal ours at S plus the embedding's bytes (card bytes of S' = total bytes of S); its probe carries the bf16 embedding on the host, so its container label is higher (the bf16 table adds 2,542,796,800 B).

