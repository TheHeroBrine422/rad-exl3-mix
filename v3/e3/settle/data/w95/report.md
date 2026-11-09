# #126 i8 vs i4 at the sub-1 levels (w87 two-sided held-out proxy)

pairs: 162 (expected 27 x 2 x 3 = 162)

## G0: i4 rows vs w87's

162 compared, 0 without a w87 row, max |rel_B ratio - 1| = 9.53e-08 -> **PASS**

## R(K) = geo mean rel(i8) / rel(i4), 27 tensors x 2 samples

| K | n | **B (held-out)** | 95% CI (tensor bootstrap) | B1 | B2 | A (in-sample) | B one-sided | inputs B | rest B | max tensor B | s_exl i8/i4 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.125 | 54 | **1.0020** | 0.9977–1.0070 | 1.0024 | 1.0016 | 1.0021 | 0.9991 | 1.0032 | 1.0005 | 1.0528 (3:attn_qg) | 0.997 | ok |
| 0.5 | 54 | **0.9993** | 0.9974–1.0012 | 0.9997 | 0.9989 | 0.9993 | 0.9988 | 0.9985 | 1.0004 | 1.0120 (3:attn_qg) | 0.996 | ok |
| 1 | 54 | **0.9999** | 0.9985–1.0017 | 0.9996 | 1.0001 | 1.0000 | 0.9995 | 1.0002 | 0.9995 | 1.0162 (3:attn_qg) | 0.997 | ok |

## per family (B, geo over layers x samples)

| family | K0.125 | K0.5 | K1 |
|---|---|---|---|
| attn_k | 1.0014 | 0.9966 | 0.9970 |
| attn_output | 1.0035 | 1.0022 | 1.0011 |
| attn_qg | 1.0141 | 0.9982 | 1.0067 |
| attn_v | 1.0002 | 1.0007 | 1.0010 |
| ffn_down | 0.9986 | 1.0000 | 1.0001 |
| ffn_gate_up | 1.0011 | 0.9996 | 0.9972 |
| ssm_out | 0.9989 | 0.9999 | 0.9996 |
| ssm_qkv | 0.9970 | 0.9972 | 0.9980 |
| ssm_z | 1.0034 | 0.9997 | 0.9984 |

## absolute held-out rel_B (geo), i4 / i8

| K | i4 | i8 |
|---|---|---|
| 0.125 | 0.3284 | 0.3291 |
| 0.5 | 0.1370 | 0.1369 |
| 1 | 0.0570 | 0.0570 |

**Rule:** PASS: switch to i8 at K0.125-1
