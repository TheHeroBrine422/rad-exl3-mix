## Quality (0.8B, fake-quant, edy trunk quant)

hash / i8 = gen-masked mean KL ratio, per seed (z = paired diff / per-document bootstrap SE), geo = geometric mean over seeds.

| K | golden | KL i8 s0 / s1 | KL hash s0 / s1 | ratio s0 (z) | ratio s1 (z) | geo |
|---|---|---|---|---|---|---|
| 2 | search | 0.4984 / 0.5046 | 0.5064 / 0.5229 | 1.0160 (+2.3) | 1.0363 (+5.7) | **1.0261** |
| 2 | verify | 0.4934 / 0.5032 | 0.5000 / 0.5216 | 1.0134 (+1.9) | 1.0367 (+6.1) | **1.0250** |
| 3 | search | 0.0901 / 0.0898 | 0.0935 / 0.0936 | 1.0377 (+4.6) | 1.0415 (+5.9) | **1.0396** |
| 3 | verify | 0.0908 / 0.0902 | 0.0931 / 0.0931 | 1.0249 (+3.3) | 1.0321 (+4.1) | **1.0285** |
| 4 | search | 0.0218 / 0.0219 | 0.0223 / 0.0221 | 1.0225 (+3.1) | 1.0098 (+1.3) | **1.0161** |
| 4 | verify | 0.0221 / 0.0223 | 0.0221 / 0.0223 | 1.0003 (+0.0) | 1.0007 (+0.1) | **1.0005** |

**Quality verdict:** FAIL (search geo > 1.010 at K2, K3, K4)

## Side checks

Reproduction (d1112's edy `k{K}ste` vs this run's `i8`, search; same recipe, expected 1.0000):

- K2: s0 1.00000 (KL 0.49842 → 0.49842), s1 1.00000 (KL 0.50455 → 0.50455)
- K3: s0 1.00000 (KL 0.09009 → 0.09009), s1 1.00000 (KL 0.08984 → 0.08984)
- K4: s0 1.00000 (KL 0.02184 → 0.02184), s1 1.00000 (KL 0.02188 → 0.02188)

IQR's A (`hiqr`, 0xB5297B, seed 0) — readout:

| K | hiqr / i8 search | hiqr / i8 verify | hiqr / hash (searched A) search |
|---|---|---|---|
| 2 | 1.0249 (+4.2) | 1.0253 (+3.9) | 1.0087 (+1.5) |
| 3 | 1.0383 (+5.4) | 1.0159 (+2.1) | 1.0006 (+0.1) |
| 4 | 1.0278 (+2.8) | 1.0291 (+3.5) | 1.0052 (+0.4) |

## Per-tensor proxy error (exllamav3's, after the scale refit): hash / i8, geo over tensors and seeds

| K | all | in_proj_qkv | in_proj_z | out_proj | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.0077 (n 300) | 1.0085 | 1.0089 | 1.0071 | 1.0120 | 1.0080 | 1.0068 | 1.0073 | 1.0074 | 1.0070 | 1.0067 |
| 3 | 1.0087 (n 300) | 1.0097 | 1.0098 | 1.0082 | 1.0112 | 1.0070 | 1.0080 | 1.0074 | 1.0088 | 1.0084 | 1.0079 |
| 4 | 1.0092 (n 300) | 1.0106 | 1.0092 | 1.0090 | 1.0132 | 1.0114 | 1.0073 | 1.0077 | 1.0090 | 1.0091 | 1.0078 |
