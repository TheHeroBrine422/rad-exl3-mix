## #39-01 gain on today's stack (held out)

27 tensors, 612 rows, grid [0.9, 0.95, 1.0, 1.05, 1.1]; complete: True

| m | geo r_B1 (both K) | r_B2 K2.5 | r_B2 K4 | r_B (B1+B2) K2.5 | r_B K4 | r_A K2.5 | r_A K4 |
|---|---|---|---|---|---|---|---|
| 0.9 | 1.0349 | 1.0284 | 1.0425 | 1.0282 | 1.0422 | 1.0279 | 1.0427 |
| 0.95 | 1.0104 | 1.0079 | 1.0137 | 1.0077 | 1.0135 | 1.0078 | 1.0138 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1.05 | 1.0037 | 1.0056 | 1.0036 | 1.0051 | 1.0031 | 1.0056 | 1.0032 |
| 1.1 | 1.0238 | 1.0236 | 1.0249 | 1.0234 | 1.0246 | 1.0240 | 1.0250 |

**Global (rule 1-3):** m* (B1) = 1; m* picked on B2 instead: 1. K2.5: held-out gain g = +0.00%, tensors agreeing 0 / 27; K4: held-out gain g = +0.00%, tensors agreeing 0 / 27. Edge: False. **Verdict: keep 1.0**

**Per family (rule 4, reported):**

| family | K | m*_f (B1) | g_f (B2) | max tensor r_B2 (both samples) | qualifies |
|---|---|---|---|---|---|
| ffn_down | 2.5 | 1 | +0.00% | 1.0000 | False |
| ffn_down | 4 | 1 | +0.00% | 1.0000 | False |
| ssm_qkv | 2.5 | 1 | +0.00% | 1.0000 | False |
| ssm_qkv | 4 | 1 | +0.00% | 1.0000 | False |
| attn_qg | 2.5 | 1 | +0.00% | 1.0000 | False |
| attn_qg | 4 | 1.05 | +0.63% | 1.0162 | False |
| attn_v | 2.5 | 1 | +0.00% | 1.0000 | False |
| attn_v | 4 | 1 | +0.00% | 1.0000 | False |
| ffn_gate_up | 2.5 | 1 | +0.00% | 1.0000 | False |
| ffn_gate_up | 4 | 1 | +0.00% | 1.0000 | False |
| ssm_out | 2.5 | 1 | +0.00% | 1.0000 | False |
| ssm_out | 4 | 1 | +0.00% | 1.0000 | False |
| ssm_z | 2.5 | 1 | +0.00% | 1.0000 | False |
| ssm_z | 4 | 1.05 | +0.01% | 1.0067 | False |
| attn_k | 2.5 | 1 | +0.00% | 1.0000 | False |
| attn_k | 4 | 1 | +0.00% | 1.0000 | False |
| attn_output | 2.5 | 1 | +0.00% | 1.0000 | False |
| attn_output | 4 | 1.05 | -0.03% | 1.0059 | False |

Families that would get a table (qualify at both K): none

**#117 order vs natural order at m = 1 (readers, reported):** order / natural rel_B
- K2.5: geo 0.9779 over 36 (ssm_qkv 0.9844, attn_qg 0.9568, attn_v 0.9829, ffn_gate_up 0.9929, ssm_z 0.9855, attn_k 0.9655); better in 36 / 36
- K4: geo 0.9781 over 36 (ssm_qkv 0.9812, attn_qg 0.9594, attn_v 0.9842, ffn_gate_up 0.9939, ssm_z 0.9847, attn_k 0.9657); better in 36 / 36

One-sided rel1 (B) and two-sided before the refit (B1+B2 via rel_norefit B1), geo over all, by m:
- m 0.9: one-sided 1.0356, before refit (B1) 1.0354
- m 0.95: one-sided 1.0106, before refit (B1) 1.0105
- m 1: one-sided 1.0000, before refit (B1) 1.0000
- m 1.05: one-sided 1.0040, before refit (B1) 1.0038
- m 1.1: one-sided 1.0230, before refit (B1) 1.0246
