# w64 CPU readout (report64.py, 931 jobs)

## Item 0 · settle64 (exllamav3's LDLQ order) vs w50's GPU tile-CD ratios (attn_qg, same blocks)

cd3 ÷ LDLQ (rel after the refit), mul1x_i8. Rule: geo over the two samples within ±0.5% of the GPU's.

| layer | K | CPU s0 / s1 | CPU geo | GPU s0 / s1 | GPU geo | CPU − GPU | ok |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 0.9826 / 0.9829 | 0.9828 | 0.9838 / 0.9894 | 0.9866 | -0.39% | yes |
| 3 | 3 | 0.9822 / 0.9863 | 0.9842 | 0.985 / 0.99 | 0.9875 | -0.33% | yes |
| 3 | 4 | 0.9842 / 0.9853 | 0.9848 | 0.9853 / 0.9899 | 0.9876 | -0.29% | yes |
| 19 | 2 | 0.9928 / 0.9922 | 0.9925 | 0.9937 / 0.9942 | 0.9939 | -0.15% | yes |
| 19 | 3 | 0.9944 / 0.9936 | 0.9940 | 0.9958 / 0.996 | 0.9959 | -0.19% | yes |
| 19 | 4 | 0.9943 / 0.9930 | 0.9937 | 0.9958 / 0.9952 | 0.9955 | -0.19% | yes |
| 43 | 2 | 0.9942 / 0.9938 | 0.9940 | 0.9954 / 0.9948 | 0.9951 | -0.11% | yes |
| 43 | 3 | 0.9949 / 0.9947 | 0.9948 | 0.9951 / 0.9957 | 0.9954 | -0.06% | yes |
| 43 | 4 | 0.9946 / 0.9940 | 0.9943 | 0.9961 / 0.9958 | 0.9959 | -0.17% | yes |

**Item 0: PASS** (9 of 9 within ±0.5%; 9 needed)

Order alone (K2, mul1_i8, LDLQ error after the refit): rev (settle64) ÷ fwd (w43's settle.py run)

| layer | s0 | s1 | cd3÷ldlq fwd s0 / s1 | rev s0 / s1 |
|---|---|---|---|---|
| 3 | 0.8756 | 0.7954 | 0.9511 / 0.9322 | 0.9789 / 0.9868 |
| 19 | 0.9826 | 0.9811 | 0.9927 / 0.9930 | 0.9928 / 0.9924 |
| 43 | 0.9843 | 0.9862 | 0.9913 / 0.9924 | 0.9922 / 0.9905 |

## Item 2 · #109 clipped i8: r = rel(i8c α) / rel(mul1x), m = 1, after the refit (10 tensors)

Median over tensors (sample 0 / sample 1); α = 1 is today's i8.

| K | α 1 | α 1.15 | α 1.33 | α 1.5 | α 1.75 | α 2 |
|---|---|---|---|---|---|---|
| 6.5 | 1.013 / 1.014 | 1.100 / 1.099 | 1.251 / 1.250 | 1.447 / 1.444 | 1.798 / 1.773 | 2.136 / 2.143 |
| 7 | 1.238 / 1.239 | 1.257 / 1.254 | 1.356 / 1.355 | 1.539 / 1.535 | 1.878 / 1.878 | 2.209 / 2.212 |
| 7.25 | 1.632 / 1.638 | 1.590 / 1.591 | 1.617 / 1.617 | 1.715 / 1.716 | 1.996 / 2.002 | 2.263 / 2.257 |
| 7.5 | 2.254 / 2.244 | 2.187 / 2.182 | 2.208 / 2.203 | 2.200 / 2.197 | 2.326 / 2.327 | 2.446 / 2.439 |
| 8 | 4.329 / 4.346 | 4.237 / 4.238 | 4.271 / 4.287 | 4.251 / 4.279 | 4.355 / 4.392 | 4.243 / 4.277 |

α\*(K) picked on sample 0 (lowest median), judged on sample 1:

| K | α* | s1 median | s1 p90 | s1 max | α = 1 s1 median | equal-error mul1 K (median, α*) |
|---|---|---|---|---|---|---|
| 6.5 | 1 | 1.014 | 1.017 | 1.017 | 1.014 | 6.49 |
| 7 | 1 | 1.239 | 1.270 | 1.331 | 1.239 | 6.83 |
| 7.25 | 1.15 | 1.591 | 1.618 | 1.735 | 1.638 | 6.89 |
| 7.5 | 1.15 | 2.182 | 2.220 | 2.347 | 2.244 | 6.90 |
| 8 | 1.15 | 4.238 | 4.333 | 4.498 | 4.346 | 6.90 |

- K7 gap: **does not close** (α* 1: median 1.239, p90 1.270; K6.5 α* 1 1.014 vs i8 1.014)
- i8 K7.25 does not join (α* 1.15: median 1.591, p90 1.618)
- i8 K7.5 does not join (α* 1.15: median 2.182, p90 2.220)
- → **negative** by the rule (K7 median > 1.10 at every α)

## Item 4 · #110 bf16 (row, 128) scales at equal bits: r = rel(variant @ K−⅛) / rel(plain @ K)

Per tensor geo over samples 0 / 1; family median (max tensor); `⅛ bit` = rel(plain@K−⅛) / rel(plain@K).

| K | family | ⅛ bit | post@K8 | pre@K8 | prepost@K8 |
|---|---|---|---|---|---|
| 4 | attn_output | 1.184 | 1.172 (1.174) | 1.169 (1.174) | 1.157 (1.162) |
| 4 | ssm_out | 1.184 | 1.173 (1.177) | 1.169 (1.172) | 1.157 (1.161) |
| 5 | attn_output | 1.182 | 1.175 (1.176) | 1.160 (1.166) | 1.153 (1.158) |
| 5 | ssm_out | 1.182 | 1.173 (1.181) | 1.162 (1.172) | 1.153 (1.164) |
| 6 | attn_output | 1.178 | 1.177 (1.178) | 1.145 (1.160) | 1.143 (1.158) |
| 6 | ssm_out | 1.179 | 1.175 (1.178) | 1.151 (1.159) | 1.148 (1.156) |
| 7 | attn_output | 1.171 | 1.170 (1.176) | 1.120 (1.158) | 1.119 (1.158) |
| 7 | ssm_out | 1.174 | 1.174 (1.190) | 1.129 (1.150) | 1.128 (1.148) |

- post@K8: **FAIL**
- pre@K8: **FAIL**
- prepost@K8: **FAIL**

## Item 5 · #106 / #103 on the order-fixed proxy: rel(arm) / rel(natural) (stored-order arms; exllamav3 quantizes
the last stored block first: `head` = heaviest quantized last (literal), `rev` = heaviest first (quantization order))

| family | K | head (literal) | rev (quantization order) | worst tensor rev | rev, samples 0+1 |
|---|---|---|---|---|---|
| attn_output | 2 | 1.1008 | 0.9706 | 0.9925 | 0.9706 |
| attn_output | 3 | 1.1005 | 0.9699 | 0.9896 | 0.9699 |
| attn_output | 4 | 1.1008 | 0.9708 | 0.9901 | 0.9708 |

attn_output (literal stored order): **fail** — 1.1008 / 1.1005 / 1.1008 (old: pass 0.928 / 0.927 / 0.927)

attn_output (quantization order): **PASS (adopt)** — 0.9706 / 0.9699 / 0.9708 (old: pass 0.928 / 0.927 / 0.927)
| ssm_out | 2 | 1.0210 | 0.9802 | 0.9927 | 0.9802 |
| ssm_out | 3 | 1.0179 | 0.9782 | 0.9924 | 0.9782 |
| ssm_out | 4 | 1.0195 | 0.9795 | 0.9922 | 0.9795 |

ssm_out (literal stored order): **fail** — 1.0210 / 1.0179 / 1.0195 (old: fail 0.986 / 0.986 / 0.988)

ssm_out (quantization order): **PASS (adopt)** — 0.9802 / 0.9782 / 0.9795 (old: fail 0.986 / 0.986 / 0.988)
| ffn_down | 2 | 1.0028 | 0.9904 | 0.9942 | 0.9902 |
| ffn_down | 3 | 1.0022 | 0.9899 | 0.9936 | 0.9899 |
| ffn_down | 4 | 1.0029 | 0.9901 | 0.9943 | 0.9900 |

ffn_down (literal stored order): **fail** — 1.0028 / 1.0022 / 1.0029 (old: fail 0.994 / 0.997 / 0.998)

ffn_down (quantization order): **fail** — 0.9904 / 0.9899 / 0.9901 (old: fail 0.994 / 0.997 / 0.998)
ffn_down robustness (sample 0, rev / head): L0 K2 0.9129 / 1.0334, L0 K3 0.9112 / 1.0317, L0 K4 0.9120 / 1.0316, L43 K2 0.9950 / 1.0039, L43 K3 0.9944 / 1.0036, L43 K4 0.9943 / 1.0036, L63 K2 0.9608 / 1.0452, L63 K3 0.9586 / 1.0396, L63 K4 0.9594 / 1.0350

Item 5 extension (reported, not deciding): all run tensors of the family, sample 0, median [min, max] of
rel(arm) / rel(natural); `free` = unconstrained block order (no head grouping), stored order as `head`.

| family | K | n | rev | head | free | rev > 1.00 |
|---|---|---|---|---|---|---|
| attn_output | 2 | 16 | 0.9579 [0.7940, 0.9895] | 1.0781 [1.0183, 1.1742] | 1.1030 [1.0226, 1.2159] | 0 |
| attn_output | 3 | 16 | 0.9595 [0.7878, 0.9904] | 1.0782 [1.0149, 1.1791] | 1.1055 [1.0264, 1.2252] | 0 |
| attn_output | 4 | 16 | 0.9591 [0.7859, 0.9920] | 1.0783 [1.0128, 1.1786] | 1.1039 [1.0199, 1.2309] | 0 |
| ssm_out | 2 | 13 | 0.9799 [0.9561, 0.9919] | 1.0311 [1.0100, 1.0581] | 1.0364 [1.0094, 1.0599] | 0 |
| ssm_out | 3 | 13 | 0.9764 [0.9535, 0.9920] | 1.0309 [1.0091, 1.0508] | 1.0333 [1.0056, 1.0658] | 0 |
| ssm_out | 4 | 13 | 0.9783 [0.9560, 0.9917] | 1.0304 [1.0087, 1.0510] | 1.0317 [1.0132, 1.0620] | 0 |
