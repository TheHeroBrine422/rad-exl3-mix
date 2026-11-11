# w100 report

## #128 sub-1 hygiene

sub1 rows: 324 (54 tensor x samples)

### 1 · C9 statistics (cur arm; M = H_pub, after the per-output-channel refit)

| K | family (α) | c_mean | c q50 (mean) | c_max | e_mean (clipped) | rows at clip | n | fires |
|---|---|---|---|---|---|---|---|---|
| 0.125 | ssm_out (3) | 2.780 | 2.828 | 2.828 | 0.490 | 95.1% | 6 | **FIRES** |
| 0.125 | attn_output (-1) | 0.725 | 0.711 | 0.985 | 0.473 | 78.0% | 6 | **FIRES** |
| 0.125 | ffn_down (0) | 1.000 | 1.000 | 1.000 | 0.448 | 67.3% | 6 | no |
| 0.5 | ssm_out (3) | 1.947 | 1.951 | 2.828 | 0.349 | 1.0% | 6 | **FIRES** |
| 0.5 | attn_output (-1) | 0.851 | 0.846 | 0.996 | 0.274 | 0.0% | 6 | no |
| 0.5 | ffn_down (0) | 1.000 | 1.000 | 1.000 | 0.294 | 2.0% | 6 | no |
| 1 | ssm_out (3) | 1.310 | 1.312 | 1.705 | 0.162 | 0.0% | 6 | **FIRES** |
| 1 | attn_output (-1) | 0.938 | 0.936 | 0.999 | 0.119 | 0.0% | 6 | no |
| 1 | ffn_down (0) | 1.000 | 1.000 | 1.000 | 0.136 | 0.0% | 6 | no |

**C9 rule: FIRES → propose α(K) = α · clip(K − 1, 0, 1).**

Real 27B codes (w92, i4, layers 2–3), in-job C9 recomputed here (G3 = |e_mean / job's record − 1|):

| K | key | α | e_mean | c_mean | c q50 | rows at clip | G3 |
|---|---|---|---|---|---|---|---|
| 0.125 | 2.linear_attn.out_proj |  | 0.496 | 2.808 | 2.828 | 4988/5120 | 5.5e-11 |
| 0.125 | 3.self_attn.o_proj |  | 0.487 | 0.715 | 0.707 | 4738/5120 | 3.3e-11 |
| 0.875 | 2.linear_attn.out_proj |  | 0.224 | 1.472 | 1.492 | 0/5120 | 1.2e-10 |
| 0.875 | 3.self_attn.o_proj |  | 0.142 | 0.926 | 0.922 | 0/5120 | 1.7e-10 |

G3 max: 1.6633283639322372e-10

### 2 · g_scale grid edge (cur arm's search; exllamav3 grid coarse 0.1..1.9)

| K | top edge (g 2.05) | bottom edge | g median | g min / max | fires (> 10%) | ext arm: geo rel2_B ext / cur (all, n changed) | inputs | rest |
|---|---|---|---|---|---|---|---|---|
| 0.125 | 0/54 | 0/54 | 1.651 | 1.631 / 1.661 | no | 1.0000 (0) | 1.0000 | 1.0000 |
| 0.5 | 0/54 | 0/54 | 1.026 | 1.009 / 1.031 | no | 1.0000 (0) | 1.0000 | 1.0000 |
| 1 | 0/54 | 0/54 | 0.822 | 0.804 / 0.827 | no | 1.0000 (0) | 1.0000 | 1.0000 |

K0.125: ext grid's g picks: median 1.651, min 1.631, max 1.661, at the new top edge (3.9+): 0

K0.5: ext grid's g picks: median 1.026, min 1.009, max 1.031, at the new top edge (3.9+): 0

K1: ext grid's g picks: median 0.822, min 0.804, max 0.827, at the new top edge (3.9+): 0

Real w92 job logs (i4, 13 tensors per level): K0.125: g 2.050; K0.875: g 0.874, 0.875, 0.876, 0.877, 0.878, 0.881

Production logs K0.125: 13/13 at the top edge (g 2.05) → **FIRES**

Production logs K0.875: 0/13 at the top edge (g 2.05) → no

**g_scale rule (proxy or production > 10% at an edge): FIRES → propose the coarse grid to 3.9 on the top side.**

Amendment 1 (report only): K0.125 i8 at fixed g vs the cur arm (proxy pick ~1.65); geo ratio, held-out two-sided rel2_B / one-sided rel1_pub:

| g | all rel2_B | all rel1_pub | inputs rel2_B | rest rel2_B | tensors×samples better (rel2_B) |
|---|---|---|---|---|---|
| 2.05 | 1.0326 | 1.0346 | 1.0447 | 1.0176 | 0/54 |
| 2.6 | 1.0762 | 1.0678 | 1.1063 | 1.0397 | 0/54 |
| 3.3 | 1.1091 | 1.1031 | 1.1498 | 1.0601 | 0/54 |

### 3 · Measured LDLQ drift (cur arm; table value 1.08 below K1 and at K1)

| K | D = geo d_all (all) | inputs | rest | min / max | d_fork (fork comment) | fires | drift arm: geo rel2_B drift / cur (n changed) |
|---|---|---|---|---|---|---|---|
| 0.125 | 1.3475 | 1.4439 | 1.2359 | 1.058 / 2.147 | 1.2438 | **FIRES** | 0.9697 (54) |
| 0.5 | 1.1838 | 1.2187 | 1.1415 | 1.071 / 1.353 | 1.1412 | no | 0.9793 (54) |
| 1 | 1.0892 | 1.1108 | 1.0628 | 1.039 / 1.187 | 1.0672 | **FIRES** | 0.9971 (54) |

### 4 · K0.125 vs a zero tensor (rel ≥ 0.97 → price as zero / drop the cell)

Real codes (w92 K0.125, i4; undamped calib H_in; Q = 0 gives 1):

| tensor | rel1 | rel2 (bn1920) | rel1 best row scalar | rows with e_n > 1 |
|---|---|---|---|---|
| 2.linear_attn.in_proj_qkv | 0.399 | 0.268 | 0.396 | 32/10240 |
| 2.linear_attn.in_proj_z | 0.296 | 0.147 | 0.294 | 30/6144 |
| 2.linear_attn.out_proj | 0.297 | 0.322 | 0.296 | 0/5120 |
| 2.mlp.gate_proj | 0.727 | 0.457 | 0.726 | 1/17408 |
| 2.mlp.up_proj | 0.784 | 0.578 | 0.783 | 0/17408 |
| 2.mlp.down_proj | 0.272 | 0.380 | 0.271 | 0/5120 |
| 3.self_attn.k_proj | 0.411 | 0.179 | 0.404 | 5/1024 |
| 3.self_attn.q_proj | 0.077 | 0.050 | 0.076 | 357/12288 |
| 3.self_attn.v_proj | 0.667 | 0.483 | 0.652 | 9/1024 |
| 3.self_attn.o_proj | 0.025 | 0.059 | 0.025 | 3/5120 |
| 3.mlp.gate_proj | 0.695 | 0.615 | 0.694 | 2/17408 |
| 3.mlp.up_proj | 0.788 | 0.702 | 0.788 | 0/17408 |
| 3.mlp.down_proj | 0.720 | 0.683 | 0.720 | 0/5120 |

Proxy (i8 K0.125, 27 tensors, geo over 2 samples): max rel2_B / rel1_pub per family:

| family | tensors: rel2_B (held out) / rel1_pub |
|---|---|
| ssm_qkv | L2 0.328 / 0.464; L30 0.344 / 0.280; L50 0.346 / 0.372 |
| ssm_z | L10 0.281 / 0.195; L34 0.211 / 0.181; L62 0.146 / 0.147 |
| ssm_out | L10 0.408 / 0.620; L30 0.416 / 0.411; L50 0.467 / 0.561 |
| attn_qg | L3 0.121 / 0.170; L19 0.292 / 0.514; L43 0.152 / 0.207 |
| attn_k | L11 0.218 / 0.303; L35 0.260 / 0.326; L63 0.280 / 0.363 |
| attn_v | L7 0.523 / 0.591; L43 0.427 / 0.455; L59 0.690 / 0.634 |
| attn_output | L19 0.525 / 0.593; L43 0.647 / 0.645; L63 0.372 / 0.350 |
| ffn_gate_up | L10 0.615 / 0.654; L43 0.444 / 0.469; L62 0.156 / 0.218 |
| ffn_down | L2 0.201 / 0.543; L30 0.434 / 0.415; L63 0.258 / 0.280 |

**Zero-floor rule: no tensor at rel ≥ 0.97: the K0.125 floor stands.**

### 5 · suh / svh scan (gpubox, read-only)

71 directories, 28160 scale tensors: zeros 0, subnormals 93, non-finite 0.

**Scan rule (zero / subnormal at K ≥ 1): FIRES [('/data/a4cost/q-i4/K1', 'model.language_model.layers.17.linear_attn.out_proj.suh')] → propose R14 clamp.**

- /data/a4cost/q-i4/K1 K?: model.language_model.layers.17.linear_attn.out_proj.suh zero 0 subnormal 1 min 5.51e-05 (median 1.51e-02)
- /data/w92/lv/sub1-0.125/level K?: model.language_model.layers.3.self_attn.o_proj.suh zero 0 subnormal 2 min 4.77e-07 (median 4.68e-03)
- /data/w92/lv/sub1-0.125/level K?: model.language_model.layers.3.self_attn.q_proj.suh zero 0 subnormal 44 min 3.58e-07 (median 7.63e-04)
- /data/w92/out/sub1-0.125 K0.125: model.language_model.layers.3.self_attn.q_proj.suh zero 0 subnormal 44 min 3.58e-07 (median 7.63e-04)
- /data/w92/out/sub1-0.125 K0.125: model.language_model.layers.3.self_attn.o_proj.suh zero 0 subnormal 2 min 4.77e-07 (median 4.68e-03)

## #129 H_out shape (σ_out × h^p), pick on (B1, O1), judge on (B2, O2)

hshape rows: 2106 (162 tensor x sample x K), arms 12 + full


### K0.5

| cell | arm* (pick X1) | g on X2 | agree | edge | proposed | swap: arm* X2 → g X1 | geo r_B,O (arm*) | full / cur (B,O) |
|---|---|---|---|---|---|---|---|---|
| all | E-so0.1-p1 | +0.40% | 22/27 | edge | no | E-so0.1-p1 → +0.46% | 0.9957 | 0.9902 |
| inputs | E-so0.1-p1 | +0.58% | 14/15 | edge | no | E-so0.1-p0.75 → +0.65% | 0.9938 | 0.9868 |
| rest | E-so0.1-p1 | +0.18% | 8/12 | edge | no | E-so0.1-p1 → +0.20% | 0.9982 | 0.9944 |
| ssm_qkv | E-so0.03-p0.75 | +1.15% | 3/3 |  | no | E-so0.03-p0.75 → +1.27% | 0.9873 | 0.9895 |
| ssm_z | E-so0.1-p0.75 | +2.16% | 3/3 | edge | **YES** | E-so0.1-p0.75 → +1.88% | 0.9802 | 0.9685 |
| ssm_out | E-so0.03-p1 | +0.16% | 2/3 |  | no | E-so0.03-p1 → +0.20% | 0.9982 | 0.9949 |
| attn_qg | E-so0.003-p0.75 | +0.33% | 2/3 | edge | no | E-so0.003-p0.75 → +0.18% | 0.9976 | 0.9863 |
| attn_k | E-so0.1-p1 | +0.49% | 3/3 | edge | no | E-so0.1-p1 → +0.55% | 0.9948 | 0.9961 |
| attn_v | E-so0.1-p0.75 | +0.54% | 3/3 | edge | no | E-so0.1-p0.75 → +0.71% | 0.9939 | 0.9940 |
| attn_output | E-so0.003-p1 | +0.20% | 2/3 | edge | no | E-so0.1-p0.75 → +0.15% | 0.9979 | 0.9999 |
| ffn_gate_up | E-so0.003-p0.75 | +0.21% | 2/3 | edge | no | E-so0.1-p1 → +0.58% | 0.9958 | 0.9841 |
| ffn_down | E-so0.01-p1 | +0.00% | 0/3 |  | no | E-so0.01-p1 → +0.00% | 1.0000 | 0.9987 |

Surface, all 27 × 2, geo r on (B, O) = held out on both sides (rows σ_out, columns p):

| σ_out \ p | 0.5 | 0.75 | 1 |
|---|---|---|---|
| 0.003 | 1.0059 | 0.9982 | 1.0011 |
| 0.01 | 1.0055 | 0.9973 | 1.0000 |
| 0.03 | 1.0059 | 0.9977 | 0.9997 |
| 0.1 | 1.0067 | 0.9967 | 0.9957 |

### K2.5

| cell | arm* (pick X1) | g on X2 | agree | edge | proposed | swap: arm* X2 → g X1 | geo r_B,O (arm*) | full / cur (B,O) |
|---|---|---|---|---|---|---|---|---|
| all | E-so0.03-p1 | +0.05% | 16/27 |  | no | E-so0.03-p1 → +0.02% | 0.9996 | 0.9830 |
| inputs | E-so0.03-p1 | +0.11% | 11/15 |  | no | E-so0.03-p1 → +0.05% | 0.9993 | 0.9762 |
| rest | E-so0.1-p1 | -0.02% | 4/12 | edge | no | E-so0.01-p1 → +0.00% | 0.9999 | 0.9917 |
| ssm_qkv | E-so0.003-p1 | -0.04% | 2/3 | edge | no | E-so0.01-p1 → +0.00% | 1.0001 | 0.9797 |
| ssm_z | E-so0.03-p1 | +0.07% | 3/3 |  | no | E-so0.03-p1 → +0.04% | 0.9995 | 0.9544 |
| ssm_out | E-so0.01-p1 | +0.00% | 0/3 |  | no | E-so0.01-p1 → +0.00% | 1.0000 | 0.9968 |
| attn_qg | E-so0.03-p1 | +0.31% | 2/3 |  | no | E-so0.003-p1 → -0.03% | 0.9984 | 0.9597 |
| attn_k | E-so0.03-p1 | +0.15% | 3/3 |  | no | E-so0.03-p1 → +0.09% | 0.9988 | 0.9947 |
| attn_v | E-so0.03-p1 | +0.04% | 2/3 |  | no | E-so0.03-p1 → +0.09% | 0.9993 | 0.9929 |
| attn_output | E-so0.1-p1 | +0.07% | 2/3 | edge | no | E-so0.1-p0.75 → -0.07% | 0.9988 | 0.9977 |
| ffn_gate_up | E-so0.1-p1 | -0.16% | 0/3 | edge | no | E-so0.01-p1 → +0.00% | 0.9997 | 0.9773 |
| ffn_down | E-so0.003-p1 | +0.28% | 3/3 | edge | no | E-so0.003-p1 → +0.12% | 0.9978 | 0.9950 |

Surface, all 27 × 2, geo r on (B, O) = held out on both sides (rows σ_out, columns p):

| σ_out \ p | 0.5 | 0.75 | 1 |
|---|---|---|---|
| 0.003 | 1.0227 | 1.0027 | 1.0000 |
| 0.01 | 1.0227 | 1.0033 | 1.0000 |
| 0.03 | 1.0235 | 1.0039 | 0.9996 |
| 0.1 | 1.0261 | 1.0066 | 1.0004 |

### K4

| cell | arm* (pick X1) | g on X2 | agree | edge | proposed | swap: arm* X2 → g X1 | geo r_B,O (arm*) | full / cur (B,O) |
|---|---|---|---|---|---|---|---|---|
| all | E-so0.03-p1 | +0.06% | 18/27 |  | no | E-so0.03-p1 → +0.04% | 0.9996 | 0.9827 |
| inputs | E-so0.003-p1 | +0.01% | 7/15 | edge | no | E-so0.03-p1 → +0.05% | 0.9995 | 0.9755 |
| rest | E-so0.03-p1 | +0.08% | 9/12 |  | no | E-so0.03-p1 → +0.02% | 0.9995 | 0.9917 |
| ssm_qkv | E-so0.03-p1 | +0.10% | 2/3 |  | no | E-so0.003-p1 → +0.18% | 0.9983 | 0.9761 |
| ssm_z | E-so0.03-p1 | +0.16% | 3/3 |  | no | E-so0.03-p1 → +0.19% | 0.9982 | 0.9526 |
| ssm_out | E-so0.003-p1 | +0.09% | 2/3 | edge | no | E-so0.003-p1 → +0.14% | 0.9991 | 0.9950 |
| attn_qg | E-so0.03-p1 | -0.05% | 1/3 |  | no | E-so0.003-p1 → +0.19% | 0.9993 | 0.9601 |
| attn_k | E-so0.01-p1 | +0.00% | 0/3 |  | no | E-so0.01-p1 → +0.00% | 1.0000 | 0.9972 |
| attn_v | E-so0.003-p1 | +0.06% | 1/3 | edge | no | E-so0.1-p1 → -0.09% | 0.9995 | 0.9923 |
| attn_output | E-so0.03-p1 | +0.20% | 3/3 |  | no | E-so0.03-p1 → +0.06% | 0.9988 | 0.9995 |
| ffn_gate_up | E-so0.01-p1 | +0.00% | 0/3 |  | no | E-so0.03-p1 → -0.06% | 1.0000 | 0.9776 |
| ffn_down | E-so0.03-p1 | +0.09% | 2/3 |  | no | E-so0.1-p1 → -0.12% | 0.9993 | 0.9949 |

Surface, all 27 × 2, geo r on (B, O) = held out on both sides (rows σ_out, columns p):

| σ_out \ p | 0.5 | 0.75 | 1 |
|---|---|---|---|
| 0.003 | 1.0232 | 1.0036 | 0.9998 |
| 0.01 | 1.0231 | 1.0031 | 1.0000 |
| 0.03 | 1.0245 | 1.0045 | 0.9996 |
| 0.1 | 1.0280 | 1.0072 | 1.0007 |

**#129: 1 of 36 cells proposed** (0.5|ssm_z).
