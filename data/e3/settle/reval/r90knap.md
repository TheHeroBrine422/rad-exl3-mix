# #90 rule 2: knap4 with sub-1 options (quality only, lambda 0)

Predicted KLD (kfit2 model) per size; change vs the same menu without sub-1 (base). Rule: >= 3% lower with the geomean family ratios (arm sub1) on menu v3e, at 256 KiB buckets. 1.25 / 1.3 / 1.4: reported only.

| menu | bpw | base KLD | sub1 KLD | change | sub1max KLD | change | trunk params at K<1 (sub1) | at K1-1.875 base -> sub1 | emb base / sub1 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| v3e | 1.25 | 0.75306 | 0.74134 | -1.56% | 0.73976 | -1.8% | 23.4% (68 tensors) | 97.5% -> 73.0% | t1 / t1 | (reported) |
| v3e | 1.3 | 0.64213 | 0.63867 | -0.54% | 0.63768 | -0.7% | 14.0% (39 tensors) | 95.6% -> 80.8% | t1 / t1 | (reported) |
| v3e | 1.4 | 0.48755 | 0.48726 | -0.06% | 0.48706 | -0.1% | 1.7% (6 tensors) | 93.5% -> 91.8% | t1 / t1 | (reported) |
| v3e | 1.5 | 0.38109 | 0.38087 | -0.06% | 0.38084 | -0.1% | 0.8% (3 tensors) | 92.0% -> 91.2% | t1.25 / t1.25 | below 3% |
| v3e | 1.75 | 0.22205 | 0.22200 | -0.02% | 0.22200 | -0.0% | 0.0% (2 tensors) | 79.6% -> 79.6% | t1.25 / t1.25 | below 3% |
| v3e | 2 | 0.13407 | 0.13407 | +0.00% | 0.13407 | +0.0% | 0.0% (0 tensors) | 51.4% -> 51.4% | t1.5 / t1.5 | below 3% |
| v3e | 2.25 | 0.08591 | 0.08591 | +0.00% | 0.08591 | +0.0% | 0.0% (0 tensors) | 14.3% -> 14.3% | t1.75 / t1.75 | below 3% |
| v3e | 2.5 | 0.05761 | 0.05761 | +0.00% | 0.05761 | +0.0% | 0.0% (0 tensors) | 6.2% -> 6.2% | t2 / t2 | below 3% |

## Menu granularity: K1.125-1.875 levels (v3e) vs knap4's default V3 ladder (v3), both without sub-1


## Solver resolution (DP bucket): base / sub1 KLD and the sub-1 change

- v3e 1.5 bpw: 256 KiB 0.38109 / 0.38087 (-0.06%)
- v3e 1.75 bpw: 256 KiB 0.22205 / 0.22200 (-0.02%)
- v3e 2 bpw: 256 KiB 0.13407 / 0.13407 (+0.00%)
- v3e 2.25 bpw: 256 KiB 0.08591 / 0.08591 (+0.00%)
- v3e 2.5 bpw: 256 KiB 0.05761 / 0.05761 (+0.00%)

## Which tensors go sub-1 (menu v3e, arm sub1, 256 KiB): per family tensors (share of trunk params), K used

- **1.25 bpw** (68 tensors, 23.4%): ffn_gate_up 18 (13.2%), ffn_down 13 (4.8%), attn_qg 6 (1.6%), ssm_qkv 7 (1.5%), ssm_z 10 (1.3%), ssm_out 6 (0.8%), attn_output 1 (0.1%), attn_k 5 (0.1%), attn_v 2 (0.0%); K0.75: 14, K0.875: 54
  - attn_k: 15:0.875 19:0.875 43:0.875 47:0.875 51:0.875
  - attn_output: 15:0.875
  - attn_qg: 15:0.875 19:0.875 43:0.875 47:0.75 51:0.875 55:0.875
  - attn_v: 15:0.875 19:0.875
  - ffn_down: 30:0.875 31:0.875 32:0.875 33:0.875 41:0.875 42:0.875 43:0.875 44:0.875 45:0.875 46:0.75 47:0.875 48:0.875 49:0.875
  - ffn_gate_up: 0:0.75 15:0.875 16:0.75 17:0.875 30:0.875 31:0.875 32:0.75 33:0.75 34:0.75 43:0.875 44:0.875 45:0.875 46:0.75 47:0.75 48:0.75 49:0.75 50:0.75 51:0.875
  - ssm_out: 38:0.875 40:0.875 41:0.75 42:0.875 45:0.875 58:0.875
  - ssm_qkv: 1:0.875 40:0.875 41:0.875 42:0.875 44:0.875 45:0.875 46:0.875
  - ssm_z: 1:0.875 40:0.875 41:0.875 42:0.875 44:0.75 45:0.875 46:0.875 48:0.875 49:0.875 50:0.875
- **1.3 bpw** (39 tensors, 14.0%): ffn_gate_up 11 (8.1%), ffn_down 8 (2.9%), attn_qg 5 (1.3%), ssm_z 6 (0.8%), ssm_qkv 2 (0.4%), ssm_out 3 (0.4%), attn_k 3 (0.1%), attn_v 1 (0.0%); K0.75: 2, K0.875: 37
  - attn_k: 15:0.875 47:0.875 51:0.875
  - attn_qg: 15:0.875 19:0.875 43:0.875 47:0.875 51:0.875
  - attn_v: 15:0.875
  - ffn_down: 41:0.875 42:0.875 43:0.875 44:0.875 45:0.875 46:0.875 47:0.875 48:0.875
  - ffn_gate_up: 0:0.75 16:0.875 31:0.875 32:0.75 33:0.875 34:0.875 46:0.875 47:0.875 48:0.875 49:0.875 50:0.875
  - ssm_out: 38:0.875 41:0.875 42:0.875
  - ssm_qkv: 42:0.875 44:0.875
  - ssm_z: 1:0.875 41:0.875 42:0.875 44:0.875 45:0.875 46:0.875
- **1.4 bpw** (6 tensors, 1.7%): ffn_gate_up 1 (0.7%), attn_qg 2 (0.5%), ffn_down 1 (0.4%), attn_k 1 (0.0%), attn_v 1 (0.0%); K0.75: 1, K0.875: 5
  - attn_k: 15:0.875
  - attn_qg: 15:0.875 47:0.875
  - attn_v: 15:0.875
  - ffn_down: 46:0.875
  - ffn_gate_up: 0:0.75
- **1.5 bpw** (3 tensors, 0.8%): ffn_gate_up 1 (0.7%), attn_k 1 (0.0%), attn_v 1 (0.0%); K0.875: 3
  - attn_k: 15:0.875
  - attn_v: 15:0.875
  - ffn_gate_up: 0:0.875
- **1.75 bpw** (2 tensors, 0.0%): attn_k 1 (0.0%), attn_v 1 (0.0%); K0.875: 2
  - attn_k: 15:0.875
  - attn_v: 15:0.875
- **2 bpw** (0 tensors, 0.0%): none; -
- **2.25 bpw** (0 tensors, 0.0%): none; -
- **2.5 bpw** (0 tensors, 0.0%): none; -
