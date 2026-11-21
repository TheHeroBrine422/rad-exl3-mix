| K | n | CPTC vs mad (geo-mean, 95% CI) | CPTC vs mul1 | mad vs mul1 | CPTC wins vs mad |
|---|---|---|---|---|---|
| 5 | 22 | -5.6% [-7.1%, -4.7%] | -3.8% [-4.8%, -3.0%] | +2.0% [+1.3%, +2.7%] | 22/22 |
| 6 | 22 | +9.2% [+5.4%, +11.5%] | +11.4% [+8.3%, +13.4%] | +2.0% [+1.2%, +2.9%] | 1/22 |

per tensor (CPTC / mad - 1):
   3/attn_k       K5  -4.6% K6  +9.7%
  27/attn_k       K5  -7.3% K6 +13.6%
   3/attn_output  K5 -17.8% K6 -22.6%
  27/attn_output  K5  -7.2% K6 +13.1%
   3/attn_qg      K5  -5.9% K6 +10.2%
  27/attn_qg      K5  -4.1% K6 +10.2%
   3/attn_v       K5  -6.4% K6 +12.6%
  27/attn_v       K5  -4.7% K6  +9.8%
   2/ffn_down     K5  -4.2% K6 +10.4%
   3/ffn_down     K5  -4.4% K6  +9.8%
  26/ffn_down     K5  -4.3% K6 +10.0%
  27/ffn_down     K5  -4.1% K6 +10.3%
   2/ffn_gate_up  K5  -4.0% K6 +10.9%
   3/ffn_gate_up  K5  -4.0% K6 +11.1%
  26/ffn_gate_up  K5  -4.2% K6 +10.8%
  27/ffn_gate_up  K5  -4.4% K6 +10.3%
   2/ssm_out      K5  -7.5% K6 +13.9%
  26/ssm_out      K5  -6.3% K6 +12.7%
   2/ssm_qkv      K5  -4.4% K6 +10.0%
  26/ssm_qkv      K5  -4.3% K6  +9.9%
   2/ssm_z        K5  -4.4% K6 +12.0%
  26/ssm_z        K5  -4.3% K6 +10.2%
