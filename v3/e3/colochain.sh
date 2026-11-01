#!/bin/bash
# after the damping sweep: damping at more depths (K3/5/7), for a per-family x depth damping table
until grep -q "DAMPSWEEP DONE" /data/logs/damp.log; do sleep 60; done
cd /data/work/fhi && OUTD=data/damp2 KS=3,5,7 DAMPS="0.025 0.01 0.005 0.0025 0.001" \
  TENSORS="1 ssm_qkv,1 ssm_out,1 ffn_down,5 ssm_z,5 ffn_gate_up,19 attn_qg,19 attn_k,19 attn_output,19 ffn_down,30 ssm_qkv,30 ssm_out,30 ffn_gate_up,47 attn_qg,47 attn_v,47 attn_output,47 ffn_gate_up,58 ssm_qkv,58 ssm_z,58 ffn_down" \
  ./dampsweep2.sh
