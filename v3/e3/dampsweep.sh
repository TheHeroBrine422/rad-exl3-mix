#!/bin/bash
# dampsweep.sh -- LDLQ damping vs K on CPU (trellis.py E3_DAMP): is exllamav3's 0.025 over-damped, more so at high K
# (the hypothesis for turboderp's K6-8 wins)? 16 tensors (every family; layers 0/10 GDN, 11/43/63 attention), damping
# 0.05/0.025/0.01/0.005/0.0025/0.001, K2/4/6/8, 128 rows, error vs the UNDAMPED Hessian. cpu1, 9 jobs x 4 threads.
cd "$(dirname "$0")"; mkdir -p data/damp; D=${OUTD:-data/damp}; mkdir -p $D
export E3_CKPT=${E3_CKPT:-/data/models/qwen38-27b-bf16} E3_HESS=${E3_HESS:-/data/challenge/hessians/calib} OMP_NUM_THREADS=4 E3_FACTORS=1.0,1.15,1.3,1.5
one() { [ -s $D/$1-$2-d$3.jsonl ] && return
  E3_DAMP=$3 nice python3 trellis.py $1 $2 mul1 ${KS:-2,4,6,8} 128 $D/$1-$2-d$3.jsonl > $D/$1-$2-d$3.log 2>&1
  echo "$(date -u +%H:%M) rc=$? $*"; }
export -f one; export D KS
TL=${TENSORS:-"0 ssm_qkv,0 ffn_gate_up,0 ffn_down,10 ssm_out,10 ssm_z,10 ssm_qkv,10 ffn_down,11 attn_qg,11 attn_k,11 attn_v,11 attn_output,11 ffn_gate_up,43 attn_output,43 ffn_down,63 attn_qg,63 ffn_down"}
IFS=, read -ra TT <<< "$TL"
for t in "${TT[@]}"; do
  for d in ${DAMPS:-0.025 0.005 0.05 0.01 0.0025 0.001}; do echo "$t $d"; done; done | xargs -P ${JOBS:-9} -L 1 bash -c 'one $0 $1 $2'
echo DAMPSWEEP DONE
