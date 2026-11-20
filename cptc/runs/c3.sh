#!/bin/bash
# c3.sh PAR THREADS BITS -- C3 v2: global CPTC tables vs mad / mul1 on 22 real 27B tensors (same rows / rotation as the 11-19 C3),
# E3_V2=1 (AVX2 search, bit-identical), E3_TB=$TB (default 1). Restartable: skips (layer,w) already in c3v.jsonl.
cd /data/work/cptcv; P=$1 T=$2 B=$3; mkdir -p logs
export OMP_NUM_THREADS=$T E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib E3_V2=1 E3_TB=${TB:-1} \
  CPTC_LIBDIR=/data/work/cptcv/v3/e3 CPTC_E3DIR=/data/work/cptcv/v3/e3 CPTC_TABLES=/data/work/cptcv/cptc/tables E3_C3_BASE=mul1,mad
L="27 ffn_down|3 ffn_down|2 ffn_down|26 ffn_down|3 ffn_gate_up|27 ffn_gate_up|2 ffn_gate_up|26 ffn_gate_up|2 ssm_qkv|26 ssm_qkv|2 ssm_out|26 ssm_out|2 ssm_z|26 ssm_z|3 attn_qg|27 attn_qg|3 attn_k|27 attn_k|3 attn_v|27 attn_v|3 attn_output|27 attn_output"
echo "$L" | tr '|' '\n' | while read l w; do grep -q "\"layer\": $l, \"w\": \"$w\"" c3v.jsonl 2>/dev/null || echo "$l $w"; done |
  xargs -P $P -L 1 sh -c 'python3 -u cptc/tensor_check.py $0 $1 '$B' 256 c3v.jsonl > logs/c3-L$0-$1.log 2>&1; echo "$(date -u +%H:%M) $0 $1 rc=$?" >> logs/c3.done'
echo "C3 ALL DONE $(date -u +%H:%M)" >> logs/c3.done
