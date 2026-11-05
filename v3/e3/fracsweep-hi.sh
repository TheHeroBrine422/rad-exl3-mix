#!/bin/bash
# fracsweep-hi.sh -- fractional-K trellis at the top (K5-8 half steps) and bottom (K0.25) of the V3 menu, with
# K4.5/5 overlapping fracsweep.sh's points. The 14 tensors of fracsweep.sh plus layers 0 (GDN) and 63 (full attention).
# cpu1: 9 tensors at once x 4 threads, one output file per tensor. Merge: cat data/fhi/*.jsonl.
cd "$(dirname "$0")"; mkdir -p data/fhi
export E3_CKPT=${E3_CKPT:-/data/models/qwen38-27b-bf16} E3_HESS=${E3_HESS:-/data/challenge/hessians/calib} OMP_NUM_THREADS=4 E3_FACTORS=0.85,1.0,1.15,1.3,1.5,1.75
KS=0.25,4.5,5,5.5,6,6.5,7,7.5,8
one() { [ -s data/fhi/$1-$2.done ] && return
  nice python3 trellis.py $1 $2 mul1 $KS 128 data/fhi/$1-$2.jsonl > data/fhi/$1-$2.log 2>&1 && date -u > data/fhi/$1-$2.done
  echo "$(date -u +%H:%M) done $1 $2"; }
export -f one; export KS
printf '%s\n' "10 ssm_out" "10 ssm_z" "10 ssm_qkv" "10 ffn_gate_up" "10 ffn_down" "11 attn_qg" "11 attn_k" "11 attn_v" \
  "11 attn_output" "43 ssm_out" "43 ssm_z" "43 ssm_qkv" "43 ffn_gate_up" "43 ffn_down" "0 ssm_qkv" "0 ffn_gate_up" \
  "0 ffn_down" "63 attn_qg" "63 attn_output" "63 ffn_gate_up" "63 ffn_down" | xargs -P 9 -L 1 bash -c 'one $0 $1'
echo FRACHI DONE $(ls data/fhi/*.done | wc -l) of 21
