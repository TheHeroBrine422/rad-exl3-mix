#!/bin/bash
# gscale-run.sh -- gscale.py on 16 tensors (every family; layers 0/10 GDN, 11/43/63 attention + MLP), colo,
# 9 jobs x 4 threads, one output per tensor in data/gs/. Merge: cat data/gs/*.jsonl.
cd "$(dirname "$0")"; mkdir -p data/gs
export E3_CKPT=${E3_CKPT:-/data/models/qwen38-27b-bf16} E3_HESS=${E3_HESS:-/data/challenge/hessians/calib} OMP_NUM_THREADS=4
one() { nice python3 gscale.py $1 $2 data/gs/$1-$2.jsonl 128 > data/gs/$1-$2.log 2>&1; echo "$(date -u +%H:%M) rc=$? $1 $2"; }
export -f one
printf '%s\n' "0 ssm_qkv" "0 ffn_gate_up" "0 ffn_down" "10 ssm_out" "10 ssm_z" "10 ssm_qkv" "10 ffn_down" "11 attn_qg" \
  "11 attn_k" "11 attn_v" "11 attn_output" "11 ffn_gate_up" "43 attn_output" "43 ffn_down" "63 attn_qg" "63 ffn_down" \
  | xargs -P 9 -L 1 bash -c 'one $0 $1'
echo GSCALE DONE
