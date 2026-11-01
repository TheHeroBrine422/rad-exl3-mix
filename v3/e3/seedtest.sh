#!/bin/bash
# seedtest.sh -- best-of-N rotation seeds: does a tensor's quantization error depend on its rotation seed in a way
# that holds across rows? 9 tensors (layers 10 GDN, 11 attention, 43 MLP/attention) x 4 sign seeds x 2 disjoint row
# samples (128 rows each), K2/3/4, scale search 0.85-1.3. Report: seedreport.py (seed effect on sample A vs sample B).
cd "$(dirname "$0")"; mkdir -p data/seed
export E3_CKPT=${E3_CKPT:-/data/models/qwen38-27b-bf16} E3_HESS=${E3_HESS:-/data/challenge/hessians/calib} OMP_NUM_THREADS=4 E3_FACTORS=0.85,1.0,1.15,1.3
one() { [ -s data/seed/$1-$2-s$3-r$4.jsonl ] && return
  E3_SEED=$3 E3_ROWSEED=$4 nice python3 trellis.py $1 $2 mul1 2,3,4 128 data/seed/$1-$2-s$3-r$4.jsonl > data/seed/$1-$2-s$3-r$4.log 2>&1
  echo "$(date -u +%H:%M) rc=$? $*"; }
export -f one
for t in "10 ssm_qkv" "10 ssm_out" "10 ffn_down" "11 attn_qg" "11 attn_k" "11 attn_output" "11 ffn_gate_up" "43 attn_output" "43 ffn_down"; do
  for s in 0 1 2 3; do for r in 0 1; do echo "$t $s $r"; done; done; done | xargs -P ${JOBS:-9} -L 1 bash -c 'one $0 $1 $2 $3'
echo SEEDTEST DONE
