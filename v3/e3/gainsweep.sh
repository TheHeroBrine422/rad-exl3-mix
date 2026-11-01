#!/bin/bash
# gainsweep.sh -- global scale ("gain") vs K and family on CPU (trellis.py scale factor = per-row RMS x f, f searched
# finely 0.8-1.8 in 0.1 steps, Hessian-weighted error): how far is the best gain from a fixed one, per family and K?
# (DESIGN §2 "scale gain per K and family": est. 0 to -4%; the GPU gaintest covers 3 layers at K2/4/6 only.)
# One job per (tensor, K): E3_FACTORS is the whole grid, trellis.py keeps the best; gainreport.py reads the per-factor
# rows from E3_ALLF output. 16 tensors x K2..8, 256 rows. Out: data/gain/.
cd "$(dirname "$0")"; mkdir -p data/gain
export E3_CKPT=${E3_CKPT:-/data/models/qwen38-27b-bf16} E3_HESS=${E3_HESS:-/data/challenge/hessians/calib} OMP_NUM_THREADS=4
export E3_FACTORS=0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8 E3_ALLF=1
one() { [ -s data/gain/$1-$2-K$3.jsonl ] && return
  nice python3 trellis.py $1 $2 mul1 $3 256 data/gain/$1-$2-K$3.jsonl > data/gain/$1-$2-K$3.log 2>&1; echo "$(date -u +%H:%M) rc=$? $*"; }
export -f one
for K in 4 6 2 8 3 5 7; do for t in "0 ssm_qkv" "0 ffn_gate_up" "0 ffn_down" "10 ssm_out" "10 ssm_z" "10 ssm_qkv" "10 ffn_down" \
  "11 attn_qg" "11 attn_k" "11 attn_v" "11 attn_output" "11 ffn_gate_up" "43 attn_output" "43 ffn_down" "63 attn_qg" "63 ffn_down"; do
  echo "$t $K"; done; done | xargs -P ${JOBS:-11} -L 1 bash -c 'one $0 $1 $2'
echo GAINSWEEP DONE
