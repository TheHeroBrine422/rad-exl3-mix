#!/bin/bash
# r96run.sh PAR -- #96 stage 1 (README "#96"): rowslab2.py on every trunk tensor of all 64 layers, one thread a job,
# one result file a tensor in $D/rs/ (finished ones skipped). Run from a copy of v3/e3 on cpu2 (W20 folder).
cd "$(dirname "$0")/.."; D=${D:-/data/work/w20}
export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib RS_HOUT=/data/work/hout27/hout-bn
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $D/rs $D/logs
for l in $(seq 0 63); do
  if [ $((l % 4)) = 3 ]; then T="attn_qg attn_k attn_v attn_output"; else T="ssm_qkv ssm_z ssm_out"; fi
  for w in ffn_down ffn_gate_up $T; do [ -s $D/rs/$l-$w.json ] || echo "$l $w"; done
done | xargs -P ${1:-16} -L 1 bash -c 'nice -n 10 python3 -W ignore rowslab2.py $0 $1 '$D'/rs/$0-$1.json > '$D'/logs/rs-$0-$1.log 2>&1'
echo R96 DONE $(ls $D/rs/*.json | wc -l)
