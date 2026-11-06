#!/bin/bash
# blockx/run.sh [PAR] -- blockx.py over the 336 trunk linears of the E1c dump (head 336 excluded), PAR (6) single-thread
# jobs round-robin, nice 10. Layout under the run dir: e1b/{aqproxy,mkmask}.py, blockx/, e0-ident.map, out/, logs/.
# Idempotent: each job skips tensors already in its out/bx-J.jsonl.
cd "$(dirname "$0")/.."; mkdir -p out logs
P=${1:-6}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 E3_CKPT=/data/models/qwen38-27b-bf16 AQ_DIR=$PWD/e1b
for j in $(seq 0 $((P-1))); do
  idx=$(seq $j $P 335 | paste -sd,)
  nice -n 10 python3 -W ignore blockx/blockx.py ${DUMP:-/data/work/aq/dump} e0-ident.map $idx out/bx-$j.jsonl > logs/bx-$j.log 2>&1 &
done
wait; echo BLOCKX DONE
