#!/bin/bash
# proxy table rows for our 4 standalone heads (hd-K*.rad) -> hproxy-ours.jsonl, on card 0 between evals
: ${RXM_WORK:=/root} ${RXM_CKPT:=/data/workspace/models/qwen38-27b-bf16} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_CKPT RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}
exec 9>${RXM_WORK}/card0.lock; flock 9; exec 7>${RXM_WORK}/ram.lock; flock -s 7
ROCR_VISIBLE_DEVICES=0 HP_FILTER=output.weight OMP_NUM_THREADS=8 ./hproxy/hproxy ${RXM_CKPT} challenge/hessians/calib hproxy-ours.jsonl \
  ours-K3=${RXM_OURS}/hd-K3.rad ours-K4=${RXM_OURS}/hd-K4.rad ours-K7=${RXM_OURS}/hd-K7.rad ours-K8=${RXM_OURS}/hd-K8.rad
echo "$(date +%H:%M) hproxy ours heads rc=$?" >> ourchain.log
