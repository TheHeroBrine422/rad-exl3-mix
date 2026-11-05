#!/bin/bash
# Embedding trellis menu sweep (sampled rows): quarter steps K1.5-6 x {row scale, G1024}, + G512 for K>=4.
# 4096 rows, 6 scale factors, per-row npz in rows3/. 16 jobs x 3 threads (cpu2 quota 48).
cd /data/work/embt
jobs() { for k in 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5 5.25 5.5 5.75 6; do
  echo "$k 5120"; echo "$k 1024"; case $k in 4*|5*|6) echo "$k 512";; esac; done; }
one() { k=$1; g=$2; [ -s rows3/out-K$k-G$g.jsonl ] && return
  E3_CKPT=/data/models/qwen38-27b-bf16 EMB_LLOYD=0 EMB_KS=$k EMB_G=$g EMB_FACTORS=0.85,0.95,1.05,1.15,1.25,1.35 \
  EMB_ROWS_OUT=rows3/r OMP_NUM_THREADS=3 nice python3 embtrellis.py 4096 rows3/out-K$k-G$g.jsonl > rows3/log-K$k-G$g.txt 2>&1; }
export -f one
jobs | xargs -P 16 -L 1 bash -c "one \$0 \$1"
echo SWEEP3 DONE $(ls rows3/out-*.jsonl | wc -l) of $(jobs | wc -l)
