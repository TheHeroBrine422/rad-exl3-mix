#!/bin/bash
# Embedding trellis menu, low end (2026-11-01): K1 and K1.25 x {row scale, G1024}, same settings as run3.sh
# (4096 rows, 6 scale factors). Adds to embmenu.jsonl.
cd /data/work/embt
one() { k=$1; g=$2; [ -s rows4/out-K$k-G$g.jsonl ] && return
  E3_CKPT=/data/models/qwen38-27b-bf16 EMB_LLOYD=0 EMB_KS=$k EMB_G=$g EMB_FACTORS=0.85,0.95,1.05,1.15,1.25,1.35 \
  EMB_ROWS_OUT=rows4/r OMP_NUM_THREADS=8 nice python3 embtrellis.py 4096 rows4/out-K$k-G$g.jsonl > rows4/log-K$k-G$g.txt 2>&1; }
export -f one
mkdir -p rows4
printf "1 5120\n1 1024\n1.25 5120\n1.25 1024\n" | xargs -P 4 -L 1 bash -c "one \$0 \$1"
echo SWEEP4 DONE $(ls rows4/out-*.jsonl | wc -l) of 4
