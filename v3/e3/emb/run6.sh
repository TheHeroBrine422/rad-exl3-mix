#!/bin/bash
# Embedding trellis menu, low end, run6 (2026-11-02): K1 and K1.25 with the scale search extended DOWNWARD (0.5 .. 1.0).
# run5 widened the top (to 1.75) and changed nothing: at K1 826 of 4096 rows sat on the LOW edge (0.85), none above 0.95.
# (4096 rows, 6 scale factors). Adds to embmenu.jsonl.
cd /data/work/embt
one() { k=$1; g=$2; [ -s rows6/out-K$k-G$g.jsonl ] && return
  E3_CKPT=/data/models/qwen38-27b-bf16 EMB_LLOYD=0 EMB_KS=$k EMB_G=$g EMB_FACTORS=0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 \
  EMB_ROWS_OUT=rows6/r OMP_NUM_THREADS=8 nice python3 embtrellis.py 4096 rows6/out-K$k-G$g.jsonl > rows6/log-K$k-G$g.txt 2>&1; }
export -f one
mkdir -p rows6
printf "1 5120\n1.25 5120\n" | xargs -P 4 -L 1 bash -c "one \$0 \$1"
echo SWEEP6 DONE $(ls rows6/out-*.jsonl | wc -l) of 2
