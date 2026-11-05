#!/bin/bash
# run7.sh PAR -- #97 embedding menu at both ends (sub1/README.md "#97"), run6's pattern (4096 rows, rng 5, one scale
# per row = G5120) with embtrellis2.py: t6.25..t8 and t0.25/0.5/0.75, t6 and t1 rerun as anchors against
# embmenu.jsonl, Lloyd e1..e8 on the same rows. Scale-factor grids follow the measured drift of the best factor
# (t4-t6 sat at 1.05-1.35 and climbing with K; t1 at the 0.85 low edge, best scale shrinks below K1).
# One job a level, 3 threads each; results in $D/emb/out-*.jsonl (finished levels skipped).
cd "$(dirname "$0")/.."; D=${D:-/data/work/w20}; mkdir -p $D/emb
HI=1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.75; LO=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
jobs() { echo "lloyd - -"; for k in 6 6.25 6.5 6.75 7 7.25 7.5 7.75 8; do echo "t $k $HI"; done
  for k in 0.25 0.5 0.75 1; do echo "t $k $LO"; done; }
one() { m=$1; k=$2; f=$3; o=$D/emb/out-$m$k.jsonl; [ -s $o ] && return
  if [ $m = lloyd ]; then env="EMB_LLOYD_BITS=1,2,3,4,5,6,7,8 EMB_KS="; else env="EMB_LLOYD=0 EMB_KS=$k EMB_FACTORS=$f"; fi
  rm -f $o.part; env E3_CKPT=/data/models/qwen38-27b-bf16 EMB_G=5120 EMB_ROWS_OUT=$D/emb/r OMP_NUM_THREADS=3 $env \
    nice -n 10 python3 -W ignore embtrellis2.py 4096 $o.part > $D/emb/log-$m$k.txt 2>&1 && mv $o.part $o; }
export -f one; export D
jobs | xargs -P ${1:-14} -L 1 bash -c 'one $0 $1 $2'
echo RUN7 DONE $(ls $D/emb/out-*.jsonl | wc -l) of $(jobs | wc -l)
