#!/bin/bash
# prepA.sh [SNAPNAME] -- w181, colo: after the refits of a data cut (snapA / snapB) are done: paired rows, the refit's
# predictions, the anomaly regression (both versions), the ranker's training features. Niced, single-threaded.
W=/data/work/w13/w181; S=$W/${1:-snapA}; A=$W/code/search/v3/alloc193b; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $A; mkdir -p $S/rank
for g in b1 b2 b3; do ln -sf refit-$g.out.json $S/draw-$g.out.json; done
( nice -n 19 python3 -I rank193.py cands $S $S/rank/cands.jsonl && nice -n 19 python3 -I rank193.py feats $S $S/rank/cands.jsonl $S/rank/feats.json ) > $S/rank/prep.log 2>&1 &
nice -n 19 python3 -I pairs193.py $S $S/pairs.jsonl > $S/pairs.log 2>&1
nice -n 19 python3 -I curv193.py $S $S $S/curv.json --no-opt --pairs $S/pairs.jsonl --pred-out $S/predref.json > $S/curv.log 2>&1
nice -n 19 python3 -I anom193.py $S $S/pairs.jsonl $S/anom.json --pred $S/predref.json > $S/anom.log 2>&1
wait; touch $S/.prep-done
