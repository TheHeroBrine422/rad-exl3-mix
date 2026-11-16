#!/bin/bash
# round-colo.sh N COMMIT [CUT] -- w181, colo: one round from the fresh state in snapL (incumbents) with the models of
# data cut CUT (default snapA): anom base + variants, rankreg (64 candidates per size, top 4) + rankanom (top 2).
N=$1; H=$2; W=/data/work/w13/w181; S=$W/${3:-snapA}; L=$W/snapL; A=$W/code/search/v3/alloc193b; O=$W/r$N
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; cd $A; mkdir -p $O
for f in menu-v2.json results.jsonl refit-b1.in.json refit-b2.in.json refit-b3.in.json refit-b1.out.json refit-b2.out.json refit-b3.out.json draw-b1.out.json draw-b2.out.json draw-b3.out.json; do ln -sf $S/$f $L/$f; done
nice -n 19 python3 -I anomvar193.py $L $L $S/anom.json $S/pairs.jsonl $W/confirmed.json $O/anom --commit $H > $O/anom.log 2>&1 &
for s in 2 2.5 3 3.5 4 4.5 5 6; do
  ( nice -n 19 python3 -I rankreg193.py gen $L $L $O/rr-$s --n 64 --sizes $s --seed $((193 + N)) > $O/rr-$s.log 2>&1 &&
    nice -n 19 python3 -I rank193.py feats $O/rr-$s $O/rr-$s/cands.jsonl $O/rr-$s/feats.json >> $O/rr-$s.log 2>&1 &&
    nice -n 19 python3 -I rankreg193.py score $S/rank/feats.json $O/rr-$s/feats.json $O/rr-$s/scores.json >> $O/rr-$s.log 2>&1 &&
    nice -n 19 python3 -I rankreg193.py pick $L $O/rr-$s $O/rr-$s/scores.json $O/rr --top 4 --anom $S/anom.json --top2 2 --commit $H >> $O/rr-$s.log 2>&1 ) &
done
wait; touch $O/.done
