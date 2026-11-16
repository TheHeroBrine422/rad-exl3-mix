#!/bin/bash
# chk200-one.sh NAME CARD CODEFILE [args]  (w187, #200 unit checks, box2): the first 3 plan steps of one trainer on card CARD
# (flock -n). NAME = old (#196's qad196.py, all defaults) or new (qad200.py --wd 0 --accum 1). Same data, same 30k fixed eval.
set -u
NAME=$1 CARD=$2 F=$3; shift 3
cd /data/q27/w187 && source env.sh
OUT=$W/chk/$NAME; mkdir -p $OUT; rm -rf $OUT/*
eval "exec {fd}>/data/q27/locks/card$CARD.lock"; flock -n $fd || { echo "card $CARD busy"; exit 75; }
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
ROCR_VISIBLE_DEVICES=$CARD $PY $CODE/v3/$F $OUT --model $MODEL --level $VL --chunks @$W/chunks200.txt \
  --calib $CALIB --pools $POOLS --genmask $G154/genmask --head $HD --head-perm $PERM --emb $EMB --ab $AB \
  --budgets 1e6 --fixed-eval 30000 --fe-every 1000 --monitor 0 --build-lock $BUILD_LOCK --max-steps 3 "$@" > $OUT/stdout.log 2>&1
echo "chk $NAME rc $?"
