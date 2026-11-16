#!/bin/bash
# tcheck196.sh NAME CARD GOLDEN [CKPT] (w182, #196) -- the torch path's rows for one arm on one judge set (tscore196.py,
# A8 as served), on card CARD under its card lock (blocking). 4.0 students: NOHEAD=1 EMBX=<t3.375 table>; 2.5: the default
# head / table. Writes torch/NAME-SET.npz (restartable: skipped when it exists). Compare with eval196.py cmp.
set -u
NAME=$1 CARD=$2 G=$3 CKPT=${4:-}
cd /data/q27/w182 && source env.sh
mkdir -p torch
SET=$(basename $G); OUT=torch/$NAME-$SET.npz
[ -e $OUT ] && { echo "skip $OUT"; exit 0; }
if [ "${NOHEAD:-0}" = 1 ]; then HA=""; else HA="--head $HD --head-perm $PERM"; fi
EMBP=${EMBX:-$EMB}
LV=${LV:-$VL}
C=""; [ -n "$CKPT" ] && C="--ckpt $CKPT"
flock -w 7200 /data/q27/locks/card$CARD.lock -c "ROCR_VISIBLE_DEVICES=$CARD $PY $CODE/v3/qad196/tscore196.py $OUT.part --golden $G --model $MODEL --level $LV $HA --emb $EMBP --ab $AB --build-lock $W/build.lock $C > logs/tscore-$NAME-$SET.log 2>&1"
rc=$?; [ $rc -eq 0 ] && [ -e $OUT.part ] && mv $OUT.part $OUT
echo "$(date -u +%H:%M) tcheck $NAME $SET rc $rc"
