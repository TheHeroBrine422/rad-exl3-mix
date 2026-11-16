#!/bin/bash
# run198.sh NAME CARD [qad196.py args]  (w186, #198, box2) -- one arm-A training job on box2 card CARD (flock -n on the card
# lock; exit 75 when the card is busy). Arm A is #196 / #191 arm A exactly (qad196.py defaults on the 2.5 incumbent,
# lr 1e-3, #181 data mix, held-out fixed eval); the only change for A-s1 is "--seed 1" (data order). Restartable: the
# trainer resumes from OUT/state.pt; a finished arm has OUT/ckpt-1e+06/.done.
set -u
NAME=$1 CARD=$2; shift 2
cd /data/q27/w186 && source env.sh
OUT=$W/runs/$NAME; mkdir -p $OUT
eval "exec {fd}>/data/q27/locks/card$CARD.lock"; flock -n $fd || { echo "card $CARD busy"; exit 75; }
ls -d $G154/gold/*/ | sed 's:/$::' > $W/chunks198.txt
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME start on card $CARD: $*" >> $W/run198.log
rm -f $OUT/STOP
# glibc fixed mmap threshold (as #191 / #196): the per-segment CPU copies of the carried states go back to the OS
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
ROCR_VISIBLE_DEVICES=$CARD nice -n 5 $PY $CODE/v3/qad196/qad196.py $OUT --model $MODEL --level $VL --chunks @$W/chunks198.txt \
  --calib $CALIB --pools $POOLS --genmask $G154/genmask --head $HD --head-perm $PERM --emb $EMB --ab $AB \
  --budgets 1e6 --fixed-eval 30000 --fe-every 1000 --monitor 0 --build-lock $BUILD_LOCK "$@" >> $OUT/stdout.log 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME rc $rc" >> $W/run198.log
exit $rc
