#!/bin/bash
# run200.sh NAME CARD BUDGET [qad200.py args]  (w187, #200, box2) -- one training job on box2 card CARD. The card lock is
# held for the whole job (flock -n: exit 75 when the card is busy); exit 76 when MemAvailable < 35 GB (a 17 GB job beside
# >= 15 GB free, RULES200). No ram.lock (the served steps take it exclusively for packs, and a 6 h training hold would
# queue them behind it). Restartable: the trainer resumes from OUT/state.pt; a finished run has OUT/ckpt-<budget>/.done.
set -u
NAME=$1 CARD=$2 BUDGET=$3; shift 3
cd /data/q27/w187 && source env.sh
OUT=$W/runs/$NAME; mkdir -p $OUT
eval "exec {fd}>/data/q27/locks/card$CARD.lock"; flock -n $fd || { echo "card $CARD busy"; exit 75; }
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AV" -ge 35 ] || { echo "MemAvailable ${AV} GB < 35: waiting"; exit 76; }
ls -d $G154/gold/*/ | sed 's:/$::' > $W/chunks200.txt          # the #196 chunk list (49 golden dirs)
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME start on card $CARD (budget $BUDGET): $*" >> $W/run200.log
rm -f $OUT/STOP
# glibc fixed mmap threshold (as #191 / #196): the per-segment CPU copies of the carried states go back to the OS
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
ROCR_VISIBLE_DEVICES=$CARD nice -n 5 $PY $CODE/v3/qad200/qad200.py $OUT --model $MODEL --level $VL --chunks @$W/chunks200.txt \
  --calib $CALIB --pools $POOLS --genmask $G154/genmask --head $HD --head-perm $PERM --emb $EMB --ab $AB \
  --rows $BUDGET --fixed-eval 30000 --fe-every 1000 --monitor 0 --build-lock $BUILD_LOCK --save-every 20 "$@" >> $OUT/stdout.log 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME rc $rc" >> $W/run200.log
exit $rc
