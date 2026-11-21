#!/bin/bash
# run214.sh ARM CARD SIZE BUDGETS [qad200.py args]  (box2, #214) -- one training job: the released SIZE pick's student (lv/SIZE from rad2level214.py) on box2 HIP
# card CARD (flock -n card lock: exit 75 when busy; exit 76 when MemAvailable < 35 GB: the 17 GB student build beside >= 15 GB free). ram.lock shared.
# Restartable: qad200 resumes from OUT/state.pt; a finished budget has OUT/ckpt-<T>/.done. Data: RULES214 (pools214.json train214 + the xcorp chunks).
set -u
ARM=$1 CARD=$2 SIZE=$3 BUDGETS=$4; shift 4
cd /data/q27/w214 && source code/v3/qad214/env214.sh
OUT=$W/runs/$ARM; mkdir -p $OUT
eval "exec {fd}>$Q/locks/card$CARD.lock"; flock -n $fd || { echo "card $CARD busy"; exit 75; }
eval "exec {rf}>$Q/locks/ram.lock"; flock -s $rf
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AV" -ge 35 ] || { echo "MemAvailable ${AV} GB < 35: waiting"; exit 76; }
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM start on card $CARD (size $SIZE, budgets $BUDGETS): $*" >> $W/run214.log
rm -f $OUT/STOP
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
ROCR_VISIBLE_DEVICES=$CARD nice -n 5 $PY $CODE/v3/qad200/qad200.py $OUT --model $MODEL --level $W/lv/$SIZE/trunk --chunks @$W/chunks214.txt \
  --calib $CALIB --pools $POOLS --pool train214 --genmask $W/genmask --head $W/lv/$SIZE/head --head-perm $W/lv/$SIZE/R16.npy --ab $W/lv/$SIZE/ab.safetensors \
  --budgets $BUDGETS --fixed-eval 20000 --fe-every 12 --monitor 0 --build-lock $BUILD_LOCK --save-every 12 "$@" >> $OUT/stdout.log 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM rc $rc" >> $W/run214.log
exit $rc
