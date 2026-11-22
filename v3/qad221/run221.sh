#!/bin/bash
# run221.sh ARM CARD SIZE BUDGETS [qad220.py args]  (box2, w221) -- one training job: w214's / w217's runner (run217.sh) with qad220.py and W = /data/q27/w221
# (flock -n card lock: exit 75 when busy; exit 76 when MemAvailable < 35 GB). Env CODE_PY overrides the trainer (default qad220.py; the default-equivalence check runs qad200.py).
# Restartable (qad221 resumes from OUT/state.pt). Data and order: w214's (pools214.json train214 + chunks214.txt), seed 0.
set -u
ARM=$1 CARD=$2 SIZE=$3 BUDGETS=$4; shift 4
cd /data/q27/w221 && source code/v3/qad221/env221.sh
TRAINER=${CODE_PY:-$CODE/v3/qad220/qad220.py}
OUT=$W/runs/$ARM; mkdir -p $OUT
eval "exec {fd}>$Q/locks/card$CARD.lock"; flock -n $fd || { echo "card $CARD busy"; exit 75; }
eval "exec {rf}>$Q/locks/ram.lock"; flock -s $rf
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AV" -ge 35 ] || { echo "MemAvailable ${AV} GB < 35: waiting"; exit 76; }
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM start on card $CARD (size $SIZE, budgets $BUDGETS): $*" >> $W/run221.log
rm -f $OUT/STOP
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
ROCR_VISIBLE_DEVICES=$CARD nice -n 5 $PY $TRAINER $OUT --model $MODEL --level $W14/lv/$SIZE/trunk --chunks @$W14/chunks214.txt \
  --calib $CALIB --pools $POOLS --pool train214 --genmask $W14/genmask --head $W14/lv/$SIZE/head --head-perm $W14/lv/$SIZE/R16.npy --ab $W14/lv/$SIZE/ab.safetensors \
  --budgets $BUDGETS --fixed-eval 20000 --fe-every 12 --monitor 0 --build-lock $BUILD_LOCK --save-every 12 "$@" >> $OUT/stdout.log 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM rc $rc" >> $W/run221.log
exit $rc
