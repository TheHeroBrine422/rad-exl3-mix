#!/bin/bash
# qrun.sh MACHINE ARM CARD SIZE BUDGETS [qad220.py args]  (w222) -- one training job of the global queue: w221's run221.sh (= w214's / w217's runner) with qenv.sh's paths. Data and order: w214's
# (pools214.json train214 + chunks214.txt, seed 0). Resumes from OUT/state.pt (restartable). The card lock is held by qcard.sh (QCARD_LOCKED=1); run by hand without it, the lock is taken
# with flock -n (exit 75 when busy). exit 76 = MemAvailable below MINRAM (waiting).
set -u
MACHINE=$1 ARM=$2 CARD=$3 SIZE=$4 BUDGETS=$5; shift 5
HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE
OUT=$QROOT/runs/$ARM; mkdir -p $OUT
if [ -z "${QCARD_LOCKED:-}" ]; then eval "exec {fd}>$(CARDLOCK $CARD)"; flock -n $fd || { echo "card $CARD busy"; exit 75; }; fi
eval "exec {rf}>$RAMLOCK"; flock -s $rf
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AV" -ge "$MINRAM" ] || { echo "MemAvailable ${AV} GB < $MINRAM: waiting"; exit 76; }
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM start on $MACHINE card $CARD (size $SIZE, budgets $BUDGETS): $*" >> $QROOT/run.log
rm -f $OUT/STOP
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
env $(GPUENV $CARD) nice -n 5 $PY $TRAINER $OUT --model $MODEL --level $IN/lv/$SIZE/trunk --chunks @$CHUNKS \
  --calib $CALIB --pools $IN/pools214.json --pool train214 --genmask $IN/genmask --head $IN/lv/$SIZE/head --head-perm $IN/lv/$SIZE/R16.npy --ab $IN/lv/$SIZE/ab.safetensors \
  --budgets $BUDGETS --fixed-eval 20000 --fe-every 12 --monitor 0 --build-lock $BUILD_LOCK --save-every 12 "$@" >> $OUT/stdout.log 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM rc $rc" >> $QROOT/run.log
exit $rc
