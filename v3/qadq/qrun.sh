#!/bin/bash
# qrun.sh MACHINE ARM CARD SIZE BUDGETS [qad220.py args]  (w222) -- one training job of the global queue: w221's run221.sh (= w214's / w217's runner) with qenv.sh's paths. Data and order: w214's
# (pools214.json train214 + chunks214.txt, seed 0). Resumes from OUT/state.pt (restartable). The card lock is held by qcard.sh (QCARD_LOCKED=1); run by hand without it, the lock is taken
# with flock -n (exit 75 when busy). exit 76 = MemAvailable below MINRAM (waiting).
# w231: CARD = A,B+C,D (groups joined by +) = data-parallel ranks, one process a group with only its cards visible (rank r = group r, gloo on
# 127.0.0.1:QDP_PORT, default 29400 + 10 x the first card); rank r > 0 logs to stdout-r<r>.log. When one rank fails the others are killed (they would
# wait in the gradient all-reduce for gloo's 2 h timeout); the rc is the first failing rank's. Without + it is the old single process.
# w233: with --pipe-procs each group A,B runs as two stage processes (one card each): A,B = 2 processes, A,B+C,D = 4 (rank = 2 x group + stage).
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
ARGS=(--model $MODEL --level $IN/lv/$SIZE/trunk --chunks @$CHUNKS \
  --calib $CALIB --pools $IN/pools214.json --pool train214 --genmask $IN/genmask --head $IN/lv/$SIZE/head --head-perm $IN/lv/$SIZE/R16.npy --ab $IN/lv/$SIZE/ab.safetensors \
  --budgets $BUDGETS --fixed-eval 20000 --fe-every 12 --monitor 0 --build-lock $BUILD_LOCK --save-every 12 "$@")
# w233: with --pipe-procs every group A,B is two processes (stage 0 on card A, stage 1 on card B; rank = 2 x group + stage, QAD_STAGES=2)
PROCS=(); STAGES=1
if [[ " $* " == *" --pipe-procs "* ]]; then
  STAGES=2; IFS=+ read -ra GRP <<< "$CARD"
  for g in "${GRP[@]}"; do IFS=, read -ra CC <<< "$g"; [ ${#CC[@]} = 2 ] || { echo "--pipe-procs: each group must be two cards (A,B): $g"; exit 2; }; PROCS+=(${CC[0]} ${CC[1]}); done
fi
if [ $STAGES = 1 ] && [[ $CARD != *+* ]]; then
  env $(GPUENV $CARD) nice -n 5 $PY $TRAINER $OUT "${ARGS[@]}" >> $OUT/stdout.log 2>&1
  rc=$?
else
  IFS=+ read -ra GRP <<< "$CARD"; c0=${GRP[0]%%,*}; PORT=${QDP_PORT:-$((29400 + 10 * c0))}
  [ $STAGES = 2 ] && GRP=("${PROCS[@]}"); W=${#GRP[@]}
  declare -A RANK_OF=(); pids=()
  for r in $(seq 0 $((W - 1))); do
    log=$OUT/stdout.log; [ $r -gt 0 ] && log=$OUT/stdout-r$r.log
    env $(GPUENV ${GRP[$r]}) RANK=$r WORLD_SIZE=$W LOCAL_RANK=0 QAD_STAGES=$STAGES MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT nice -n 5 $PY $TRAINER $OUT "${ARGS[@]}" >> $log 2>&1 &
    pids+=($!); RANK_OF[$!]=$r
  done
  rc=0; left=$W
  while [ $left -gt 0 ]; do
    wait -n -p done_pid "${pids[@]}"; r_rc=$?; left=$((left - 1))
    if [ $r_rc -ne 0 ] && [ $rc -eq 0 ]; then
      rc=$r_rc; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM rank ${RANK_OF[$done_pid]} rc $r_rc: stopping the other ranks" >> $QROOT/run.log
      for p in "${pids[@]}"; do [ $p != $done_pid ] && kill $p 2>/dev/null; done
    fi
    pids=($(for p in "${pids[@]}"; do [ $p != $done_pid ] && echo $p; done))
  done
fi
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM rc $rc" >> $QROOT/run.log
exit $rc
