#!/bin/bash
# w98.sh [STEP ...] -- #138-01 (../w98.md): the refit arms on fixed codes, gpubox card 0. Default STEPS:
#   jobs K2.5:L10 K2.5:L39 K4:L10 K4:L39 K0.5:L10 K0.5:L39 score report
#   jobs     mkjobs98.py specs (layers 10,39 at K2.5 / K4 / K0.5 i8, seed 0, the staged HEAD settings file)
#   K:L      one layer's group jobs at K, one after another, as rqah's worker runs them (`python -m rxq job`, PYTHONPATH =
#            pipeline + fork, RXQ_REFIT_CPU_FALLBACK=1, under swap27's memguard), with RXQ_REFIT_ARMS (every arm in the
#            same job, same codes) and RXQ_REFIT_ARMS_WQ=$A/arms/K<K>; card0.lock then ram.lock shared for the unit
#   score    f2calc98.py over each K's arm weights (CPU, nice 10, 8 threads, ram.lock shared)
#   report   report98.py -> $A/report.txt (checks c1-c3, the table, the verdict)
# Every step skips finished work (.done markers after the data); rerun after a failure or STOP (touch $A/STOP).
set -u
: ${RXM_WORK:=/root}
A=${A:-/data/w98}; CODE=$A/code; EXL3=$A/exl3; W=$CODE/v3/e3/settle/w98; PY=/data/venv-exl3/bin/python
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib HOUT=/data/swap27/hout-bn T=/data/swap27/tools
GUARD_MB=${GUARD_MB:-3000} ARMS=none,hout,undamped,hout+undamped
STEPS=${*:-jobs K2.5:L10 K2.5:L39 K4:L10 K4:L39 K0.5:L10 K0.5:L39 score report}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$A/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
ENVQ=(PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1)
mkdir -p $A/out $A/arms $A/jobs $A/logs; cd $A
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
die() { log "W98 STOP: $*"; exit 1; }

jobs_() {
  for K in 2.5 4 0.5; do
    [ -f $A/jobs/K$K/jobs.jsonl ] && continue
    env "${ENVQ[@]}" $PY $W/mkjobs98.py $K i8 0 $MODEL $HESS $HOUT $A/jobs/K$K --layers 10,39 >> $A/result.txt 2>&1 || die "mkjobs K$K"
  done
}

unit() {  # unit K L
  local K=$1 L=$2 o=$A/out/K$1; local d=$o/L$2.done
  [ -f $d ] && return 0; [ -f $A/STOP ] && { log "STOP before K$K L$L"; exit 0; }
  local s=$(date +%s); mkdir -p $o
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    log "K$K L$L: card 0 held after $(( $(date +%s) - s )) s"
    for sp in $A/jobs/K$K/L$L-g*.json; do
      n=$(basename $sp .json); jo=$o/$n; mkdir -p $jo
      $PY -I $CODE/v3/e3/settle/w92/clean92.py $jo || exit 1
      t0=$(date +%s%N)
      $T/memguard.sh $GUARD_MB nice -n 5 env "${ENVQ[@]}" RXQ_REFIT_ARMS=$ARMS RXQ_REFIT_ARMS_WQ=$A/arms/K$K \
        $PY -m rxq job $sp -o $jo --device 0 >> $jo/rxq.log 2>&1 || { echo "JOBFAIL $n (rc $?)"; exit 1; }
      echo "K$K $n $(( ($(date +%s%N) - t0) / 1000000 ))" >> $A/wall-ms.txt
    done ) >> $A/logs/K$K-L$L.log 2>&1
  local rc=$?
  [ $rc = 0 ] && touch $d
  log "K$K L$L: rc $rc, $(( $(date +%s) - s )) s wall (incl. lock wait)"; return $rc
}

score() {
  for K in 2.5 4 0.5; do
    [ -f $A/arms/K$K/.scored ] && continue
    ( exec 7>$RXM_WORK/ram.lock; flock -s 7
      THREADS=8 nice -n 10 $PY -I $W/f2calc98.py $A/arms/K$K $MODEL $HESS $HOUT $CODE/pipeline ) >> $A/logs/f2-K$K.log 2>&1 \
      && touch $A/arms/K$K/.scored && log "score K$K: done" || die "f2calc98 K$K (see logs/f2-K$K.log)"
  done
}

for st in $STEPS; do
  case $st in
    jobs) jobs_ ;;
    K*:L*) k=${st%%:*}; l=${st##*:}; unit ${k#K} ${l#L} || die "unit $st" ;;
    score) score ;;
    report) $PY -I $W/report98.py $A /data/w92/out/fs0 > $A/report.txt.part 2>&1 && mv $A/report.txt.part $A/report.txt \
              && log "report: $(tail -1 $A/report.txt)" || die "report98 (see $A/report.txt.part)" ;;
    *) die "unknown step $st" ;;
  esac
done
log "w98 steps done: $STEPS"
