#!/bin/bash
# run233.sh CARDS NAME SCRIPT [flags...] (box2; setsid nohup) -- w233 profiling: waits for the listed cards' locks (never preempts), holds them, runs
# SCRIPT (prof233.py | qad220.py) from /data/q27/w233/code on those cards (ROCR_VISIBLE_DEVICES=CARDS, comma list) with w214's data / student flags
# (qrun.sh's), OUT = /data/q27/w233/runs/NAME. Extra env passes through. Log: /data/q27/w233/run233.log
set -u
MYCARDS=$1 NAME=$2 SCRIPT=$3; shift 3
source /data/q27/w233/code/v3/qadq/qenv.sh box2      # sets CARDS / CODE (the queue's): ours are MYCARDS / W233CODE
W=/data/q27/w233; W233CODE=$W/code; OUT=$W/runs/$NAME; mkdir -p $OUT
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "$NAME waiting for cards $MYCARDS"
if [ "${W233_LOCKED:-}" != 1 ]; then for c in $(echo $MYCARDS | tr ",+" "  "); do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done; fi   # W233_LOCKED=1: the caller holds them
log "$NAME cards $MYCARDS held"
case $SCRIPT in prof233.py) S=$W233CODE/v3/qad233/prof233.py;; qad220.py) S=$W233CODE/v3/qad220/qad220.py;; *) S=$SCRIPT;; esac
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864 TORCHINDUCTOR_COMPILE_THREADS=1   # w233: no inductor compile-worker pool (32 workers, ~16 GB host RAM a trainer process)
t0=$(date +%s)
ARGS=(--model $MODEL --level $IN/lv/2.50/trunk --chunks @$CHUNKS \
  --calib $CALIB --pools $IN/pools214.json --pool train214 --genmask $IN/genmask --head $IN/lv/2.50/head --head-perm $IN/lv/2.50/R16.npy --ab $IN/lv/2.50/ab.safetensors \
  --fixed-eval 20000 --fe-every 12 --monitor 0 --build-lock $BUILD_LOCK --save-every 12 "$@")
if [[ " $* " == *" --pipe-procs "* ]]; then      # w233: one stage process a card, as qrun.sh launches them: MYCARDS = groups of S cards joined by + (A,B / A,B+C,D / A,B,C,D)
  IFS=+ read -ra GRP <<< "$MYCARDS"; IFS=, read -ra CC <<< "${GRP[0]}"; ST=${#CC[@]}; PROCS=()
  for g in "${GRP[@]}"; do IFS=, read -ra CC <<< "$g"; PROCS+=("${CC[@]}"); done
  WS=${#PROCS[@]}; PORT=$((29600 + 10 * ${PROCS[0]})); pids=()
  for r in $(seq 0 $((WS - 1))); do
    lg=$OUT/stdout.log; [ $r -gt 0 ] && lg=$OUT/stdout-r$r.log
    env ROCR_VISIBLE_DEVICES=${PROCS[$r]} RANK=$r WORLD_SIZE=$WS LOCAL_RANK=0 QAD_STAGES=$ST MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT $PY $S $OUT "${ARGS[@]}" >> $lg 2>&1 < /dev/null &
    pids+=($!)
  done
  rc=0; for k in "${pids[@]}"; do wait -n "${pids[@]}"; r_=$?; if [ $r_ != 0 ] && [ $rc = 0 ]; then rc=$r_; kill "${pids[@]}" 2>/dev/null; fi; done
else
  env ROCR_VISIBLE_DEVICES=$MYCARDS $PY $S $OUT "${ARGS[@]}" >> $OUT/stdout.log 2>&1 < /dev/null; rc=$?
fi
log "$NAME rc $rc ($(( $(date +%s) - t0 )) s)"
