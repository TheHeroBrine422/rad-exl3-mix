#!/bin/bash
# chain233e.sh (box2; setsid nohup) -- w233 GPU block 3 (one holder of cards 2,3), code3 with --pipe-procs on the fin12 plan:
#   sS:  steps 1-20 from scratch, py-spy (all threads --idle, then --gil) on each stage process during steps ~9-15 (where the host time goes);
#   sW1: resumed at step 120 (deep traces ~120k; = fin-w1 of the old pipeline: 29.8 s / step), 8 steps, step 124 profiled;
#   sW2: resumed at step 200 (~200k; = fin-w2: 45.4 s / step), 8 steps, step 204 profiled.
set -u
W=/data/q27/w233; C=$W/code3/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233e waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233e holds cards 2,3"
FL="--budgets 3e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --V 12 --plan231 $W/plan-fin12.json --split-layers 32 \
 --attn-gqa --state-card-gb 4 --state-budget-gb 8 --card-mem-gb 28.5 --pipe-xfer host --snap-at none --pipe-procs"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
spy() {   # spy OUT FROM_STEPS: py-spy on both stage processes once train.jsonl has FROM lines
  local OUT=$1 N=$2
  until [ "$(grep -c '"what": "main"' $OUT/train.jsonl 2>/dev/null)" -ge $N ]; do sleep 5; [ -e $OUT/.end ] && return; done
  for r in 0 1; do
    P=$(pgrep -f "^/data/q27/venv/bin/python $C/pipe233.py $OUT " | while read p; do grep -q "RANK=$r" <(tr '\0' '\n' < /proc/$p/environ) && echo $p; done | head -1)
    py-spy record -p $P -r 40 -d 40 --idle --threads -f raw -o $OUT/pyspy-all-r$r.txt --nonblocking > $OUT/pyspy-all-r$r.log 2>&1 &
  done; wait
  for r in 0 1; do
    P=$(pgrep -f "^/data/q27/venv/bin/python $C/pipe233.py $OUT " | while read p; do grep -q "RANK=$r" <(tr '\0' '\n' < /proc/$p/environ) && echo $p; done | head -1)
    py-spy record -p $P -r 40 -d 30 --gil --threads -f raw -o $OUT/pyspy-gil-r$r.txt --nonblocking > $OUT/pyspy-gil-r$r.log 2>&1 &
  done; wait
}
win() {   # win NAME FROM_STEP NSTEPS SPY_AT
  local N=$1 S=$2 OUT=$W/runs/$1
  [ -e $OUT/.end ] && { log "$N done before"; return; }
  rm -rf $OUT; mkdir -p $OUT
  [ $S -gt 0 ] && $PY $C/mkstate233.py $W/runs/fin-w0/state.pt $S $OUT/state.pt
  W233_PROF_STEPS=$((S + 4)) $C/run233.sh 2,3 $N $C/pipe233.py $FL --max-steps $((S + $3)) &
  local R=$!
  [ $4 -gt 0 ] && spy $OUT $4
  wait $R; touch $OUT/.end
}
win sS 0 20 8
win sW1 120 8 5
win sW2 200 8 5
log "chain233e finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3" >> $W/cards.log
