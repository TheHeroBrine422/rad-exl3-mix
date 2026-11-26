#!/bin/bash
# chain233w.sh (box2; setsid nohup) -- w233 final 4-card check of the r6 recipe as 2 dp x 2 stages (cards 0,1+2,3; code15): the dp100 recipe (20 steps,
# vs dp100 24.34 / d22 18.53 s per step) and fin24 W1 / W2 (vs g22-W1 25.3 / g22-W2 29.1), timelines, host RSS of the 4 processes, rank walls
set -u
W=/data/q27/w233; C=$W/code15/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233w waiting for cards 0,1,2,3"
for c in 0 1 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233w holds cards 0,1,2,3"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
COM="--lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --attn-gqa --snap-at none --state-early"
R6="--pipe-procs 2 --split-layers 32 --state-card-gb 6.5,5.5 --state-budget-gb 10.5,9.5 --card-mem-gb 29.5 --park-pool-gb 12 --evict-fit --pool-strict --pipe-ahead 4"
DP="--budgets 6e6 --cap-keep /data/q27/queue/slices/jlong-ids.json --plan-ext /data/q27/queue/ext/ext-plan-226c.json --ext-root /data/q27/queue/ext --lr-steps 3e-4:1.5e6,1e-4:5.6e6 --max-steps 20"
FIN="--budgets 6e6 --V 24 --plan231 $W/plan-fin24.json"
run() {   # run NAME CARDS FROM_STEP flags...
  local N=$1 CA=$2 S=$3; shift 3; local OUT=$W/runs/$N
  [ -e $OUT/.end ] && { log "$N done before"; return; }
  rm -rf $OUT; mkdir -p $OUT; [ $S -gt 0 ] && $PY $C/mkstate233.py $W/runs/fin-w0/state.pt $S $OUT/state.pt
  W233_TL=1 $C/run233.sh $CA $N $W/code15/v3/qad220/qad220.py "$@"; touch $OUT/.end
}
run f22-dp 0,1+2,3 0 $COM $R6 $DP
run f22-W2 0,1+2,3 200 $COM $R6 $FIN --max-steps 208
run f22-W1 0,1+2,3 120 $COM $R6 $FIN --max-steps 128
log "chain233w finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 0,1 (keeps 2,3 idle)" >> $W/cards.log
