#!/bin/bash
# chain233l.sh (box2; setsid nohup) -- w233 (c): the finals layouts on all four cards (one holder of cards 0-3), code10 (= code9 + the S-card run233) + --state-early:
#   d22 / d14: w231-dp100's recipe (ext-plan-226c, cap 262144, V 24) from scratch, 20 steps, as 2 dp ranks x 2 stages (cards 0,1+2,3) and as 1 rank x 4
#              stages (0,1,2,3; bounds 16,32,48): vs dp100 (2 ranks x the one-process pipeline: ~30 s / step)
#   g22-W1 / g14-W1, g22-W2 / g14-W2: the fin24 plan (finals-like mix, V 24, 4 deep traces from step 0) resumed at 120 / 200 (traces ~120k / ~200k), 8 steps;
#              W2 with the per-segment timeline (W233_TL=1)
set -u
W=/data/q27/w233; C=$W/code10/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233l waiting for cards 0,1,2,3"
for c in 0 1 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233l holds cards 0,1,2,3"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
COM="--lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --attn-gqa --snap-at none --state-early"
L22="--pipe-procs 2 --split-layers 32 --state-card-gb 4 --state-budget-gb 8 --card-mem-gb 28.5 --park-pool-gb 16"
L14="--pipe-procs 4 --stage-bounds 16,32,48 --state-card-gb 8 --state-budget-gb 8 --card-mem-gb 28.5 --park-pool-gb 4"
DP="--budgets 6e6 --cap-keep /data/q27/queue/slices/jlong-ids.json --plan-ext /data/q27/queue/ext/ext-plan-226c.json --ext-root /data/q27/queue/ext --lr-steps 3e-4:1.5e6,1e-4:5.6e6 --max-steps 20"
run() {   # run NAME CARDS FROM_STEP [ENV] flags...
  local N=$1 CA=$2 S=$3 E=$4; shift 4; local OUT=$W/runs/$N
  [ -e $OUT/.end ] && { log "$N done before"; return; }
  rm -rf $OUT; mkdir -p $OUT; [ $S -gt 0 ] && $PY $C/mkstate233.py $W/runs/fin-w0/state.pt $S $OUT/state.pt
  env $E $C/run233.sh $CA $N $W/code10/v3/qad220/qad220.py "$@"; touch $OUT/.end
}
run d22 0,1+2,3 0 X=1 $COM $L22 $DP
run d14 0,1,2,3 0 X=1 $COM $L14 $DP
FIN="--budgets 6e6 --V 24 --plan231 $W/plan-fin24.json"
run g22-W1 0,1+2,3 120 X=1 $COM $L22 $FIN --max-steps 128
run g14-W1 0,1,2,3 120 X=1 $COM $L14 $FIN --max-steps 128
run g22-W2 0,1+2,3 200 W233_TL=1 $COM $L22 $FIN --max-steps 208
run g14-W2 0,1,2,3 200 W233_TL=1 $COM $L14 $FIN --max-steps 208
log "chain233l finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 0,1,2,3" >> $W/cards.log
