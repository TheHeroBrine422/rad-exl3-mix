#!/bin/bash
# chain233g.sh (box2; setsid nohup) -- w233 GPU block 5 (cards 2,3, one holder), code5: --budget-evict-first (park idle streams' states before stalling
# the pipeline) on the deep windows W1 (resumed at 120) and W2 (at 200), 8 steps each (vs sW1 24.6 / sW2 ~31.5 s / step without it)
set -u
W=/data/q27/w233; C=$W/code5/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233g waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233g holds cards 2,3"
FL="--budgets 3e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --V 12 --plan231 $W/plan-fin12.json --split-layers 32 \
 --attn-gqa --state-card-gb 4 --state-budget-gb 8 --card-mem-gb 28.5 --pipe-xfer host --snap-at none --pipe-procs --budget-evict-first"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
for X in "eW1 120" "eW2 200"; do
  set -- $X; N=$1; S=$2; OUT=$W/runs/$N
  [ -e $OUT/.end ] && continue
  rm -rf $OUT; mkdir -p $OUT; $PY $C/mkstate233.py $W/runs/fin-w0/state.pt $S $OUT/state.pt
  W233_PROF_STEPS=$((S + 4)) $C/run233.sh 2,3 $N $C/pipe233.py $FL --max-steps $((S + 8)); touch $OUT/.end
done
log "chain233g finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3" >> $W/cards.log
