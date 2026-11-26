#!/bin/bash
# chain233k.sh (box2; setsid nohup) -- w233 crash test (b): w231's deep300 recipe (D256-25: ext-plan-226c, cap 262144, V 24 on one card pair, state
# budget 8, cap 28.5) UNSERIALIZED with --pipe-procs (2 stage processes) --state-early --park-pool-gb 16, from step 180 (w231-deep300's step-216
# parameters / Adam state re-pointed to step 180 by mkstate233: the data from 180 on is the original's; w231's crashes were in steps 188 and 219) to 310.
# Restartable: a rerun resumes from runs/crash300/state.pt. Holds cards 2,3 for the whole run.
set -u
W=/data/q27/w233; C=$W/code9/v3/qad233; PY=/data/q27/venv/bin/python; OUT=$W/runs/crash300
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233k waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233k holds cards 2,3"
mkdir -p $OUT
[ -e $OUT/state.pt ] || $PY $C/mkstate233.py /data/q27/queue/runs/w231-deep300/state.pt 180 $OUT/state.pt
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
$C/run233.sh 2,3 crash300 $W/code9/v3/qad220/qad220.py --budgets 6e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 \
  --cap-keep /data/q27/queue/slices/jlong-ids.json --plan-ext /data/q27/queue/ext/ext-plan-226c.json --ext-root /data/q27/queue/ext --lr-steps 3e-4:1.5e6,1e-4:5.6e6 \
  --split-layers 32 --state-card-gb 4 --attn-gqa --snap-at none --max-steps 310 --state-budget-gb 8 --card-mem-gb 28.5 --pipe-procs --state-early --park-pool-gb 16
log "chain233k finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3" >> $W/cards.log
