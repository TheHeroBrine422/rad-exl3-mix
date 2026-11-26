#!/bin/bash
# chain233m.sh (box2; setsid nohup) -- w233 crash test (b), resumed: crash300 from its state.pt (step 204) to 310, same recipe as chain233k, park pool
# 20 GB (crash300 had pool misses at 16 GB -> pageable parks; it was OOM-killed by the host at step 211, not a GPU fault). Cards 2,3, one holder.
set -u
W=/data/q27/w233; C=$W/code10/v3/qad233; OUT=$W/runs/crash300
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233m waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233m holds cards 2,3 (crash300 resumes from $(/data/q27/venv/bin/python -c "import torch; print(torch.load('$OUT/state.pt', weights_only=False)['st']['step'])"))"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
$C/run233.sh 2,3 crash300 $W/code10/v3/qad220/qad220.py --budgets 6e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 \
  --cap-keep /data/q27/queue/slices/jlong-ids.json --plan-ext /data/q27/queue/ext/ext-plan-226c.json --ext-root /data/q27/queue/ext --lr-steps 3e-4:1.5e6,1e-4:5.6e6 \
  --split-layers 32 --state-card-gb 4 --attn-gqa --snap-at none --max-steps 310 --state-budget-gb 8 --card-mem-gb 28.5 --pipe-procs --state-early --park-pool-gb 20
log "chain233m finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3 (crash300 done)" >> $W/cards.log
