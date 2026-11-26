#!/bin/bash
# chain233d.sh (box2; setsid nohup) -- w233 GPU block 2 (one holder of cards 2,3): (0) test233 on card 2 (pinned async embed / targets = old, on a card);
# (1) the 2-card equivalence: the fin12 plan from step 0, 20 steps, --pipeline (one process, eqP) vs --pipe-procs (two stage processes, eqS), code3
# (state.pt parameters + the loss curve compared by cmp233.py); W233_PROF_STEPS=12: one profiled step each (GPU busy per card).
set -u
W=/data/q27/w233; C=$W/code3/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233d waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233d holds cards 2,3"
(cd $C && ROCR_VISIBLE_DEVICES=2 QAD_PIPELINE=$W/code3/pipeline $PY test233.py cuda:0 > $W/test233-gpu.txt 2>&1); log "test233 on card 2: $(tail -n 1 $W/test233-gpu.txt)"
FL="--budgets 3e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --V 12 --plan231 $W/plan-fin12.json --split-layers 32 \
 --attn-gqa --state-card-gb 4 --state-budget-gb 8 --card-mem-gb 28.5 --pipe-xfer host --snap-at none --max-steps 20"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256 W233_PROF_STEPS=12
for N in eqP eqS; do
  [ -e $W/runs/$N/state.pt ] && { log "$N done before"; continue; }
  rm -rf $W/runs/$N
  if [ $N = eqP ]; then $C/run233.sh 2,3 $N $C/pipe233.py $FL --pipeline; else $C/run233.sh 2,3 $N $C/pipe233.py $FL --pipe-procs; fi
done
log "chain233d finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3" >> $W/cards.log
