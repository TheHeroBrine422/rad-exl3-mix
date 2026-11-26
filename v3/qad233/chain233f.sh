#!/bin/bash
# chain233f.sh (box2; setsid nohup) -- w233 GPU block 4 (one holder of cards 2,3), code4 (N-stage --pipe-procs):
#   eqS4: fin12 steps 1-20 with --pipe-procs (2 stages) = eqP (one-process --pipeline) bit for bit? (cmp233.py)
#   aW1-2 / aW1-4: the W1 window (resumed at 120, deep traces ~120k, 8 steps) with --pipe-ahead 2 / 4 (vs sW1 = ahead 0: 23.7 s / step)
set -u
W=/data/q27/w233; C=$W/code4/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233f waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233f holds cards 2,3"
FL="--budgets 3e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --V 12 --plan231 $W/plan-fin12.json --split-layers 32 \
 --attn-gqa --state-card-gb 4 --state-budget-gb 8 --card-mem-gb 28.5 --pipe-xfer host --snap-at none --pipe-procs"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
if [ ! -e $W/runs/eqS4/state.pt ]; then
  rm -rf $W/runs/eqS4; W233_PROF_STEPS=12 $C/run233.sh 2,3 eqS4 $C/pipe233.py $FL --max-steps 20
  $PY $C/cmp233.py $W/runs/eqP $W/runs/eqS4 > $W/runs/eqS4/cmp-eqP.txt 2>&1; log "eqS4 vs eqP: $(grep -E 'train_kl|parameters' $W/runs/eqS4/cmp-eqP.txt | tr '\n' ' ')"
fi
for AH in 2 4; do
  N=aW1-$AH; OUT=$W/runs/$N
  [ -e $OUT/.end ] && continue
  rm -rf $OUT; mkdir -p $OUT; $PY $C/mkstate233.py $W/runs/fin-w0/state.pt 120 $OUT/state.pt
  W233_PROF_STEPS=124 $C/run233.sh 2,3 $N $C/pipe233.py $FL --max-steps 128 --pipe-ahead $AH; touch $OUT/.end
done
log "chain233f finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3" >> $W/cards.log
