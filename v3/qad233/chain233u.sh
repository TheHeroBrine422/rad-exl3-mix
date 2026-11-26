#!/bin/bash
# chain233u.sh (box2; setsid nohup) -- w233 optimization pass 2 (cards 2,3; code15): W2 / W3 with the pass-1 settings (per-stage budgets 10.5,9.5, cap 29.5)
# + --evict-fit (park the smallest covering state, no PIPE_INFLIGHT stall), pool 13; per-segment timelines; bit for bit vs the bases
set -u
W=/data/q27/w233; C=$W/code15/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233u waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233u holds cards 2,3"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
FL="--budgets 3e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --V 12 --plan231 $W/plan-fin12.json --split-layers 32 \
 --attn-gqa --pipe-xfer host --snap-at none --pipe-procs --state-early"
B3="--state-card-gb 6.5,5.5 --state-budget-gb 10.5,9.5 --card-mem-gb 29.5 --park-pool-gb 12 --evict-fit --pool-strict --pipe-ahead 4"
run() {   # run NAME FROM_STEP REF flags...
  local N=$1 S=$2 REF=$3; shift 3; local OUT=$W/runs/$N
  [ -e $OUT/.end ] && { log "$N done before"; return; }
  rm -rf $OUT; mkdir -p $OUT; $PY $C/mkstate233.py $W/runs/fin-w0/state.pt $S $OUT/state.pt
  W233_TL=1 W233_PROF_STEPS=$((S + 4)) $C/run233.sh 2,3 $N $C/pipe233.py "$@" --max-steps $((S + 8)); touch $OUT/.end
  [ -n "$REF" ] && $PY $C/cmp233.py $W/runs/$REF $OUT > $OUT/cmp-$REF.txt 2>&1; log "$N vs $REF: $(grep -E 'train_kl|wall|parameters' $OUT/cmp-$REF.txt 2>/dev/null | tr '\n' ' ')"
}
run oW2-r6 200 yW2 $FL $B3
run oW3-r6 238 oW3-base $FL $B3
log "chain233u finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3 (pass 2 done)" >> $W/cards.log
