#!/bin/bash
# chain233v.sh (box2; setsid nohup) -- w233 cycle 2 (cards 2,3; code15): stage 1 is now ~95 % busy (the bottleneck): move work off it.
#   r7: r6 + --kl-keep (no head/KL recompute on stage 1);  r8: r6 with split 34 (budgets 10,10);  r9: r8 + --kl-keep. W2 (vs yW2) and W0 (vs yW0)
set -u
W=/data/q27/w233; C=$W/code15/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233v waiting for cards 2,3"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233v holds cards 2,3"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
FL="--budgets 3e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --V 12 --plan231 $W/plan-fin12.json \
 --attn-gqa --pipe-xfer host --snap-at none --pipe-procs --state-early --card-mem-gb 29.5 --park-pool-gb 12 --evict-fit --pool-strict --pipe-ahead 4"
R6="--split-layers 32 --state-card-gb 6.5,5.5 --state-budget-gb 10.5,9.5"
R8="--split-layers 34 --state-card-gb 6,6 --state-budget-gb 10,10"
run() {   # run NAME FROM_STEP REF flags...
  local N=$1 S=$2 REF=$3; shift 3; local OUT=$W/runs/$N
  [ -e $OUT/.end ] && { log "$N done before"; return; }
  rm -rf $OUT; mkdir -p $OUT; [ $S -gt 0 ] && $PY $C/mkstate233.py $W/runs/fin-w0/state.pt $S $OUT/state.pt
  W233_TL=1 W233_PROF_STEPS=$((S + 4)) $C/run233.sh 2,3 $N $C/pipe233.py "$@" --max-steps $((S + 8)); touch $OUT/.end
  [ -n "$REF" ] && $PY $C/cmp233.py $W/runs/$REF $OUT > $OUT/cmp-$REF.txt 2>&1; log "$N vs $REF: $(grep -E 'train_kl|wall|parameters' $OUT/cmp-$REF.txt 2>/dev/null | tr '\n' ' ')"
}
run oW2-r7 200 yW2 $FL $R6 --kl-keep
run oW2-r8 200 yW2 $FL $R8
run oW2-r9 200 yW2 $FL $R8 --kl-keep
run oW0-r7 0 yW0 $FL $R6 --kl-keep
run oW0-r8 0 yW0 $FL $R8
log "chain233v finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3 (cycle 2 done)" >> $W/cards.log
