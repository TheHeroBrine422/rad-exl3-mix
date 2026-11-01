#!/bin/bash
# chain1b.sh -- resume chain1's evals after gate 2e's both-card collection (chain1 paused 2026-10-31 22:45 CT: its
# per-eval card-1 lock kept gate 2e, holding card 0, waiting). Same queue; the eval cut off then is re-claimed.
: ${RXM_WORK:=/root}; export RXM_WORK
W=/data/v3levels PY=/data/venv-exl3/bin/python L=/data/v3levels/chain1.log
cd $RXM_WORK
log() { echo "$(date -u +%H:%M) $*" | tee -a $L; }
until grep -q "^DONE" /data/yaqa/full64.log 2>/dev/null || ! pgrep -f "^/bin/bash ./gate2e.sh" > /dev/null; do sleep 60; done
for t in $(ls evq/claim); do case $t in v3e-*|v3s-*) [ -f evq/scores/$t.f32 ] || rmdir evq/claim/$t;; esac; done
log "chain1 evals resumed"
Q=$W/chain1.q
./auto/evalq2.sh 1 $Q >> $W/chain1-eval.log 2>&1
S=$RXM_WORK/evq/scores
for G in golden-fit3 golden-ml; do
  for t in le2 le3 le4 le5 t2 t2.5 t3 t3.5 t4; do
    [ -f $S/v3e-$t-$G.f32 ] && log "emb $t $G vs base6: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/$G $S/v3e-base6-$G.f32 $S/v3e-$t-$G.f32)"
  done
done
for t in o6 td6 o6d; do
  [ -f $S/v3s-$t-fit3.f32 ] && log "swap $t vs base5: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-fit3 $S/v3s-base5-fit3.f32 $S/v3s-$t-fit3.f32)"
done
[ -f $S/v3s-td6-fit3.f32 ] && log "swap td6 vs o6: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-fit3 $S/v3s-o6-fit3.f32 $S/v3s-td6-fit3.f32)"
[ -f $S/v3s-o6d-fit3.f32 ] && log "swap o6d vs o6: $($PY -I $RXM_WORK/hproxy/paired.py $RXM_WORK/golden-fit3 $S/v3s-o6-fit3.f32 $S/v3s-o6d-fit3.f32)"
rm -f $W/emb/emb-K*.rad                                    # scratch (54 GB reflink copies, 2.5 GB unshared each)
log "chain1 done"
