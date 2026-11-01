#!/bin/bash
# ladder2.sh CARD -- the full uniform requant (user, 2026-11-01) on qtf2: both cards pull levels from one shared queue
# (QUEUE, one K per line, popped under a lock), so a card never idles while levels remain. Settings: int8-rounded
# codebook, per-tensor input damping sweep, output scales per $OFF (empty = always on; else ourq --out-scales-off),
# B13 code stamp (a resume across code / settings changes is refused). Head NOT included (donors are separate).
# Run one per card:  ladder2.sh 0 & ladder2.sh 1 &       Out /data/v3ladder2/K<k>/, log /data/v3ladder2/ladder-c<CARD>.log
: ${RXM_WORK:=/root}; export RXM_WORK
C=$1 D=/data/v3ladder2 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
Q=${QUEUE:-$D/queue.txt} OFF=${OFF:-}
mkdir -p $D; log() { echo "$(date -u +%H:%M:%S) card $C $*" | tee -a $D/ladder-c$C.log; }
X="--codebook i8 --qtf2 --frac-ext $SRC --sigma-in-sweep 0.025,0.005,0.001"
[ -n "$OFF" ] && X="$X --out-scales-off $OFF"
log "start with: $X"
pop() {   # next level not done and not taken; prints nothing when the queue is empty
  ( flock 8
    while read -r k; do
      [ -z "$k" ] && continue
      [ -f $D/K$k/model.safetensors.index.json ] && [ $(ls $D/K$k/q-L*.safetensors 2>/dev/null | wc -l) -eq 64 ] && continue
      [ -f $D/K$k.taken ] && kill -0 $(cat $D/K$k.taken) 2>/dev/null && continue
      echo $$ > $D/K$k.taken; echo $k; break      # $$ = this card's driver (a subshell keeps it)
    done < $Q ) 8>$D/queue.lock
}
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build
while k=$(pop) && [ -n "$k" ]; do
  s=$(date +%s)
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    $PY -I ourq/ourq.py -K $k -o $D/K$k --device 0 $X >> $D/K$k.log 2>&1 )
  rc=$?
  n=$(ls $D/K$k/q-L*.safetensors 2>/dev/null | wc -l)
  [ $rc = 0 ] && [ $n = 64 ] && log "K$k done in $(( ($(date +%s) - s) / 60 )) min" || log "K$k FAIL rc=$rc ($n/64 layers; resumes on rerun)"
  rm -f $D/K$k.taken
done
log "queue empty"
