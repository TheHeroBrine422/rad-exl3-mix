#!/bin/bash
# ladder2.sh CARD -- the full uniform requant (user, 2026-11-01) on qtf2: both cards pull levels from one shared queue
# (QUEUE, one K per line, popped under a lock), so a card never idles while levels remain. Settings: int8-rounded
# codebook, per-tensor input damping sweep, output scales per $OFF (empty = always on; else ourq --out-scales-off),
# B13 code stamp (a resume across code / settings changes is refused). Head NOT included (donors are separate).
# A queue line "RUN:<script>" is an exclusive slot: the card that pops it waits until no other level is running,
# then runs the script with exclusive host RAM (head donor remake: needs ~26 GB; ladder levels hold shared RAM).
# A queue line "a,b,c" is a group: one ourq process quantizes those levels layer by layer, reusing each tensor's
# K-independent work (output identical to one run per level: v3/levels/multik.sh). Group logs G<a_b_c>.log.
# A line "mul1:K" (or "mul1:a,b") quantizes with the plain mul1 codebook into $D/mul1/K<k>; other lines use i8 into
# $D/K<k>. env SWEEP (input-group damping grid), QUEUE.
# Run one per card:  ladder2.sh 0 & ladder2.sh 1 &       Out /data/v3ladder2/K<k>/, log /data/v3ladder2/ladder-c<CARD>.log
: ${RXM_WORK:=/root}; export RXM_WORK
C=$1 D=${LADDER_D:-/data/v3ladder2} PY=${LADDER_PY:-/data/venv-exl3/bin/python} SRC=/data/exl3frac/src
Q=${QUEUE:-$D/queue.txt} OFF=${OFF:-}
mkdir -p $D; log() { echo "$(date -u +%H:%M:%S) card $C $*" | tee -a $D/ladder-c$C.log; }
SWEEP=${SWEEP:-0.025,0.005,0.001}
X="--qtf2 --frac-ext $SRC --sigma-in-sweep $SWEEP"
cbof() { case $1 in mul1:*) echo mul1;; *) echo i8;; esac; }          # line -> codebook
rootof() { case $1 in mul1:*) echo $D/mul1;; *) echo $D;; esac; }     # line -> output root
levof() { echo ${1#mul1:}; }                                          # line -> level list
[ -n "$OFF" ] && X="$X --out-scales-off $OFF"
log "start with: $X"
pop() {   # next level not done and not taken; prints nothing when the queue is empty
  ( flock 8
    while read -r k; do
      [ -z "$k" ] && continue
      case $k in RUN:*) [ -f $D/$(basename ${k#RUN:}).done ] && continue; [ -f $D/slot.taken ] && kill -0 $(cat $D/slot.taken) 2>/dev/null && continue
                        echo $$ > $D/slot.taken; echo $k; break;; esac
      R=$(rootof $k) lv=$(levof $k); mkdir -p $R
      done_all=1; for kk in ${lv//,/ }; do
        [ -f $R/K$kk/model.safetensors.index.json ] && [ $(ls $R/K$kk/q-L*.safetensors 2>/dev/null | wc -l) -eq 64 ] || done_all=0
      done; [ $done_all = 1 ] && continue
      g=$(cbof $k)_${lv//,/_}
      [ -f $D/K$g.taken ] && kill -0 $(cat $D/K$g.taken) 2>/dev/null && continue
      echo $$ > $D/K$g.taken; echo $k; break      # $$ = this card's driver (a subshell keeps it)
    done < $Q ) 8>$D/queue.lock
}
others_running() {   # another card's level is in progress
  local t; for t in $D/K*.taken; do [ -f "$t" ] && kill -0 $(cat "$t") 2>/dev/null && return 0; done; return 1
}
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build
while k=$(pop) && [ -n "$k" ]; do
  s=$(date +%s)
  if [ "${k#RUN:}" != "$k" ]; then
    sc=${k#RUN:}; log "slot $sc: waiting for the other card's level"
    while others_running; do sleep 30; done
    ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock 7; CARD=$C $sc >> $D/slot.log 2>&1 ) \
      && touch $D/$(basename $sc).done && log "slot $sc done in $(( ($(date +%s) - s) / 60 )) min" || log "slot $sc FAIL (slot.log)"
    rm -f $D/slot.taken; continue
  fi
  R=$(rootof $k) lv=$(levof $k) cb=$(cbof $k); g=${cb}_${lv//,/_}
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    $PY -I ourq/ourq.py -K $lv -o "$R/K{K}" --device 0 --codebook $cb $X >> $D/K$g.log 2>&1 )
  rc=$?
  n=""; for kk in ${lv//,/ }; do n="$n $(ls $R/K$kk/q-L*.safetensors 2>/dev/null | wc -l)"; done
  [ $rc = 0 ] && [ -z "$(echo $n | tr ' ' '\n' | grep -v '^64$')" ] && log "K$k done in $(( ($(date +%s) - s) / 60 )) min" \
    || log "K$k FAIL rc=$rc (layers per level:$n; resumes on rerun)"
  rm -f $D/K$g.taken
done
log "queue empty"
