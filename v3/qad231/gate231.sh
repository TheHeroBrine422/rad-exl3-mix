#!/bin/bash
# gate231.sh LISTFILE CARD... (box2; setsid nohup) -- w231 GPU runs: waits for the listed cards' locks (never preempts), holds them, then runs each line of LISTFILE
#   ARM CARDS SIZE BUDGETS flags...     CARDS with letters for the listed cards in order (A B C D): "A,B" = one process on two cards (--split-layers),
#                                       "A,B+C,D" = two data-parallel ranks (qrun.sh), "p2p231 MODE SECONDS" = the transfer stress test on A,B (no trainer)
# through the queue's runner (qrun.sh: w214's data / student paths) with w231's trainer copy (/data/q27/w231/code). Outputs QROOT/runs/<ARM> (ARM must start with
# w231-); a line is skipped when runs/<ARM>/gate231.done exists, else its run dir is wiped (NOWIPE=1: resumed from state.pt). A crash is logged, not retried.
# Extra environment for the trainer (e.g. AMD_SERIALIZE_KERNEL=3) passes through. Log: /data/q27/w231/gate231.log
set -u
LIST=$1; shift; CARDSL=("$@")
W=/data/q27/w231; QROOT=/data/q27/queue; CODE=$W/code
export CODE_PY=$CODE/v3/qad220/qad220.py QCARD_LOCKED=1
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/gate231.log; }
log "waiting for cards ${CARDSL[*]}"
for c in "${CARDSL[@]}"; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
log "cards ${CARDSL[*]} held"
sub() { local x=$1 k=0; for L in A B C D; do [ $k -lt ${#CARDSL[@]} ] && x=${x//$L/${CARDSL[$k]}}; k=$((k + 1)); done; echo $x; }
while read -r ARM CARDS SIZE BUD FLAGS; do
  case "$ARM" in ''|\#*) continue;; esac
  if [ "$ARM" = p2p231 ]; then                 # p2p231 MODE SECONDS: the stress test on the first two cards
    log "p2p231 $CARDS $SIZE s on cards ${CARDSL[0]},${CARDSL[1]}"
    ROCR_VISIBLE_DEVICES=${CARDSL[0]},${CARDSL[1]} /data/q27/venv/bin/python $CODE/v3/qad231/p2p231.py $CARDS $SIZE >> $W/p2p231-$CARDS.log 2>&1 < /dev/null
    log "p2p231 $CARDS rc $? (log $W/p2p231-$CARDS.log)"; continue
  fi
  case "$ARM" in w231-*) ;; *) log "skip $ARM: not a w231- arm"; continue;; esac
  [ -e $QROOT/runs/$ARM/gate231.done ] && { log "$ARM done before"; continue; }
  CARDS=$(sub $CARDS)
  [ -n "${NOWIPE:-}" ] || rm -rf $QROOT/runs/$ARM
  log "$ARM start (cards $CARDS): $SIZE $BUD $FLAGS"
  t0=$(date +%s)
  while true; do eval "$CODE/v3/qadq/qrun.sh box2 $ARM $CARDS $SIZE $BUD $FLAGS" < /dev/null; rc=$?; [ $rc = 76 ] || break; log "$ARM: RAM floor, waiting"; sleep 120; done
  log "$ARM rc $rc ($(( $(date +%s) - t0 )) s)"
  [ $rc = 0 ] && touch $QROOT/runs/$ARM/gate231.done
done < $LIST
log "list $LIST finished; cards ${CARDSL[*]} released"
