#!/bin/bash
# gate226.sh CARDA CARDB LISTFILE  (box2) -- w226 GPU tests: holds both card locks (waits for them; never preempts), then runs each line of LISTFILE
#   ARM CARDS SIZE BUDGETS flags...      CARDS = one card (A or B) or "A,B" (the trainer sees both: --split-layers)
# through the queue's runner (qrun.sh: w214's data / student / judge paths) with the trainer in GATE_CODE (default the queue's). Outputs QROOT/runs/<ARM> (ARM must start
# with w226-); a line is skipped when runs/<ARM>/gate226.done exists, else its run dir is wiped and it runs from scratch. Log: /data/q27/w226/gate226.log
set -u
A=$1 B=$2 LIST=$3
W=/data/q27/w226; QROOT=/data/q27/queue; CODE=${GATE_CODE:-$QROOT/code}
export CODE_PY=$CODE/v3/qad220/qad220.py QCARD_LOCKED=1
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/gate226.log; }
exec {la}>/data/q27/locks/card$A.lock; exec {lb}>/data/q27/locks/card$B.lock
log "waiting for cards $A + $B"; flock $la; flock $lb; log "cards $A + $B held"
while read -r ARM CARDS SIZE BUD FLAGS; do
  case "$ARM" in ''|\#*) continue;; esac
  case "$ARM" in w226-*) ;; *) log "skip $ARM: not a w226- arm"; continue;; esac
  [ -e $QROOT/runs/$ARM/gate226.done ] && { log "$ARM done before"; continue; }
  CARDS=${CARDS//A/$A}; CARDS=${CARDS//B/$B}
  [ -n "${NOWIPE:-}" ] || rm -rf $QROOT/runs/$ARM          # NOWIPE=1: resume the run from its state.pt
  log "$ARM start (cards $CARDS): $SIZE $BUD $FLAGS"
  t0=$(date +%s)
  while true; do eval "$CODE/v3/qadq/qrun.sh box2 $ARM $CARDS $SIZE $BUD $FLAGS" < /dev/null; rc=$?; [ $rc = 76 ] || break; log "$ARM: RAM floor, waiting"; sleep 120; done
  log "$ARM rc $rc ($(( $(date +%s) - t0 )) s)"
  [ $rc = 0 ] && touch $QROOT/runs/$ARM/gate226.done
done < $LIST
log "list $LIST finished; cards $A + $B released"
