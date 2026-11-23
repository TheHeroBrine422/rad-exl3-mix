#!/bin/bash
# gate225.sh CARDA CARDB LISTFILE  (box2) -- w225's GPU gates: holds both card locks (waits for them; never preempts), then runs each line of LISTFILE
#   ARM CARDS SIZE BUDGETS flags...      CARDS = one card (A or B) or "A,B" (the trainer sees both: --split-layers)
# through the queue's runner (qrun.sh: w214's data / student / judge paths) with w225's trainer copy (CODE_PY). Outputs QROOT/runs/<ARM> (ARM must start
# with w225-); a line is skipped when runs/<ARM>/gate225.done exists, else its run dir is wiped and it runs from scratch. Log: /data/q27/w225/gate225.log
set -u
A=$1 B=$2 LIST=$3
W=/data/q27/w225; CODE=$W/code; QROOT=/data/q27/queue
export CODE_PY=$CODE/v3/qad220/qad220.py QCARD_LOCKED=1
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/gate225.log; }
exec {la}>/data/q27/locks/card$A.lock; exec {lb}>/data/q27/locks/card$B.lock
log "waiting for cards $A + $B"; flock $la; flock $lb; log "cards $A + $B held"
while read -r ARM CARDS SIZE BUD FLAGS; do
  case "$ARM" in ''|\#*) continue;; esac
  case "$ARM" in w225-*) ;; *) log "skip $ARM: not a w225- arm"; continue;; esac
  [ -e $QROOT/runs/$ARM/gate225.done ] && { log "$ARM done before"; continue; }
  CARDS=${CARDS//A/$A}; CARDS=${CARDS//B/$B}
  rm -rf $QROOT/runs/$ARM
  log "$ARM start (cards $CARDS): $SIZE $BUD $FLAGS"
  t0=$(date +%s)
  while true; do eval "$CODE/v3/qadq/qrun.sh box2 $ARM $CARDS $SIZE $BUD $FLAGS" < /dev/null; rc=$?; [ $rc = 76 ] || break; log "$ARM: RAM floor, waiting"; sleep 120; done
  log "$ARM rc $rc ($(( $(date +%s) - t0 )) s)"
  [ $rc = 0 ] && touch $QROOT/runs/$ARM/gate225.done
done < $LIST
log "list $LIST finished; cards $A + $B released"
