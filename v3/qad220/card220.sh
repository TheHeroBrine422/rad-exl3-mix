#!/bin/bash
# card220.sh CARD JOBS -- a card's dispatcher for w220: waits until w217's queue / dispatcher / reader for this card is gone (never takes a card between w217's arms) and the card
# lock is free, then takes the next job of the shared list JOBS (cursor in JOBS.cursor under flock), runs it, repeats until none is left. Restartable: finished work is skipped by its outputs
# (a restart with the cursor file removed re-walks the list). Job lines:
#   T ARM SIZE BUDGETS READS flags...     train (run220.sh), then the served reads READS (comma list of ckpt names, or -) on j217 (serve220.sh)
#   B LABEL SIZE flags...                 speed / equivalence run (bench220.sh: a few steps; skipped when runs/LABEL/bench.done exists)
CARD=$1; JOBS=$2; CUR=$JOBS.cursor
cd /data/q27/w220
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> q220.log; }
w217gone() { ! pgrep -f "[c]ard217.sh $CARD " > /dev/null && ! pgrep -f "[r]eader217.sh $CARD " > /dev/null && ! pgrep -f "[p]ull217.sh $CARD " > /dev/null && ! pgrep -f "[q]217.sh .* $CARD " > /dev/null && ! pgrep -f "[s]erve217.sh .* $CARD( |$)" > /dev/null; }
waitcard() { until w217gone; do sleep 120; done; }
while true; do
  waitcard
  exec {lk}> $JOBS.lock; flock $lk
  n=$(cat $CUR 2>/dev/null || echo 0); LINE=$(grep -v '^#' $JOBS | grep -v '^$' | sed -n "$((n+1))p")
  [ -n "$LINE" ] && echo $((n+1)) > $CUR
  exec {lk}>&-
  [ -z "$LINE" ] && break
  set -- $LINE; K=$1; shift
  case "$K" in
    T) ARM=$1; SIZE=$2; BUD=$3; READS=$4; shift 4
       LASTN=$(python3 -c "print('%.0e' % float('${BUD##*,}'))")
       if [ ! -e runs/$ARM/ckpt-$LASTN/.done ]; then
         while true; do waitcard; code/v3/qad220/run220.sh $ARM $CARD $SIZE $BUD "$@"; rc=$?; case $rc in 75|76) sleep 120;; *) break;; esac; done
         say "train $ARM card $CARD rc $rc"
       fi
       for ck in ${READS//,/ }; do
         [ "$ck" = - ] && continue
         code/v3/qad220/serve220.sh $ARM $SIZE $ck $CARD >> logs/reader220-$CARD.log 2>&1; say "read $ARM-$ck card $CARD rc $?"
       done ;;
    B) LABEL=$1; SIZE=$2; shift 2
       if [ ! -e runs/$LABEL/bench.done ]; then
         while true; do waitcard; code/v3/qad220/bench220.sh $LABEL $CARD $SIZE "$@"; rc=$?; case $rc in 75|76) sleep 120;; *) break;; esac; done
         say "bench $LABEL card $CARD rc $rc"
       fi ;;
  esac
done
say "card $CARD dispatcher done"
