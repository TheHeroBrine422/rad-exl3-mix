#!/bin/bash
# card220.sh CARD LIST -- one card's sequential queue for w220 (restartable: finished steps are skipped by their outputs). Before each step it waits until w217's own
# queue / reader for this card is gone (never takes a card between w217's arms) and the card lock is free. LIST lines:
#   T ARM SIZE BUDGETS [qad220 flags]      train (run220.sh; skipped when runs/ARM/ckpt-<last budget>/.done exists); retried while the card is busy (75) / RAM short (76)
#   R ARM SIZE CKPT                        served read on j217 (serve220.sh; skipped when scores/ARM-CKPT.json exists)
#   B LABEL SIZE [qad220 flags]            speed / equivalence run: bench220.sh (a few steps; skipped when runs/LABEL/bench.done exists)
CARD=$1; LIST=$2
cd /data/q27/w220
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> q220.log; }
w217gone() { ! pgrep -f "[c]ard217.sh $CARD " > /dev/null && ! pgrep -f "[r]eader217.sh $CARD " > /dev/null && ! pgrep -f "[q]217.sh .* $CARD " > /dev/null; }
waitcard() { until w217gone; do sleep 120; done; }
while read -r K REST; do
  [ -z "$K" ] && continue; case "$K" in \#*) continue;; esac
  set -- $REST
  case "$K" in
    T) ARM=$1; SIZE=$2; BUD=$3; shift 3
       LASTN=$(python3 -c "print('%.0e' % float('${BUD##*,}'))")
       [ -e runs/$ARM/ckpt-$LASTN/.done ] && continue
       while true; do waitcard; code/v3/qad220/run220.sh $ARM $CARD $SIZE $BUD "$@"; rc=$?; case $rc in 75|76) sleep 120;; *) break;; esac; done
       say "train $ARM card $CARD rc $rc" ;;
    R) waitcard; code/v3/qad220/serve220.sh $1 $2 $3 $CARD >> logs/reader220-$CARD.log 2>&1
       say "read $1-$3 card $CARD rc $?" ;;
    B) LABEL=$1; SIZE=$2; shift 2
       [ -e runs/$LABEL/bench.done ] && continue
       while true; do waitcard; code/v3/qad220/bench220.sh $LABEL $CARD $SIZE "$@"; rc=$?; case $rc in 75|76) sleep 120;; *) break;; esac; done
       say "bench $LABEL card $CARD rc $rc" ;;
  esac
done < $LIST
say "card $CARD queue $LIST done"
