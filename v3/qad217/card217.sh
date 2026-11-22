#!/bin/bash
# card217.sh CARD LIST -- one card's sequential queue (restartable: a finished arm / read is skipped by its outputs). LIST lines:
#   T ARM SIZE BUDGETS [qad200 flags]      train (q217.sh; skipped when runs/ARM/ckpt-<last budget>/.done exists)
#   W ARM                                  wait until runs/ARM/ckpt-5e+05/.done exists (a training started by an earlier driver)
#   R RUNDIR ARM SIZE CKPT                 served read on j217 (serve217.sh; skipped when scores/ARM-CKPT.json exists)
CARD=$1; LIST=$2
cd /data/q27/w217
while read -r K REST; do
  set -- $REST
  case "$K" in
    T) ARM=$1; SIZE=$2; BUD=$3; shift 3
       LASTN=$(python3 -c "print('%.0e' % float('${BUD##*,}'))")
       [ -e runs/$ARM/ckpt-$LASTN/.done ] && continue
       code/v3/qad217/q217.sh $ARM $CARD $SIZE $BUD "$@" ;;
    W) while [ ! -e runs/$1/ckpt-5e+05/.done ]; do sleep 20; done ;;
    R) code/v3/qad217/serve217.sh $1 $2 $3 $4 $CARD >> logs/reader217-$CARD.log 2>&1
       echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $2-$4 rc $?" >> logs/reader217-$CARD.log ;;
  esac
done < $LIST
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') card $CARD queue $LIST done" >> q217.log
