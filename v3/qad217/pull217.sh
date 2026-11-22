#!/bin/bash
# pull217.sh CARD JOBS -- a card's dispatcher: takes the next job of the shared list JOBS (cursor in JOBS.cursor, under flock), runs it, repeats until none is left.
# Job line: `T ARM SIZE BUDGETS READS SEED flags...` -- train ARM (q217.sh; skipped when its last budget's checkpoint is done), then the served reads READS (comma list of
# ckpt names, e.g. ckpt-2e+05,ckpt-5e+05). SEED = a w214 arm whose state.pt / items.json / fixed-eval-docs.json are copied into runs/ARM first (an extension of that
# run, resumed from its saved state), or `-`. Restartable: finished trainings / reads are skipped by their outputs (a restart with the cursor file removed re-walks the list).
CARD=$1; JOBS=$2; CUR=$JOBS.cursor
cd /data/q27/w217
while true; do
  exec {lk}> $JOBS.lock; flock $lk
  n=$(cat $CUR 2>/dev/null || echo 0); LINE=$(grep -v '^#' $JOBS | grep -v '^$' | sed -n "$((n+1))p")
  [ -n "$LINE" ] && echo $((n+1)) > $CUR
  exec {lk}>&-
  [ -z "$LINE" ] && break
  set -- $LINE; K=$1; ARM=$2; SIZE=$3; BUD=$4; READS=$5; SEED=$6; shift 6
  [ "$K" = T ] || continue
  if [ "$SEED" != - ] && [ ! -e runs/$ARM/state.pt ]; then
    mkdir -p runs/$ARM; for f in state.pt items.json fixed-eval-docs.json; do cp /data/q27/w214/runs/$SEED/$f runs/$ARM/$f.part && mv runs/$ARM/$f.part runs/$ARM/$f; done
  fi
  LASTN=$(python3 -c "print('%.0e' % float('${BUD##*,}'))")
  [ -e runs/$ARM/ckpt-$LASTN/.done ] || code/v3/qad217/q217.sh $ARM $CARD $SIZE $BUD "$@"
  for ck in ${READS//,/ }; do
    code/v3/qad217/serve217.sh /data/q27/w217 $ARM $SIZE $ck $CARD >> logs/reader217-$CARD.log 2>&1
    echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM-$ck rc $?" >> logs/reader217-$CARD.log
  done
done
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') card $CARD pull queue $JOBS done" >> q217.log
