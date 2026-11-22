#!/bin/bash
# card221.sh CARD JOBS -- w221's per-card dispatcher (cards 2 and 3 only): takes the next job of the shared list JOBS (cursor JOBS.cursor under flock), trains it (run221.sh,
# retried while the card lock is busy / RAM short), then the served reads on j217 (serve221.sh), repeats until the list is empty. Restartable: finished trainings / reads are skipped
# by their outputs (remove the cursor file to re-walk the list). Job line:  T ARM SIZE BUDGETS READS flags...   (READS = comma list of ckpt names)
CARD=$1; JOBS=$2; CUR=$JOBS.cursor
case $CARD in 2|3) ;; *) echo "w221 uses cards 2 and 3 only"; exit 2;; esac
cd /data/q27/w221
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> q221.log; }
while true; do
  exec {lk}> $JOBS.lock; flock $lk
  n=$(cat $CUR 2>/dev/null || echo 0); LINE=$(grep -v '^#' $JOBS | grep -v '^$' | sed -n "$((n+1))p")
  [ -n "$LINE" ] && echo $((n+1)) > $CUR
  exec {lk}>&-
  [ -z "$LINE" ] && break
  set -- $LINE; K=$1; shift
  [ "$K" = T ] || continue
  ARM=$1; SIZE=$2; BUD=$3; READS=$4; shift 4
  LASTN=$(python3 -c "print('%.0e' % float('${BUD##*,}'))")
  if [ ! -e runs/$ARM/ckpt-$LASTN/.done ]; then
    while true; do code/v3/qad221/run221.sh $ARM $CARD $SIZE $BUD "$@"; rc=$?; case $rc in 75|76) sleep 120;; *) break;; esac; done
    say "train $ARM card $CARD rc $rc"
  fi
  for ck in ${READS//,/ }; do
    code/v3/qad221/serve221.sh $ARM $SIZE $ck $CARD >> logs/reader221-$CARD.log 2>&1; say "read $ARM-$ck card $CARD rc $?"
  done
done
say "card $CARD dispatcher done"
