#!/bin/bash
# reader217.sh CARD LIST -- the served reads on one card: LIST lines `RUNDIR ARM SIZE CKPT` (RUNDIR /data/q27/w217 or /data/q27/w214; CKPT e.g. ckpt-5e+05). For each,
# wait until RUNDIR/runs/ARM/CKPT/.done exists, then serve217.sh. Restartable. The card lock is taken inside score217.sh (training holds it while it runs).
CARD=$1; LIST=$2
cd /data/q27/w217
while true; do
  left=0
  while read -r RD ARM SIZE CK; do
    [ -z "$RD" ] && continue; case "$RD" in \#*) continue;; esac
    [ -s scores/$ARM-$CK.json ] && continue
    if [ -e $RD/runs/$ARM/$CK/.done ]; then
      code/v3/qad217/serve217.sh $RD $ARM $SIZE $CK $CARD >> logs/reader217-$CARD.log 2>&1
      echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM-$CK rc $?" >> logs/reader217-$CARD.log
    else left=1; fi
  done < $LIST
  [ $left -eq 0 ] && break
  sleep 60
done
echo "reader $CARD done" >> logs/reader217-$CARD.log
