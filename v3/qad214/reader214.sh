#!/bin/bash
# reader214.sh CARD LIST -- the served reads of #214's checkpoints on one card: LIST lines `ARM SIZE CKPT` (e.g. `25A 2.50 ckpt-5e+05`); for each, wait until
# runs/ARM/CKPT/.done exists (or runs/ARM/SKIP exists: skip the arm's remaining lines), then serve214.sh (swap + radiance on the judge slice). Restartable.
CARD=$1; LIST=$2
cd /data/q27/w214
while true; do
  left=0
  while read -r ARM SIZE CK; do
    [ -z "$ARM" ] && continue; case "$ARM" in \#*) continue;; esac
    [ -s scores/$ARM-$CK.json ] && continue
    [ -e runs/$ARM/SKIP ] && continue
    if [ -e runs/$ARM/$CK/.done ]; then
      code/v3/qad214/serve214.sh $ARM $SIZE $CK $CARD >> logs/reader214-$CARD.log 2>&1
      echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM-$CK rc $?" >> logs/reader214-$CARD.log
    else left=1; fi
  done < $LIST
  [ $left -eq 0 ] && break
  sleep 60
done
echo "reader $CARD done" >> logs/reader214-$CARD.log
