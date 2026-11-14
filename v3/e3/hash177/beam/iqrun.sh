#!/bin/bash
# iqrun.sh "CARDS" QUEUE -- #177 (w139) ds8: iqh.py units from QUEUE lines "NAME K CODE [ROWS]" (IQR-V4-form head donors via
# libquant on the card), each on the first free card of CARDS (card lock, polled every 20 s) + ram.lock shared; a unit
# whose wq / time file exists is skipped by iqh.py itself; claimed by mkdir. Stop: touch /data/q27/w139h/STOPIQ. Log
# /data/q27/w139h/iq/run.log. (w139)
source /data/q27/tools/env-ds8.sh; W=/data/q27/w139h; LK=/data/q27/locks; cd $W
log() { echo "$(date -u +%T) $*" >> $W/iq/run.log; }
while read -r name k code rows; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue; [ -f $W/STOPIQ ] && break
  u=$W/iq/$name-K$k${rows:+-r$rows}; mkdir -p $W/iq; mkdir $u.claim 2>/dev/null || continue
  while :; do for c in $1; do exec 5>$LK/card$c.lock; flock -n 5 && break 2; exec 5>&-; done; sleep 20; done
  log "$name K$k $code ${rows:-all} card $c"
  ( exec 7>$LK/ram.lock; flock -s 7
    ABQ_DEVICES=0 ROCR_VISIBLE_DEVICES=$c OMP_NUM_THREADS=8 nice -n 5 $PY -I iqh.py $name $k $code ${rows:+--rows $rows} >> $W/iq/$name-K$k.log 2>&1
    log "$name K$k rc $?" )
  exec 5>&-; rmdir $u.claim
done < $2
log "worker ($1) done"
