#!/bin/bash
# devq.sh -- short GPU dev jobs into idle-card gaps. Queue /data/devq/queue: "NAME [need:FILE] CMD..." (a job with
# need:FILE is skipped until FILE exists; CMD sees $CARD; the card
# is selected via ROCR_VISIBLE_DEVICES). Each pending job runs on the first card whose lock is free (non-blocking try,
# card 0 then 1, every 20 s), holding only that card's lock (+ ram.lock shared): never blocks the long chains, which
# just wait the job's few minutes. Done: /data/devq/NAME.rc, log /data/devq/NAME.log. Lines can be appended anytime.
: ${RXM_WORK:=/root}; D=/data/devq; mkdir -p $D; touch $D/queue
cd $RXM_WORK
while true; do
  job=$(while read -r n rest; do [ -z "$n" ] || [ "${n:0:1}" = "#" ] && continue; [ -f $D/$n.rc ] || [ -f $D/$n.run ] && continue
    set -- $rest; case $1 in need:*) [ -e "${1#need:}" ] || continue;; esac; echo "$n"; break; done < $D/queue)
  [ -z "$job" ] && { sleep 20; continue; }
  cmd=$(awk -v n=$job '$1 == n {$1 = ""; if ($2 ~ /^need:/) $2 = ""; print; exit}' $D/queue)
  for c in 0 1 1; do                   # non-blocking try on both, then queue up for card 1 (evals there are minutes)
    exec 9>$RXM_WORK/card$c.lock
    if flock -n 9 || { [ $c = 1 ] && [ -n "$TRIED" ] && flock -w 600 9; }; then
      touch $D/$job.run; echo "$(date -u +%H:%M) $job on card $c" >> $D/devq.log
      ( exec 7>$RXM_WORK/ram.lock; flock -s 7; export CARD=$c ROCR_VISIBLE_DEVICES=$c; bash -c "$cmd" ) > $D/$job.log 2>&1
      echo $? > $D/$job.rc; rm -f $D/$job.run; echo "$(date -u +%H:%M) $job rc=$(cat $D/$job.rc)" >> $D/devq.log
      flock -u 9; break
    fi
    exec 9>&-; [ $c = 1 ] && TRIED=1
  done
  TRIED=; sleep 20
done
