#!/bin/bash
# q200.sh CARD JOBS (w187, #200, box2) -- one card's queue. JOBS lines: NAME|BUDGET|CKPT|FLAGS|DEP (DEP: a file that must
# exist first, or empty). Per line: training (run200.sh; retried on a busy card / low RAM), then the served read
# (eval196.py on card CARD: pack under conv + ram exclusive, scores under the card lock, the mix deleted after scoring).
# Restart = rerun the same queue: finished training and served reads are skipped. Queue log: $W/q200.log.
CARD=$1; JOBS=$2
cd /data/q27/w187 && source env.sh
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $W/q200.log; }
disk_wait() { while [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -lt 100 ]; do say "/data below 100 GB: waiting"; sleep 600; done; }
while IFS='|' read -r NAME BUDGET CK FLAGS DEP; do
  [ -z "$NAME" ] && continue
  case "$NAME" in \#*) continue;; esac
  if [ ! -e runs/$NAME/$CK/.done ]; then
    while [ -n "$DEP" ] && [ ! -e "$DEP" ]; do sleep 120; done
    while true; do
      ./code/v3/qad200/run200.sh "$NAME" "$CARD" "$BUDGET" $FLAGS; rc=$?
      case $rc in 75|76) sleep 120;; *) break;; esac
    done
    say "train $NAME card $CARD rc $rc"
    [ $rc -eq 0 ] || exit 1
  fi
  if [ ! -e served/$NAME/result.json ]; then
    disk_wait
    $PY $CODE/v3/qad200/eval196.py $NAME served --vlevel $VL --template $TPL --host $HOSTC --pack-home /data/q27/rad10-e157b/home \
      --card $CARD --ckpt $W/runs/$NAME/$CK --slice x181g=$G1 --slice fe38=$G2 > logs/served-$NAME.out 2>&1
    rc=$?; say "served $NAME card $CARD rc $rc"
    [ $rc -eq 0 ] || exit 1
  fi
done < "$JOBS"
say "queue card $CARD done ($JOBS)"
