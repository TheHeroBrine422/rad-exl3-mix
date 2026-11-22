#!/bin/bash
# q217.sh ARM CARD SIZE BUDGETS [qad200.py flags] -- run217.sh retried while the card is busy (75) or RAM is short (76). Log /data/q27/w217/q217.log.
ARM=$1; CARD=$2; SIZE=$3; BUDGETS=$4; shift 4
cd /data/q27/w217
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> q217.log; }
while true; do
  code/v3/qad217/run217.sh $ARM $CARD $SIZE $BUDGETS "$@"; rc=$?
  case $rc in 75|76) sleep 120;; *) break;; esac
done
say "train $ARM card $CARD rc $rc"
