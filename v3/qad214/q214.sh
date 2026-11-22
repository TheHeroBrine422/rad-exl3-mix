#!/bin/bash
# q214.sh ARM CARD SIZE BUDGETS [qad200.py flags] -- one arm's training job, retried while the card is busy (75) or RAM is short (76); restartable (qad200 resumes
# from state.pt). Log /data/q27/w214/q214.log. The served reads are reader214.sh's (one reader a card, reads each finished checkpoint).
ARM=$1; CARD=$2; SIZE=$3; BUDGETS=$4; shift 4
cd /data/q27/w214
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> q214.log; }
while true; do
  code/v3/qad214/run214.sh $ARM $CARD $SIZE $BUDGETS "$@"; rc=$?
  case $rc in 75|76) sleep 120;; *) break;; esac
done
say "train $ARM card $CARD rc $rc"
