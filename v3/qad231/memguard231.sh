#!/bin/bash
# memguard231.sh ARM FINAL_CKPT [FLOOR_GB] [ONLY_CARDS] (box2; setsid nohup) -- w231: memguard226.sh for any deep run (T2 or T4): while QROOT/runs/ARM trains,
# a MemAvailable below FLOOR_GB (default 10) stops it cleanly (touch runs/ARM/STOP: every rank sees it in the same step, rank 0 saves state.pt, all exit; the job
# is then marked failed "0(no-final-checkpoint)": rm failed/ + claims/ of its key to resume). ONLY_CARDS (e.g. "2 3", T2 runs): those cards take only R / U jobs
# meanwhile (QROOT/only-card<N>). Logs MemAvailable and each rank's step / host RSS every 5 min to QROOT/logs/memguard-ARM.log. Ends when
# runs/ARM/FINAL_CKPT/.done exists or after a STOP (removing its only-card files).
set -u
ARM=$1 FIN=$2 FLOOR=${3:-10} ONLY=${4:-}; Q=${QROOT:-/data/q27/queue}; R=$Q/runs/$ARM; L=$Q/logs/memguard-$ARM.log
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
for c in $ONLY; do echo "R U" > $Q/only-card$c; done
log "start: floor $FLOOR GB, final $FIN${ONLY:+, cards $ONLY take only R / U}"
n=0
while true; do
  AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ -e $R/$FIN/.done ]; then log "final checkpoint written: done"; break; fi
  if [ -e $R/STOP ]; then log "STOP file present (someone else stopped it): done"; break; fi
  if [ "$AV" -lt "$FLOOR" ]; then touch $R/STOP; log "MemAvailable $AV GB < $FLOOR: STOP touched (state.pt, resumable)"; break; fi
  if [ $((n % 5)) = 0 ]; then log "MemAvailable $AV GB, run: $(tail -n 1 $R/train.jsonl 2>/dev/null | grep -o '"step": [0-9]*\|"rss_gib": [0-9.]*\|"rank_wall": \[[0-9., ]*\]\|"rank_rss_gib": \[[0-9., ]*\]' | tr '\n' ' ')"; fi
  n=$((n + 1)); sleep 60
done
for c in $ONLY; do rm -f $Q/only-card$c; done; log "end"
