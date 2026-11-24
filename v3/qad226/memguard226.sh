#!/bin/bash
# memguard226.sh ARM [FLOOR_GB] (box2; setsid nohup) -- lead 11-24: while the deep run QROOT/runs/ARM trains, cards 2 / 3 take only R / U jobs (QROOT/only-card2,3)
# and, when MemAvailable drops below FLOOR_GB (default 10), the run is stopped cleanly (touch runs/ARM/STOP: the trainer saves state.pt and exits; the T2 job
# is then marked failed "0(no-final-checkpoint)": rm failed/ + claims/ of its key to resume). Logs MemAvailable and the run's host RSS every 5 min to
# /data/q27/w226/memguard-ARM.log. Ends (removing the only-card files) when the run's final checkpoint exists or it was stopped.
set -u
ARM=$1 FLOOR=${2:-10}; Q=/data/q27/queue; R=$Q/runs/$ARM; L=/data/q27/w226/memguard-$ARM.log
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
echo "R U" > $Q/only-card2; echo "R U" > $Q/only-card3
log "start: cards 2 / 3 take only R / U; floor $FLOOR GB"
n=0
while true; do
  AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ -e $R/train.jsonl ] && ls $R/ckpt-1e+07/.done > /dev/null 2>&1; then log "final checkpoint written: done"; break; fi
  if [ "$AV" -lt "$FLOOR" ]; then touch $R/STOP; log "MemAvailable $AV GB < $FLOOR: STOP touched (state.pt, resumable)"; break; fi
  if [ $((n % 5)) = 0 ]; then log "MemAvailable $AV GB, run: $(tail -n 1 $R/train.jsonl 2>/dev/null | grep -o '"step": [0-9]*\|"rss_gib": [0-9.]*' | tr '\n' ' ')"; fi
  n=$((n + 1)); sleep 60
done
rm -f $Q/only-card2 $Q/only-card3; log "only-card files removed"
