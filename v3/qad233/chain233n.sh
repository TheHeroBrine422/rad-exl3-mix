#!/bin/bash
# chain233n.sh (box2; setsid nohup) -- w233: (1) wait until the box has >= 95 GB MemAvailable (the 1-card screens with their compile pools ended), then
# the crash test resume (chain233m: crash300 from its state.pt to 310 on cards 2,3); (2) then the 4-card layout window (chain233l).
set -u
W=/data/q27/w233; C=$W/code10/v3/qad233
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233n waiting for MemAvailable >= 95 GB"
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 95 ]; do sleep 30; done
rm -f $W/runs/crash300/STOP
bash $C/chain233m.sh
bash $C/chain233l.sh
