#!/bin/bash
# gold96.sh -- w96 Fisher pass over the calib golden on gpubox's CPU (the golden is 42 GB and stays there): row groups
# (mkgroups96 calib) then fisher96 with 4 processes, nice 19, /root/ram.lock shared. Idempotent. Out /data/w96/.
set -u
W=/data/w96 C=/data/w96/code PY=/data/venv-exl3/bin/python G=/root/challenge/golden/calib
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/gold96.log; }
cd $W
[ -s $W/groups-calib.npy ] || $PY -I $C/v3/yaqa/w96/mkgroups96.py calib $G $C/eval/pools/pools.json $C/eval/pools/pools-v3.json \
  $C/v3/yaqa/w96/split96.json $W/groups-calib.npy >> $W/gold96.log 2>&1 || { log "groups FAIL"; exit 1; }
if [ ! -s $W/fisher-calib.npz ]; then
  log "fisher: waiting for ram.lock (shared)"
  ( exec 7>/root/ram.lock; flock -s 7; log "fisher start"
    OMP_NUM_THREADS=1 nice -n 19 ionice -c3 $PY -I $C/v3/yaqa/w96/fisher96.py $G $W/groups-calib.npy $W/fisher-calib.npz 4 ) >> $W/gold96.log 2>&1
  [ -s $W/fisher-calib.npz ] && log "fisher OK" || { log "fisher FAIL"; exit 1; }
fi
log GOLD96 DONE
