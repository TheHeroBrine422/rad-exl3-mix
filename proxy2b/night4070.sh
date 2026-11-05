#!/bin/bash
# night4070.sh -- the 4070 queue for the night of 2026-11-02/03 (no gpubox), one job after another (VRAM cap 7.2 GB).
# Each step is idempotent; a failed step is logged and the queue moves on. Stop: touch /data/yb/night.stop (after the
# current step). Order: what feeds the 27B run first.
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp L=/data/yb/night4070.log
log() { echo "$(date -u +%T) $*" | tee -a $L; }
while pgrep -f "^/bin/bash $P/[y]aqaFact1[12].sh" > /dev/null; do sleep 60; done
step() { [ -f /data/yb/night.stop ] && { log "stop file: $1 not started"; exit 0; }
  log "start $1"; s=$(date +%s); shift_name=$1; shift
  "$@" > /data/logs/night-$shift_name.log 2>&1 && log "done $shift_name ($(( ($(date +%s) - s) / 60 )) min)" \
    || log "FAILED $shift_name (rc $?, /data/logs/night-$shift_name.log)"; }
step w10-hout-attrib bash $X/w10-hout-attrib/run.sh
step w01-genmask     bash $X/w01-genmask/run.sh
step fact13          bash $P/yaqaFact13.sh
step w03-noGU        bash $X/w03-swiglu-diag/run.sh
step w02-fixedscale  bash $X/w02-fixedscale/run.sh all
step w07-genh       bash $X/w07-genh/run.sh
step w07-seqsib     bash $X/w07-seqsib/run.sh
step w07-denoise    bash $X/w07-denoise/run.sh
step w06-heal        bash $X/w06-heal/run.sh
log "NIGHT4070 DONE"
