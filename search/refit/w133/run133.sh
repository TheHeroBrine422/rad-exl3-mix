#!/bin/bash
# #173 (w133) colo runner: one unit per line of UNITS (ev133.py arguments after the command word), P parallel jobs,
# single-threaded, nice 19. Finished fits / preds are skipped by ev133.py itself (fit files) or here (pred files).
#   run133.sh UNITS P LOG
set -u
cd "$(dirname "$0")"
export MENU=/data/work/w13/w133/data/menu129h.json OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
UNITS=$1; P=${2:-6}; LOG=${3:-/data/work/w13/w133/out/run.log}
echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') start $UNITS P=$P" >> $LOG
grep -v '^#' "$UNITS" | xargs -P "$P" -I{} bash -c 'ulimit -v 4000000; t0=$(date +%s); nice -n 19 python3 -I ev133.py {} >> '"$LOG"'.units 2>&1; echo "$(TZ=America/Chicago date "+%m-%d %H:%M:%S CST") rc $? $(( $(date +%s) - t0 ))s {}" >> '"$LOG"
echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') done $UNITS" >> $LOG
