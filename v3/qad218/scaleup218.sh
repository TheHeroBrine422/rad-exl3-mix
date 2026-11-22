#!/bin/bash
# scaleup218.sh DISPATCHER_PID (iceman, #218 scaling follow-up 11-22) -- between two queue jobs on card 0: waits for the queue dispatcher
# (stopped with `touch /data/queue/stop-card0`, so it exits after its running job; never killed) to exit, then on card 0 (gpu218.sh: card lock,
# RAM floor): the 10th 100k-row training chunk's gradient at the released codes (train chunk 9 -> runs/c9) and the check gradient on all 34 tune
# docs (runs/gtune34), then hands the card back to the queue (qwait.sh: removes stop-card0, starts qcard.sh).
set -u
P=$1; cd /data/w218; G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 60;; *) return $rc;; esac; done; }
while kill -0 $P 2>/dev/null; do sleep 30; done
say "scaleup: card 0 free (dispatcher $P gone)"
[ -f runs/c9/grad.json ] || run logs/scaleup.log $PY $C/probe218.py grad runs/c9 --chunk 9
[ -f runs/gtune34/grad.json ] || run logs/scaleup.log $PY $C/probe218.py grad runs/gtune34 --grad-set tune
say "scaleup: card 0 back to the queue"
exec setsid /data/queue/code/v3/qadq/qwait.sh iceman 0 < /dev/null >> /data/queue/logs/qcard-0.out 2>&1
