#!/bin/bash
# final218.sh (iceman /data/w218, #218 11-21 after the line searches) -- the judge reads: the tune pick of the redesigned step (r0p = the
# preconditioned direction at eta 0.2, the lowest tune KL) and o0 (a fresh encode of the true weight at the released scales: the
# encode-to-encode noise floor), then the scale-only comparison on round 0's documents (chunks 0-2, 304k rows) with its judge read.
set -u
cd /data/w218; G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 30;; *) return $rc;; esac; done; }
say "final: judge r0p, o0"; run logs/judge.log $PY $C/probe218.py eval runs/judge --set judge --variants r0p=runs/r0/encp/eta-0.2,o0=runs/r0/enco/eta-0 || exit 1
say "final: scale-only chunks 0-2"; run logs/scale.log $PY $C/probe218.py scale runs/scale --chunks 0-2 --compile-scales || exit 1
say "final done"
