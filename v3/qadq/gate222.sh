#!/bin/bash
# gate222.sh MACHINE CODEDIR ARM BUDGETS [qad220 flags] (w222) -- one gate run of a candidate trainer (CODEDIR/v3/qad220/qad220.py, a copy of the queue's code dir with the candidate patch) on whichever card of the machine's queue
# is between jobs: a blocking flock (5 s) on each card lock in turn, so it waits for a free card and never preempts a running job (the dispatchers compete for the same locks). Runs qrun.sh with CODE_PY set, SIZE 2.50.
# Result: QROOT/gates/ARM.rc and the run in QROOT/runs/ARM. Start with setsid nohup.
set -u
MACHINE=$1 CODEDIR=$2 ARM=$3 BUD=$4; shift 4
HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE; mkdir -p $QROOT/gates
while true; do
  for c in $CARDS; do
    eval "exec {fd}>$(CARDLOCK $c)"
    if flock -w 5 $fd; then
      qsay "gate $ARM on card $c (trainer $CODEDIR)"
      QCARD_LOCKED=1 CODE_PY=$CODEDIR/v3/qad220/qad220.py $HERE0/qrun.sh $MACHINE $ARM $c 2.50 $BUD "$@"; rc=$?
      case $rc in 76) exec {fd}>&-; sleep 120; continue 2;; esac
      echo $rc > $QROOT/gates/$ARM.rc; qsay "gate $ARM rc $rc"; exit $rc
    fi
    exec {fd}>&-
  done
done
