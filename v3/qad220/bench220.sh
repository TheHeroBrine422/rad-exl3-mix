#!/bin/bash
# bench220.sh LABEL CARD SIZE [qad220 flags] -- a short run for the speed table / equivalence checks: w214's data and order, 2.5e5-row budget, no fixed eval, no checkpoint,
# `--max-steps ${STEPS:-5}` (state.pt at the end). Env CODE_PY overrides the trainer (qad200.py for the default-equivalence check). Writes runs/LABEL/bench.done on rc 0.
set -u
LABEL=$1 CARD=$2 SIZE=$3; shift 3
cd /data/q27/w220 && source code/v3/qad220/env220.sh
case $LABEL in d200*) export CODE_PY=$CODE/v3/qad200/qad200.py;; esac      # the default-equivalence reference: qad200.py (w217's trainer)
code/v3/qad220/run220.sh $LABEL $CARD $SIZE 2.5e5 --max-steps ${STEPS:-5} "$@"; rc=$?
[ $rc -eq 0 ] && touch $W/runs/$LABEL/bench.done
exit $rc
