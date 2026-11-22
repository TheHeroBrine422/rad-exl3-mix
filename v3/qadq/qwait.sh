#!/bin/bash
# qwait.sh MACHINE CARD PID [PID...] (w222) -- hand a card over: wait until the old dispatchers (PIDs) have exited, then run qcard.sh MACHINE CARD. Start with setsid nohup.
MACHINE=$1 CARD=$2; shift 2
HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE
qsay "qwait card $CARD: waiting for old dispatchers $*"
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
qsay "qwait card $CARD: old dispatchers gone, starting qcard"
exec $HERE0/qcard.sh $MACHINE $CARD
