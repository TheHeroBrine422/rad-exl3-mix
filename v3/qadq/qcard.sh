#!/bin/bash
# qcard.sh MACHINE CARD  (w222) -- the dispatcher of the global QAD queue: one per card (box2 cards 0-3, iceman card 0). Takes the next ready job of QROOT/jobs.txt (qq.py), holds the
# card lock for the whole job (training, then the served reads), appends one results.tsv row per read, repeats. Idle (no ready job): polls jobs.txt every 60 s, so appending / reordering lines is
# all it takes to feed it. Restartable: a claim of a dead dispatcher is taken over (the trainer resumes from state.pt, finished checkpoints / reads are skipped by their outputs).
# Stop between jobs: `touch QROOT/stop` (all cards) or `QROOT/stop-card<N>`; remove the file and start the dispatcher again. Never kills a training.
set -u
MACHINE=$1 CARD=$2
HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE
case " $CARDS " in *" $CARD "*) ;; *) echo "card $CARD is not in $MACHINE's queue (cards: $CARDS)"; exit 2;; esac
mkdir -p $QROOT/dispatch $QROOT/logs; touch $QROOT/jobs.txt
eval "exec {inst}>$QROOT/dispatch/card$CARD.lock"; flock -n $inst || { echo "a dispatcher for card $CARD already runs"; exit 3; }
echo $$ > $QROOT/dispatch/card$CARD.pid
Q="$PY $HERE/qq.py"
stopped() { [ -e $QROOT/stop ] || [ -e $QROOT/stop-card$CARD ]; }
qsay "qcard $MACHINE card $CARD start (pid $$)"
while ! stopped; do
  $Q peek $QROOT || { sleep 60; continue; }
  eval "exec {lk}>$(CARDLOCK $CARD)"
  flock -w 60 $lk || { exec {lk}>&-; continue; }
  stopped && break
  LINE=$($Q claim $QROOT $CARD $$) || { exec {lk}>&-; sleep 5; continue; }
  eval "set -- $LINE"; KIND=$1 KEY=$2 ARM=$3 SIZE=$4 BUD=$5 READS=$6 CKD=$7 CTL=$8 ST=$9; shift 9
  qsay "card $CARD takes $KEY ($KIND $ARM $SIZE; $ST)"
  export QCARD_LOCKED=1
  fin() { $Q finish $QROOT $KEY $1 $2; qsay "card $CARD $KEY: $1 rc $2"; }
  ok=1; rc=0
  case $KIND in
  T|B)
    LASTN=$($PY -c "print('%.0e' % float('${BUD##*,}'))")
    if [ "$KIND" = B ] || [ ! -e $QROOT/runs/$ARM/ckpt-$LASTN/.done ]; then
      while true; do $HERE/qrun.sh $MACHINE $ARM $CARD $SIZE $BUD "$@"; rc=$?; case $rc in 75|76) sleep 120;; *) break;; esac; done
      qsay "card $CARD train $ARM rc $rc"
      if [ -e $QROOT/cancel/$KEY ]; then fin cancelled $rc; ok=0
      elif [ $rc -ne 0 ]; then fin fail $rc; ok=0
      elif [ "$KIND" = T ] && [ ! -e $QROOT/runs/$ARM/ckpt-$LASTN/.done ]; then fin fail "0(no-final-checkpoint)"; ok=0
      fi
    fi
    if [ $ok = 1 ] && [ "$KIND" = B ]; then touch $QROOT/runs/$ARM/bench.done; fin ok $rc; ok=0; fi
    if [ $ok = 1 ]; then
      for ck in ${READS//,/ }; do
        [ "$CTL" = - ] && CN=- || CN=$CTL-$ck
        $HERE/qread.sh $MACHINE $ARM-$ck $SIZE $QROOT/runs/$ARM/$ck $CARD $CN >> $QROOT/logs/qread-$CARD.log 2>&1; r=$?; qsay "card $CARD read $ARM-$ck rc $r"
        [ $r -eq 0 ] || { rc=$r; ok=2; }
      done
      [ $ok = 2 ] && fin fail $rc || fin ok 0
    fi;;
  R)
    [ "$CTL" = - ] && CN=- || CN=$CTL-$(basename $CKD)
    $HERE/qread.sh $MACHINE $ARM-$(basename $CKD) $SIZE $CKD $CARD $CN >> $QROOT/logs/qread-$CARD.log 2>&1; r=$?; qsay "card $CARD read $ARM-$(basename $CKD) rc $r"
    [ $r -eq 0 ] && fin ok 0 || fin fail $r;;
  esac
  unset QCARD_LOCKED
  $PY $HERE/qrow.py fill $QROOT --golden $J217 --ctldir $CTLDIR >> $QROOT/logs/qrow.log 2>&1
  exec {lk}>&-
done
qsay "qcard $MACHINE card $CARD stopped"
