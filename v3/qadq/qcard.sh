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
QQ="$PY $HERE/qq.py"
stopped() { [ -e $QROOT/stop ] || [ -e $QROOT/stop-card$CARD ]; }
qsay "qcard $MACHINE card $CARD start (pid $$)"
while ! stopped; do
  $QQ peek $QROOT $CARD || { sleep 60; continue; }
  eval "exec {lk}>$(CARDLOCK $CARD)"
  flock -w 60 $lk || { exec {lk}>&-; continue; }
  stopped && break
  LINE=$($QQ claim $QROOT $CARD $$ --caps M,U,T2) || { exec {lk}>&-; sleep 5; continue; }
  eval "set -- $LINE"; KIND=$1 KEY=$2 ARM=$3 SIZE=$4 BUD=$5 READS=$6 CKD=$7 CTL=$8 ST=$9; shift 9
  qsay "card $CARD takes $KEY ($KIND $ARM $SIZE; $ST)"
  export QCARD_LOCKED=1
  fin() { $QQ finish $QROOT $KEY $1 $2; qsay "card $CARD $KEY: $1 rc $2"; }
  ok=1; rc=0
  case $KIND in
  M)   # two-card job (the teacher tap pass): own card's lock held, the partner's taken too (45 min at most, else the claim is released), the trainer gets both cards
    PARTNER=${1#partner=}; shift
    eval "exec {lk2}>$(CARDLOCK $PARTNER)"; got=0
    for try in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45; do flock -w 60 $lk2 && { got=1; break; }; stopped && break; done
    if [ $got = 0 ]; then qsay "card $CARD $KEY: partner card $PARTNER not free in 45 min, claim released"; $QQ unclaim $QROOT $KEY; exec {lk2}>&-; unset QCARD_LOCKED; exec {lk}>&-; sleep 30; continue; fi
    AVG=$(df -BG --output=avail $CKD 2>/dev/null | tail -1 | tr -dc 0-9); mkdir -p $CKD; AVG=$(df -BG --output=avail $CKD | tail -1 | tr -dc 0-9)
    if [ "$AVG" -lt "${QMINFREE_GB:-200}" ]; then qsay "card $CARD $KEY: only ${AVG} GB free under $CKD (< ${QMINFREE_GB:-200}), not started"; fin fail "disk"; exec {lk2}>&-; unset QCARD_LOCKED; exec {lk}>&-; continue; fi
    while true; do $HERE/qrun.sh $MACHINE $ARM "$CARD,$PARTNER" $SIZE $BUD --make-taps $CKD "$@" {inst}>&-; rc=$?; case $rc in 75|76) sleep 120;; *) break;; esac; done
    qsay "card $CARD+$PARTNER $ARM taps rc $rc"
    if [ -e $QROOT/cancel/$KEY ]; then fin cancelled $rc; elif [ $rc -eq 0 ] && [ -e $CKD/.done ]; then fin ok 0; else fin fail $rc; fi
    exec {lk2}>&-;;
  T|B|T2)
    LASTN=$($PY -c "print('%.0e' % float('${BUD##*,}'))"); TC=$CARD
    if [ "$KIND" = T2 ]; then PARTNER=${1#partner=}; shift; fi
    if [ "$KIND" = B ] || [ ! -e $QROOT/runs/$ARM/ckpt-$LASTN/.done ]; then
      if [ "$KIND" = T2 ]; then   # w226: two-card training: the partner's lock too (45 min at most, else the claim is released), the trainer sees both cards
        eval "exec {lk2}>$(CARDLOCK $PARTNER)"; got=0
        for try in $(seq 45); do flock -w 60 $lk2 && { got=1; break; }; stopped && break; done
        if [ $got = 0 ]; then qsay "card $CARD $KEY: partner card $PARTNER not free in 45 min, claim released"; $QQ unclaim $QROOT $KEY; exec {lk2}>&-; unset QCARD_LOCKED; exec {lk}>&-; sleep 30; continue; fi
        TC=$CARD,$PARTNER
      fi
      tries=0
      while true; do $HERE/qrun.sh $MACHINE $ARM $TC $SIZE $BUD "$@" {inst}>&-; rc=$?
        case $rc in 75|76) sleep 120; continue;; esac
        # w226: a two-card run that crashed after a saved state resumes (up to 3 times): g226-gate hit a one-off illegal memory access at step 13, the resumed run passed it
        if [ "$KIND" = T2 ] && [ $rc -ne 0 ] && [ $tries -lt 3 ] && [ -e $QROOT/runs/$ARM/state.pt ] && [ ! -e $QROOT/cancel/$KEY ] && [ ! -e $QROOT/runs/$ARM/STOP ]; then
          tries=$((tries + 1)); qsay "card $TC $ARM rc $rc: resuming from state.pt (retry $tries of 3)"; sleep 30; continue; fi
        break; done
      qsay "card $TC train $ARM rc $rc"
      if [ "$KIND" = T2 ]; then $QQ freepartner $QROOT $KEY; exec {lk2}>&-; qsay "card $CARD $KEY: partner card $PARTNER released"; fi
      if [ -e $QROOT/cancel/$KEY ]; then fin cancelled $rc; ok=0
      elif [ $rc -ne 0 ]; then fin fail $rc; ok=0
      elif [ "$KIND" = T ] && [ ! -e $QROOT/runs/$ARM/ckpt-$LASTN/.done ]; then fin fail "0(no-final-checkpoint)"; ok=0
      fi
    fi
    if [ $ok = 1 ] && [ "$KIND" = B ]; then touch $QROOT/runs/$ARM/bench.done; fin ok $rc; ok=0; fi
    if [ $ok = 1 ]; then
      [ "$READS" = - ] && READS=
      for ck in ${READS//,/ }; do
        [ "$CTL" = - ] && CN=- || CN=$CTL-$ck
        $HERE/qread.sh $MACHINE $ARM-$ck $SIZE $QROOT/runs/$ARM/$ck $CARD $CN >> $QROOT/logs/qread-$CARD.log 2>&1 {inst}>&-; r=$?; qsay "card $CARD read $ARM-$ck rc $r"
        [ $r -eq 0 ] || { rc=$r; ok=2; }
      done
      [ $ok = 2 ] && fin fail $rc || fin ok 0
    fi;;
  U)   # a released container read as it is (fragility test: does the unhealed model derail on the blow-up documents?)
    RADF=$QROOT/rel/host-emb/qwen3.8-27b-exl3-v3-${SIZE}bpw.rad
    if [ ! -e $RADF.ok ]; then mkdir -p $QROOT/rel; hf download Hero422/qwen3.8-27b-exl3-v3-rad host-emb/qwen3.8-27b-exl3-v3-${SIZE}bpw.rad --local-dir $QROOT/rel >> $QROOT/logs/hf-dl.log 2>&1 && touch $RADF.ok; fi
    if [ -e $RADF.ok ]; then $HERE/qscore.sh $MACHINE $ARM $RADF $CARD >> $QROOT/logs/qread-$CARD.log 2>&1 {inst}>&-; r=$?; else r=5; fi
    qsay "card $CARD read $ARM (released $SIZE) rc $r"; [ $r -eq 0 ] && fin ok 0 || fin fail $r;;
  R)
    [ "$CTL" = - ] && CN=- || CN=$CTL-$(basename $CKD)
    $HERE/qread.sh $MACHINE $ARM-$(basename $CKD) $SIZE $CKD $CARD $CN >> $QROOT/logs/qread-$CARD.log 2>&1 {inst}>&-; r=$?; qsay "card $CARD read $ARM-$(basename $CKD) rc $r"
    [ $r -eq 0 ] && fin ok 0 || fin fail $r;;
  esac
  unset QCARD_LOCKED
  $HERE/qimport.sh $MACHINE >> $QROOT/logs/qimport.log 2>&1
  $PY $HERE/qrow.py fill $QROOT --golden $J217 --ctldir $CTLDIR >> $QROOT/logs/qrow.log 2>&1
  exec {lk}>&-
done
qsay "qcard $MACHINE card $CARD stopped"
