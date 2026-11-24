#!/bin/bash
# qread.sh MACHINE NAME SIZE CKPTDIR CARD [CTLNAME] -- the served read of one checkpoint on j217 (w221's serve221.sh): swap214.py (the released container + the checkpoint's scales), qscore.sh,
# the patched container deleted, then one row appended to QROOT/results.tsv (qrow.py; CTLNAME = the control read's score name for the paired column). Skips what exists.
# NAME = score name. SIZE = 2.50 / 3.00 / 4.00. Needs the disk (>= 100 GB free).
set -u
MACHINE=$1 N=$2 SIZE=$3 CKD=$4 CARD=$5 CTLN=${6:--}
HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE
[ -e $CKD/.done ] || { echo "$N: checkpoint $CKD not finished"; exit 3; }
mkdir -p $QROOT/patched $QROOT/logs
if [ ! -s $QROOT/scores/$N.json ]; then
  while [ "$(df -BG --output=avail $QROOT | tail -1 | tr -dc 0-9)" -lt 100 ]; do sleep 120; done
  RAD=$IN/rel/host-emb/qwen3.8-27b-exl3-v3-${SIZE}bpw.rad; P=$QROOT/patched/$N.rad
  [ -e $P ] || $PY $HERE/swap222.py $RAD $IN/lv/$SIZE $CKD $P --verify > $QROOT/logs/swap-$N.log 2>&1 || { echo "swap failed"; exit 4; }
  $HERE/qscore.sh $MACHINE $N $P $CARD; rc=$?
  [ $rc -eq 0 ] || exit $rc
  rm -f $P
fi
case $N in JL-*|JC-*) echo "$N: JL / JC judge read (no results.tsv row; lead readout: qad226/jl226.py [--golden DIR])"; exit 0;; esac
ARM=${N%-ckpt-*}; CK=ckpt-${N##*-ckpt-}
$PY $HERE/qrow.py add $QROOT --machine $MACHINE --card $CARD --name $N --arm $ARM --ckpt $CK --size $SIZE --golden $J217 --ctldir $CTLDIR --ctl $CTLN --run $(dirname $CKD)
