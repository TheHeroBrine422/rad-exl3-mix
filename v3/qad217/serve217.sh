#!/bin/bash
# serve217.sh RUNDIR ARM SIZE CKPTNAME CARD -- the served read of one checkpoint on j217: swap214.py (the released container + the checkpoint's scales),
# score217.sh, then the patched container is deleted. RUNDIR = the directory that holds runs/ARM (w217 for the eps arms, w214 for the no-eps controls).
# Score name ARM-CKPT in w217/scores. Skips what exists. Needs /data >= 100 GB free.
set -u
RUNDIR=$1 ARM=$2 SIZE=$3 CK=$4 CARD=$5
cd /data/q27/w217 && source code/v3/qad217/env217.sh
N=$ARM-$CK; [ -s $W/scores/$N.json ] && { echo "$N done"; exit 0; }
[ -e $RUNDIR/runs/$ARM/$CK/.done ] || { echo "$N: checkpoint not finished"; exit 3; }
while [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -lt 100 ]; do sleep 120; done
RAD=$W14/rel/host-emb/qwen3.8-27b-exl3-v3-${SIZE}bpw.rad; P=$W/patched/$N.rad; mkdir -p $W/patched $W/logs
[ -e $P ] || $PY $CODE/v3/qad214/swap214.py $RAD $W14/lv/$SIZE $RUNDIR/runs/$ARM/$CK $P --verify > $W/logs/swap-$N.log 2>&1 || { echo "swap failed"; exit 4; }
$CODE/v3/qad217/score217.sh $N $P $CARD; rc=$?
[ $rc -eq 0 ] && rm -f $P
exit $rc
