#!/bin/bash
# serve214.sh ARM SIZE CKPTNAME CARD -- the served read of one checkpoint: swap214.py (the released container + the checkpoint's scales, f16 planes), then
# score214.sh on the card (judge slice j214), then the patched container is deleted. Skips what exists (scores/ARM-CKPT.json). Needs /data >= 100 GB free.
set -u
ARM=$1 SIZE=$2 CK=$3 CARD=$4
cd /data/q27/w214 && source code/v3/qad214/env214.sh
N=$ARM-$CK; [ -s $W/scores/$N.json ] && { echo "$N done"; exit 0; }
[ -e $W/runs/$ARM/$CK/.done ] || { echo "$N: checkpoint not finished"; exit 3; }
while [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -lt 100 ]; do sleep 120; done
RAD=$W/rel/host-emb/qwen3.8-27b-exl3-v3-${SIZE}bpw.rad; P=$W/patched/$N.rad; mkdir -p $W/patched
[ -e $P ] || $PY $CODE/v3/qad214/swap214.py $RAD $W/lv/$SIZE $W/runs/$ARM/$CK $P --verify > $W/logs/swap-$N.log 2>&1 || { echo "swap failed"; exit 4; }
$CODE/v3/qad214/score214.sh $N $P $CARD; rc=$?
[ $rc -eq 0 ] && rm -f $P
exit $rc
