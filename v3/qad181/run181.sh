#!/bin/bash
# run181.sh NAME VLEVEL HEADDIR CARDS [qad181.py args] (w154, #181) -- one QAD run on this host's CARDS (comma list,
# HIP indices), data-parallel (torchrun, one whole student a card), every card lock held for the whole run (flock -n:
# never queues on a held card; exits 75 if one is busy). Restartable: rerun the same line (OUT/state.pt resumes; the
# plan does not depend on the card count, so a run may continue on another host / card set after copying OUT).
set -u
NAME=$1 VL=$2 HD=$3 CARDS=$4; shift 4
cd /data/q27/w154 && source env.sh
OUT=$W/runs/$NAME; mkdir -p $OUT
for c in ${CARDS//,/ }; do
  eval "exec {fd}>/data/q27/locks/card$c.lock"; flock -n $fd || { echo "card $c busy"; exit 75; }
done
N=$(echo ${CARDS//,/ } | wc -w)
ls -d $W/gold/*/ | sed 's:/$::' > $W/chunks181.txt
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME start on $(hostname -s) cards $CARDS: $*" >> $W/run181.log
ROCR_VISIBLE_DEVICES=$CARDS nice -n 5 $PY -m torch.distributed.run --standalone --nproc_per_node $N $CODE/v3/qad181/qad181.py $OUT \
  --model $MODEL --level $VL --chunks @$W/chunks181.txt --calib $CALIB --pools $POOLS --head $HD --head-perm $PERM "$@" >> $OUT/stdout.log 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME rc $rc" >> $W/run181.log
exit $rc
