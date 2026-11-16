#!/bin/bash
# run196.sh NAME CARD [qad196.py args] (w182, #196, box2) -- one qad196.py job on box2 card CARD (one R9700; the card lock
# is held for the whole job: flock -n, exits 75 when the card is busy). The student is the 2.5 incumbent's served form by
# default (env.sh: VL = #191's vl/inc2.5m); LV=/path overrides the virtual level (the 4.0 student). Restartable: rerunning
# the same line resumes from OUT/state.pt (finished arms are skipped by their ckpt .done markers).
set -u
NAME=$1 CARD=$2; shift 2
cd /data/q27/w182 && source env.sh
LV=${LV:-$VL}
# RULES196: the 4.0 arms run at lr 1e-4 (2.5: the default 1e-3). Set here from the level, so no queue line can forget it (11-15 fix)
LRARG=""; case "$LV" in *inc4.0m*) LRARG="--lr 1e-4";; esac
OUT=$W/runs/$NAME; mkdir -p $OUT
eval "exec {fd}>/data/q27/locks/card$CARD.lock"; flock -n $fd || { echo "card $CARD busy"; exit 75; }
ls -d $G154/gold/*/ | sed 's:/$::' > $W/chunks196.txt
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME start on card $CARD (level $LV, head ${NOHEAD:+bf16}, emb ${EMBX:-$EMB}): $*" >> $W/run196.log
rm -f $OUT/STOP
# glibc fixed mmap threshold (as #191): the per-segment CPU copies of the carried states are returned to the OS
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
# NOHEAD=1: the torch student keeps the checkpoint's bf16 lm_head (the 4.0 student: its u3.75 head has no torch level dir yet,
# see the report); EMBX overrides the token table (the 4.0 student's decoded K3.375 table)
HEADARGS="--head $HD --head-perm $PERM"; [ "${NOHEAD:-0}" = 1 ] && HEADARGS=""
EMBP=${EMBX:-$EMB}
ROCR_VISIBLE_DEVICES=$CARD nice -n 5 $PY $CODE/v3/qad196/qad196.py $OUT --model $MODEL --level $LV --chunks @$W/chunks196.txt \
  --calib $CALIB --pools $POOLS --genmask $G154/genmask $HEADARGS --emb $EMBP --ab $AB \
  --budgets 1e6 --fixed-eval 30000 --fe-every 1000 --monitor 0 --build-lock $W/build.lock $LRARG "$@" >> $OUT/stdout.log 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $NAME rc $rc" >> $W/run196.log
exit $rc
