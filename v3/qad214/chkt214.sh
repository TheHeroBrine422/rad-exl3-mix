#!/bin/bash
# chkt214.sh SIZE CARD [CKPT] -- the loader check on the torch path: the unhealed (or CKPT's) student of the released SIZE pick, A8 on, scored on the judge slice
# (tscore191.py); compare with chkstu214.py against the served rows. Output chk/student-SIZE[-ckpt].npz.
SIZE=$1 CARD=$2 CK=${3:-}
cd /data/q27/w214; source code/v3/qad214/env214.sh
exec {fd}> $Q/locks/card$CARD.lock; flock $fd
exec {rf}> $Q/locks/ram.lock; flock -s $rf
mkdir -p chk; N=student-$SIZE${CK:+-$(basename $(dirname $CK))-$(basename $CK)}
ROCR_VISIBLE_DEVICES=$CARD $PY code/v3/qad191/tscore191.py chk/$N.npz --golden slices/golden-j214 --model $MODEL --level lv/$SIZE/trunk \
  --head lv/$SIZE/head --head-perm lv/$SIZE/R16.npy --ab lv/$SIZE/ab.safetensors --build-lock $BUILD_LOCK ${CK:+--ckpt $CK} > logs/chkt-$N.log 2>&1
echo rc=$? >> logs/chkt-$N.log
