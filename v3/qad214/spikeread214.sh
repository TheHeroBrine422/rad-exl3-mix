#!/bin/bash
# spikeread214.sh CARD ARM SIZE CKPT... -- the spike diagnosis: the served read of each checkpoint on slices/golden-spike214 (the six calib windows of 40Q's plan step 24,
# 3,412 rows) next to spk-ctl-<SIZE> (the unhealed release on the same windows). Patched containers are deleted after scoring.
CARD=$1 ARM=$2 SIZE=$3; shift 3
cd /data/q27/w214; source code/v3/qad214/env214.sh
for CK in "$@"; do
  N=spk-$ARM-$CK; P=$W/patched/$N.rad
  [ -s scores/$N.json ] && continue
  $PY $CODE/v3/qad214/swap214.py $W/rel/host-emb/qwen3.8-27b-exl3-v3-${SIZE}bpw.rad $W/lv/$SIZE $W/runs/$ARM/$CK $P > logs/swap-$N.log 2>&1 || exit 4
  $CODE/v3/qad214/score214.sh $N $P $CARD $W/slices/golden-spike214 && rm -f $P
done
