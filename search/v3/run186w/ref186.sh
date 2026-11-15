#!/bin/bash
# ref186.sh NAME MODEL [GOLDEN] -- score one reference container on golden-t186n (#186, w164) on the first free card
# of CARDS (default: 7, ds8; box2: CARDS="3 2"). The lead's ref.sh with this lane's output folder. One card at a
# time: the card lock is held for the whole run. Rows file = NAME.json.rows (refs186.py reads it).
cd /data/q27/w164/refs; N=$1; M=$2; G=${3:-/data/q27/w164/slices/golden-t186n}
[ -s $N.json.rows ] && exit 0
while :; do for c in ${CARDS:-7}; do
  exec 9>/data/q27/locks/card$c.lock; if flock -n 9; then
    flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=/data/q27/rad10-e157b/home /data/q27/rad10-e157b/install/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $N.json.part > $N.log 2>&1
    rc=$?; [ -f $N.json.part.rows ] && mv $N.json.part.rows $N.json.rows; echo "$N rc $rc card $c"; exit $rc; fi; exec 9>&-; done; sleep 20; done
