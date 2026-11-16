#!/bin/bash
# score197.sh NAME MANIFEST GOLDEN -- #197 (w185): one whole-slice KLD read of a serve manifest (mkmanifest197.py) on the
# first free card of CARDS (default 0 2; box2 cards 1 and 3 are w182's QAD). Same engine flags as the live search's
# score_cmd (bf16 KV, --tp 1, --max-model-len 16384, --max-num-seqs 1, --host-headroom-mib 8192); the binary is the
# vload build of radiance e29c765b (rad-vload/install), which reads the manifest's final/ trunk lines. Holds the card
# lock for the run and ram.lock shared. Restartable: an existing NAME.json.rows is skipped. Outputs in scores/.
set -eu
# the vload build keeps every level shard it reads open (64 per level): raise the 1024 soft limit (hard 524288)
ulimit -n 65536
N=$1; M=$2; G=$3; W=/data/q27/w185
BIN=${BIN:-$W/rad-vload/install/bin/radiance}; HOMEX=${HOMEX:-$W/rad-vload/install/share/radiance}
mkdir -p $W/scores $W/logs
[ -s $W/scores/$N.json.rows ] && { echo "$N skipped (rows present)"; exit 0; }
cd $W/scores
while :; do for c in ${CARDS:-0 2}; do
  exec 9>/data/q27/locks/card$c.lock
  if flock -n 9; then
    echo "$(date -u +%FT%TZ) $N start card $c" >> $W/logs/score.status
    flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$HOMEX $BIN --model $M --tp 1 \
      --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 \
      --kld-ref $G --kld-out $N.json.part > $W/logs/$N.log 2>&1
    rc=$?
    if [ $rc = 0 ] && [ -f $N.json.part.rows ]; then mv $N.json.part.rows $N.json.rows; fi
    [ -f $N.json.part ] && mv $N.json.part $N.json
    echo "$(date -u +%FT%TZ) $N rc $rc card $c" >> $W/logs/score.status
    exit $rc
  fi
  exec 9>&-
done; sleep 20; done
