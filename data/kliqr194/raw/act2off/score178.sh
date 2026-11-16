#!/bin/bash
# score178.sh NAME MODEL GOLDEN [VAR=VALUE ...] -- w178 (#194-02): one radiance KLD read of MODEL on GOLDEN, the same shape as
# w177's tools/score.sh (bf16 KV, --tp 1, --max-model-len 16384, --max-num-seqs 16, --host-headroom-mib 8192), with extra
# environment after the golden (e.g. RADIANCE_ACT_TWO=0). Binary: w177's install, read only. Takes the first free card of
# CARDS (default "1 2 3": card 0 stays free for the speed rows) under /data/q27/locks/card<N>.lock; ram.lock shared.
# Restartable: an existing NAME.json.rows is skipped. Outputs: rows/NAME.json.rows, rows/NAME.json.
W=/data/q27/w178; N=$1; M=$2; G=$3; shift 3; INST=/data/q27/w177/install
mkdir -p $W/rows $W/logs; cd $W/rows
[ -s $N.json.rows ] && { echo "$(date -u +%FT%TZ) $N skipped (rows present)" >> $W/logs/score.status; exit 0; }
while :; do for c in ${CARDS:-1 2 3}; do
  exec 9>/data/q27/locks/card$c.lock
  if flock -n 9; then
    echo "$(date -u +%FT%TZ) $N start card $c env[$*]" >> $W/logs/score.status
    flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c "$@" $INST/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 16 --host-headroom-mib 8192 --kld-ref $G --kld-out $N.json.part > $W/logs/$N.log 2>&1
    rc=$?
    if [ $rc = 0 ] && [ -f $N.json.part.rows ]; then mv $N.json.part.rows $N.json.rows; fi
    [ -f $N.json.part ] && mv $N.json.part $N.json
    echo "$(date -u +%FT%TZ) $N rc $rc card $c" >> $W/logs/score.status; exit $rc
  fi
  exec 9>&-
done; sleep 20; done
