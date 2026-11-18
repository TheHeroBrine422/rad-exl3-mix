#!/bin/bash
# run205.sh CARD... (box2) -- score every /data/q27/mad205/mixes/*.mix (pick + mad) on the 6 phase-2 slices with the w201 build
# (qc-exl3-v3-eval, byte-identical to the search build), one worker per card pulling from a shared queue (mkdir claims).
# Output /data/q27/mad205/scores/<mix>-<slice>.json(.rows) + docs/<mix>-<slice>.json ({"m": [[KL sum, positions] per doc]}).
# Restartable (finished outputs skipped; claims of dead workers: rm -rf /data/q27/mad205/claims before restarting). Stop: touch STOP.
set -u
Z=/data/q27/mad205; B=/data/q27/w201/rad21; mkdir -p $Z/scores $Z/docs $Z/claims $Z/logs
declare -A G=([final750]=/data/q27/final/golden-final750 [s195c6]=/data/q27/w180/slices/golden-s195c6 [s195c7]=/data/q27/w180/slices/golden-s195c7
  [s195c8]=/data/q27/w180/slices/golden-s195c8 [s195c9]=/data/q27/w180/slices/golden-s195c9 [s195c10]=/data/q27/w180/slices/golden-s195c10)
worker() {
  local c=$1
  for sl in final750 s195c6 s195c7 s195c8 s195c9 s195c10; do for m in $Z/mixes/*-mad.mix; do   # picks: phase-2 rows from the coordinator (same bytes, w201)
    [ -f $Z/STOP ] && return
    local n=$(basename $m .mix) o=$Z/scores/$(basename $m .mix)-$sl
    [ -f $o.json ] && continue; mkdir $Z/claims/$n-$sl 2>/dev/null || continue
    t0=$(date +%s)
    flock /data/q27/locks/card$c.lock flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$B/home $B/install/bin/radiance \
      --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 16448 --max-num-seqs 1 --host-headroom-mib 8192 \
      --kld-ref ${G[$sl]} --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    rc=$?
    [ $rc = 0 ] && /data/q27/venv/bin/python -I -c "
import json,sys; sys.path.insert(0,'/data/q27/srch183/code/pipeline'); from rxrun.hostd import doc_sums
o,_=doc_sums('${G[$sl]}','$o.json.rows'); json.dump({'m':o},open('$Z/docs/$n-$sl.json','w'))" || rc=2
    echo "$(date -u +%H:%M:%S) card $c $n $sl rc=$rc $(( $(date +%s) - t0 )) s" >> $Z/run205.log
    [ $rc = 0 ] || rmdir $Z/claims/$n-$sl
  done; done
}
for c in "$@"; do worker $c & done; wait; echo "$(date -u +%H:%M:%S) ALL DONE" >> $Z/run205.log
