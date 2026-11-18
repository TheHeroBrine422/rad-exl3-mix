#!/bin/bash
# iqr206.sh (ds8) -- IQR-V4 (StillDeadcode/qwen3.8-27b-iqr-v4, 13 sizes) on the 6 phase-2 slices for a 1:1 paired comparison with
# the frozen V3 picks (report only; the picks are frozen). One lane per card: lane i takes models i, i+8, ...: download (or local
# copy) -> 6 slices with the coordinator's build and score command -> per-document sums (hostd.doc_sums) -> delete the download.
# Output /data/q27/iqr206/{scores,docs}/<model>-<slice>.*; log iqr206.log. Restartable (finished slices skipped). Stop: touch STOP.
set -u
Z=/data/q27/iqr206; D=$Z/dl; B=/data/q27/w173/rad10; mkdir -p $Z/scores $Z/docs $D
IQ=StillDeadcode/qwen3.8-27b-iqr-v4
M=(iq2r iq2r-l iq3r-s iq3r iq3r-l iq4r-s iq4r iq4r-l iq5r-s iq5r iq6r-s iq6r iq7r)
declare -A G=([final750]=/data/q27/final/golden-final750 [s195c6]=/data/q27/w180/slices/golden-s195c6 [s195c7]=/data/q27/w180/slices/golden-s195c7
  [s195c8]=/data/q27/w180/slices/golden-s195c8 [s195c9]=/data/q27/w180/slices/golden-s195c9 [s195c10]=/data/q27/w180/slices/golden-s195c10)
log() { echo "$(date -u +%H:%M:%S) $*" >> $Z/iqr206.log; }
lane() {
  local c=$1 i
  for ((i=c; i<${#M[@]}; i+=8)); do
    local n=${M[$i]} f=qwen3.8-27b-${M[$i]}-v4.rad loc dl=0 left=0
    for sl in final750 s195c6 s195c7 s195c8 s195c9 s195c10; do [ -f $Z/docs/$n-$sl.json ] || left=1; done
    [ $left = 0 ] && continue
    loc=/data/q27/w153/ref/$f
    if [ ! -f $loc ]; then
      loc=$D/$f
      if [ ! -f $loc ]; then
        while [ $(df --output=avail -B1G /data | tail -1) -lt 200 ]; do sleep 60; done
        log "card $c $n download"; hf download $IQ $f --local-dir $D > $Z/dl-$n.log 2>&1 || { log "card $c $n DOWNLOAD FAILED"; continue; }
      fi; dl=1
    fi
    for sl in final750 s195c6 s195c7 s195c8 s195c9 s195c10; do
      [ -f $Z/STOP ] && return
      local o=$Z/scores/$n-$sl; [ -f $Z/docs/$n-$sl.json ] && continue
      t0=$(date +%s)
      flock /data/q27/locks/card$c.lock flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$B/home $B/install/bin/radiance \
        --model $loc --tp 1 --kv-cache-dtype bf16 --max-model-len 16448 --max-num-seqs 1 --host-headroom-mib 8192 \
        --kld-ref ${G[$sl]} --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
      rc=$?
      [ $rc = 0 ] && { /data/q27/venv/bin/python -I -c "
import json,sys; sys.path.insert(0,'/data/q27/srch183/code/pipeline'); from rxrun.hostd import doc_sums
o,_=doc_sums('${G[$sl]}','$o.json.rows'); json.dump({'m':o,'bytes':$(stat -c %s $loc)},open('$Z/docs/$n-$sl.json','w'))" || rc=2; }
      log "card $c $n $sl rc=$rc $(( $(date +%s) - t0 )) s"
    done
    [ $dl = 1 ] && [ -z "$(ls $Z/docs/$n-*.json 2>/dev/null | wc -l | grep -vx 6)" ] && rm -f $loc && log "card $c $n deleted"
  done
}
for c in 0 1 2 3 4 5 6 7; do lane $c & done; wait; log "ALL DONE"
