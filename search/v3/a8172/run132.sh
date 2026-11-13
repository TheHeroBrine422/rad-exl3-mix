#!/bin/bash
# w132 (#172) step 1, on ds8: the search incumbents at 2.5 / 3.5 / 5.0 / 6.0 (drv/state.jsonl, 11-13 08:30 CST) remixed
# from their hostd .mix files without (f16) and with "act <w> a8g128" on every trunk weight, A8 scored on golden-ver3-8k
# and s171b (f16 rows: the search's own scores/, same rad10 build; one f16 rescore as a determinism + timing check).
# Locks as hostd (ram_mode shared): radmix under conv.lock + ram.lock shared, scoring under one free card lock + ram.lock
# shared, one run at a time. Restartable: finished scores are skipped. Mixes deleted after use.
# Phase 2 (PREV=1): the incumbents each one replaced (prev of the same size, f16 s171b rows exist), A8 on s171b only:
# was the f16 accept decision the same under A8?
set -u
Q=/data/q27; W=$Q/w132; mkdir -p $W/mix $W/scores
BASE=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad
V3=$Q/goldens/golden-ver3-8k; SB=$Q/w131/slices/golden-s171b
TAGS="v3s171-2.5-explore-00419 v3s171-3.5-block-00483 v3s171-5-pair-00449 v3s171-6-explore-00372"
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/run.log; }

mkmix() {  # tag mode -> $W/mix/tag-mode.rad
  local t=$1 m=$2 mf=$W/mix/$1-$2.mix out=$W/mix/$1-$2.rad
  if [ $m = f16 ]; then cp $Q/mixes/$t.mix $mf
  else { cat $Q/mixes/$t.mix; awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $Q/mixes/$t.mix; } > $mf; fi
  [ -f $out ] && return 0
  rm -f $out.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $Q/tools/radmix $out.part $BASE $mf >> $W/mix/$t-$m.mixlog 2>&1 \
    && mv $out.part $out || { log "radmix $t $m FAILED"; return 1; }
  log "mix $t $m bytes $(stat -c %s $out)"
}

score() {  # tag mode golden name
  local t=$1 m=$2 g=$3 n=$4 o=$W/scores/$1-$2-$4
  [ -f $o.json.rows ] && return 0
  while :; do
    for c in 0 1 2 3 4 5 6 7; do
      exec 8>$Q/locks/card$c.lock
      if flock -n 8; then
        log "score $t $m $n card $c"
        flock -s $Q/locks/ram.lock bash -c "ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$Q/rad10/home $Q/rad10/install/bin/radiance --model $W/mix/$t-$m.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $g --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
        rc=$?; exec 8>&-
        log "score $t $m $n rc $rc $(grep 'I kld: [0-9]* / ' $o.log | tail -n 1)"
        return $rc
      fi
      exec 8>&-
    done
    sleep 20
  done
}

log "start pid $$ PREV=${PREV:-0}"
if [ "${PREV:-0}" = 1 ]; then
  for t in v3s171-2.5-block-00413 v3s171-3.5-pair-00396 v3s171-5-pair-00409 v3s171-6-inc-00358; do
    mkmix $t a8 || continue; score $t a8 $SB s171b; rm -f $W/mix/$t-a8.rad
  done; log "done PREV"; exit 0
fi
for t in $TAGS; do
  mkmix $t f16 || continue
  if [ $t = v3s171-5-pair-00449 ]; then score $t f16 $V3 ver3-8k; fi
  rm -f $W/mix/$t-f16.rad
  mkmix $t a8 || continue
  score $t a8 $V3 ver3-8k
  score $t a8 $SB s171b
  rm -f $W/mix/$t-a8.rad
done
log "done"
