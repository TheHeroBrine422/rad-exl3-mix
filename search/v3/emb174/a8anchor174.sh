#!/bin/bash
# a8anchor174.sh -- (w135, #174, ds8; lead note 11-13 10:15 AM CST: search paused, anchor evals on A8 containers) the
# embedding anchor swaps in A8: a search incumbent's hostd .mix + "token_embd.weight <donor>" + "act <w> a8g128" on every
# trunk weight (w132's A8 mix rule, run132.sh), radmix'ed on V2 3.00, scored on s171b with exactly w132's radiance command
# (rad10, bf16 KV, max-model-len 16384), so each pairs with w132's A8 score of the same incumbent
# (/data/q27/w132/scores/<tag>-a8-s171b.json.rows). One eval at a time on one free card (card lock; ram.lock shared;
# radmix under conv.lock); mixes deleted after scoring. Restartable: finished scores are skipped.
set -u
Q=/data/q27; W=$Q/emb/a8; mkdir -p $W/mix $W/scores
BASE=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; SB=$Q/w131/slices/golden-s171b
RUNS="v3s171-2.5-explore-00419:e2 v3s171-3.5-block-00483:e2 v3s171-3.5-block-00483:e4 v3s171-5-pair-00449:e6
      v3s171-6-explore-00372:e5 v3s171-6-explore-00372:e7"
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/run.log; }

mkmix() {  # tag emb -> $W/mix/tag-emb-a8.rad
  local t=$1 e=$2 n=$1-$2-a8; local mf=$W/mix/$n.mix out=$W/mix/$n.rad
  grep -q '^token_embd' $Q/mixes/$t.mix && { log "$t.mix already sets token_embd"; return 1; }
  { cat $Q/mixes/$t.mix; echo "token_embd.weight $Q/emb/emb-$e.rad"; awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $Q/mixes/$t.mix; } > $mf
  [ -f $out ] && return 0
  rm -f $out.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $Q/tools/radmix $out.part $BASE $mf >> $W/mix/$n.mixlog 2>&1 \
    && mv $out.part $out || { log "radmix $n FAILED"; return 1; }
  log "mix $n bytes $(stat -c %s $out)"
}

score() {  # name -> $W/scores/name-s171b.json(.rows)
  local n=$1 o=$W/scores/$1-s171b
  [ -f $o.json.rows ] && return 0
  while :; do
    for c in 0 1 2 3 4 5 6 7; do
      exec 8>$Q/locks/card$c.lock
      if flock -n 8; then
        log "score $n card $c"
        flock -s $Q/locks/ram.lock bash -c "ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$Q/rad10/home $Q/rad10/install/bin/radiance --model $W/mix/$n.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $SB --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
        rc=$?; exec 8>&-
        log "score $n rc $rc $(grep 'I kld: [0-9]* / ' $o.log | tail -n 1)"
        return $rc
      fi
      exec 8>&-
    done
    sleep 20
  done
}

log "start pid $$"
for r in $RUNS; do
  t=${r%%:*}; e=${r#*:}; n=$t-$e-a8
  [ -f $W/scores/$n-s171b.json.rows ] && continue
  mkmix $t $e || continue
  score $n
  rm -f $W/mix/$n.rad
done
log "done"
