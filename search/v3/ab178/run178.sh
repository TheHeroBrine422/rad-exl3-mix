#!/bin/bash
# run178.sh -- RULES-178 on ds8 (w142, #178). Arms G / R come from abgptq.py (CPU, started apart: $W/gq/ab-{G,R}.safetensors);
# the trellis arms are quantized here (abtrellis.py on the first free card, under its card lock + ram.lock shared).
# Per size: the incumbent's hostd .mix + "act <w> a8g128" on every blk.* weight (w132's form) -> radmix once (conv.lock +
# ram.lock shared); per arm a copy (reflinked where the filesystem can; ds8 cannot), abpatch.py writes the arm's a/b into it, scored on golden-s171d under the
# first free card lock + ram.lock shared (hostd's score command), then deleted. Arm B (no a/b change) = w132's RULES-value
# score of the same mix (w132/value/scores/<tag>-s171d), reused. Restartable: finished scores are skipped.
set -u
Q=/data/q27; W=$Q/w142; mkdir -p $W/mix $W/scores
BASE=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; G=$Q/w131/slices/golden-s171d
declare -A TAG=([2.5]=v3s171-2.5-block-00626 [4.0]=v3s171-4-block-00420)
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" >> $W/run.log; }

card() {  # run "$@" under the first free card lock (+ ram.lock shared), ROCR_VISIBLE_DEVICES set; waits
  while :; do
    for c in 0 1 2 3 4 5 6 7; do
      exec 8>$Q/locks/card$c.lock
      if flock -n 8; then
        log "card $c: $1"
        flock -s $Q/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c bash -c "$2"; local rc=$?
        exec 8>&-; return $rc
      fi
      exec 8>&-
    done
    sleep 30
    # nothing free after a pass: queue on one card (blocking flock waiters otherwise win every release)
    local c=$(( (RANDOM % 4) * 2 + 1 ))
    exec 8>$Q/locks/card$c.lock; flock 8
    log "card $c (queued): $1"
    flock -s $Q/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c bash -c "$2"; local rc=$?
    exec 8>&-; return $rc
  done
}

mkmix() {  # size -> $W/mix/<size>-B.rad
  local t=${TAG[$1]} mf=$W/mix/$1-B.mix out=$W/mix/$1-B.rad
  [ -f $out ] && return 0
  { cat $Q/mixes/$t.mix; awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $Q/mixes/$t.mix; } > $mf
  rm -f $out.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $Q/tools/radmix $out.part $BASE $mf > $W/mix/$1-B.mixlog 2>&1 \
    && mv $out.part $out || { log "radmix $1 FAILED"; return 1; }
  log "mix $1 $t bytes $(stat -c %s $out) acts $(grep -c '^act ' $mf)"
}

arm() {  # size arm ab.safetensors
  local s=$1 a=$2 src=$3 o=$W/scores/$1-$2-s171d r=$W/mix/$1-$2.rad
  [ -f $o.json.rows ] && return 0
  mkmix $s || return 1
  rm -f $r; cp --reflink=auto $W/mix/$s-B.rad $r || { log "copy $s $a FAILED"; rm -f $r; return 1; }   # ds8 /data: no reflink (radmix: 0 cloned), a full copy
  /data/q27/venv/bin/python $W/code/abpatch.py $r $src >> $W/run.log 2>&1 || { log "abpatch $s $a FAILED"; rm -f $r; return 1; }
  card "score $s $a" "RADIANCE_HOME=$Q/rad10/home $Q/rad10/install/bin/radiance --model $r --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
  log "score $s $a rc $? $(grep 'documents in' $o.log)"
  rm -f $r
}

log "start pid $$"
if [ ! -f $W/tq/ab-T6.safetensors ]; then
  card "trellis a/b K4,5,6" "source $Q/tools/env-ds8.sh; cd $W && python code/abtrellis.py $Q/models/qwen38-27b-bf16 $Q/challenge/hessians/calib tq -K 4,5,6 > tq.log 2>&1"
  log "trellis rc $? $(tail -n 1 $W/tq.log)"
fi
for s in 2.5 4.0; do mkmix $s; done
for s in 2.5 4.0; do       # the trellis arms first (ready), then G / R once abgptq.py has written them
  ( for K in 4 5 6; do arm $s T$K $W/tq/ab-T$K.safetensors; done
    while [ ! -f $W/gq/ab-R.safetensors ]; do sleep 60; done
    arm $s G $W/gq/ab-G.safetensors; arm $s R $W/gq/ab-R.safetensors ) &
done
wait
rm -f $W/mix/*-B.rad
log "done"
