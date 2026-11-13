#!/bin/bash
# [M1=1] run-value.sh [WORKER] -- RULES-value.md (M1=1: amendment 1's arm) (w132, #172) on ds8: the 12 initial picks + 12 current incumbents as A8 mixes
# (hostd .mix + "act <w> a8g128" on every trunk weight), each scored on golden-s171c and golden-s171d. No argument:
# starts 8 workers (worker i: card i, tags i, i+8, i+16). radmix under conv.lock + ram.lock shared (and >= 150 GB free
# on /data first); scoring under card<i>.lock + ram.lock shared (hostd's score command). Restartable; mixes deleted.
set -u
Q=/data/q27; W=$Q/w132/value; mkdir -p $W/mix $W/scores
BASE=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; S=$Q/w131/slices
TAGS=(v3s171-1.5-init-00000 v3s171-1.75-init-00001 v3s171-2-init-00002 v3s171-2.5-init-00003 v3s171-3-init-00004
      v3s171-3.5-init-00005 v3s171-4-init-00006 v3s171-4.5-init-00007 v3s171-5-init-00008 v3s171-5.5-init-00009
      v3s171-6-init-00010 v3s171-6.5-init-00011
      v3s171-1.5-explore-00487 v3s171-1.75-pair-00495 v3s171-2-partial-00593 v3s171-2.5-block-00626 v3s171-3-block-00500
      v3s171-3.5-pair-00510 v3s171-4-block-00420 v3s171-4.5-solve-00599 v3s171-5-pair-00449 v3s171-5.5-thompson-00540
      v3s171-6-explore-00372 v3s171-6.5-pair-00588)
MIXSRC=$Q/mixes
if [ -n "${M1:-}" ]; then                 # amendment 1: w133's M1 picks (mkm1mix.py -> $W/m1mix/m1-<size>.mix)
  TAGS=(m1-1.5 m1-1.75 m1-2 m1-2.5 m1-3 m1-3.5 m1-4 m1-4.5 m1-5 m1-5.5 m1-6 m1-6.5 m1-7); MIXSRC=$W/m1mix
fi
[ -n "${T7:-}" ] && TAGS=(v3s171-7-init-00012)   # amendment 1: 7.0's initial pick (= its current incumbent's assignment), worker 0
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" >> $W/run.log; }
if [ $# = 0 ]; then
  for i in 0 1 2 3 4 5 6 7; do M1=${M1:-} setsid nohup bash $0 $i > $W/worker$i.out 2>&1 < /dev/null & done; exit 0
fi
c=$1; log "worker $c start pid $$"
for ((j = c; j < ${#TAGS[@]}; j += 8)); do
  t=${TAGS[$j]}; mf=$W/mix/$t-a8.mix; out=$W/mix/$t-a8.rad
  if [ -f $W/scores/$t-s171c.json.rows ] && [ -f $W/scores/$t-s171d.json.rows ]; then continue; fi
  { cat $MIXSRC/$t.mix; awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $MIXSRC/$t.mix; } > $mf
  if [ ! -f $out ]; then
    while [ $(df --output=avail -B1G /data | tail -n 1) -lt 150 ]; do log "worker $c: < 150 GB free, waiting"; sleep 60; done
    rm -f $out.part
    flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $Q/tools/radmix $out.part $BASE $mf > $W/mix/$t.mixlog 2>&1 && mv $out.part $out \
      || { log "radmix $t FAILED"; continue; }
    log "mix $t bytes $(stat -c %s $out) acts $(grep -c '^act ' $mf)"
  fi
  for sl in s171c s171d; do
    o=$W/scores/$t-$sl; [ -f $o.json.rows ] && continue
    flock $Q/locks/card$c.lock flock -s $Q/locks/ram.lock bash -c "ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$Q/rad10/home $Q/rad10/install/bin/radiance --model $out --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $S/golden-$sl --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
    log "score $t $sl card $c rc $? $(grep 'documents in' $o.log)"
  done
  rm -f $out
done
log "worker $c done"
