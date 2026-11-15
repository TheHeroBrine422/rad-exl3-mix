#!/bin/bash
# gold195.sh CARD_A CARD_B -- bf16 27B golden of the #195 non-Latin batch (the lead's gold3.sh command: TP2, bf16 KV, top-512).
# Waits for both card locks (flock, never preempts), holds ram.lock shared. Restartable: a finished golden is skipped.
set -u
Q=/data/q27; G=$Q/w180/batch; R=$Q/rad10-perf; M=$Q/models/qwen38-27b-bf16; c=xn8k-b195
a=$1 b=$2; L=$G/gold195.log; log() { echo "$(date -u +%FT%T) $*" >> $L; }
[ -f $G/gold/$c/.done ] && exit 0
rm -rf $G/gold/$c; mkdir -p $G/gold
( exec 5>$Q/locks/card$a.lock 6>$Q/locks/card$b.lock 7>$Q/locks/ram.lock; flock 5; flock 6; flock -s 7
  s=$(date +%s); log "start $c on $a,$b"
  ROCR_VISIBLE_DEVICES=$a,$b RADIANCE_HOME=$R/home $R/install/bin/radiance --model $M --tp 2 --kv-cache-dtype bf16 \
    --max-model-len 8448 --max-num-seqs 4 --host-headroom-mib 6144 --kld-record $G/gold/$c --kld-corpus $G/$c.jsonl \
    --kld-top 512 > $G/rec-$c.log 2>&1; rc=$?
  [ $rc = 0 ] && { cp $G/$c.jsonl.meta.jsonl $G/gold/$c/corpus.meta.jsonl; touch $G/gold/$c/.done; }
  log "rec $c rc=$rc $(( ($(date +%s) - s) / 60 )) min" )
