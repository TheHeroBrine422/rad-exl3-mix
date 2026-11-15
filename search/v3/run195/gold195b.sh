#!/bin/bash
# gold195b.sh CARD_A CARD_B CHUNK... -- bf16 27B goldens (the lead's gold3.sh command: TP2, bf16 KV, top-512) for corpora
# $G/CHUNK.jsonl (+ .meta.jsonl) on box2. A lane polls its two card locks without blocking (never holds one card while waiting
# for the other, never preempts), then claims the next unclaimed chunk. Restartable: a finished golden (.done) is skipped.
set -u
Q=/data/q27; G=$Q/w180/batch; R=$Q/rad10-perf; M=$Q/models/qwen38-27b-bf16
a=$1 b=$2; shift 2; L=$G/gold195.log; log() { echo "$(date -u +%FT%T) $*" >> $L; }
while :; do
  todo=""; for c in "$@"; do [ -f $G/gold/$c/.done ] || [ -f $G/claim-$c ] || { todo=$c; break; }; done
  [ -z "$todo" ] && break
  ( exec 5>$Q/locks/card$a.lock 6>$Q/locks/card$b.lock 7>$Q/locks/ram.lock
    flock -n 5 || exit 9; flock -n 6 || exit 9; flock -s 7
    c=$todo; ( set -o noclobber; echo "$a,$b" > $G/claim-$c ) 2>/dev/null || exit 8
    rm -rf $G/gold/$c; mkdir -p $G/gold; s=$(date +%s); log "start $c on $a,$b"
    ROCR_VISIBLE_DEVICES=$a,$b RADIANCE_HOME=$R/home $R/install/bin/radiance --model $M --tp 2 --kv-cache-dtype bf16 \
      --max-model-len 8448 --max-num-seqs 4 --host-headroom-mib 6144 --kld-record $G/gold/$c --kld-corpus $G/$c.jsonl \
      --kld-top 512 > $G/rec-$c.log 2>&1; rc=$?
    [ $rc = 0 ] && { cp $G/$c.jsonl.meta.jsonl $G/gold/$c/corpus.meta.jsonl; touch $G/gold/$c/.done; } || rm -f $G/claim-$c
    log "rec $c rc=$rc $(( ($(date +%s) - s) / 60 )) min" )
  [ $? = 9 ] && sleep 30
done
log "lane $a,$b DONE"
