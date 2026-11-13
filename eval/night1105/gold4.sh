#!/bin/bash
# gold4.sh LANE CARD_A CARD_B -- as gold3.sh, for the 40M-token xcorp QAD corpus (xq-c00..39, top-256) -- bf16 27B reference goldens (TP2, radiance --kld-record, exact wire, bf16 KV) for the
# night1105 corpora chunks never recorded (b3 + tr32k-b2-c2); lanes share a queue by claim files. Lead, 2026-11-13.
set -u
Q=/data/q27; G=$Q/gold4; R=$Q/rad10-perf; M=$Q/models/qwen38-27b-bf16
lane=$1 a=$2 b=$3; L=$G/lane$lane.log; log() { echo "$(date -u +%T) $*" >> $L; }
for c in $(cd $G/corpora; ls xq-c*.jsonl | grep -v meta | sed "s/.jsonl$//"); do
  [ -f $G/gold/$c/.done ] && continue
  ( set -o noclobber; echo $lane > $G/claim-$c ) 2>/dev/null || continue
  len=8448
  rm -rf $G/gold/$c; mkdir -p $G/gold
  ( exec 5>$Q/locks/card$a.lock 6>$Q/locks/card$b.lock 7>$Q/locks/ram.lock; flock 5; flock 6; flock -s 7
    s=$(date +%s); log "start $c on $a,$b"
    ROCR_VISIBLE_DEVICES=$a,$b RADIANCE_HOME=$R/home $R/install/bin/radiance --model $M --tp 2 --kv-cache-dtype bf16 \
      --max-model-len $len --max-num-seqs 4 --host-headroom-mib 6144 --kld-record $G/gold/$c --kld-corpus $G/corpora/$c.jsonl \
      --kld-top 256 > $G/rec-$c.log 2>&1; rc=$?
    [ $rc = 0 ] && { cp $G/corpora/$c.jsonl.meta.jsonl $G/gold/$c/corpus.meta.jsonl; touch $G/gold/$c/.done; }
    log "rec $c rc=$rc $(( ($(date +%s) - s) / 60 )) min" )
  [ -f $G/gold/$c/.done ] || rm -f $G/claim-$c
done
log "lane $lane DONE"
