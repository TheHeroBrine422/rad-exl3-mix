#!/bin/bash
# kldenv.sh CARD TAG=HOME=PACK[=ENV]... -- whole-model KLD rows (score.sh's settings: rad10-perf, bf16 KV, max-model-len
# 16384, one sequence; golden-search27c-c) per TAG with ENV ("K=V,K=V") for the server, under card + ram (shared) locks;
# out res/kldr/TAG.json(.rows); finished skipped. (w152: the #177-02 decode-path KLD re-measure after fdd480c)
C=$1; shift; Q=/data/q27; W=$Q/w152; G=$Q/goldens/golden-search27c-c; I=$Q/rad10-perf/install; mkdir -p $W/res/kldr
for row in "$@"; do
  IFS='=' read -r t H P EV <<< "$row"; o=$W/res/kldr/$t
  [ -f $o.json ] && continue
  s=$(date +%s); ulimit -n 65536
  flock -s $Q/locks/ram.lock flock $Q/locks/card$C.lock env ${EV//,/ } ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H timeout 3600 \
    $I/bin/radiance --model $P --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 \
    --kld-ref $G --kld-out $o.json.part > $o.log 2>&1; rc=$?
  [ $rc = 0 ] && mv $o.json.part $o.json && mv $o.json.part.rows $o.json.rows
  echo "$(date -u +%T) kldr $t rc=$rc $(( $(date +%s) - s ))s"
done
