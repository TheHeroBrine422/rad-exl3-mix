#!/bin/bash
# long2card.sh CARD -- remaining full-length long checks (1.5-4.5), shared by both cards via mkdir claims
cd /root; mkdir -p auto/longclaim
for T in 2.00 2.50 3.00 3.50 4.00 4.50; do
  t=long-v2-$T
  grep -q "\"tag\": \"$t\", \"bytes\"" longmix.jsonl && continue
  mkdir auto/longclaim/$T 2>/dev/null || continue
  s=$(date +%s); LV_TP=1 LV_CARD=$1 ./hproxy/longverify.sh $t /data/release/qwen3.8-27b-exl3-v2-${T}bpw.rad -
  echo "$(date -u +%H:%M) [long c$1] $t: $(grep "\"tag\": \"$t\"" longmix.jsonl | tail -1 | cut -c1-150) ($(( ($(date +%s) - s) / 60 )) min)" >> auto/chain.log
done
