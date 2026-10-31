#!/bin/bash
# fullrun.sh CARD -- one worker a card (both may run): the whole public corpus (596 docs, 1.69M positions, bf16 KV) on
# each final container. Whole documents where the KV cache holds the 245k-token ones (4096-token steps, else 2048:
# a smaller arena leaves the cache more room); else the 200k-capped golden (only docs 435 and 531 are cut).
# One JSON line a model to /root/fullfinal.jsonl; per-position scores kept in /root/fullscores/<tag>.f32.
C=${1:-1}
cd /root; mkdir -p fullscores ffclaim
attempt() {  # GOLDEN MAXLEN STEP [MAXTOK]
  G=$1
  if [ -n "$4" ]; then export RADIANCE_KLD_MAXTOK=$4; else unset RADIANCE_KLD_MAXTOK; fi
  w=$(BIN=/root/radiance-kldgold-fast CARD=$C ./gold.sh $m $1 corpus/public.jsonl /root/ff-$n bf16 $2 --max-num-batched-tokens $3)
}
for n in sc2.00-e3 plain2.00-e3 sc4.00-e4 sc3.00-e4 sc2.20-e3 sc6.00-e6 sc5.00-e5 sc1.80-e3 sc1.60-e3 sc1.40-e3 \
         plain2.50-e3 plain3.00-e4 plain3.50-e4 plain4.00-e4 plain5.00-e5 plain6.00-e6; do
  grep -q "\"tag\": \"$n\"" /root/fullfinal.jsonl 2>/dev/null && continue
  m=/root/models/final/q38-27b-$n.rad
  [ -f $m ] || { grep -q "^$n rc=[1-9]" /root/models/final/conv.out && continue; }
  while [ ! -f $m ] && ! grep -q "^$n rc=[1-9]" /root/models/final/conv.out; do sleep 60; done
  [ -f $m ] || continue
  mkdir /root/ffclaim/$n 2>/dev/null || continue      # the other card's worker has it
  attempt golden/public 262144 4096
  grep -q "can never be scored" /root/ff-$n.log && attempt golden/public 262144 2048
  grep -q "can never be scored" /root/ff-$n.log && attempt golden-cap200k 202000 2048 200000
  python3 -c "
import json; r=json.load(open('/root/ff-$n.json'))
print(json.dumps({'tag':'$n','card':$C,'golden':'$G','run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'median':r['kld']['median'],'p99':r['kld']['p99'],'p99_9':r['kld']['p99_9'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio'],'long_agentic':r.get('kld_mean_long_agentic')}))" >> /root/fullfinal.jsonl \
    || echo "{\"tag\": \"$n\", \"run\": \"$w\", \"fail\": 1}" >> /root/fullfinal.jsonl
  [ -f /root/ff-$n/score.f32 ] && mv /root/ff-$n/score.f32 /root/fullscores/$n.f32
  rm -rf /root/ff-$n
done
echo "FULLDONE card $C" >> /root/fullfinal.jsonl
