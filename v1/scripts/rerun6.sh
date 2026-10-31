#!/bin/bash
# rerun6.sh CARD TAG -- the 6.00 containers: the KV pool holds ~170k, so the 160k-capped golden (docs 435, 531 cut).
C=$1 n=$2; cd /root
m=/root/models/final/q38-27b-$n.rad; G=golden-cap160k
export RADIANCE_KLD_MAXTOK=160000
w=$(BIN=/root/radiance-kldgold-fast CARD=$C ./gold.sh $m $G corpus/public.jsonl /root/ff-$n bf16 162000 --max-num-batched-tokens 2048)
python3 -c "
import json; r=json.load(open('/root/ff-$n.json'))
print(json.dumps({'tag':'$n','card':$C,'golden':'$G','run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'median':r['kld']['median'],'p99':r['kld']['p99'],'p99_9':r['kld']['p99_9'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio'],'long_agentic':r.get('kld_mean_long_agentic')}))" >> /root/fullfinal.jsonl \
  || echo "{\"tag\": \"$n\", \"run\": \"$w\", \"fail\": 1}" >> /root/fullfinal.jsonl
[ -f /root/ff-$n/score.f32 ] && mv /root/ff-$n/score.f32 /root/fullscores/$n.f32
rm -rf /root/ff-$n
