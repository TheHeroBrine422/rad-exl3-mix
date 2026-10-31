#!/bin/bash
# embrun.sh TAG FMT... -- after the queued runs (rlong.jsonl holds two DONE lines: the ladder's and the 3.00
# rerun's), each /root/emb/TAG-FMT.rad on the 100k corpus; one JSON line a variant to /root/emb.jsonl
cd /root
while [ "$(grep -c DONE /root/rlong.jsonl 2>/dev/null)" -lt 2 ]; do sleep 30; done
export RADIANCE_KLD_MAXTOK=4096
T=$1; shift
for f in "$@"; do
  m=/root/emb/$T-$f.rad
  while [ ! -f $m ] && ! grep -q "$T $f FAIL" /root/emb/prep.log; do sleep 20; done
  [ -f $m ] || { echo "{\"model\":\"$T\",\"emb\":\"$f\",\"fail\":\"prep\"}" >> /root/emb.jsonl; continue; }
  w=$(./gold.sh $m golden100k corpus100k.jsonl /root/re-$T-$f bf16 8192 --max-num-batched-tokens 4096)
  python3 -c "
import json; r=json.load(open('/root/re-$T-$f.json'))
print(json.dumps({'model':'$T','emb':'$f','run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio']}))" >> /root/emb.jsonl \
    || echo "{\"model\":\"$T\",\"emb\":\"$f\",\"run\":\"$w\",\"fail\":1}" >> /root/emb.jsonl
  rm -rf /root/re-$T-$f $m
done
