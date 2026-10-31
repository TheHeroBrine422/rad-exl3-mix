#!/bin/bash
# run100k.sh -- the 100k-position subset (52 docs, one a source, each cut to 4096 tokens), bf16 KV; /root/r100k.jsonl
cd /root
export RADIANCE_KLD_MAXTOK=4096
for b in ${MODELS:-1.40 1.60 1.80 2.00 2.20 3.00 4.00 5.00 6.00}; do
  w=$(./gold.sh /root/models/q38-27b-exl3-${b}bpw.rad golden100k corpus100k.jsonl /root/r100k-$b bf16 8192 --max-num-batched-tokens 4096)
  python3 -c "
import json; r=json.load(open('/root/r100k-$b.json'))
print(json.dumps({'bpw':'$b','run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio'],'nonfinite':r['nonfinite_rows']}))" >> /root/r100k.jsonl \
    || echo "{\"bpw\":\"$b\",\"run\":\"$w\",\"fail\":1}" >> /root/r100k.jsonl
  rm -rf /root/r100k-$b
done
echo DONE >> /root/r100k.jsonl
