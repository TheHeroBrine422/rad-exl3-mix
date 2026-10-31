#!/bin/bash
# runlong.sh -- after run100k: one 100k-token agent trace + 8 docs of 8192 tokens (165,527 positions), bf16 KV,
# per-doc caps from the corpus lines; one JSON line a model to /root/rlong.jsonl, full reports in /root/rlong-<bpw>.json
cd /root
while ! grep -q DONE /root/r100k.jsonl 2>/dev/null; do sleep 30; done
unset RADIANCE_KLD_MAXTOK
for b in ${MODELS:-1.40 1.60 1.80 2.00 2.20 3.00 4.00 5.00 6.00}; do
  w=$(./gold.sh /root/models/q38-27b-exl3-${b}bpw.rad goldenlong corpuslong.jsonl /root/rlong-$b bf16 131072 --max-num-batched-tokens 4096)
  python3 -c "
import json; r=json.load(open('/root/rlong-$b.json'))
print(json.dumps({'bpw':'$b','run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio'],'nonfinite':r['nonfinite_rows'],
 'long_by_pos':{k:round(v['kld'],5) for k,v in r['by_position_long_docs'].items()}}))" >> /root/rlong.jsonl \
    || echo "{\"bpw\":\"$b\",\"run\":\"$w\",\"fail\":1}" >> /root/rlong.jsonl
  rm -rf /root/rlong-$b
done
echo DONE >> /root/rlong.jsonl
