#!/bin/bash
# full.sh -- the public corpus, whole documents, bf16 KV, inline scoring; one JSON line a model to /root/full.jsonl
cd /root
run() {  # BPW GOLDEN MAXLEN [env]
  b=$1 g=$2 ml=$3
  w=$(./gold.sh /root/models/q38-27b-exl3-${b}bpw.rad $g corpus/public.jsonl /root/full-$b bf16 $ml)
  python3 -c "
import json; r=json.load(open('/root/full-$b.json'))
print(json.dumps({'bpw':'$b','run':'$w','golden':'$g','positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio'],'nonfinite':r['nonfinite_rows']}))" >> /root/full.jsonl \
    || echo "{\"bpw\":\"$b\",\"run\":\"$w\",\"fail\":1}" >> /root/full.jsonl
  rm -rf /root/full-$b
}
for b in 4.00 1.40 1.60 1.80 2.00 2.20 3.00 5.00; do run $b golden/public 262144; done
# 6.00: KV holds ~208k tokens; its one longer document (245k) is cut to its first 200k
[ -d golden-cap200k ] || python3 -I golden_pick.py --maxtok 200000 golden/public corpus/public.jsonl golden-cap200k /tmp/unused.jsonl all
RADIANCE_KLD_MAXTOK=200000 run 6.00 golden-cap200k 202000
echo FULLDONE >> /root/full.jsonl
