#!/bin/bash
# after the queue: the long-context 1.40 the collisions killed twice (same recorder, one at a time)
while pgrep -f "^/bin/bash /root/queue.sh" >/dev/null; do sleep 30; done
cd /root; w=$(./gold.sh /root/models/q38-27b-exl3-1.40bpw.rad goldenlong corpuslong.jsonl /root/q-long-1.40 bf16 131072 --max-num-batched-tokens 4096)
python3 -c "
import json; r=json.load(open(\"/root/q-long-1.40.json\"))
print(json.dumps({\"set\":\"long\",\"model\":\"1.40\",\"emb\":\"bf16\",\"run\":\"$w\",\"kld\":r[\"kld\"][\"mean\"],\"p99\":r[\"kld\"][\"p99\"],\"p99_9\":r[\"kld\"][\"p99_9\"],\"top1\":r[\"top1_agreement\"],\"long_by_pos\":{k:round(v[\"kld\"],5) for k,v in r[\"by_position_long_docs\"].items()}}))" >> /root/results.jsonl
echo AFTERDONE >> /root/results.jsonl
