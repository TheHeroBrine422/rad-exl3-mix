#!/bin/bash
# aftersplice.sh -- when tp2long.sh is done: each capped 5.00/6.00 run + its TP2 long docs -> uncapped full-corpus
# score (fullscores/<tag>.full.f32) and one JSON line in /root/fulluncapped.jsonl
cd /root
until grep -q TP2LONGDONE tp2long.out 2>/dev/null; do sleep 60; done
for p in "sc5.00-e5 200000" "plain5.00-e5 200000" "sc6.00-e6 160000" "plain6.00-e6 160000"; do
  set -- $p
  [ -f tp2long/$1.f32 ] || { echo "{\"tag\": \"$1\", \"fail\": \"no tp2 long run\"}" >> fulluncapped.jsonl; continue; }
  python3 -I splicelong.py golden/public $2 fullscores/$1.f32 golden-tp2long tp2long/$1.f32 fullscores/$1.full.f32 > /dev/null &&
  python3 -I aggscore.py golden/public fullscores/$1.full.f32 | python3 -c "
import json,sys; r=json.load(sys.stdin)
print(json.dumps({'tag':'$1','golden':'golden/public (capped TP1 + TP2 long docs)','positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'p99_9':r['kld']['p99_9'],'top1':r['top1_agreement'],'ppl_ratio':r['ppl']['ratio'],'long_agentic':r.get('kld_mean_long_agentic')}))" >> fulluncapped.jsonl \
  || echo "{\"tag\": \"$1\", \"fail\": \"splice\"}" >> fulluncapped.jsonl
done
echo SPLICEDONE >> fulluncapped.jsonl
