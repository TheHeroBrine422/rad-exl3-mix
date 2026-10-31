#!/bin/bash
# finalrun.sh -- e2e.sh (card 0) on every final container as finalconv.sh produces it; skips tags already in e2e.jsonl
cd /root
for n in sc4.00-e5 sc4.00-e6 sc4.00-e3 plain2.00-e3 sc1.40-e3 sc1.60-e3 sc1.80-e3 sc2.20-e3 sc3.00-e4 sc5.00-e5 sc6.00-e6 \
         plain2.50-e3 plain3.00-e4 plain3.50-e4 plain4.00-e4 plain5.00-e5 plain6.00-e6; do
  grep -q "\"tag\": \"$n\", \"prompt_tok\"" /root/e2e.jsonl 2>/dev/null && continue
  m=/root/models/final/q38-27b-$n.rad
  while [ ! -f $m ] && ! grep -q "^$n rc=[1-9]" /root/models/final/conv.out; do sleep 30; done
  [ -f $m ] && /root/e2e.sh $m $n || echo "{\"tag\":\"$n\",\"fail\":\"convert\"}" >> /root/e2e.jsonl
done
echo FINALDONE >> /root/e2e.jsonl
