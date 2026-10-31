#!/bin/bash
# tp2long.sh -- the public corpus's two ~245k-token traces (docs 435, 531) at --tp 2 on both cards, per model;
# sc4.00 first (its TP1 run scored them whole: the long-context check), then the capped 5.00/6.00 rows.
cd /root
python3 -I golden_pick.py golden/public corpus/public.jsonl golden-tp2long corpus/tp2long.jsonl 435 531 > /dev/null
for n in sc4.00-e4 sc5.00-e5 plain5.00-e5 sc6.00-e6 plain6.00-e6; do
  [ -f tp2long/$n.f32 ] && continue
  w=$(RHOME=/root/dev-home TP=2 CARD=0,1 BIN=/root/radiance-kldgold-fast ./gold2.sh /root/models/final/q38-27b-$n.rad \
      golden-tp2long corpus/tp2long.jsonl /root/tp2l-$n bf16 262144 --max-num-batched-tokens 4096)
  mkdir -p tp2long; [ -f tp2l-$n/score.f32 ] && mv tp2l-$n/score.f32 tp2long/$n.f32
  echo "$n $w" >> tp2long.out; rm -rf tp2l-$n
done
echo TP2LONGDONE >> tp2long.out
