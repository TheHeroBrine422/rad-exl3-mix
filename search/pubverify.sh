#!/bin/bash
# pubverify.sh TAG MODEL.rad [MIXFILE] -- the public held-out score of one finalist: the whole public corpus, every
# document whole (TP2 over both cards: the KV pool holds the 245k-token traces at any size), the official rows.
# Needs both cards: takes both card locks. One JSON line to $RXM_WORK/pubmix.jsonl, per-row scores in $RXM_WORK/fullscores/.
: ${RXM_WORK:=/root} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast} ${RXM_MIX:=/data/mix}; export RXM_WORK RXM_KLDBIN RXM_MIX   # paths (README: Paths)
cd ${RXM_WORK}
T=$1 M=$2 X=${3:--}
exec 8>${RXM_WORK}/card0.lock 9>${RXM_WORK}/card1.lock; flock 8; flock 9
if [ "$X" != "-" ]; then ./hproxy/radmix ${RXM_MIX}/pub-$T.rad $M $X 2> ${RXM_WORK}/pub-$T.radmix.log || exit 1; M=${RXM_MIX}/pub-$T.rad; fi
w=$(RHOME=${RXM_WORK}/dev-home TP=2 CARD=0,1 HEADROOM=8192 BIN=${RXM_KLDBIN} ./gold2.sh $M golden/public corpus/public.jsonl \
    ${RXM_WORK}/pub-$T bf16 262144 --max-num-batched-tokens 4096)
python3 -c "
import json; r=json.load(open('${RXM_WORK}/pub-$T.json'))
print(json.dumps({'tag':'$T','bytes':$(stat -c %s $M),'run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'p99_9':r['kld']['p99_9'],'top1':r['top1_agreement'],'long_agentic':r.get('kld_mean_long_agentic')}))" >> ${RXM_WORK}/pubmix.jsonl \
  || echo "{\"tag\": \"$T\", \"run\": \"$w\", \"fail\": 1}" >> ${RXM_WORK}/pubmix.jsonl
[ -f ${RXM_WORK}/pub-$T/score.f32 ] && mv ${RXM_WORK}/pub-$T/score.f32 ${RXM_WORK}/fullscores/$T.f32
rm -rf ${RXM_WORK}/pub-$T ${RXM_MIX}/pub-$T.rad
