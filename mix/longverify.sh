#!/bin/bash
# longverify.sh TAG MODEL.rad [MIXFILE] -- the long-context check of one finalist (2026-10-31): the two verify-pool
# 250k agent traces (calib 905 rich, 2675 express; golden-long2), whole, TP2. Was pubverify.sh: the whole public corpus, every
# document whole (TP2 over both cards: the KV pool holds the 245k-token traces at any size), the official rows.
# Needs both cards: takes both card locks. One JSON line to $RXM_WORK/longmix.jsonl, per-row scores in $RXM_WORK/fullscores/.
: ${RXM_WORK:=/root} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast} ${RXM_MIX:=/data/mix}; export RXM_WORK RXM_KLDBIN RXM_MIX   # paths (README: Paths)
cd ${RXM_WORK}; mkdir -p longscores
T=$1 M=$2 X=${3:--}
# LV_TP=1 LV_CARD=N (2026-10-31): one card (mixes whose 262144-token bf16 KV fits beside the weights); default TP2
if [ "${LV_TP:-2}" = 1 ]; then exec 8>${RXM_WORK}/card${LV_CARD:-1}.lock; flock 8
else exec 8>${RXM_WORK}/card0.lock 9>${RXM_WORK}/card1.lock; flock 8; flock 9; fi
if [ "$X" != "-" ]; then ./hproxy/radmix ${RXM_MIX}/long-$T.rad $M $X 2> ${RXM_WORK}/long-$T.radmix.log || exit 1; M=${RXM_MIX}/long-$T.rad; fi
w=$(RHOME=${RXM_WORK}/dev-home TP=${LV_TP:-2} CARD=${LV_CARD:-0,1} HEADROOM=8192 BIN=${RXM_KLDBIN} ./gold2.sh $M golden-long2 corpus/long2.jsonl \
    ${RXM_WORK}/long-$T bf16 262144 --max-num-batched-tokens 4096)
python3 -c "
import json; r=json.load(open('${RXM_WORK}/long-$T.json'))
print(json.dumps({'tag':'$T','bytes':$(stat -c %s $M),'run':'$w','positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'p99_9':r['kld']['p99_9'],'top1':r['top1_agreement'],'long_agentic':r.get('kld_mean_long_agentic')}))" >> ${RXM_WORK}/longmix.jsonl \
  || echo "{\"tag\": \"$T\", \"run\": \"$w\", \"fail\": 1}" >> ${RXM_WORK}/longmix.jsonl
[ -f ${RXM_WORK}/long-$T/score.f32 ] && mv ${RXM_WORK}/long-$T/score.f32 ${RXM_WORK}/longscores/$T.f32
rm -rf ${RXM_WORK}/long-$T ${RXM_MIX}/long-$T.rad
