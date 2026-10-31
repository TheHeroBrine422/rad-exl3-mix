#!/bin/bash
# evalq.sh CARD QUEUE GOLDEN CORPUS -- a worker: for each queue line "TAG BASE.rad MIXFILE" (MIXFILE may be "-"),
# build the mix by reflink (radmix), score it on GOLDEN/CORPUS on CARD, keep the per-row scores in
# $RXM_WORK/evq/scores/TAG.f32 and one line in $RXM_WORK/evq/results.jsonl. Tags are claimed (mkdir), so any number of
# workers share a queue; lines appended to QUEUE later are picked up. Exits on a line "END".
: ${RXM_WORK:=/root} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast} ${RXM_MIX:=/data/mix}; export RXM_WORK RXM_KLDBIN RXM_MIX   # paths (README: Paths)
C=$1 Q=$2 G=$3 CO=$4
cd ${RXM_WORK}; mkdir -p evq/claim evq/scores ${RXM_MIX}
export HEADROOM=${HEADROOM:-16384}
export RADIANCE_KLD_MAXTOK=${MAXTOK:-8192}
exec 9>${RXM_WORK}/card$C.lock
while true; do
  got=""
  while read -r tag base mix; do
    [ "$tag" = END ] && { [ -z "$got" ] && exit 0; break; }
    [ -z "$tag" ] || [ "${tag:0:1}" = "#" ] && continue
    [ -f evq/scores/$tag.f32 ] && continue
    # claim under a per-queue lock: mkdir alone raced on the overlay filesystem (two cards ran one tag)
    exec 8>>$Q.claimlock; flock 8
    if [ -f evq/scores/$tag.f32 ] || ! mkdir evq/claim/$tag 2>/dev/null; then flock -u 8; continue; fi
    flock -u 8
    got=1
    m=${RXM_MIX}/$tag.rad
    if [ "$mix" = "-" ]; then : > ${RXM_MIX}/$tag.empty; mix=${RXM_MIX}/$tag.empty; fi
    ./hproxy/radmix $m $base $mix 2> evq/claim/$tag/radmix.log || { echo "{\"tag\": \"$tag\", \"fail\": \"radmix\"}" >> evq/results.jsonl; continue; }
    flock 9; exec 7>${RXM_WORK}/ram.lock; flock -s 7   # card, then host RAM shared (a head quantization takes it whole)
    w=$(CARD=$C BIN=${RXM_KLDBIN} ./gold2.sh $m $G $CO ${RXM_WORK}/evq/run-$tag bf16 16384 --max-num-batched-tokens 4096)
    flock -u 7; flock -u 9
    if [ -f evq/run-$tag/score.f32 ] && python3 -I aggscore.py $G evq/run-$tag/score.f32 > evq/claim/$tag/agg.json 2>/dev/null; then
      mv evq/run-$tag/score.f32 evq/scores/$tag.f32
      python3 -c "
import json; r=json.load(open('evq/claim/$tag/agg.json'))
print(json.dumps({'tag':'$tag','card':$C,'run':'$w','bytes':$(stat -c %s $m),'positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'top1':r['top1_agreement']}))" >> evq/results.jsonl
    else
      echo "{\"tag\": \"$tag\", \"run\": \"$w\", \"fail\": \"score\"}" >> evq/results.jsonl
      cp evq/run-$tag.log evq/claim/$tag/ 2>/dev/null
    fi
    rm -rf evq/run-$tag evq/run-$tag.log evq/run-$tag.json evq/run-$tag.err $m
  done < $Q
  [ -z "$got" ] && sleep 30
done
