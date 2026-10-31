#!/bin/bash
# evalq2.sh CARD QUEUE -- eval worker for the overnight pipeline. Queue lines: "TAG BASE.rad MIX|- GOLDEN CORPUS".
# Before every job it scans QUEUE.prio first, then QUEUE (both re-read each time, so lines can be added anywhere).
# Tags are claimed under a lock (mkdir evq/claim/TAG); scored ones are skipped. A mix bigger than TP2_BYTES (default 26e9,
# about 7.7 bpw) runs at --tp 2 on both cards (locks card0 then card1, always in that order, so no cycle with the other
# worker). Exits when QUEUE has an END line and nothing is left; otherwise waits for work. Results as evalq.sh.
: ${RXM_WORK:=/root} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast} ${RXM_MIX:=/data/mix}; export RXM_WORK RXM_KLDBIN RXM_MIX   # paths (README: Paths)
C=$1 Q=$2
cd ${RXM_WORK}; mkdir -p evq/claim evq/scores ${RXM_MIX}
export HEADROOM=${HEADROOM:-16384} RADIANCE_KLD_MAXTOK=${MAXTOK:-8192}
TP2_BYTES=${TP2_BYTES:-26000000000}
touch $Q.prio
declare -A DEFER                       # TP2 tags waiting for card 1 (quantizer holds it): skipped until the time
next() {   # prints the first unclaimed, unscored line of Q.prio then Q; claims it
  exec 8>>$Q.claimlock; flock 8
  local f tag rest
  for f in $Q.prio $Q; do
    while read -r tag rest; do
      [ -z "$tag" ] || [ "${tag:0:1}" = "#" ] || [ "$tag" = END ] && continue
      [ -f evq/scores/$tag.f32 ] && continue
      [ -n "${DEFER[$tag]}" ] && [ $(date +%s) -lt ${DEFER[$tag]} ] && continue
      mkdir evq/claim/$tag 2>/dev/null || continue
      echo "$tag $rest"; flock -u 8; return 0
    done < $f
  done
  flock -u 8; return 1
}
while true; do
  if ! line=$(next); then
    grep -qx END $Q && exit 0
    sleep 30; continue
  fi
  read -r tag base mix G CO <<< "$line"
  m=${RXM_MIX}/$tag.rad
  if [ "$mix" = "-" ]; then : > ${RXM_MIX}/$tag.empty; mix=${RXM_MIX}/$tag.empty; fi
  ./hproxy/radmix $m $base $mix 2> evq/claim/$tag/radmix.log || { echo "{\"tag\": \"$tag\", \"fail\": \"radmix\"}" >> evq/results.jsonl; rm -f $m; continue; }
  if [ $(stat -c %s $m) -gt $TP2_BYTES ]; then
    exec 5>${RXM_WORK}/card0.lock 6>${RXM_WORK}/card1.lock; flock 5
    if ! flock -w 10 6; then           # card 1 busy (quantizing): put it back, try again in 5 min, do other work
      flock -u 5; rm -rf evq/claim/$tag $m; DEFER[$tag]=$(( $(date +%s) + 300 )); continue
    fi
    exec 7>${RXM_WORK}/ram.lock; flock -s 7
    w=$(TP=2 CARD=0,1 RHOME=${RXM_WORK}/dev-home BIN=${RXM_KLDBIN} ./gold2.sh $m $G $CO ${RXM_WORK}/evq/run-$tag bf16 16384)
    card=01
    flock -u 7; flock -u 6; flock -u 5
  else
    exec 5>${RXM_WORK}/card$C.lock; flock 5; exec 7>${RXM_WORK}/ram.lock; flock -s 7
    # pack documents into steps below 20 GB (2026-10-30: calib-A 92 -> 85 s; KLD within 0.005%, not byte-identical)
    if [ $(stat -c %s $m) -lt 20000000000 ]; then PK="--max-num-batched-tokens 4096 --prefill-span 2"; S=8; else PK=""; S=1; fi
    w=$(SEQS=$S CARD=$C BIN=${RXM_KLDBIN} ./gold2.sh $m $G $CO ${RXM_WORK}/evq/run-$tag bf16 16384 $PK)
    card=$C
    flock -u 7; flock -u 5
  fi
  if [ -f evq/run-$tag/score.f32 ] && python3 -I aggscore.py $G evq/run-$tag/score.f32 > evq/claim/$tag/agg.json 2>/dev/null; then
    mv evq/run-$tag/score.f32 evq/scores/$tag.f32
    python3 -c "
import json; r=json.load(open('evq/claim/$tag/agg.json'))
print(json.dumps({'tag':'$tag','card':'$card','run':'$w','bytes':$(stat -c %s $m),'golden':'$G','positions':r['positions'],'kld':r['kld']['mean'],'p99':r['kld']['p99'],'top1':r['top1_agreement']}))" >> evq/results.jsonl
  else
    echo "{\"tag\": \"$tag\", \"run\": \"$w\", \"fail\": \"score\"}" >> evq/results.jsonl
    cp evq/run-$tag.log evq/claim/$tag/ 2>/dev/null
  fi
  rm -rf evq/run-$tag evq/run-$tag.log evq/run-$tag.json evq/run-$tag.err $m
done
