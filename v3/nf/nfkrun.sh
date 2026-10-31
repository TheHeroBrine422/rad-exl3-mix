#!/bin/bash
# nfkrun.sh (copy of e1brun.sh, RHOME default $RXM_WORK/nfk/home) TAG MODEL.rad GOLDEN CORPUS [VAR=VALUE...] -- one E1b eval: MODEL scored on GOLDEN with the E1b build
# ($RXM_WORK/e1b/home: the frozen install + the E1b exl3_r4d.so), the eval worker's packed settings (evalq2.sh, mixes
# < 20 GB), the extra env (EXL3_AQ_MASK, EXL3_AQ_GROUP, EXL3_AQ_CLIP). Waits for card 0 between evals (card lock,
# then ram.lock shared, as evalq2.sh). Score -> $RXM_WORK/e1b/scores/TAG.f32, one JSON line -> $RXM_WORK/e1b/results.jsonl.
: ${RXM_WORK:=/root} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast}; export RXM_WORK RXM_KLDBIN   # paths (README: Paths)
T=$1 M=$2 G=$3 CO=$4; shift 4
cd ${RXM_WORK}; mkdir -p e1b/scores e1b/runs
for kv in "$@"; do export "$kv"; done
export EXL3_AQ_LOG=${RXM_WORK}/e1b/runs/$T.map HEADROOM=16384 RADIANCE_KLD_MAXTOK=8192
rm -f $EXL3_AQ_LOG
exec 5>${RXM_WORK}/card0.lock; flock 5; exec 7>${RXM_WORK}/ram.lock; flock -s 7
s=$(date +%s)
w=$(SEQS=8 CARD=0 RHOME=${RHOME:-${RXM_WORK}/nfk/home} BIN=${RXM_KLDBIN} ./gold2.sh $M $G $CO ${RXM_WORK}/e1b/runs/$T bf16 16384 \
      --max-num-batched-tokens 4096 --prefill-span 2)
flock -u 7; flock -u 5
if [ -f e1b/runs/$T/score.f32 ] && python3 -I aggscore.py $G e1b/runs/$T/score.f32 > e1b/runs/$T.agg 2>/dev/null; then
  mv e1b/runs/$T/score.f32 e1b/scores/$T.f32
  python3 -c "
import json, sys; r = json.load(open('e1b/runs/$T.agg'))
print(json.dumps({'tag': '$T', 'model': '$M', 'golden': '$G', 'env': sys.argv[1:], 'run': '$w', 'wall': $(date +%s) - $s,
                  'kld': r['kld']['mean'], 'p99': r['kld']['p99'], 'top1': r['top1_agreement']}))" "$@" >> e1b/results.jsonl
  tail -1 e1b/results.jsonl
else
  echo "{\"tag\": \"$T\", \"fail\": \"score\", \"run\": \"$w\"}" | tee -a e1b/results.jsonl
fi
rm -rf e1b/runs/$T e1b/runs/$T.log e1b/runs/$T.json e1b/runs/$T.err
