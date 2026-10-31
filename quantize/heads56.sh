#!/bin/bash
# heads56.sh -- after round 1 (queue-r1) is scored: our K6 and K5 heads (they OOMed before the refit_scales fix),
# each quantized under card0 + ram.lock exclusive, converted into a head-swap container on sc4.00-e4, proxy-scored
# (head row only -> hproxy-ours.jsonl), then both head-swap evals on calib-A on both cards.
: ${RXM_WORK:=/root} ${RXM_CKPT:=/data/workspace/models/qwen38-27b-bf16} ${RXM_VENV:=/data/venv-exl3} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_CKPT RXM_VENV RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}; . ${RXM_VENV}/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%H:%M) [heads56] $*" >> ourchain.log; }
pending() { awk '$1!="END"{print $1}' evq/queue-r1.txt | while read t; do [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done; }
until [ -z "$(pending)" ]; do sleep 60; done
log "round 1 done, starting heads K6 K5"
Q=evq/queue-heads56.txt; : > $Q
for K in 6 5; do
  ( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock 7
    ROCR_VISIBLE_DEVICES=0 python -I ourq/ourq.py -K $K -o ${RXM_OURS}/K$K --only-head 2>&1 | grep -v -e Warning -e "cpu = _conv" >> ${RXM_WORK}/ourq-K$K.log; exit ${PIPESTATUS[0]} ) \
  || { log "head K$K quantize FAIL"; continue; }
  ./ourq/ourconv.sh ${RXM_OURS}/hd-K$K.rad sc4.00-e4 ${RXM_OURS}/K$K output.weight >> ourchain.log 2>&1 || { log "head K$K conv FAIL"; continue; }
  ( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock -s 7
    ROCR_VISIBLE_DEVICES=0 HP_FILTER=output.weight OMP_NUM_THREADS=8 ./hproxy/hproxy ${RXM_CKPT} \
      challenge/hessians/calib /tmp/hp-hd$K.jsonl ours-K$K=${RXM_OURS}/hd-K$K.rad > hproxy-hd$K.log 2>&1 ) \
    && grep '"w": "output.weight"' /tmp/hp-hd$K.jsonl >> hproxy-ours.jsonl && rm -f /tmp/hp-hd$K.jsonl || log "head K$K hproxy FAIL"
  echo "head-sc4.00-e4-ours-K$K ${RXM_OURS}/hd-K$K.rad -" >> $Q
  log "head K$K done"
done
echo END >> $Q
for c in 0 1; do nohup ./hproxy/evalq.sh $c $Q golden-calA corpus/calA.jsonl > evq/worker${c}h56.log 2>&1 < /dev/null & done
wait
log "head K6/K5 swap evals done"
