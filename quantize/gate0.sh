#!/bin/bash
# gate0.sh -- card 0 after calibration: verify the knapsack frontier on calib half B beside card 1, then quantize
# full models (NOHEADS: each full model quantizes its own head).
: ${RXM_WORK:=/root} ${RXM_VENV:=/data/venv-exl3}; export RXM_WORK RXM_VENV   # paths (README: Paths)
cd ${RXM_WORK}
V=evq/queue-verify.txt
until [ -f $V ]; do sleep 60; done
nohup ./hproxy/evalq.sh 0 $V golden-calB corpus/calB.jsonl > evq/worker0v.log 2>&1 < /dev/null &
until [ -z "$(awk '$1 ~ /^v-(knap|ref)-/ {print $1}' $V | while read t; do
        [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done)" ]; do sleep 120; done
echo "$(date +%H:%M) [c0] frontier verification scored; profiling the quantizer (layer 0, K=4 and 8)" >> ourchain.log
( . ${RXM_VENV}/bin/activate; export ROCR_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec 9>${RXM_WORK}/card0.lock; flock 9
  for K in 4 8; do python -I ourq/qprof.py $K 0 ${RXM_WORK}/qprof-K$K.json > ${RXM_WORK}/qprof-K$K.log 2>&1; done )
echo "$(date +%H:%M) [c0] profile in ${RXM_WORK}/qprof-K{4,8}.json; card 0 resumes quantization" >> ourchain.log
NOHEADS=1 exec ./ourq/ourchain2.sh 0
