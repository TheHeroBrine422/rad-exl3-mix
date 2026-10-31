#!/bin/bash
# retryhead.sh K -- one head that failed (GPU OOM from fragmentation): retried with expandable segments after card 0's
# head loop, under the same locks (card 0, then ram.lock exclusive), then converted and queued for its swap eval.
: ${RXM_WORK:=/root} ${RXM_VENV:=/data/venv-exl3} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_VENV RXM_OURS   # paths (README: Paths)
K=$1; cd ${RXM_WORK}; . ${RXM_VENV}/bin/activate
export ROCR_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q -E "\[c0\] head K3 (queued|quantize FAIL|conv FAIL)" ourchain.log; do sleep 60; done
( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock 7
  python -I ourq/ourq.py -K $K -o ${RXM_OURS}/K$K --only-head 2>&1 | grep -v -e Warning -e "cpu = _conv" >> ${RXM_WORK}/ourq-K$K.log; exit ${PIPESTATUS[0]} ) \
  && ./ourq/ourconv.sh ${RXM_OURS}/hd-K$K.rad sc4.00-e4 ${RXM_OURS}/K$K output.weight >> ourchain.log 2>&1 \
  && echo "head-sc4.00-e4-ours-K$K ${RXM_OURS}/hd-K$K.rad -" >> evq/queue-cal.txt \
  && echo "$(date +%H:%M) [retry] head K$K queued" >> ourchain.log || echo "$(date +%H:%M) [retry] head K$K FAIL again" >> ourchain.log
