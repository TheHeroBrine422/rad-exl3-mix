#!/bin/bash
# vtchain.sh -- after the first batch's preps: golden-ml, ml evals of the references and of the variants already
# queued, then the remaining calib-ranked plans and the guard-ranked plans (embtier-run.sh queues calA + ml evals).
: ${RXM_WORK:=/root}; export RXM_WORK
cd /data/vt
./vtmlslice.sh > ml-slice.log 2>&1 || exit 1
while ls emb/*.part > /dev/null 2>&1; do sleep 20; done
( exec 8>>$RXM_WORK/auto/q.claimlock; flock 8
  echo "vt-ml-base-bf16 /data/vt/emb/base.rad - golden-ml corpus/ml.jsonl" >> $RXM_WORK/auto/q.prio
  echo "vt-ml-ref-sc4.00-e4 $RXM_WORK/models/final/q38-27b-sc4.00-e4.rad - golden-ml corpus/ml.jsonl" >> $RXM_WORK/auto/q.prio
  for f in emb/vt-emb-*.rad; do t=$(basename $f .rad); echo "${t/vt-emb/vt-ml} /data/vt/$f - golden-ml corpus/ml.jsonl" >> $RXM_WORK/auto/q.prio; done )
./embtier-run.sh 100000:4,1 50000:4,1 10000:4,90000:3,1 100000:3,1 30000:4,2 30000:3,2 20000:4,30000:3,2
VT_RANK=rank-guard-counts.npy VT_PFX=g ./embtier-run.sh 100000:4,2 50000:4,2 30000:4,2 100000:4,1 50000:4,1
echo VTCHAINDONE >> emb/prep.log
