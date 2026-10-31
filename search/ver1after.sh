#!/bin/bash
# after round 3: k2 mixes + the reference frontier on the fresh verify-pool slice golden-ver1, both cards
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final}; export RXM_WORK RXM_MODELS   # paths (README: Paths)
cd ${RXM_WORK}
until grep -q "round 3 done" ourchain.log; do sleep 60; done
Q=evq/queue-ver1.txt; : > $Q
for t in sc1.40-e3 sc1.60-e3 sc1.80-e3 sc2.00-e3 sc2.20-e3 plain2.50-e3 sc3.00-e4 plain3.00-e4 plain3.50-e4 sc4.00-e4 sc5.00-e5 sc6.00-e6; do
  echo "w-ref-$t ${RXM_MODELS}/q38-27b-$t.rad -" >> $Q; done
for m in knap/mixes/knap-k2-sc4.00-e4-*.mix; do n=$(basename $m .mix); echo "w-$n ${RXM_MODELS}/q38-27b-sc4.00-e4.rad ${RXM_WORK}/$m" >> $Q; done
echo END >> $Q
for c in 0 1; do nohup ./hproxy/evalq.sh $c $Q golden-ver1 corpus/ver1.jsonl > evq/worker${c}ver1.log 2>&1 < /dev/null & done
echo "$(date +%H:%M) verify-pool check (golden-ver1): 12 refs + 18 k2 mixes, tags w-*" >> ourchain.log
wait
echo "$(date +%H:%M) verify-pool check done" >> ourchain.log
