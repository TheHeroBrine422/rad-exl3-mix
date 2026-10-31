#!/bin/bash
# gateA.sh -- after calib-B verification: the same corrected mixes + reference containers on calib half A (the half
# the sensitivities were fitted on), to separate the knapsack model's own error from the A/B slice difference.
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final}; export RXM_WORK RXM_MODELS   # paths (README: Paths)
cd ${RXM_WORK}
V=evq/queue-verify.txt; QA=evq/queue-verifyA.txt
until [ -z "$(awk '{print $1}' $V | while read t; do [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done)" ]; do sleep 60; done
: > $QA
for m in knap/mixes/knap-fix-*.mix; do n=$(basename $m .mix); echo "a-${n} ${RXM_MODELS}/q38-27b-sc4.00-e4.rad ${RXM_WORK}/$m" >> $QA; done
for t in sc2.00-e3 plain2.00-e3 plain2.50-e3 sc3.00-e4 plain3.50-e4 plain4.00-e4 sc5.00-e5 sc6.00-e6; do
  echo "a-ref-$t ${RXM_MODELS}/q38-27b-$t.rad -" >> $QA; done
echo END >> $QA
for p in $(pgrep -f "^/bin/bash ./hproxy/evalq.sh [01] evq/queue-verify.txt"); do kill $p; done
nohup ./hproxy/evalq.sh 0 $QA golden-calA corpus/calA.jsonl > evq/worker0a.log 2>&1 < /dev/null &
nohup ./hproxy/evalq.sh 1 $QA golden-calA corpus/calA.jsonl > evq/worker1a.log 2>&1 < /dev/null &
echo "$(date +%H:%M) calib-B verification done; calib-A check of the same mixes + refs started" >> ourchain.log
# when the calib-A pass is scored: the report (both halves, bootstrap intervals)
until [ -z "$(awk "\$1!=\"END\" {print \$1}" evq/queue-verifyA.txt | while read t; do [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done)" ]; do sleep 60; done
python3 -I knap/abreport.py > knap/abreport.txt 2>&1
echo "$(date +%H:%M) A/B report: ${RXM_WORK}/knap/abreport.txt" >> ourchain.log
