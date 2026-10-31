#!/bin/bash
# aftercal.sh -- once the calibration queue (and the head swaps the quantizer chain queues) is scored: solve the
# knapsack on the GPU box ($RXM_WORK/knap), then verify on calib half B (golden-calB, never used for fitting): every frontier
# mix plus the reference containers, on card 1 (queue $RXM_WORK/evq/queue-verify.txt, which later stages keep extending).
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final}; export RXM_WORK RXM_MODELS   # paths (README: Paths)
cd ${RXM_WORK}
Q=evq/queue-cal.txt
pending() { awk '$1 != "END" && $1 !~ /^#/ && NF==3 {print $1}' $Q | while read t; do
  [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done; }
until grep -q -E "head K3 (queued|quantize FAIL|conv FAIL)" ourchain.log && [ -z "$(pending)" ]; do sleep 120; done
echo END >> $Q
mkdir -p knap; cp hproxy/knap.py knap/
ln -sf ${RXM_WORK}/hproxy-c0.jsonl knap/hproxy-c0.jsonl; ln -sf ${RXM_WORK}/hproxy-c1.jsonl knap/hproxy-c1.jsonl
cp ${RXM_WORK}/radls.jsonl knap/; cp evq/results.jsonl knap/evq-results.jsonl
python3 -I knap/knap.py knap 2.0,2.25,2.5,2.75,2.99,3.25,3.5,3.75,3.99,4.5,5.0,5.49,6.0 > knap/frontier.txt 2> knap/knap.err
V=evq/queue-verify.txt
for m in knap/mixes/*.mix; do n=$(basename $m .mix); echo "v-$n ${RXM_MODELS}/q38-27b-sc4.00-e4.rad ${RXM_WORK}/$m" >> $V; done
for t in sc2.00-e3 plain2.00-e3 plain2.50-e3 sc3.00-e4 plain3.50-e4 sc4.00-e4 plain4.00-e4 sc5.00-e5 sc6.00-e6; do
  echo "v-ref-$t ${RXM_MODELS}/q38-27b-$t.rad -" >> $V; done
# card 1's calibration worker exits at END; this one takes over and keeps polling for more
while pgrep -f "evalq.sh 1 evq/queue-cal.txt" > /dev/null; do sleep 30; done
nohup ./hproxy/evalq.sh 1 $V golden-calB corpus/calB.jsonl > evq/worker1v.log 2>&1 < /dev/null &
echo "aftercal: $(ls knap/mixes | wc -l) mixes queued for verification $(date +%H:%M)" >> ourchain.log
