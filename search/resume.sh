#!/bin/bash
# resume.sh -- after the GPU box restarts: release claims of evals that never finished, drop half-written outputs, and
# restart every stage of the queue (each one skips what is already done). Run as root on the GPU box.
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours} ${RXM_MIX:=/data/mix}; export RXM_WORK RXM_OURS RXM_MIX   # paths (README: Paths)
cd ${RXM_WORK}
pgrep -f "evalq.sh|ourchain2?.sh|aftercal.sh|oursverify.sh|radiance-kldgold|ourq.py|rad-convert" > /dev/null && {
  echo "something is still running; not resuming"; pgrep -af "evalq.sh|ourchain2?.sh|aftercal.sh|oursverify.sh|radiance-kldgold|ourq.py|rad-convert" | cut -c1-100; exit 1; }
# evals: a claim without a score or a result line was in flight
for d in evq/claim/*/; do t=$(basename $d)
  [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\"" evq/results.jsonl || { rm -rf $d evq/run-$t* ${RXM_MIX}/$t.rad; echo "released $t"; }
done
# a fail line from the crash itself would block nothing (claims decide), but a stale END must not stop the worker early
rm -f ${RXM_OURS}/*.part ${RXM_OURS}/K*/*.part
# ourchain2 skips full models that already have a container; finished ones are never redone
for f in ${RXM_OURS}/*.rad.part; do rm -f $f; done
# the quantizer's per-layer shards are whole files (written .part then renamed): ourq resumes past them
grep -q "^END$" evq/queue-cal.txt || nohup ./hproxy/evalq.sh 1 evq/queue-cal.txt golden-calA corpus/calA.jsonl > evq/worker1.log 2>&1 < /dev/null &
rm -rf ${RXM_WORK}/ourq/claim/K*   # in-flight full-model claims: ourq resumes per layer
nohup ./ourq/ourchain2.sh 0 > ${RXM_WORK}/ourchain2-0.out 2>&1 < /dev/null &
nohup ./ourq/ourchain2.sh 1 > ${RXM_WORK}/ourchain2-1.out 2>&1 < /dev/null &
if [ -f evq/queue-verify.txt ]; then
  nohup ./hproxy/evalq.sh 1 evq/queue-verify.txt golden-calB corpus/calB.jsonl > evq/worker1v.log 2>&1 < /dev/null &
else
  nohup ./hproxy/aftercal.sh > ${RXM_WORK}/aftercal.out 2>&1 < /dev/null &
fi
nohup ./hproxy/oursverify.sh > /dev/null 2>&1 < /dev/null &
echo "resumed $(date +%H:%M)" >> ourchain.log
