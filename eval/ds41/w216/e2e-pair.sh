#!/bin/bash
# w216: engine_e2e_test's preemption_by_recompute case on the pre-cleanup build (w213) and on ours, card3 under its lock
L=/data/q27/w216/logs
exec {fd}> /data/q27/locks/card3.lock; flock -w 1200 $fd || { echo "card3 busy"; exit 0; }
export HIP_VISIBLE_DEVICES=3
for t in w213 w216; do
  cd /data/q27/$t/src/build && timeout 900 ./tests/engine_e2e_test preemption_by_recompute_continues_each_stream_exactly > $L/t-e2e-preempt-$t.log 2>&1
  echo "$t RC=$? $(grep -E 'ok |FAIL|bracket' $L/t-e2e-preempt-$t.log | tr '\n' ' ' | cut -c1-300)"
done
echo PAIR_DONE
