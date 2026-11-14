#!/bin/bash
# q8.sh -- card 1, after q7: is the mad decode's KS counter sharing a race? Greedy decode with logprobs (gdec) on both mad
# packs: mbase again (mbase2: run-to-run), mfix (qc-exl3-v3 + kctr-key.patch: counters keyed by (stream, output)), mord
# (qc-exl3-v3 with exl3i's rad_kernel_concurrent 0: every launch ordered, the race-free reference). (w152)
W=/data/q27/w152; cd $W; Q=/data/q27
while pgrep -f "^bash tools/q[4-7][.]sh" > /dev/null; do sleep 20; done
while [ ! -f b10mord/home/.ready ] && pgrep -f "plugb.sh 10 /data/q27/w152/src-mord" > /dev/null || [ ! -d b10mord/home ]; do sleep 20; done
for m in $Q/w150/packs/q27v3-635bac02-k2.5-mad-a8.rad $Q/w150/packs/q27v3-144dd6e1-k4.125-mad-a8.rad; do
  bash tools/gdec.sh 1 $m mbase2 $W/b10mbase/home
  bash tools/gdec.sh 1 $m mfix $W/b10mfix/home
  bash tools/gdec.sh 1 $m mord $W/b10mord/home
done
echo "$(date -u +%T) q8 DONE" >> logs/q.log
