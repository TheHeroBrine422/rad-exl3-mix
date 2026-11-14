#!/bin/bash
# q2.sh -- after q1 on card 1: base's 8k prefill profile, decode numerics (gdec) base / base again / step / step2 on both
# packs, KLD step2 (def, c64), e2e round 3 (base, step, step2). (w152)
W=/data/q27/w152; cd $W
while pgrep -f "^bash tools/q1.sh" > /dev/null; do sleep 20; done
UNITS=p8 bash tools/profq.sh 1 base $W/b10base/home
for p in v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8; do
  bash tools/gdec.sh 1 $p base $W/b10base/home; bash tools/gdec.sh 1 $p base2 $W/b10base/home
  bash tools/gdec.sh 1 $p step $W/b10step/home; bash tools/gdec.sh 1 $p step2 $W/b10step2/home
done
for p in v3-k4.125-r16k4-a8 v3-k2.5-r16k2.5-a8; do bash tools/kld.sh 1 $p step2-def step2-c64 >> logs/kld.log 2>&1; done
ROWS="base=$W/b10base/home step=$W/b10step/home step2=$W/b10step2/home" bash tools/e2.sh 1 3 v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8
echo "$(date -u +%T) q2 DONE" >> logs/q.log
