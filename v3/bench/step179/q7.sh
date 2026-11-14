#!/bin/bash
# q7.sh -- card 1, after q6: KLD rows of the rebased head (mstep3 = b73675c on qc-exl3-v3 9f0a059) on the mad K2.5 pack,
# ROCm 10.1 (vs mstep = b02aa9a, mbase) and the frozen 7.14 install (mbase-714 vs mstep3-714), def mode. (w152)
W=/data/q27/w152; cd $W; Q=/data/q27
while pgrep -f "^bash tools/q[456][.]sh" > /dev/null; do sleep 20; done
bash tools/kld.sh 1 $Q/w150/packs/q27v3-635bac02-k2.5-mad-a8.rad mstep3-def mbase-714-def mstep3-714-def >> logs/kld.log 2>&1
echo "$(date -u +%T) q7 DONE" >> logs/q.log
