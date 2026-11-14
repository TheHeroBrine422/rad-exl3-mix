#!/bin/bash
# q1.sh -- first after-queue on card 1: e2e round 2 (base, step interleaved), KLD rows base / step (def, c64) on both
# packs, the step home's profile queue. (w152)
W=/data/q27/w152; cd $W
ROWS="base=$W/b10base/home step=$W/b10step/home" bash tools/e2.sh 1 2 v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8
for p in v3-k4.125-r16k4-a8 v3-k2.5-r16k2.5-a8; do bash tools/kld.sh 1 $p base-def step-def base-c64 step-c64 >> logs/kld.log 2>&1; done
bash tools/profq.sh 1 step $W/b10step/home
echo "$(date -u +%T) q1 DONE" >> logs/q.log
