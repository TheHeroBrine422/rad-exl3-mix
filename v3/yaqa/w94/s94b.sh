#!/bin/bash
# s94b.sh -- (post hoc, report-only, after G2 failed) #125 per-stratum sums for the weighted joint estimate; else as s94.sh: on colo after q94.sh: head-only KL per dump row of every proxy arm (hq94.py score), 8 jobs
# (2 sets x 4 tag groups), single-threaded, nice 19. Waits for Q94DONE. Restartable (finished tag files are skipped).
cd /data/work/w13/w94 && . ./env.sh
until grep -q Q94DONE logs/q94.log; do sleep 30; done
H="nice -n 19 python3 -W ignore code/v3/yaqa/w94/hq94.py score2 out"
T=(real/one/K3 real/one/K4 raw/one/K3 raw/one/K4 mr/one/K3 mr/one/K4 mr/two:0.01/K3 mr/two:0.01/K4 mr/two:0.1/K3
   mr/two:0.1/K4 mr/two:1/K3 mr/two:1/K4 mr-r4/one/K3 mr-r4/one/K4 mr-r8/one/K3 mr-r8/one/K4)
for s in search qad; do
  for g in 0 1 2 3; do
    $H $s in/dump/$s in/sim-$s.npz ${T[@]:$((g * 4)):4} > logs/score2-$s-$g.log 2>&1 &
  done
done
wait
echo S94B DONE >> logs/q94.log
