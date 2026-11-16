#!/bin/bash
# wait-s4seed.sh (w187, #200): S4seed = mean of the four 1 M seeds (D0 = #191's L0B0T0Q0D0 (w170), A-s1 (w186), s2, s3):
# waits for their .done markers, then soup200.sh on card 2 (the soup's GPU step takes card 2's lock like a scoring step).
cd /data/q27/w187 && source env.sh
M1=/data/q27/w170/runs/L0B0T0Q0D0/ckpt-1e+06 M2=/data/q27/w186/runs/A-s1/ckpt-1e+06
M3=$W/runs/s2/ckpt-1e+06 M4=$W/runs/s3/ckpt-1e+06
for m in $M1 $M2 $M3 $M4; do while [ ! -e $m/.done ]; do sleep 300; done; done
./code/v3/qad200/soup200.sh S4seed 2 $M1 $M2 $M3 $M4
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') wait-s4seed done rc $?" >> $W/q200.log
