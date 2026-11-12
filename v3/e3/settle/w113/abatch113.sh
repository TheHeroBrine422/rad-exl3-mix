#!/bin/bash
# abatch113.sh -- w113 #148-01 driver: a113.sh's remaining units in batches of 4 per card hold (hold113.sh), then f2 and
# report (outside any card hold). Restartable (finished units are skipped).
S=/data/w113/code/v3/e3/settle/w113
U=(); for s in 0 1; do for K in 2.5 4; do for a in N P T V; do U+=($a:$K:$s); done; done; done
for i in 0 4 8 12; do bash $S/hold113.sh "${U[@]:$i:4}" || exit 1; [ -f /data/w113/a/STOP ] && exit 0; sleep 5; done
bash $S/a113.sh f2 report
