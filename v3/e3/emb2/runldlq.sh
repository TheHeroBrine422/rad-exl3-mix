#!/bin/bash
# runldlq.sh -- the #119 embedding LDLQ screen on colo: H in {I, M0, F} x K in {1, 1.5, 2, 3}, MAXJ jobs at once, THR
# threads each; skips arms whose output exists (restartable). Outputs out/ldlq-<H>-K<k>.jsonl, logs out/ldlq-*.log.
# Caveat: it skips finished arms only, not running ones: restart it only after its arms have exited (11-08: a restart
# while K2 ran launched two duplicates, killed by hand).
cd /data/work/w13/w83
MAXJ=${MAXJ:-6}; THR=${THR:-2}
for K in ${KS:-2 3 1 1.5}; do for H in ${HS:-I M0 F}; do
  o=out/ldlq-$H-K$K.jsonl; [ -s $o ] && continue
  while [ $(pgrep -fc '^python3 code/embldlq.py') -ge $MAXJ ]; do sleep 30; done
  OMP_NUM_THREADS=$THR nohup nice -n 19 python3 code/embldlq.py $H $K $o > out/ldlq-$H-K$K.log 2>&1 < /dev/null &
  sleep 5
done; done
wait; echo ALLDONE >> out/runldlq.done
