#!/bin/bash
# tile CD + H_out on attn_qg at K4: the 27B "does no harm" pairs (w76, pipeline/tools/CDHOUT27.md), as a card-0 queue
# insert (the lead schedules it; same pattern as v3/e3/tilecd/insert-40-cd27.sh). Arms bn4 then bn4s1, then the verdict.
# ~3-4 h of card 0 if both no-op gates pass (H_out quant of 16 q_proj x 2 runs x 2 arms + 2 scores), ~1 h of CPU (two
# packs under conv.lock + ram.lock exclusive). Idempotent: a rerun resumes. Q = the queue's log (default /data/q1106/q.log).
Q=${Q:-/data/q1106/q.log}
cd /data/cdhout27 || exit 1
for p in "bn4 /data/swap27/q-bn4/K4" "bn4s1 /data/swap27/q-bn4s1/K4"; do set -- $p
  ARM=$1 BASE=$2 bash /data/cdhout27/code/pipeline/tools/cdhout27.sh >> /data/cdhout27/run.out 2>&1 < /dev/null
  rc=$?; [ $rc = 0 ] && echo "$(date -u +%T) cdhout27 $1 done" >> $Q || echo "$(date -u +%T) cdhout27 $1 FAIL exit $rc (/data/cdhout27/$1/cdhout27.log)" >> $Q
done
/data/venv-exl3/bin/python -I /data/cdhout27/code/pipeline/tools/cdhout27verdict.py /data/cdhout27 bn4 bn4s1 >> /data/cdhout27/result.txt 2>&1
echo "$(date -u +%T) cdhout27 verdict written (/data/cdhout27/result.txt)" >> $Q
