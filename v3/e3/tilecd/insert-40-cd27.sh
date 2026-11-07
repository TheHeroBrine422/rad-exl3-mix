#!/bin/bash
# #77 tile CD (cd3) on attn_qg: the 27B "does no harm" pairs (w50, v3/e3/settle/CD27.md). Runs as a q2.sh insert,
# before the next arm. ~1.3 h if both no-op gates pass (card 0 ~25 min: 2 x (noop + cd quant of 16 q_proj + score);
# CPU ~55 min: 2 packs under conv.lock), ~2.3 h if they fail (a control pack + score per arm). Idempotent: a rerun resumes.
cd /data/cd27 || exit 1
for p in "s1 /data/swap27/q-s1/K2.5" "base /data/v3ladder2/K2.5"; do set -- $p
  ARM=$1 BASE=$2 CTL=$1 bash /data/cd27/code/cd27.sh >> /data/cd27/run.out 2>&1 < /dev/null
  rc=$?; [ $rc = 0 ] && echo "$(date -u +%T) cd27 $1 done" >> /data/q1106/q.log || echo "$(date -u +%T) cd27 $1 FAIL exit $rc (/data/cd27/$1/cd27.log)" >> /data/q1106/q.log
done
/data/venv-exl3/bin/python -I /data/cd27/code/cd27verdict.py /data/cd27 s1 base >> /data/cd27/result.txt 2>&1
echo "$(date -u +%T) cd27 verdict written (/data/cd27/result.txt)" >> /data/q1106/q.log
