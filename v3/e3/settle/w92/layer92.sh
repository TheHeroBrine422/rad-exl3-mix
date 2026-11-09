#!/bin/bash
# layer92.sh OD JD NAME... -- w92.sh's job runner (it holds card0.lock + ram.lock shared around this): the named rxq jobs
# one after another, each as rad-quant-at-home's rxq_quant worker runs it (`python -m rxq job SPEC -o OUT --device 0`,
# PYTHONPATH = pipeline + fork, RXQ_REFIT_CPU_FALLBACK=1), under swap27's memguard. Env from w92.sh: PY W T GUARD_MB
# CODE EXL3. A tensor saved but not logged by an earlier killed try is removed first (clean92.py, as rqah's run()).
set -u
od=$1 jd=$2; shift 2
for n in "$@"; do o=$od/$n; mkdir -p $o
  $PY -I $W/clean92.py $o || exit 1
  s=$(date +%s%N)
  $T/memguard.sh $GUARD_MB nice -n 5 env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src \
    RXQ_REFIT_CPU_FALLBACK=1 $PY -m rxq job $jd/$n.json -o $o --device 0 >> $o/rxq.log 2>&1 || { echo "JOBFAIL $n (rc $?)"; exit 1; }
  echo "$n $(( ($(date +%s%N) - s) / 1000000 ))" >> $od/wall-ms.txt
done
