#!/bin/bash
# noise.sh [NSEQ] -- bf16 vs fp32 backward on the same NSEQ (default 8) qad sequences: rows of both kept in fp32,
# labels shared (the fp32 run samples them), then cmp.py. Waits for gate2.sh to finish (both cards).
Y=/data/yaqa N=${1:-8} PY=/data/venv-exl3/bin/python
cd $Y
until grep -q "gate2 done" gate2.log 2>/dev/null; do sleep 30; done
run() {  # OUT [env...]
  local o=$1; shift
  ( exec 8>/root/card0.lock 9>/root/card1.lock; flock 8; flock 9
    env YAQA_ROWS_F32=1 YAQA_START=0 "$@" $PY -I collect.py $o $N 2048 32 ) > $o.log 2>&1
}
rm -rf $Y/n32 $Y/n16
run $Y/n32 YAQA_FP32=1 && run $Y/n16 YAQA_LABELS=$Y/n32 || { echo "noise run FAIL" >> noise.log; exit 1; }
grep "^seq" n32.log n16.log | tail -4 >> noise.log
$PY -I cmp.py $Y/n32 $Y/n16 >> noise.log 2>&1
echo "noise done" >> noise.log
