#!/bin/bash
# retry45.sh -- K4.5 head segfaulted in refit_scales (hipLaunchKernel, 2026-11-01 21:44 UTC; K3.5 fine): one retry after
# the queue, alone; on success curve A is re-planned and re-priced with it. A second segfault = deterministic: debug.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/alone.log; }
while pgrep -f "^/bin/bash ./(halfcurve|lowcurve|lowcurve2).sh" > /dev/null; do sleep 20; done
cp $H/K4.5.log $H/K4.5-segv1.log 2>/dev/null
$Y/headalone.sh 4.5
[ -f $H/K4.5/wq/lm_head.rowerr.npz ] || { log "K4.5 retry FAILED again: deterministic, needs debugging"; exit 1; }
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
exec 9>/root/card0.lock; flock 9
HS_TRAIN=qad HS_FLOOR=3 HS_KS=3,3.5,4,4.5,5,5.5,6 HS_BUDGETS=$(bud 3.5 5.5) \
  $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-A-plan.json >> $H/alone.log 2>&1
ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/curve-A-plan.json $CK $H/curve-A.json > $H/curve-A.json.log 2>&1
[ -s $H/curve-A.json ] && log "curve A re-priced with K4.5" || log "curve A re-price FAIL"
