#!/bin/bash
# lowprelim.sh -- preliminary low-K head curve from the donors that exist (K2, K3, K3.5, K4): uniform u2-u4 and slab
# plans with every floor >= K2, budgets 2.0-3.5 at 1/8 steps, priced by headsim. Log /data/yaqa/headslab2/alone.log.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/alone.log; }
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
exec 9>$RXM_WORK/card0.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7
HS_TRAIN=qad HS_FLOOR=2 HS_FLOOR_ALL=1 HS_KS=2,3,3.5,4 HS_BUDGETS=$(bud 2 3.5) \
  $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/lcurve-plan.json >> $H/alone.log 2>&1
ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/lcurve-plan.json $CK $H/lcurve.json > $H/lcurve.json.log 2>&1
[ -s $H/lcurve.json ] && log "low prelim curve priced" || log "low prelim headsim FAIL"
