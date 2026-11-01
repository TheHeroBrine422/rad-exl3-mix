#!/bin/bash
# lowcurve.sh -- low head donors for the < 2.5 bpw sizes (V2's 1.50 / 1.75 picked a K2 head): K2.25, 2.5, 2.75 alone
# (headalone.sh; K2 exists), then curve C: donors K2-4, slab tails allowed down to K2 (HS_FLOOR=2), budgets 2.0-3.5 at
# 1/8 steps, plus the uniform u2..u4; priced by headsim. Waits for halfcurve.sh. Log /data/yaqa/headslab2/alone.log.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/alone.log; }
while pgrep -f "^/bin/bash ./halfcurve.sh" > /dev/null; do sleep 20; done
for K in 2.25 2.5 2.75; do [ -f $H/K$K/wq/lm_head.rowerr.npz ] || $Y/headalone.sh $K; done
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
exec 9>/root/card0.lock; flock 9
HS_TRAIN=qad HS_FLOOR=2 HS_KS=2,2.25,2.5,2.75,3,3.5,4 HS_BUDGETS=$(bud 2 3.5) \
  $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-C-plan.json >> $H/alone.log 2>&1
ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/curve-C-plan.json $CK $H/curve-C.json > $H/curve-C.json.log 2>&1
[ -s $H/curve-C.json ] && log "low curve C priced" || log "curve C headsim FAIL"
log "lowcurve done"
