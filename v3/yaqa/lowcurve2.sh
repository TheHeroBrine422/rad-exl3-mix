#!/bin/bash
# lowcurve2.sh -- research: head donors K1, K1.5 (alone), then curve D: donors K1-3, slab tails down to K1, budgets 1.5-3.0 at 1/8, uniform u1..u3. Waits for lowcurve.sh.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/alone.log; }
while pgrep -f "^/bin/bash ./(halfcurve|lowcurve).sh" > /dev/null; do sleep 20; done
for K in 1 1.5; do [ -f $H/K$K/wq/lm_head.rowerr.npz ] || $Y/headalone.sh $K; done
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
exec 9>/root/card0.lock; flock 9
HS_TRAIN=qad HS_FLOOR=1 HS_KS=1,1.5,2,2.25,2.5,2.75,3 HS_BUDGETS=$(bud 1.5 3) \
  $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-D-plan.json >> $H/alone.log 2>&1
ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/curve-D-plan.json $CK $H/curve-D.json > $H/curve-D.json.log 2>&1
[ -s $H/curve-D.json ] && log "low curve D priced" || log "curve D headsim FAIL"
log "lowcurve2 done"
