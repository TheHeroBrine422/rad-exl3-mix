#!/bin/bash
# halfcurve.sh -- half-step head donors K4.5-7.5 one at a time (headalone.sh), then the slab-head curve re-planned and
# priced with every donor K3-8 at half steps (two headsim passes). Log /data/yaqa/headslab2/alone.log.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/alone.log; }
for K in 4.5 5.5 6.5 7.5; do [ -f $H/K$K/wq/lm_head.rowerr.npz ] || $Y/headalone.sh $K; done
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
export HS_TRAIN=qad HS_FLOOR=3
exec 9>/root/card0.lock; flock 9
HS_KS=3,3.5,4,4.5,5,5.5,6 HS_BUDGETS=$(bud 3.5 5.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-A-plan.json >> $H/alone.log 2>&1
HS_KS=5,5.5,6,6.5,7,7.5,8 HS_BUDGETS=$(bud 5.5 7.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-B-plan.json >> $H/alone.log 2>&1
for p in A B; do
  ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/curve-$p-plan.json $CK $H/curve-$p.json > $H/curve-$p.json.log 2>&1
  [ -s $H/curve-$p.json ] && log "half-step curve $p priced" || log "curve $p headsim FAIL"
done
log "halfcurve done"
