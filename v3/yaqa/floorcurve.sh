#!/bin/bash
# floorcurve.sh -- curves A/B re-planned with every tail floor (HS_FLOOR_ALL=1) and priced by headsim; the curve per
# budget = the best floor by exact head KL. Waits for headset.sh. Log /data/yaqa/headslab2/alone.log.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/alone.log; }
while pgrep -f "^/bin/bash ./headset.sh" > /dev/null; do sleep 20; done
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
export HS_TRAIN=qad HS_FLOOR=3 HS_FLOOR_ALL=1
exec 9>$RXM_WORK/card0.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7
HS_KS=3,3.5,4,4.5,5,5.5,6 HS_BUDGETS=$(bud 3.5 5.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/fcurve-A-plan.json >> $H/alone.log 2>&1
HS_KS=5,5.5,6,6.5,7,8 HS_BUDGETS=$(bud 5.5 7.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/fcurve-B-plan.json >> $H/alone.log 2>&1
for p in A B; do
  ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/fcurve-$p-plan.json $CK $H/fcurve-$p.json > $H/fcurve-$p.json.log 2>&1
  [ -s $H/fcurve-$p.json ] && log "floor curve $p priced" || log "floor curve $p headsim FAIL"
done
