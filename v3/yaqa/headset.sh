#!/bin/bash
# headset.sh -- head quantizer settings (pre-requant checklist, head): K3 and K5 x {mean-row on: default damping, 0.01,
# 0.005}, each alone (exclusive host RAM), against the existing default donors (mean-row off, default damping). Priced
# by headsim per variant: a QDIR with K3 + K5 of that variant, plans u3 / u5 (+ s4). Rules (fixed 2026-11-01):
# mean-row on where it gains; damping: one value if K3 and K5 agree, else every remade donor runs all three and keeps
# the lowest headsim KL. Waits for halfcurve.sh. Out /data/yaqa/headset/, log run.log.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 O=/data/yaqa/headset PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
while pgrep -f "^/bin/bash ./halfcurve.sh" > /dev/null || pgrep -f "^/bin/bash /root/ourq/exl3frac2/qtf2run.sh" > /dev/null; do sleep 20; done
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
for V in mr mr-d0.01 mr-d0.005; do
  X="--head-mean-row"; case $V in *d0.01) X="$X --sigma-reg 0.01";; *d0.005) X="$X --sigma-reg 0.005";; esac
  for K in 3 5; do
    D=$O/$V/K$K; mkdir -p $O/$V; [ -f $D/wq/lm_head.rowerr.npz ] && continue
    ( exec 9>$RXM_WORK/card0.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7
      $PY -I ourq/ourq.py -K $K -o $D --only-head --device 0 --save-wq --head-perm $H/perm.npy $X > $D.log 2>&1 )
    [ -f $D/wq/lm_head.rowerr.npz ] && log "$V K$K OK: $(grep -o '"proxy_err": [0-9.e-]*' $D/proxy.jsonl)" || log "$V K$K FAIL"
  done
done
mkdir -p $O/default; for K in 3 5; do ln -sfn $H/K$K $O/default/K$K; done
for V in default mr mr-d0.01 mr-d0.005; do
  [ -s $O/$V.json ] && [ $V != default ] && continue
  ( exec 9>$RXM_WORK/card0.lock; flock 9
    HS_TRAIN=qad HS_FLOOR=3 HS_KS=3,5 HS_BUDGETS=4 $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $O/$V $O/$V-plan.json >> $O/run.log 2>&1
    $PY -I $Y/headsim.py $DUMP $O/$V $H/perm.npy $O/$V-plan.json $CK $O/$V.json > $O/$V.json.log 2>&1 )
  [ -s $O/$V.json ] && log "$V priced: $(python3 -c "
import json; p = json.load(open('$O/$V.json'))['plans']; print({n: f\"{v['kl']:.6f}\" for n, v in p.items()})")" || log "$V headsim FAIL"
done
log "headset done"
