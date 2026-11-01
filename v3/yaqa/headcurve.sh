#!/bin/bash
# headcurve.sh -- the slab-head menu for the solver (card 0, after a4time; log /data/yaqa/headslab2/curve.log):
#  1. donor heads at the missing half steps K3.5 / 4.5 / 5.5 / 6.5 / 7.5 and K8, plain mul1 codebook (the head only runs
#     at decode / on the last prefill token, never on the int8 path), same qad-Fisher row order as K3-K7
#  2. 3-slab plans at budgets K3.5 ... K7.5 in 1/8 steps (33) + every uniform donor, in two donor sets (headsim keeps
#     every donor on the card, ~2.5 GB each): A = K3-6 (budgets 3.5-5.5), B = K5-8 (budgets 5.5-7.5; 5.5 in both)
#  3. headsim.py prices every plan (exact head-only KL, calib-A head inputs) -> curve-A.json, curve-B.json
# Then removes /data/yaqa/headcurve.pending (levels.sh card 0 waits for it).
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
log() { echo "$(date -u +%H:%M) $*" | tee -a $H/curve.log; }
trap 'rm -f /data/yaqa/headcurve.pending' EXIT
cd ${RXM_WORK}
export EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
for K in 3.5 4.5 5.5 6.5 7.5 8; do
  [ -f $H/K$K/wq/lm_head.rowerr.npz ] && continue
  ( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock 7
    ROCR_VISIBLE_DEVICES=0 $PY -I ourq/ourq.py -K $K -o $H/K$K --only-head --device 0 --save-wq --head-perm $H/perm.npy \
      --frac-ext /data/exl3frac/src 2>&1 | grep -v -e Warning -e "cpu = _conv" ) > $H/K$K.log 2>&1
  [ -f $H/K$K/wq/lm_head.rowerr.npz ] && log "K$K head quantized" || log "K$K head FAIL"
done
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
export HS_TRAIN=qad HS_FLOOR=3
HS_KS=3,3.5,4,4.5,5,5.5,6 HS_BUDGETS=$(bud 3.5 5.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-A-plan.json >> $H/curve.log 2>&1
HS_KS=5,5.5,6,6.5,7,7.5,8 HS_BUDGETS=$(bud 5.5 7.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-B-plan.json >> $H/curve.log 2>&1
for p in A B; do
  [ -s $H/curve-$p.json ] && continue
  ( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock -s 7
    ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/curve-$p-plan.json $CK $H/curve-$p.json ) > $H/curve-$p-sim.log 2>&1
  [ -s $H/curve-$p.json ] && log "curve $p priced" || log "curve $p headsim FAIL (see curve-$p-sim.log)"
done
log "headcurve done"
