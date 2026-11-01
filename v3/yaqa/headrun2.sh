#!/bin/bash
# headrun2.sh -- the row-slab LM head test, redone (headrun.sh's whole-model scores were invalid: the qwen35_exl3
# plugin's bf16-head fallback wrote bf16 bytes into the f32 logits buffer; and its design had in-sample and floor
# issues, local/mine/merged-2026-10-31.md A2). After gate2c.sh (card 0):
#  1. vocab permutation from the qad pool's Fisher only (HS_TRAIN=qad): the judging data (calib-A head dump, golden-ml)
#     come from the fit pool and are out of sample
#  2. lm_head quantized at K2..K7 in that row order (card 0, ram.lock exclusive, ~5 min each)
#  3. plans at the bytes of uniform K4 / K5 / K6, floor K3 (HS_FLOOR)
#  4. headsim.py: exact head-only KL of every plan on the 32,768 calib-A head inputs, paired vs uniform at equal bytes
#     with a block bootstrap. The whole-model confirmation (fixed plugin, golden-ml guard + a verify slice) is for the
#     finalist only, later.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16
mkdir -p $H; cd ${RXM_WORK}
log() { echo "$(date -u +%H:%M) $*" | tee -a $H/run.log; }
until grep -q "gate2c done" $Y/gate2c.log 2>/dev/null || ! pgrep -f "^/bin/bash /data/yaqa/gate2c.sh" > /dev/null; do sleep 60; done
export HS_TRAIN=qad HS_FLOOR=3
[ -f $H/perm.npy ] || $PY -I $Y/headslab.py perm $Y/hs/headsens.npz $H/perm.npy >> $H/run.log 2>&1 || { log "perm FAIL"; exit 1; }
export EXL3_QT_OPTIMIZED=1
for K in 4 3 5 6 2 7; do
  [ -f $H/K$K/wq/lm_head.rowerr.npz ] && continue
  ( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock 7
    ROCR_VISIBLE_DEVICES=0 $PY -I ourq/ourq.py -K $K -o $H/K$K --only-head --device 0 --save-wq --head-perm $H/perm.npy 2>&1 \
      | grep -v -e Warning -e "cpu = _conv" ) > $H/K$K.log 2>&1
  [ -f $H/K$K/wq/lm_head.rowerr.npz ] && log "K$K head quantized: $(grep -o 'proxy [0-9.e-]*' $H/K$K.log)" || log "K$K head FAIL"
done
$PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/plan.json 2>&1 | tee -a $H/run.log
( exec 9>${RXM_WORK}/card0.lock 7>${RXM_WORK}/ram.lock; flock 9; flock -s 7
  ROCR_VISIBLE_DEVICES=0 $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $H/plan.json $CK $H/sim.json ) > $H/sim.log 2>&1
grep -v "^[0-9]*/[0-9]* [0-9]*s$" $H/sim.log | tee -a $H/run.log
log "headrun2 done"
