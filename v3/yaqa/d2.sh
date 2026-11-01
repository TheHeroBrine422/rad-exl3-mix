#!/bin/bash
# d2.sh -- mining merge D2: is the input-side proxy in-sample? (after gate2e.sh; log /data/d2/d2.log)
#  1. split-half input Hessians H_A / H_B of layers 3 (attention), 10 (GDN), 43 (attention), 1200 non-verify windows
#     (collect.py YAQA_HSPLIT, both cards)
#  2. ours quantized on H_A (ourq.py --hin hA, plain mul1 like turboderp's) at K4 / K6, input-group damping
#     0.025 / 0.005 / 0.001, dequantized weights kept (--save-wq)
#  3. d2score.py: every arm + turboderp's uniform K4 / K6 downloads, proxy error on H_A, H_B and the published H
: ${RXM_WORK:=/root}; export RXM_WORK
D=/data/d2 Y=/data/yaqa PY=/data/venv-exl3/bin/python R=/data/workspace/references
mkdir -p $D; cd $RXM_WORK
log() { echo "$(date -u +%H:%M) $*" | tee -a $D/d2.log; }
until grep -q "gate2e done" $Y/gate2e.log 2>/dev/null; do
  pgrep -f "^/bin/bash ./gate2e.sh" > /dev/null || { log "gate2e not running and not done: stop"; exit 1; }; sleep 60; done
LAY=3,10,43
if [ ! -f $D/hsplit/hB/model.language_model.layers.43.post_attention.input.safetensors ]; then
  ( exec 8>$RXM_WORK/card0.lock 9>$RXM_WORK/card1.lock 7>$RXM_WORK/ram.lock; flock 8; flock 9; flock -s 7
    cd $Y && YAQA_HSPLIT=$LAY $PY -I collect.py $D/hsplit 1200 2048 ) > $D/hsplit.log 2>&1
  grep -q "^DONE hsplit" $D/hsplit.log || { log "hsplit FAIL"; exit 1; }
fi
log "split Hessians: $(ls $D/hsplit/hA | wc -l) groups per half"
export EXL3_QT_OPTIMIZED=1
q() {  # CARD K SIGMA
  local o=$D/o-K$2-s$3
  ( exec 9>$RXM_WORK/card$1.lock; flock 9
    for l in 3 10 43; do
      [ -f $o/q-L$(printf %02d $l).safetensors ] && continue
      ROCR_VISIBLE_DEVICES=$1 $PY -I ourq/ourq.py -K $2 -o $o --layers $l-$l --device 0 --hin $D/hsplit/hA --sigma-reg-in $3 --save-wq 2>&1 \
        | grep -v -e Warning -e "cpu = _conv"
    done ) >> $o.log 2>&1
  log "ours K$2 sigma $3: $(ls $o/q-L*.safetensors 2>/dev/null | wc -l)/3 layers"
}
( q 0 4 0.025; q 0 4 0.005; q 0 4 0.001 ) &
( q 1 6 0.025; q 1 6 0.005; q 1 6 0.001 ) &
wait
A=""; for K in 4 6; do for s in 0.025 0.005 0.001; do A="$A o$K-$s:ours:$D/o-K$K-s$s"; done; done
A="$A td4:td:$R/td-4.00bpw td6:td:$R/td-6.00bpw"
( exec 9>$RXM_WORK/card0.lock 7>$RXM_WORK/ram.lock; flock 9; flock -s 7
  ROCR_VISIBLE_DEVICES=0 $PY -I $Y/d2score.py $D/score.jsonl $LAY $D/hsplit/hA $D/hsplit/hB $RXM_WORK/challenge/hessians/calib $A ) > $D/score.log 2>&1
log "scored $(wc -l < $D/score.jsonl) rows"
rm -rf $D/o-K*/wq                          # dequantized weights: scratch (scores kept)
log "d2 done"
