#!/bin/bash
# headwm.sh -- #34 slab-head whole-model check + the 8.0 floor split (TODO 2026-11-05; rule: README.md next to it).
# gpubox, card 0 (card 1 only for the optional bf16-floor arm). Idempotent; rerun after any failure.
#  1. head Fisher from the published calib golden (CPU, niced), qad-pool rows only (HS_TRAIN=qad)  -> headsens.npz
#  2. vocab permutation (headslab.py perm); 3. lm_head quantized at K3..K7 in that order (card 0, ram.lock exclusive)
#  4. plans at uniform-K5 / K6 bytes (headslab.py plan: u5 / s5-3 / u6 / s6-3)
#  5. each plan's head written bf16-dequantized into a reflink copy of the bf16 container, swapped into V2 4.00
#     (radmix: output.weight only) -> every head arm has the same bytes; only the head's values differ
#  6. scored on golden-search27 (card 0), paired: s5-3 vs u5, s6-3 vs u6; sanity / floor arms: V2 4.00 as is (its
#     8-bit EXL3 head) and V2 4.00 + the true bf16 head (also checks the plugin's bf16-head path: finite, lower KL)
#  7. floor split on V2 8.00: as is vs + bf16 head; optional (both cards free): the bf16 model itself on TP2 (engine floor)
: ${RXM_WORK:=/root}; export RXM_WORK
H=/data/headslab3 S=/data/swap27 T=$S/tools PY=/data/venv-exl3/bin/python I=/data/install-116 HOME_=$S/home
Y=/workspace/rad-exl3-mix/v3/yaqa; [ -d $Y ] || Y=/data/yaqa   # headslab.py / headsens.py: copied next to this script on gpubox
G=$S/golden-search27 CAL=/data/challenge/golden/calib POOLS=/data/challenge/pools-v3.json
BF=/data/models-rad/qwen38-27b-bf16.rad V4=/data/release/qwen3.8-27b-exl3-v2-4.00bpw.rad V8=/data/release/qwen3.8-27b-exl3-v2-8.00bpw.rad
SRC=/data/exl3frac/src; mkdir -p $H; cd $H
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $H/result.txt; }
export HS_TRAIN=qad HS_FLOOR=3
# 1-2
[ -f headsens.npz ] || { nice -n 10 $PY -I $Y/headsens.py $CAL $POOLS headsens.npz 4 > headsens.log 2>&1 && log "headsens: $(tail -n 1 headsens.log)" || { log "headsens FAIL"; exit 1; }; }
[ -f perm.npy ] || $PY -I $Y/headslab.py perm headsens.npz perm.npy >> run.log 2>&1 || { log "perm FAIL"; exit 1; }
# card work waits for the night chain (tp2night.sh) to finish: after the K4 arms (user)
until grep -q "NIGHT DONE" /data/night1105/night.log 2>/dev/null; do sleep 300; done
# 3
for K in 5 4 6 3 7; do
  [ -f K$K/wq/lm_head.rowerr.npz ] && continue
  ( exec 5>$RXM_WORK/card0.lock 7>$RXM_WORK/ram.lock; flock 5; flock 7; cd $RXM_WORK
    ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build \
      $PY -I ourq/ourq.py -K $K -o $H/K$K --only-head --device 0 --save-wq --head-perm $H/perm.npy --qtf2 --frac-ext $SRC \
      --scaled-hessian --sigma-reg 0.025 --codebook i8 ) > K$K.log 2>&1
  [ -f K$K/wq/lm_head.rowerr.npz ] && log "head K$K: $(grep -o 'proxy [0-9.e-]*' K$K.log | tail -n 1)" || { log "head K$K FAIL (K$K.log)"; exit 1; }
done
# 4
[ -f plan.json ] || { $PY -I $Y/headslab.py plan headsens.npz perm.npy $H plan.json > plan.log 2>&1 && log "plans: $($PY -c "import json; print(sorted(json.load(open('plan.json'))['plans']))")" || { log "plan FAIL"; exit 1; }; }
# 5-6
mixhead() { arm=$1 base=$2 plan=$3 M=$H/mix-$1.rad; [ -f $M ] && return 0; D=$H/donor-$1.rad
  cp --reflink=always $BF $D || { log "reflink FAIL"; return 1; }
  [ "$plan" = bf16 ] || $PY -I $Y/headslab.py build plan.json $plan perm.npy $H $D >> run.log 2>&1 || { log "build $plan FAIL"; return 1; }
  echo "output.weight $D" > $H/$arm.mix
  $T/radmix $M.part $base $H/$arm.mix > mix-$arm.log 2>&1 && mv $M.part $M || { log "radmix $arm FAIL"; return 1; }
  rm -f $D; }   # the mix keeps a reflink clone of the head bytes
score() { arm=$1 M=$2 tp=${3:-1}; [ -f $H/$arm.json ] && return 0
  ( if [ $tp = 2 ]; then exec 5>$RXM_WORK/card0.lock 6>$RXM_WORK/card1.lock; flock 5; flock 6; DEV=0,1; else exec 5>$RXM_WORK/card0.lock; flock 5; DEV=0; fi
    exec 7>$RXM_WORK/ram.lock; flock -s 7
    ROCR_VISIBLE_DEVICES=$DEV RADIANCE_HOME=$HOME_ $I/bin/radiance --model $M --tp $tp --kv-cache-dtype bf16 --max-model-len 16384 \
      --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $H/$arm.json.part > $H/score-$arm.log 2>&1 \
      && mv $H/$arm.json.part $H/$arm.json; [ -f $H/$arm.json.part.rows ] && mv $H/$arm.json.part.rows $H/$arm.json.rows )
  [ -f $H/$arm.json.rows ] && log "$arm scored" || log "score $arm FAIL (score-$arm.log)"; }
pair() { [ -f $H/$1.json.rows ] && [ -f $H/$2.json.rows ] && log "pair $1 -> $2: $($PY -I $T/paired.py $G $H/$1.json.rows $H/$2.json.rows 2>&1 | tail -n 1)"; }
score v4 $V4
for p in u5 s5-3 u6 s6-3 bf16; do mixhead h4-$p $V4 $p && score h4-$p $H/mix-h4-$p.rad; done
pair h4-u5 h4-s5-3; pair h4-u6 h4-s6-3; pair v4 h4-bf16; pair h4-bf16 h4-s6-3
# 7
[ -f $V8 ] || nice $T/memguard.sh 2500 /data/venv-exl3/bin/hf download Hero422/qwen3.8-27b-exl3-v2-rad $(basename $V8) --local-dir /data/release > $H/dl8.log 2>&1
if [ -f $V8 ]; then score v8 $V8; mixhead h8-bf16 $V8 bf16 && score h8-bf16 $H/mix-h8-bf16.rad; pair v8 h8-bf16; fi
[ -f /data/night1105/STOP2 ] || { score bf16 $BF 2; pair bf16 v8; }   # engine floor: bf16 vs the published golden (TP2 if card 1 is ours)
log "HEADWM DONE"
