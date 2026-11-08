#!/bin/bash
# headwm90.sh -- #34-02 / #99 (../HEADWM.md) under its amendment 1 (11-08, w90): headwm.sh's arms with the head plans
# rebuilt from the corrected rowerr and every arm scored on both parts of golden-search27b. gpubox, card 0 only (card 1
# never). Idempotent; rerun after any failure. Stop: touch /data/w90/hw/STOP (between steps).
#   0. fix: headslab3's broken K*/wq/lm_head.rowerr.npz, plan.json and the s5-3 mix -> broken-rowerr-1108/; the
#      corrected rowerr (/data/w90/rowerr, rowerr90.py) installed in their place
#   1. plans: headslab.py plan (HS_TRAIN=qad HS_FLOOR=3, as headwm.sh) -> headslab3/plan.json (= /data/w90/rowerr/plan.json)
#   2. mixes as headwm.sh's mixhead (bf16-dequantized head over a reflink of the bf16 container, radmix output.weight
#      into V2 4.00 / 8.00), under conv.lock + ram.lock EXCLUSIVE (lead 11-07 pack rule)
#   3. scores (radiance, bf16 KV, card 0 under card0.lock + ram.lock shared) on golden-search27b-x and -c
#   4. pairs (pair84.py, pooled over the parts) and verdict90.py (amendment 1's rule)
: ${RXM_WORK:=/root}
H=/data/headslab3 S=/data/swap27 T=$S/tools PY=/data/venv-exl3/bin/python I=/data/install-116 HOME_=$S/home Y=/data/yaqa
A=/data/w90; O=$A/hw; W=$A/code/v3/yaqa/w90; P84=$A/code84/v3/e3/settle/w84
GX=$A/golden-search27b-x GC=$A/golden-search27b-c
BF=/data/models-rad/qwen38-27b-bf16.rad V4=/data/release/qwen3.8-27b-exl3-v2-4.00bpw.rad V8=/data/release/qwen3.8-27b-exl3-v2-8.00bpw.rad
export HS_TRAIN=qad HS_FLOOR=3
mkdir -p $O; cd $H
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/result.txt; }
stop() { [ -f $O/STOP ] && { log "STOP"; exit 0; }; }
for g in $GX $GC; do [ -f $g/kld.json ] || { log "FAIL no golden $g"; exit 1; }; done
# 0
if [ ! -f $H/rowerr.fixed ]; then
  B=$H/broken-rowerr-1108; mkdir -p $B
  for K in 3 4 5 6 7; do
    [ -f /data/w90/rowerr/K$K/wq/lm_head.rowerr.npz ] || { log "FAIL no corrected rowerr K$K"; exit 1; }
    [ -f $B/K$K.lm_head.rowerr.npz ] || mv $H/K$K/wq/lm_head.rowerr.npz $B/K$K.lm_head.rowerr.npz
    cp /data/w90/rowerr/K$K/wq/lm_head.rowerr.npz $H/K$K/wq/lm_head.rowerr.npz
  done
  for f in plan.json plan.log mix-h4-s5-3.rad h4-s5-3.mix mix-h4-s5-3.log score-h4-s5-3.log; do [ -e $H/$f ] && mv $H/$f $B/; done
  touch $H/rowerr.fixed; log "fix: broken rowerr / plan / s5-3 mix moved to $B, corrected rowerr installed"
fi
# 1
if [ ! -f $H/plan.json ]; then
  nice -n 10 $PY -I $Y/headslab.py plan headsens.npz perm.npy $H plan.json > plan.log 2>&1 || { log "FAIL plan"; exit 1; }
  cmp -s <(grep -v "^ *$" plan.log) <(grep -v "^ *$" /data/w90/rowerr/plan.log) && s="= /data/w90/rowerr/plan.log" || s="DIFFERS from /data/w90/rowerr/plan.log"
  log "plans rebuilt ($s): $(grep -E '^(s5-3|s6-3) ' plan.log | tr -s ' ' | tr '\n' ';')"
fi
# 2
mixhead() { local arm=$1 base=$2 plan=$3 M=$H/mix-$1.rad D=$H/donor-$1.rad; [ -f $M ] && return 0; stop
  ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; s=$(date +%s); rm -f $M.part $D
    cp --reflink=always $BF $D || exit 1
    if [ "$plan" != bf16 ]; then nice -n 10 $PY -I $Y/headslab.py build plan.json $plan perm.npy $H $D >> run90.log 2>&1 || exit 2; fi
    echo "output.weight $D" > $H/$arm.mix
    $T/radmix $M.part $base $H/$arm.mix > mix-$arm.log 2>&1 && mv $M.part $M || exit 3
    rm -f $D; echo "$(date -u +%H:%M:%S) mix $arm ($plan on $(basename $base)): $(( $(date +%s) - s ))s" >> $O/result.txt )
  [ -f $M ] || { rm -f $M.part $D; log "FAIL mix $arm"; return 1; }; }
# 3
score() { local arm=$1 M=$2 part=$3 g; [ $part = x ] && g=$GX || g=$GC; local out=$O/$arm-$part
  [ -f $out.json.rows ] && return 0; stop
  [ -f $M ] || { log "FAIL score $arm-$part: no mix $M"; return 1; }
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$HOME_ $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
      --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $g --kld-out $out.json.part > $out.log 2>&1 \
      && mv $out.json.part $out.json; [ -f $out.json.part.rows ] && mv $out.json.part.rows $out.json.rows
    echo "$(date -u +%H:%M:%S) score $arm-$part: $(( $(date +%s) - s ))s" >> $O/result.txt )
  [ -f $out.json.rows ] || { log "FAIL score $arm-$part ($out.log)"; return 1; }; }
both() { score $1 $2 x && score $1 $2 c; }
both v4 $V4
for p in bf16 u5 s5-3 u6 s6-3; do mixhead h4-$p $V4 $p && both h4-$p $H/mix-h4-$p.rad; done
# 4 (#34-02 first; #99 after)
$PY -I $W/verdict90.py $O $P84 $GX $GC 34 | tee -a $O/result.txt
stop
[ -f $V8 ] || { nice $T/memguard.sh 2500 /data/venv-exl3/bin/hf download Hero422/qwen3.8-27b-exl3-v2-rad $(basename $V8) --local-dir /data/release > $O/dl8.log 2>&1
                [ -f $V8 ] && log "V2 8.00 downloaded" || log "FAIL V2 8.00 download ($O/dl8.log)"; }
if [ -f $V8 ]; then both v8 $V8; mixhead h8-bf16 $V8 bf16 && both h8-bf16 $H/mix-h8-bf16.rad; fi
$PY -I $W/verdict90.py $O $P84 $GX $GC 99 | tee -a $O/result.txt
log "HEADWM90 DONE"
