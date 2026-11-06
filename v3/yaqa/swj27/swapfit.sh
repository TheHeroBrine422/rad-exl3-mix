#!/bin/bash
# swapfit.sh [STEP...] -- the 27B swj swap test (rule and commands: v3/yaqa/SWJ27.md), gpubox, card 0. No requant: the
# existing K2.5 swap arms get C10 gate / up output scales fitted on a qad slice, then are packed, mixed into V2 3.00 and
# scored exactly as swap27k.sh scores them (swap27.sh's functions, same home / build / golden-search27).
# Steps (default "meta plan fit apply arms rule"; each skips finished work, rerun after any failure):
#   meta   qad slice: calib_slice.py (pools-v3, TOK tokens, cap 4096, seed 0) -> $O/swj27-qad (tokens only, mkmeta.py; the ledger use)
#   plan   swjfit27.py --plan-only on the CPU (sources' keys, stage split under VRAM_CAP_GB 30); no GPU
#   fit    swjfit27.py, all arms in one staged bf16 pass, card 0 (card0.lock + ram.lock shared for the whole pass)
#   apply  swjapply.py: each arm's level with its scales -> $O/lv/<arm>swj (CPU, one shard at a time)
#   arms   per arm: pack (exl3i, mul1i8, rate 2.5) + mix into V2 3.00 + census + score -> $O/<arm>swj.json(.rows)
#   rule   paired.py <arm> -> <arm>swj per seed pair, and SWJ27.md's verdict (rule.py)
# Env: ARMS (default "base s1"; "bn bns1" once q1106 has made bns1; K4: "base4 base4s1", V2 4.00), TOK (default 1048576), FUSED (default 1:
# swjtri.py's kernel; 0 = the exact torch path, ~10x slower on the 27B), NOCACHE (default 1: use_cache=False, mkhin27's
# forward, the one run on gpubox's ROCm; 0 = mkgold27's fresh-cache forward, the 0.8B gate's), CODE (code dir), O (output dir).
set -uo pipefail
: ${CODE:=/data/q27/w36/code} ${O:=/data/swj27} ${TOK:=1048576} ${FUSED:=1} ${NOCACHE:=1} ${ARMS:=base s1}
STEPS=${*:-meta plan fit apply arms rule}
PYB=/data/venv-exl3/bin/python MODEL=/data/workspace/models/qwen38-27b-bf16 GOLD=/data/challenge/golden/calib
CORPUS=/data/challenge/corpus/calib.jsonl POOLS=/data/challenge/pools-v3.json
Q=/data/swap27
declare -A LV=([base]=/data/v3ladder2/K2.5 [s1]=$Q/q-s1/K2.5 [bn]=$Q/q-bn/K2.5 [bns1]=$Q/q-bns1/K2.5
               [base4]=$Q/q-base4/K4 [base4s1]=$Q/q-base4s1/K4 [bn4]=$Q/q-bn4/K4)
X=$CODE/v3/yaqa/swj27
# swap27.sh's settings and functions (TRUNK, score, the paths of home / install / golden / tools), outputs redirected to $O
source <(sed -n '/^: \${RXM_WORK/,/^quant() /p' /data/swap27/tools/swap27.sh | sed '$d')
S=$O C=0; mkdir -p $O; cd $O
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/result.txt; }
want() { [[ " $STEPS " == *" $1 "* ]]; }
memok() { [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -ge ${1:-4} ]; }
for a in $ARMS; do [ $(ls ${LV[$a]}/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || { log "arm $a: ${LV[$a]} has not 64 layers"; exit 1; }; done
V2of() { case $1 in *4|*4s1) echo /data/release/qwen3.8-27b-exl3-v2-4.00bpw.rad;; *) echo /data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad;; esac; }
Kof() { case $1 in *4|*4s1) echo 4;; *) echo 2.5;; esac; }
FITARMS=$(for a in $ARMS; do [ -f $O/fit/$a/.done ] || echo -n "$a "; done); FITARMS=${FITARMS% }
SRCS=$(for a in $FITARMS; do echo -n " --src $a=${LV[$a]}:level"; done)
FIT=(env ROCR_VISIBLE_DEVICES=0 VRAM_CAP_GB=30 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
     nice -n 5 $PYB $X/swjfit27.py $MODEL $O/swj27-qad $O/fit --scratch $O/scr-${FITARMS// /-} $SRCS $([ $FUSED = 1 ] && echo --fused) $([ $NOCACHE = 1 ] && echo --no-cache))

if want meta && [ ! -f $O/swj27-qad/kld.json ]; then
  args=$(POOLS=$POOLS $PYB -I $CODE/mix/calib_slice.py $CORPUS $TOK 4096 0 qad 2> $O/swj27-qad.slice.log) || { log "calib_slice FAIL"; exit 1; }
  echo "$args" > $O/swj27-qad.docs
  $PYB -I $X/mkmeta.py $GOLD $O/swj27-qad $args | tee -a $O/result.txt; log "slice: $(cat $O/swj27-qad.slice.log)"
fi
if want plan; then
  memok 4 || { log "plan: MemAvailable < 4 GB, not started"; exit 1; }
  [ -n "$FITARMS" ] && env ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= SWJ_DEV=cpu VRAM_CAP_GB=30 nice -n 19 $PYB $X/swjfit27.py $MODEL \
    $O/swj27-qad $O/fit --scratch $O/plan-tmp $SRCS $([ $FUSED = 1 ] && echo --fused) $([ $NOCACHE = 1 ] && echo --no-cache) --plan-only 2>&1 | grep -v -i warn | tail -1 | tee -a $O/result.txt
fi
if want fit && [ -n "$FITARMS" ]; then
  log "fit ($FITARMS, $(python3 -c "import json; print(json.load(open('$O/swj27-qad/kld.json'))['tokens'])") tokens, fused $FUSED): start"; s=$(date +%s)
  flock /root/card0.lock flock -s /root/ram.lock "${FIT[@]}" >> $O/fit-${FITARMS// /-}.log 2>&1 < /dev/null
  grep -q "^.*SWJFIT27 DONE" $O/fit-${FITARMS// /-}.log && log "fit: done in $(( ($(date +%s) - s) / 60 )) min ($(grep -h 'fused check' $O/fit-${FITARMS// /-}.log | tail -1 | cut -c10-))" \
    || { log "fit: not finished (fit-${FITARMS// /-}.log; rerun resumes)"; exit 1; }
fi
if want apply; then
  for a in $ARMS; do [ -f $O/lv/${a}swj/.done ] && continue
    memok 4 || { log "apply: MemAvailable < 4 GB"; exit 1; }
    nice -n 19 $PYB -I $X/swjapply.py ${LV[$a]} $O/fit/$a $O/lv/${a}swj | tail -1 | tee -a $O/result.txt
    [ -f $O/lv/${a}swj/.done ] || { log "apply $a FAIL"; exit 1; }
  done
fi
# pack + mix + census (swap27k.sh's packmixk at rate K), then swap27.sh's score
packmix() { arm=$1 src=$2 P=$S/pack-$1.rad M=$S/mix-$1.rad
  [ -f $M ] && [ -f $S/census-$arm.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || { log "$arm: $src has not 64 layers"; return 1; }
  TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u | sed "s|\$| exl3i src=$src codebook=mul1i8 rate=$K|" > $S/$arm.recipe
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd /workspace; s=$(date +%s)
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $S/$arm.recipe --set quantization_config.quant_method=exl3 -o $P.part > $S/conv-$arm.log 2>&1 && mv $P.part $P
      echo "pack $arm rc=$? $(( $(date +%s) - s ))s" ) | tee -a $S/result.txt
    [ -f $P ] || { log "pack $arm FAIL ($S/conv-$arm.log)"; return 1; }
  fi
  TRUNK | sed "s|\$| $P|" > $S/$arm.mix
  $T/radmix $M.part $V2 $S/$arm.mix > $S/mix-$arm.log 2>&1 && mv $M.part $M || { log "radmix $arm FAIL"; return 1; }
  $PY -I $T/census.py $M --mix $S/$arm.mix --base $V2 > $S/census-$arm.txt 2>&1 || { log "census $arm FAIL"; return 1; }
  grep -qE "exl3i  *$K(\.0+)?  *mul1i8 *f16  *336 " $S/census-$arm.txt || { log "$arm: not 336 exl3i K$K trunk weights"; return 1; }
  touch $S/census-$arm.ok; rm -f $P; log "$arm: mix ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"; }
if want arms; then
  for a in $ARMS; do [ -f $O/lv/${a}swj/.done ] || { log "arms: $a not applied"; exit 1; }
    K=$(Kof $a) V2=$(V2of $a); packmix ${a}swj $O/lv/${a}swj && score ${a}swj || exit 1
  done
fi
if want rule; then
  for a in $ARMS; do [ -f $Q/$a.json.rows ] && [ -f $O/${a}swj.json.rows ] || continue
    grep -q "pair $a -> ${a}swj:" $O/result.txt && continue
    log "pair $a -> ${a}swj: $($PY -I $T/paired.py $G $Q/$a.json.rows $O/${a}swj.json.rows 2>&1 | tail -n 1)"; done
  $PYB -I $X/rule.py $O/result.txt $ARMS | tee -a $O/result.txt
fi
