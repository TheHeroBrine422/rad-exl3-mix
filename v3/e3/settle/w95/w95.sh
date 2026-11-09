#!/bin/bash
# w95.sh [STEP ...] -- #126 step 3 (../w95.md): the card-0 smoke of i8 at the bottom of the ladder, gpubox. Score only.
# K1 i8 and K0.125 i8 as whole levels (256 group jobs each) from the patched v3.toml (i8 from K0.125, settings stamp
# 80b5e6d7...), through the donors' job path exactly as w92 (#122) ran K2.5: rxq.ladder.level_jobs -> `python -m rxq job`
# -> rxq assemble -> rxq.post apply (C9 only; swj is codebook-independent and skipped here) -> rxq.mix recipe ->
# rad-convert -> radmix into V2 3.00 -> census (336 exl3i K mul1i8 trunk weights) -> radiance KLD on w92's
# golden-search27c (both parts). Passes if every step runs and every scored row's KL is finite.
# Default STEPS: wait k1 k0125 post:k1 post:k0125 pack:k1 pack:k0125 score:k1 score:k0125 report.
#   wait   until w92's and w94's drivers have been gone for 10 min (the task: card 0 only after w92 / w94)
# Every step skips finished work (w92's helpers: .part -> rename); rerun after a failure or a STOP ($A/STOP).
# Staged by the VM: $A/code = rad-exl3-mix worktree /workspace/wt/w95-rxm (HEAD 5b7caff + w95's patch), $A/exl3 = the
# fork tree w92 staged (6242f7ae). Log $A/result.txt.
set -u
: ${RXM_WORK:=/root}
A=${A:-/data/w95}; CODE=$A/code; EXL3=$A/exl3; W=$CODE/v3/e3/settle/w92; W95=$CODE/v3/e3/settle/w95; PY=/data/venv-exl3/bin/python
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib HOUT=/data/swap27/hout-bn
V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad T=/data/swap27/tools
I8=/data/install-116 H8=/data/swap27/home        # every K2.5 pair's install / home (exl3i: any rate in 1/8 steps, mul1i8)
G92=/data/w92 GUARD_MB=${GUARD_MB:-3000}
STEPS=${*:-wait k1 k0125 post:k1 post:k0125 pack:k1 pack:k0125 score:k1 score:k0125 report}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$A/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
declare -A LEVEL=([k1]=1-i8 [k0125]=0.125-i8) KOF=([k1]=1 [k0125]=0.125)
mkdir -p $A/out $A/lv $A/mix $A/scores $A/logs; cd $A
export PY W T GUARD_MB CODE EXL3
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
stopped() { [ -f $A/STOP ]; }
die() { log "W95 STOP: $*"; exit 1; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge ${1:-6} ] || die "MemAvailable $m GB < ${1:-6} GB ($2)"; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }

# w92 / w94 done: their drivers gone for 10 min AND w92's result.txt ends a run ("W92 DONE" / "VERDICT"), or gone for
# 60 min (a stopped driver nobody restarts). 11-09: a 10-min gap alone fired between a w92 STOP and its relaunch.
waitothers() {
  local quiet=0
  while :; do
    stopped && die "STOP while waiting"
    if pgrep -f 'settle/w92/w92\.sh' > /dev/null || pgrep -f 'w94/code/run94\.sh' > /dev/null; then quiet=0
    else quiet=$((quiet + 1)); fi
    if [ $quiet -ge 10 ] && tail -3 $G92/result.txt | grep -qE 'W92 DONE|W92 VERDICT'; then break; fi
    [ $quiet -ge 60 ] && break
    sleep 60
  done
  log "wait: w92 / w94 drivers gone for $quiet min ($(tail -1 $G92/result.txt | cut -c1-80))"
}

jobs() { local tag=$1 lv=$2 seed=$3; shift 3; local jd=$A/jobs/$tag od=$A/out/$tag
  if [ ! -f $jd/jobs.jsonl ]; then
    rm -rf $jd.part; "${RX[@]}" $W/mkjobs92.py $lv $seed $MODEL $HESS $HOUT $jd.part "$@" > $A/logs/mkjobs-$tag.log 2>&1 && mv $jd.part $jd \
      || die "mkjobs $tag ($A/logs/mkjobs-$tag.log)"
    log "$(tail -1 $A/logs/mkjobs-$tag.log)"; fi
  mkdir -p $od
  for l in $(seq -f %02g 0 63); do
    local todo=()
    for s in $jd/L$l-g*.json; do [ -f "$s" ] || continue; local n=$(basename $s .json)
      [ -f $od/$n/job.json ] && [ "$($PY -c "import json, os; s = json.load(open('$s')); print(all(os.path.exists(f'$od/$n/t-{k}.safetensors') for k in s['keys']))")" = True ] && continue
      todo+=($n); done
    [ ${#todo[@]} = 0 ] && continue
    stopped && die "STOP before $tag L$l"
    memok 6 "$tag L$l"
    cardrun bash $W/layer92.sh $od $jd "${todo[@]}" || die "$tag: a job failed (L$l: $od/*/rxq.log); rerun resumes"
  done
  log "$tag: $(ls $od | grep -c '^L') jobs done, $(awk '{s += $2} END {printf "%.0f s wall", s / 1000}' $od/wall-ms.txt)"
}
assemble() { local tag=$1; shift; local d=$A/lv/$tag/level
  [ -f $d/stamp.json ] && return 0
  rm -rf $d; mkdir -p $A/lv/$tag
  nice -n 10 "${RX[@]}" -m rxq assemble $d $A/out/$tag/L* --template $A/template.json "$@" > $A/lv/$tag/assemble.log 2>&1 \
    || { tail -3 $A/lv/$tag/assemble.log; die "assemble $tag"; }
  log "$tag assembled: $(tail -1 $A/lv/$tag/assemble.log) ($(ls $d/q-L*.safetensors | wc -l) shards, $(ls $d/c9/scales 2>/dev/null | wc -l) C9 writers)"
}
level() { local tag=$1; jobs $tag ${LEVEL[$tag]} 0; assemble $tag; }
post() { local t=$1; local F=$A/lv/$t/final   # two statements (w92's 11-09 bug: one `local` expands $t first)
  [ -f $F/.done ] && return 0
  nice -n 10 "${RX[@]}" -m rxq.post apply $A/lv/$t/level $A/lv/$t/level/c9 $F > $A/lv/$t/post.log 2>&1 || die "post apply $t"
  [ -f $F/.done ] || die "post $t: no .done"; log "post $t (C9 only): $(tail -1 $A/lv/$t/post.log)"
}
packmix() { local tag=$1 src=$2 K=$3 P=$A/mix/pack-$1.rad M=$A/mix/mix-$1.rad
  [ -f $M ] && [ -f $A/mix/census-$tag.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || die "$tag: $src has not 64 layers"
  stopped && die "STOP before pack $tag"
  "${RX[@]}" -m rxq.mix recipe $src > $A/mix/$tag.recipe || die "recipe $tag"
  local mcb=$(grep -o 'codebook=[a-z0-9]*' $A/mix/$tag.recipe | sort -u | cut -d= -f2)
  [ "$mcb" = mul1i8 ] || die "$tag: recipe codebook '$mcb' (want mul1i8)"
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; cd /workspace; s=$(date +%s); rm -f $P.part
      RADIANCE_HOME=$H8 RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I8/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $A/mix/$tag.recipe --set quantization_config.quant_method=exl3 -o $P.part > $A/mix/conv-$tag.log 2>&1 && mv $P.part $P
      echo "$(date -u +%H:%M:%S) pack $tag rc=$? $(( $(date +%s) - s ))s" ) | tee -a $A/result.txt
    [ -f $P ] || { rm -f $P.part; die "pack $tag FAIL ($A/mix/conv-$tag.log)"; }
  fi
  TRUNK | sed "s|\$| $P|" > $A/mix/$tag.mix
  $T/radmix $M.part $V2 $A/mix/$tag.mix > $A/mix/mix-$tag.log 2>&1 && mv $M.part $M || die "radmix $tag"
  $PY -I $T/census.py $M --mix $A/mix/$tag.mix --base $V2 > $A/mix/census-$tag.txt 2>&1 || die "census $tag ($A/mix/census-$tag.txt)"
  grep -qE "exl3i  *$K(\.0+)?  *$mcb *f16  *336 " $A/mix/census-$tag.txt || die "$tag: not 336 exl3i K$K $mcb trunk weights"
  touch $A/mix/census-$tag.ok; rm -f $P
  log "$tag: mix ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}
TRUNK() { "${RX[@]}" -c "from rxq import mix; print(''.join(n + chr(10) for n in mix.trunk_names()), end='')"; }
score1() { local tag=$1 part=$2 G=$G92/golden-search27c-$2 o=$A/scores/$1-$2
  [ -f $o.json.rows ] && return 0
  stopped && die "STOP before score $tag"
  cardrun bash -c "s=\$(date +%s); ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H8 $I8/bin/radiance --model $A/mix/mix-$tag.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $A/logs/score-$tag-$part.log 2>&1; rc=\$?
    [ \$rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo \"\$(date -u +%H:%M:%S) score $tag-$part rc=\$rc \$(( \$(date +%s) - s ))s\"" | tee -a $A/result.txt
  [ -f $o.json.rows ] || die "score $tag-$part FAIL ($A/logs/score-$tag-$part.log)"
}

[ -f $A/template.json ] || cp $G92/template.json $A/template.json
log "w95 start: steps $STEPS; settings $("${RX[@]}" -m rxq stamp)"
for st in $STEPS; do
  stopped && die "STOP"
  case $st in
    wait) waitothers;;
    k1|k0125) level $st;;
    post:*) post ${st#post:};;
    pack:*) t=${st#pack:}; packmix $t $A/lv/$t/final ${KOF[$t]};;
    score:*) t=${st#score:}; for p in x c; do score1 $t $p; done;;
    report) $PY -I $W95/smoke95.py $A | tee $A/smoke95.txt | tail -1 | while read l; do log "$l"; done;;
    *) die "unknown step $st";;
  esac
done
log "W95 DONE ($STEPS)"
