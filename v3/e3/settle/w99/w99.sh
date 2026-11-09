#!/bin/bash
# w99.sh [STEP ...] -- #128 / #129 (../w99.md), gpubox card 0. Default STEPS:
#   prov mix hyg wait score:h verdict:h k6 swj post pack:k6 hyg:k6 score:k6 verdict:k6 report
# Every step skips finished work (outputs .part -> rename); rerun after any failure or STOP (touch $A/STOP).
#   prov      donor provenance: trunk of the stored swap27 mixes == later fresh packs of the same levels (trunk99.py);
#             waits for w92's fresh basectl / s1ctl mixes. Any DIFFERS stops the run (the rule's fallback: fresh packs)
#   mix       hb0 hp0 hb1 hp1: radmix into V2 3.00, even layers' trunk from the K4 donor, odd from the K2.5 donor
#             (conv.lock + ram.lock exclusive); census --mix --base (donor bytes, base elsewhere), 160 K4 + 176 K2.5
#   hyg       rxq.pairs mixes hbS hpS --arm <trunk>  (PAIR OK required before a score is read)
#   wait      until the w92 / w95 / w98 drivers have been gone for 10 min (the task: card 0 behind their GPU steps)
#   score:h   hp0 hb0 hp1 hb1 on both parts of golden-search27c (card0 + ram shared), verdict:h = verdict99.py h
#   k6        q27v3-ed97adb7-k6-i8 seed 0: w92's mkjobs92 specs, `python -m rxq job` one at a time (layer92.sh), assemble
#   swj       rxq.swjfit on the k6 level (SWJ27's qad meta, fused, no-cache; card0 + ram shared)
#   post      k6swj = apply LEVEL SWJ,LEVEL/c9 ; k6c9 = apply LEVEL LEVEL/c9 (same codes)
#   pack:k6   rxq.mix recipe -> rad-convert (install-116) -> radmix into V2 8.00 -> census (336 exl3i K6)
#   hyg:k6    rxq.pairs mixes k6swj k6c9 --arm ffn_gate,ffn_up ; score:k6, verdict:k6 = verdict99.py k6
set -u
: ${RXM_WORK:=/root}
A=${A:-/data/w99}; CODE=$A/code; EXL3=$A/exl3; W=$CODE/v3/e3/settle/w92; W9=$CODE/v3/e3/settle/w99; PY=/data/venv-exl3/bin/python
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib HOUT=/data/swap27/hout-bn
V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad V8=/data/release/qwen3.8-27b-exl3-v2-8.00bpw.rad META=/data/swj27/swj27-qad
T=/data/swap27/tools I8=/data/install-116 H8=/data/swap27/home GUARD_MB=${GUARD_MB:-3000} SW=/data/swap27
STEPS=${*:-prov mix hyg wait score:h verdict:h k6 swj post pack:k6 hyg:k6 score:k6 verdict:k6 report}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$A/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
TRUNKARM=attn_qg,attn_k,attn_v,attn_output,ssm_qkv,ssm_z,ssm_out,ffn_gate,ffn_up,ffn_down
# item 1 arms: tag -> "K4 donor mix, K2.5 donor mix"
declare -A EVEN=([hb0]=$SW/mix-bn4.rad [hp0]=$SW/mix-base4.rad [hb1]=$SW/mix-bn4s1.rad [hp1]=$SW/mix-base4s1.rad)
declare -A ODD=([hb0]=$SW/mix-bn.rad [hp0]=$SW/mix-base.rad [hb1]=$SW/mix-bns1.rad [hp1]=$SW/mix-s1.rad)
mkdir -p $A/out $A/lv $A/mix $A/scores $A/logs; cd $A
export PY W T GUARD_MB CODE EXL3
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
stopped() { [ -f $A/STOP ]; }
die() { log "W99 STOP: $*"; exit 1; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge ${1:-6} ] || die "MemAvailable $m GB < ${1:-6} GB ($2)"; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
TRUNK() { "${RX[@]}" -c "from rxq import mix; print(''.join(n + chr(10) for n in mix.trunk_names()), end='')"; }

# ------------------------------------------------------------------------------------------------------- item 1
prov() {
  local pairs="/data/cdhout27/mix-bn4ctl.rad:$SW/mix-bn4.rad /data/cdhout27/mix-bn4s1ctl.rad:$SW/mix-bn4s1.rad"
  pairs+=" /data/w92/mix/mix-basectl.rad:$SW/mix-base.rad /data/w92/mix/mix-s1ctl.rad:$SW/mix-s1.rad"
  for p in $pairs; do local f=${p%%:*} s=${p#*:} o=$A/prov-$(basename ${p#*:} .rad)
    [ -f $o.ok ] && continue
    while [ ! -f /data/w92/mix/census-$(basename $f .rad | sed 's/^mix-//').ok ] && [[ $f == /data/w92/* ]]; do
      stopped && die "STOP while waiting for $f"; sleep 120; done
    nice -n 10 "${RX[@]}" $W9/trunk99.py $f $s > $o.txt 2>&1; local rc=$?
    log "prov $(basename $s) vs fresh $(basename $f): $(tail -1 $o.txt)"
    [ $rc = 0 ] && touch $o.ok || die "stored donor $s differs from its fresh pack: the rule's fallback (fresh packs) applies"
  done
}
mixh() { local tag=$1 M=$A/mix/mix-$1.rad
  [ -f $M ] && [ -f $A/mix/census-$tag.ok ] && return 0
  stopped && die "STOP before mix $tag"
  TRUNK | while read n; do l=$(echo $n | cut -d. -f2); echo "$n $([ $((l % 2)) = 0 ] && echo ${EVEN[$tag]} || echo ${ODD[$tag]})"; done > $A/mix/$tag.mix
  ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; rm -f $M.part
    $T/radmix $M.part $V2 $A/mix/$tag.mix > $A/mix/mix-$tag.log 2>&1 && mv $M.part $M ) || die "radmix $tag"
  $PY -I $T/census.py $M --mix $A/mix/$tag.mix --base $V2 > $A/mix/census-$tag.txt 2>&1 || die "census $tag ($A/mix/census-$tag.txt)"
  grep -qE "exl3i  *4  *mul1i8 *f16  *160 " $A/mix/census-$tag.txt && grep -qE "exl3i  *2.5  *mul1i8 *f16  *176 " $A/mix/census-$tag.txt \
    || die "$tag: not 160 K4 + 176 K2.5 exl3i trunk weights"
  touch $A/mix/census-$tag.ok
  log "$tag: mix ready (K4 even from $(basename ${EVEN[$tag]}), K2.5 odd from $(basename ${ODD[$tag]})), $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}
hyg() { local a=$1 b=$2 arm=$3 f=$A/mix/hyg-$1-vs-$2
  [ -f $f.ok ] && return 0
  nice -n 10 "${RX[@]}" -m rxq.pairs mixes $A/mix/mix-$a.rad $A/mix/mix-$b.rad --arm $arm > $f.txt 2>&1
  local r=$(tail -1 $f.txt); log "hygiene $a vs $b: ${r:0:240}"
  [[ "$r" == "PAIR OK"* ]] && touch $f.ok
}
waitothers() { local quiet=0
  while :; do stopped && die "STOP while waiting"
    if ps -eo args | grep -qE '^(/bin/)?bash /data/w9[258]/'; then quiet=0; else quiet=$((quiet + 1)); fi
    [ $quiet -ge 10 ] && break; sleep 60; done
  log "wait: w92 / w95 / w98 drivers gone for 10 min"
}
score1() { local tag=$1 part=$2 G=$A/golden-search27c-$2 o=$A/scores/$1-$2
  [ -f $o.json.rows ] && return 0
  stopped && die "STOP before score $tag"; memok 6 "score $tag"
  cardrun bash -c "s=\$(date +%s); ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H8 $I8/bin/radiance --model $A/mix/mix-$tag.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $A/logs/score-$tag-$part.log 2>&1; rc=\$?
    [ \$rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo \"\$(date -u +%H:%M:%S) score $tag-$part rc=\$rc \$(( \$(date +%s) - s ))s\"" | tee -a $A/result.txt
  [ -f $o.json.rows ] || die "score $tag-$part FAIL ($A/logs/score-$tag-$part.log)"
}

# ------------------------------------------------------------------------------------------------------- item 2
k6level() { local tag=k6 jd=$A/jobs/k6 od=$A/out/k6
  if [ ! -f $jd/jobs.jsonl ]; then
    rm -rf $jd.part; "${RX[@]}" $W/mkjobs92.py 6-i8 0 $MODEL $HESS $HOUT $jd.part > $A/logs/mkjobs-k6.log 2>&1 && mv $jd.part $jd \
      || die "mkjobs k6 ($A/logs/mkjobs-k6.log)"
    log "$(tail -1 $A/logs/mkjobs-k6.log)"; fi
  mkdir -p $od
  for l in $(seq -f %02g 0 63); do
    local todo=()
    for s in $jd/L$l-g*.json; do [ -f "$s" ] || continue; local n=$(basename $s .json)
      [ -f $od/$n/job.json ] && [ "$($PY -c "import json, os; s = json.load(open('$s')); print(all(os.path.exists(f'$od/$n/t-{k}.safetensors') for k in s['keys']))")" = True ] && continue
      todo+=($n); done
    [ ${#todo[@]} = 0 ] && continue
    stopped && die "STOP before k6 L$l"; memok 6 "k6 L$l"
    cardrun bash $W/layer92.sh $od $jd "${todo[@]}" || die "k6: a job failed (L$l: $od/*/rxq.log); rerun resumes"
  done
  [ -f $od/.logged ] || { log "k6: $(ls $od | grep -c '^L') jobs done, $(awk '{s += $2} END {printf "%.0f s wall", s / 1000}' $od/wall-ms.txt)"; touch $od/.logged; }
  local d=$A/lv/k6/level
  [ -f $d/stamp.json ] && return 0
  rm -rf $d; mkdir -p $A/lv/k6
  nice -n 10 "${RX[@]}" -m rxq assemble $d $od/L* --template $A/template.json > $A/lv/k6/assemble.log 2>&1 \
    || { tail -3 $A/lv/k6/assemble.log; die "assemble k6"; }
  log "k6 assembled: $(tail -1 $A/lv/k6/assemble.log) ($(ls $d/q-L*.safetensors | wc -l) shards, $(ls $d/c9/scales 2>/dev/null | wc -l) C9 writers)"
}
swj() { [ -f $A/swj/k6/.done ] && return 0
  stopped && die "STOP before swj"; memok 8 "swj"; local s=$(date +%s); log "swj k6: start"
  cardrun env ROCR_VISIBLE_DEVICES=0 VRAM_CAP_GB=30 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    nice -n 5 "${RX[@]}" -m rxq.swjfit $MODEL $META $A/swj --scratch $A/swjscr --src k6=$A/lv/k6/level:level --fused --no-cache \
    >> $A/logs/swj.log 2>&1 < /dev/null
  [ -f $A/swj/k6/.done ] || die "swj k6 not done ($A/logs/swj.log; rerun resumes)"
  log "swj k6: done in $(( ($(date +%s) - s) / 60 )) min ($(grep -h 'fused check' $A/logs/swj.log | tail -1 | cut -c10-))"
}
post() { local t=$1 sc=$2 F=$A/lv/$1/final
  [ -f $F/.done ] && return 0
  mkdir -p $A/lv/$t
  nice -n 10 "${RX[@]}" -m rxq.post apply $A/lv/k6/level $sc $F > $A/lv/$t/post.log 2>&1 || die "post apply $t"
  [ -f $F/.done ] || die "post $t: no .done"; log "post $t: $(tail -1 $A/lv/$t/post.log)"
}
packmix() { local tag=$1 src=$2 K=6 P=$A/mix/pack-$1.rad M=$A/mix/mix-$1.rad
  [ -f $M ] && [ -f $A/mix/census-$tag.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || die "$tag: $src has not 64 layers"
  stopped && die "STOP before pack $tag"
  "${RX[@]}" -m rxq.mix recipe $src > $A/mix/$tag.recipe || die "recipe $tag"
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; cd /workspace; s=$(date +%s); rm -f $P.part
      RADIANCE_HOME=$H8 RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I8/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $A/mix/$tag.recipe --set quantization_config.quant_method=exl3 -o $P.part > $A/mix/conv-$tag.log 2>&1 && mv $P.part $P
      echo "$(date -u +%H:%M:%S) pack $tag rc=$? $(( $(date +%s) - s ))s" ) | tee -a $A/result.txt
    [ -f $P ] || { rm -f $P.part; die "pack $tag FAIL ($A/mix/conv-$tag.log)"; }
  fi
  TRUNK | sed "s|\$| $P|" > $A/mix/$tag.mix
  ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; rm -f $M.part
    $T/radmix $M.part $V8 $A/mix/$tag.mix > $A/mix/mix-$tag.log 2>&1 && mv $M.part $M ) || die "radmix $tag"
  $PY -I $T/census.py $M --mix $A/mix/$tag.mix --base $V8 > $A/mix/census-$tag.txt 2>&1 || die "census $tag ($A/mix/census-$tag.txt)"
  grep -qE "exl3i  *$K(\.0+)?  *mul1i8 *f16  *336 " $A/mix/census-$tag.txt || die "$tag: not 336 exl3i K$K mul1i8 trunk weights"
  touch $A/mix/census-$tag.ok; rm -f $P
  log "$tag: mix ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}

# ------------------------------------------------------------------------------------------------------------- main
[ -f $A/template.json ] || cp /data/w92/template.json $A/template.json
log "w99 start: steps $STEPS; settings $("${RX[@]}" -m rxq stamp)"
for st in $STEPS; do
  stopped && die "STOP"
  case $st in
    prov) prov;;
    mix) for t in hb0 hp0 hb1 hp1; do mixh $t; done;;
    hyg) for s in 0 1; do hyg hb$s hp$s $TRUNKARM || die "hb$s vs hp$s refused"; done;;
    wait) waitothers;;
    score:h) for t in hp0 hb0 hp1 hb1; do for p in x c; do score1 $t $p; done; done;;
    verdict:h) $PY -I $W9/verdict99.py $A h | tee $A/verdict-h.txt | tail -1 | while read l; do log "$l"; done;;
    k6) k6level;;
    swj) swj;;
    post) post k6swj $A/swj/k6,$A/lv/k6/level/c9; post k6c9 $A/lv/k6/level/c9;;
    pack:k6) packmix k6c9 $A/lv/k6c9/final; packmix k6swj $A/lv/k6swj/final;;
    hyg:k6) hyg k6swj k6c9 ffn_gate,ffn_up || die "k6swj vs k6c9 refused";;
    score:k6) for t in k6c9 k6swj; do for p in x c; do score1 $t $p; done; done;;
    verdict:k6) $PY -I $W9/verdict99.py $A k6 | tee $A/verdict-k6.txt | tail -1 | while read l; do log "$l"; done;;
    report) grep -h "VERDICT" $A/verdict-h.txt $A/verdict-k6.txt 2>/dev/null | while read l; do log "$l"; done;;
    *) die "unknown step $st";;
  esac
done
log "W99 DONE ($STEPS)"
