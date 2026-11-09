#!/bin/bash
# w92.sh [STEP ...] -- #122 (../w92.md): full-stack pass-1 levels from v3.toml through the donors' job path, gpubox card 0.
# Default STEPS: slice sub1 fs0 fs1 swjA ctl post:fs0 post:fs1 pack:fs0 pack:fs1 refs score:A verdict k1 swjB post:k1
#                pack:k1 score:B report. Every step skips finished work (outputs written .part -> rename); rerun after any
# failure or a STOP (touch $A/STOP: checked between jobs / steps).
#   slice     golden-search27c-{c,x} (calib_slice + golden_pick; pickx84 on xc8k-b2-c2), CPU, ram.lock shared
#   sub1      K0.875 / K0.125 i4 jobs of layers 2-3 (the sub-1 damping check) + partial assembly
#   fs0 fs1 k1   a level's 256 group jobs (mkjobs92.py specs, `python -m rxq job`, one at a time; card0 + ram shared per
#             layer), then rxq assemble (+ template)
#   swjA swjB rxq.swjfit over fs0+fs1 (A) / k1 (B) in one staged pass (card0 + ram shared for the pass)
#   post:T    rxq.post apply LEVEL SWJ,LEVEL/c9 FINAL
#   ctl       fresh controls basectl / s1ctl packed into V2 3.00 (CTL_FRESH=1)
#   pack:T    rxq.mix recipe FINAL -> rad-convert (conv.lock + ram.lock EXCLUSIVE) -> radmix into V2 3.00 -> census
#   refs      hygiene of every mix vs its control (rxq.pairs mixes, trunk arm); bn+swj reference mixes repacked fresh
#             only if their stored mix fails it
#   score:A|B every mix of the K2.5 pairs (A) / k1 (B) on both parts (radiance, bf16 KV, card0 + ram shared)
#   verdict   verdict92.py (the gate); report = verdict + k1 + sub1 + timings
# gpubox paths; Log $A/result.txt (one line a step), per-job logs under $A/out/<tag>/<job>/rxq.log.
set -u
: ${RXM_WORK:=/root}
A=${A:-/data/w92}; CODE=$A/code; EXL3=$A/exl3; W=$CODE/v3/e3/settle/w92; PY=/data/venv-exl3/bin/python
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib HOUT=/data/swap27/hout-bn
V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad META=/data/swj27/swj27-qad T=/data/swap27/tools
I8=/data/install-116 H8=/data/swap27/home        # K2.5 packs / scores: every earlier K2.5 pair's install / home
I4=/data/install-117 H4=/data/a4cost/home         # K1 i4: the only home with the mul1i4 importer (#46-03)
GX=$A/golden-search27c-x GC=$A/golden-search27c-c SEED=9222 GUARD_MB=${GUARD_MB:-3000}
STEPS=${*:-slice sub1 fs0 fs1 swjA ctl post:fs0 post:fs1 pack:fs0 pack:fs1 refs score:A verdict k1 swjB post:k1 pack:k1 score:B report}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$A/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
declare -A LEVEL=([fs0]=2.5-i8 [fs1]=2.5-i8 [k1]=1-i4) SEEDOF=([fs0]=0 [fs1]=1 [k1]=0)
declare -A CTLLV=([basectl]=/data/v3ladder2/K2.5 [s1ctl]=/data/swap27/q-s1/K2.5)
declare -A REFLV=([bnswj]=/data/swj27/lv/bnswj [bns1swj]=/data/swj27/lv/bns1swj)
TRUNKARM=attn_qg,attn_k,attn_v,attn_output,ssm_qkv,ssm_z,ssm_out,ffn_gate,ffn_up,ffn_down
mkdir -p $A/out $A/lv $A/mix $A/scores $A/logs; cd $A
export PY W T GUARD_MB CODE EXL3
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
stopped() { [ -f $A/STOP ]; }
die() { log "W92 STOP: $*"; exit 1; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge ${1:-6} ] || die "MemAvailable $m GB < ${1:-6} GB ($2)"; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }

# ------------------------------------------------------------------------------------------------------------ slice
slice() {
  if [ ! -f $GC/kld.json ]; then
    docs=$(EXCLUDE_GOLDEN=/data/swap27/golden-search27,/data/w91/golden-search27b-c POOLS=/root/challenge/pools-v3.json \
           python3 -I $T/calib_slice.py /data/challenge/corpus/calib.jsonl 30000 8192 $SEED search 2> $GC.slice.log) || die "calib_slice"
    echo "$docs" > $GC.docs; rm -rf $GC.part
    ( exec 7>$RXM_WORK/ram.lock; flock -s 7; nice -n 10 $PY -I $T/golden_pick.py /data/challenge/golden/calib \
        /data/challenge/corpus/calib.jsonl $GC.part $GC.jsonl $docs ) > $GC.pick.log 2>&1 && mv $GC.part $GC || die "golden_pick (calib part)"
    log "slice calib part: $(cat $GC.slice.log)"
  fi
  if [ ! -f $GX/kld.json ]; then
    rm -rf $GX.part
    ( exec 7>$RXM_WORK/ram.lock; flock -s 7; nice -n 10 $PY -I $CODE/v3/e3/settle/w84/pickx84.py /data/night1105/gold/xc8k-b2-c2 \
        /data/night1105/corpora/chunks/xc8k-b2-c2.jsonl /data/night1105/gold/xc8k-b2-c2/corpus.meta.jsonl 65 $SEED $GX.part $GX.jsonl \
        --exclude-orig /data/hin27/mX/kld.json ) 2> $GX.pick.log && mv $GX.part $GX && mv $GX.part.docs $GX.docs || die "pickx84 (xcorp part)"
    log "slice xcorp part: $(head -1 $GX.pick.log)"
  fi
  for g in $GC $GX; do [ -f $g.summary ] || $PY -c "import json, numpy as np; m = json.load(open('$g/kld.json')); g = np.fromfile('$g/gen.u8', np.uint8)
print('$(basename $g)', len(m['docs']), 'docs', sum(d['tokens'] for d in m['docs']), 'tokens', m['rows'], 'rows', int(g.sum()), 'scored')" > $g.summary
    log "$(cat $g.summary)"; done
}

# ------------------------------------------------------------------------------------------------------------- jobs
# one level's jobs: TAG LEVEL SEED [--layers A-B]; specs in $A/jobs/TAG, outputs in $A/out/TAG/<LNN-gG>
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
    || { cat $A/lv/$tag/assemble.log | tail -3; die "assemble $tag"; }
  log "$tag assembled: $(tail -1 $A/lv/$tag/assemble.log) ($(ls $d/q-L*.safetensors | wc -l) shards, $(ls $d/c9/scales 2>/dev/null | wc -l) C9 writers)"
}
level() { local tag=$1; jobs $tag ${LEVEL[$tag]} ${SEEDOF[$tag]}; assemble $tag; }
sub1() {
  for lv in 0.875-i4 0.125-i4; do local tag=sub1-${lv%-i4}; jobs $tag $lv 0 --layers 2-3
    [ -f $A/lv/$tag/level/stamp.json ] || assemble $tag --keys-from; done
  [ -f $A/sub1.txt ] || "${RX[@]}" $W/sub1chk92.py $A > $A/sub1.txt || die "sub1chk92"
  log "$(tail -1 $A/sub1.txt) ($A/sub1.txt)"
}

# ---------------------------------------------------------------------------------------------------- swj and post
swj() { local name=$1; shift; local srcs=() todo=()
  for t in "$@"; do [ -f $A/swj/$t/.done ] || { todo+=($t); srcs+=(--src $t=$A/lv/$t/level:level); }; done
  [ ${#todo[@]} = 0 ] && return 0
  stopped && die "STOP before swj $name"; memok 8 "swj $name"; s=$(date +%s)
  log "swj $name (${todo[*]}): start"
  cardrun env ROCR_VISIBLE_DEVICES=0 VRAM_CAP_GB=30 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    nice -n 5 "${RX[@]}" -m rxq.swjfit $MODEL $META $A/swj --scratch $A/swjscr-$name "${srcs[@]}" --fused --no-cache \
    >> $A/logs/swj-$name.log 2>&1 < /dev/null
  for t in "${todo[@]}"; do [ -f $A/swj/$t/.done ] || die "swj $name: $t not done ($A/logs/swj-$name.log; rerun resumes)"; done
  log "swj $name: done in $(( ($(date +%s) - s) / 60 )) min ($(grep -h 'fused check' $A/logs/swj-$name.log | tail -1 | cut -c10-))"
}
post() { local t=$1; local F=$A/lv/$t/final   # two statements: in one `local`, $t expanded to the caller's t (11-09 bug)
  [ -f $F/.done ] && return 0
  nice -n 10 "${RX[@]}" -m rxq.post apply $A/lv/$t/level $A/swj/$t,$A/lv/$t/level/c9 $F > $A/lv/$t/post.log 2>&1 || die "post apply $t"
  [ -f $F/.done ] || die "post $t: no .done"; log "post $t: $(tail -1 $A/lv/$t/post.log)"
}

# ---------------------------------------------------------------------------------------------------------- packs
# packmix TAG LEVEL K CB INSTALL HOME -> $A/mix/mix-TAG.rad (+ census); swap27k packmixk with the 11-07 lock rule
packmix() { local tag=$1 src=$2 K=$3 cb=$4 I=$5 H=$6 P=$A/mix/pack-$1.rad M=$A/mix/mix-$1.rad
  [ -f $M ] && [ -f $A/mix/census-$tag.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || die "$tag: $src has not 64 layers"
  stopped && die "STOP before pack $tag"
  if [ "$cb" = from-level ]; then "${RX[@]}" -m rxq.mix recipe $src > $A/mix/$tag.recipe || die "recipe $tag"
  else TRUNK_PATS | sed "s|\$| exl3i src=$src codebook=$cb rate=$K|" > $A/mix/$tag.recipe; fi
  local mcb=$(grep -o 'codebook=[a-z0-9]*' $A/mix/$tag.recipe | sort -u | cut -d= -f2)
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; cd /workspace; s=$(date +%s); rm -f $P.part
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I/bin/rad-convert models/qwen38-27b-bf16 \
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
TRUNK_PATS() { TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u; }
hyg() { local a=$1 b=$2 f=$A/mix/hyg-$1-vs-$2
  [ -f $f.ok ] && return 0
  "${RX[@]}" -m rxq.pairs mixes $A/mix/mix-$a.rad $3 --arm $TRUNKARM > $f.txt 2>&1
  local r=$(tail -1 $f.txt); log "hygiene $a vs $b: ${r:0:200}"
  [[ "$r" == "PAIR OK"* ]] && touch $f.ok
}
refs() {
  hyg fs0 basectl $A/mix/mix-basectl.rad || die "fs0 vs basectl refused"
  [ -f $A/mix/mix-fs1.rad ] && { hyg fs1 s1ctl $A/mix/mix-s1ctl.rad || die "fs1 vs s1ctl refused"; }
  for r in bnswj:basectl bns1swj:s1ctl; do local t=${r%:*} c=${r#*:}
    [ -f $A/mix/mix-$t.rad ] || ln -s /data/swj27/mix-$t.rad $A/mix/mix-$t.rad       # the stored #104 mix first
    if ! hyg $t $c $A/mix/mix-$c.rad; then
      log "$t: the stored mix fails hygiene vs $c: packed fresh from ${REFLV[$t]}"
      rm -f $A/mix/mix-$t.rad $A/mix/census-$t.ok; packmix $t ${REFLV[$t]} 2.5 mul1i8 $I8 $H8; hyg $t $c $A/mix/mix-$c.rad || die "$t fresh vs $c refused"
    fi; done
}

# ---------------------------------------------------------------------------------------------------------- score
score1() { local tag=$1 part=$2 I=$3 H=$4 G=$A/golden-search27c-$2 o=$A/scores/$1-$2
  [ -f $o.json.rows ] && return 0
  stopped && die "STOP before score $tag"
  cardrun bash -c "s=\$(date +%s); ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H $I/bin/radiance --model $A/mix/mix-$tag.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $A/logs/score-$tag-$part.log 2>&1; rc=\$?
    [ \$rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo \"\$(date -u +%H:%M:%S) score $tag-$part rc=\$rc \$(( \$(date +%s) - s ))s\"" | tee -a $A/result.txt
  [ -f $o.json.rows ] || die "score $tag-$part FAIL ($A/logs/score-$tag-$part.log)"
}

# ------------------------------------------------------------------------------------------------------------- main
[ -f $A/template.json ] || $PY -c "import json; c = json.load(open('$MODEL/config.json')); c['quantization_config'] = {'quant_method': 'exl3'}
json.dump(c, open('$A/template.json', 'w'), indent=1)"
log "w92 start: steps $STEPS; settings $("${RX[@]}" -m rxq stamp)"
for st in $STEPS; do
  stopped && die "STOP"
  case $st in
    slice) slice;;
    sub1) sub1;;
    fs0|fs1|k1) level $st;;
    swjA) swj A fs0 $([ -f $A/lv/fs1/level/stamp.json ] && echo fs1);;
    swjB) swj B k1;;
    post:*) post ${st#post:};;
    ctl) for c in basectl s1ctl; do packmix $c ${CTLLV[$c]} 2.5 mul1i8 $I8 $H8; done;;
    pack:k1) packmix k1 $A/lv/k1/final 1 from-level $I4 $H4;;
    pack:*) packmix ${st#pack:} $A/lv/${st#pack:}/final 2.5 from-level $I8 $H8;;
    refs) refs;;
    score:A) for t in basectl fs0 bnswj s1ctl fs1 bns1swj; do [ -f $A/mix/mix-$t.rad ] || continue
               for p in x c; do score1 $t $p $I8 $H8; done; done;;
    score:B) for p in x c; do score1 k1 $p $I4 $H4; done;;
    verdict) $PY -I $W/verdict92.py $A | tee $A/verdict92.txt | tail -1 | while read l; do log "$l"; done;;
    report) $PY -I $W/verdict92.py $A --k1 | tee $A/k1.txt | tail -1 | while read l; do log "$l"; done;;
    *) die "unknown step $st";;
  esac
done
log "W92 DONE ($STEPS)"
