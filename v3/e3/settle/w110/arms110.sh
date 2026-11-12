#!/bin/bash
# arms110.sh [STEP ...] -- #145 (../w110.md): the scale-only arms on the three rxrun levels, gpubox. Default steps:
#   c9 finals packs mixes hyg score verdict
#   c9      per level (as soon as it is assembled): rxq.post c9 with today's settings -> gate G1 vs LEVEL/c9 (c9g1.py),
#           and with e_clip 0.3 (cfg/v3-e3.toml: one line differs) -> the e3 scales. CPU, nice, ram.lock shared
#   finals  per level (after rxrun's DONE): fS = apply LEVEL SWJ (C9 off), fSE = apply LEVEL SWJ,E3, fC = apply LEVEL C9
#           (swj off); T = rxrun's final (SWJ,C9)
#   packs   per level: pT = rad-convert of T's recipe (install-116); version packs with --reuse pT: pO (writers from fS),
#           pE (writers from fSE), pN (ffn_gate_up from fC); conv.lock + ram.lock exclusive, memguard
#   mixes   radmix into V2 3.00: M1 (all layers K1) 6 arms, M2 (even K0.5 / odd K1.5) 12 arms; census
#   hyg     rxq.pairs mixes ARM T --arm <changed families> --planes svh  (PAIR OK required)
#   score   radiance KLD on golden-search27d-x / -c (card0.lock + ram.lock shared)
#   verdict verdict110.py -> $A/verdict.txt
# Restartable: every output is written .part -> rename or has an .ok; rerun after any failure. Stop: touch $A/STOP.
set -u
: ${RXM_WORK:=/root}
A=/data/w110; CODE=$A/code; W=$CODE/v3/e3/settle/w110; LV=$A/lv; PY=/data/venv-exl3/bin/python ST=159591b2
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib
V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad T=/data/swap27/tools I8=/data/install-116 H8=/data/swap27/home
STEPS=${*:-c9 finals packs mixes hyg score verdict}
export ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RX=(env PYTHONPATH=$CODE/pipeline:$A/exl3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src $PY)
declare -A KN=([k05]=0.5 [k1]=1 [k15]=1.5)
lvdir() { echo $LV/q27v3-$ST*-k${KN[$1]}-i8; }
mkdir -p $A/c9 $A/fin $A/pack $A/mix $A/scores $A/logs $A/cfg; cd $A
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
die() { log "ARMS110 STOP: $*"; exit 1; }
stopped() { [ -f $A/STOP ] && die "STOP file"; return 0; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge $1 ] || die "MemAvailable $m GB < $1 GB ($2)"; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
exrun() { ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; "$@" ); }
shrun() { ( exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
TRUNK() { "${RX[@]}" -c "from rxq import mix; print(''.join(n + chr(10) for n in mix.trunk_names()), end='')"; }

# ------------------------------------------------------------------------------------------------------------ c9
c9cfg() { local c=$A/cfg/v3-e3.toml; [ -f $c ] && return 0
  sed -E 's/^e_clip = 0\.5( |$)/e_clip = 0.3\1/' $CODE/pipeline/config/v3.toml > $c.part
  [ "$(diff $CODE/pipeline/config/v3.toml $c.part | grep -c '^[<>]')" = 2 ] || die "e3 config: not exactly one line changed"
  mv $c.part $c; log "cfg e3: $(diff $CODE/pipeline/config/v3.toml $c | grep '^>')"
}
c9one() { local k=$1 L; L=$(lvdir $k)/level
  [ -f $A/c9/$k.ok ] && return 0
  while [ ! -f $L/stamp.json ]; do stopped; sleep 120; done
  [ -f $L/c9/.done ] || die "$k: $L/c9 not done (no in-job C9?)"
  for arm in cur:$CODE/pipeline/config/v3.toml e3:$A/cfg/v3-e3.toml; do local n=${arm%%:*} cf=${arm#*:}
    [ -f $A/c9/$k-$n/.done ] && continue
    shrun nice -n 15 "${RX[@]}" -m rxq.post c9 $L $A/c9/$k-$n --config $cf --model $MODEL --hess $HESS > $A/logs/c9-$k-$n.log 2>&1 \
      || die "c9 $k $n ($A/logs/c9-$k-$n.log)"
  done
  local r; r=$("${RX[@]}" $W/c9g1.py $A/c9/$k-cur $L/c9 2>&1 | tail -1); log "c9 $k: $r"
  [[ "$r" == "G1 PASS"* ]] || die "G1 $k"
  # amendment 1: e3 = the in-job c with only the rows the e_clip change moves replaced
  r=$("${RX[@]}" $W/c9e3.py $A/c9/$k-cur $A/c9/$k-e3 $L/c9 $A/c9/$k-e3x 2>&1 | tail -1); log "c9 $k e3x: $r"
  [ -f $A/c9/$k-e3x/.done ] || die "c9e3 $k"
  touch $A/c9/$k.ok
}
c9() { c9cfg; for k in k15 k1 k05; do c9one $k & done; wait; for k in k15 k1 k05; do [ -f $A/c9/$k.ok ] || die "c9 $k"; done; }

# --------------------------------------------------------------------------------------------------------- finals
finals() {
  for k in k15 k1 k05; do local D; D=$(lvdir $k)
    while [ ! -f $D/DONE ]; do stopped; sleep 300; done
    local SW=$LV/swj/$(basename $D); [ -f $SW/.done ] || die "$k: swj not done"
    for f in fS:$SW fSE:$SW,$A/c9/$k-e3x fC:$D/level/c9; do local n=${f%%:*} sc=${f#*:} F=$A/fin/$k-${f%%:*}
      [ -f $F/.done ] && continue; rm -rf $F $F.scales
      shrun nice -n 10 "${RX[@]}" -m rxq.post apply $D/level $sc $F > $A/logs/fin-$k-$n.log 2>&1 && [ -f $F/.done ] || die "apply $k $n"
      log "final $k-$n: $(tail -1 $A/logs/fin-$k-$n.log)"
    done
    [ -e $A/fin/$k-fT ] || ln -s $D/final $A/fin/$k-fT
  done
}

# ---------------------------------------------------------------------------------------------------------- packs
# version -> (families taken from it, final dir)
packone() { local k=$1 v=$2 P=$A/pack/$1-$2.rad R=$A/pack/$1-$2.recipe D; D=$(lvdir $k)
  [ -f $P ] && return 0; stopped; memok 24 "pack $k-$v"
  case $v in
    T) cp $D/final.recipe $R ;;
    O) sed -E "/(attn_output|ssm_out)\.weight /s|src=[^ ]+|src=$A/fin/$k-fS|" $D/final.recipe > $R ;;
    E) sed -E "/(attn_output|ssm_out)\.weight /s|src=[^ ]+|src=$A/fin/$k-fSE|" $D/final.recipe > $R ;;
    N) sed -E "/ffn_gate_up\.weight /s|src=[^ ]+|src=$A/fin/$k-fC|" $D/final.recipe > $R ;;
  esac
  [ $v = T ] || [ "$(diff $D/final.recipe $R | grep -c '^>')" = $([ $v = N ] && echo 1 || echo 2) ] || die "recipe $k-$v: wrong number of changed rules"
  local reuse=""; [ $v = T ] || reuse="--reuse $A/pack/$k-T.rad"
  exrun bash -c "cd /workspace; s0=\$(date +%s); rm -f $P.part
    RADIANCE_HOME=$H8 RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I8/bin/rad-convert models/qwen38-27b-bf16 --recipe $R \
      --set quantization_config.quant_method=exl3 $reuse -o $P.part > $A/logs/conv-$k-$v.log 2>&1 && mv $P.part $P
    echo \"\$(date -u +%H:%M:%S) pack $k-$v rc=\$? \$(( \$(date +%s) - s0 ))s\" >> $A/result.txt"
  [ -f $P ] || { rm -f $P.part; die "pack $k-$v FAIL ($A/logs/conv-$k-$v.log)"; }
}
packs() { for k in k15 k1 k05; do for v in T O E N; do packone $k $v; done; done; }

# ---------------------------------------------------------------------------------------------------------- mixes
# M1 tags k1-<c9><swj>: c9 in c/o/e, swj in s/n. M2 tags sp-<c9@0.5><c9@1.5>-<swj@0.5><swj@1.5>.
M1="k1-cs k1-os k1-es k1-cn k1-on k1-en"
M2="sp-cc-ss sp-oo-ss sp-ee-ss sp-cc-nn sp-oo-nn sp-ee-nn sp-oc-ss sp-co-ss sp-ec-ss sp-ce-ss sp-cc-ns sp-cc-sn"
# pack for a weight of level k, c9 version c (c/o/e), swj version s (s/n)
wpack() { local k=$1 c=$2 s=$3 n=$4
  case $n in
    *attn_output*|*ssm_out*) case $c in c) echo $A/pack/$k-T.rad;; o) echo $A/pack/$k-O.rad;; e) echo $A/pack/$k-E.rad;; esac;;
    *ffn_gate_up*) [ $s = s ] && echo $A/pack/$k-T.rad || echo $A/pack/$k-N.rad;;
    *) echo $A/pack/$k-T.rad;;
  esac; }
mixfile() { local tag=$1
  TRUNK | while read n; do local l=$(echo $n | cut -d. -f2)
    case $tag in
      k1-*) local c=${tag:3:1} s=${tag:4:1}; echo "$n $(wpack k1 $c $s $n)";;
      sp-*) local c5=${tag:3:1} c15=${tag:4:1} s5=${tag:6:1} s15=${tag:7:1}
            if [ $((l % 2)) = 0 ]; then echo "$n $(wpack k05 $c5 $s5 $n)"; else echo "$n $(wpack k15 $c15 $s15 $n)"; fi;;
    esac; done; }
mixone() { local tag=$1 M=$A/mix/mix-$1.rad
  [ -f $M ] && [ -f $A/mix/census-$tag.ok ] && return 0; stopped; memok 8 "mix $tag"
  mixfile $tag > $A/mix/$tag.mix
  exrun bash -c "rm -f $M.part; $T/radmix $M.part $V2 $A/mix/$tag.mix > $A/logs/radmix-$tag.log 2>&1 && mv $M.part $M" || die "radmix $tag"
  $PY -I $T/census.py $M --mix $A/mix/$tag.mix --base $V2 > $A/mix/census-$tag.txt 2>&1 || die "census $tag ($A/mix/census-$tag.txt)"
  case $tag in
    k1-*) grep -qE "exl3i  *1  *mul1i8 *f16  *336 " $A/mix/census-$tag.txt || die "$tag: not 336 exl3i K1 trunk weights";;
    sp-*) grep -qE "exl3i  *0.5  *mul1i8 *f16  *160 " $A/mix/census-$tag.txt && grep -qE "exl3i  *1.5  *mul1i8 *f16  *176 " $A/mix/census-$tag.txt \
            || die "$tag: not 160 K0.5 + 176 K1.5 exl3i trunk weights";;
  esac
  touch $A/mix/census-$tag.ok; log "mix $tag ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}
mixes() { for t in $M1 $M2; do mixone $t; done; }

# ------------------------------------------------------------------------------------------------------------ hyg
# changed families of TAG vs its mix's T (k1-cs / sp-cc-ss)
armfam() { local tag=$1 f=""
  case $tag in k1-*) [ ${tag:3:1} != c ] && f="attn_output,ssm_out"; [ ${tag:4:1} != s ] && f="${f:+$f,}ffn_gate,ffn_up";;
               sp-*) [ "${tag:3:2}" != cc ] && f="attn_output,ssm_out"; [ "${tag:6:2}" != ss ] && f="${f:+$f,}ffn_gate,ffn_up";; esac
  echo $f; }
tref() { case $1 in k1-*) echo k1-cs;; sp-*) echo sp-cc-ss;; esac; }
hyg() {
  for t in $M1 $M2; do local r=$(tref $t) f=$A/mix/hyg-$t; [ $t = $r ] && continue; [ -f $f.ok ] && continue
    nice -n 10 "${RX[@]}" -m rxq.pairs mixes $A/mix/mix-$t.rad $A/mix/mix-$r.rad --arm $(armfam $t) --planes svh > $f.txt 2>&1
    local x=$(tail -1 $f.txt); log "hygiene $t vs $r: ${x:0:260}"
    if [[ "$x" == "PAIR OK"* ]]; then touch $f.ok
    elif [ "$(grep -c '^MISMATCH' $f.txt)" = 1 ] && grep -q '^MISMATCH the arm is identical too' $f.txt; then
      echo $r > $A/mix/$t.same; log "hygiene $t: identical to $r (no-op arm: scored as $r)"
    else die "hygiene $t"; fi
  done
}

# ---------------------------------------------------------------------------------------------------------- score
score1() { local tag=$1 part=$2 G=$A/golden-search27d-$2 o=$A/scores/$1-$2
  [ -f $o.json.rows ] && return 0; stopped; memok 6 "score $tag"
  cardrun bash -c "s0=\$(date +%s); RADIANCE_HOME=$H8 $I8/bin/radiance --model $A/mix/mix-$tag.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $A/logs/score-$tag-$part.log 2>&1; rc=\$?
    [ \$rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo \"\$(date -u +%H:%M:%S) score $tag-$part rc=\$rc \$(( \$(date +%s) - s0 ))s\" >> $A/result.txt"
  [ -f $o.json.rows ] || die "score $tag-$part FAIL ($A/logs/score-$tag-$part.log)"
}
score() {
  for t in $M1 $M2; do local r=$(tref $t)
    [ $t = $r ] || [ -f $A/mix/hyg-$t.ok ] || [ -f $A/mix/$t.same ] || die "score $t: no hygiene"
    [ -f $A/mix/$t.same ] && continue
    for p in x c; do score1 $t $p; done; done
}

verdict() { nice -n 10 $PY $W/verdict110.py $A > $A/verdict.txt.part 2>&1 && mv $A/verdict.txt.part $A/verdict.txt || die "verdict"
  log "verdict: $(tail -1 $A/verdict.txt)"; }

log "arms110 start: $STEPS"
for st in $STEPS; do
  case $st in c9|finals|packs|mixes|hyg|score|verdict) $st;; *) die "unknown step $st";; esac
done
log "ARMS110 DONE ($STEPS)"
