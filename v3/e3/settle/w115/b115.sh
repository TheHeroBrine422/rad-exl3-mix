#!/bin/bash
# b115.sh [STEP ...] -- w115 #152-02 / #153-02 (../w115.md): 27B K2.5 KL pairs for the arms that passed (a), gpubox card 0.
# Inputs (written by hand from report115's GATES line before launch): $B/arms ("R", "C" or "R C"), $B/cd-families (C's
# extra tile CD families, comma list), $B/slice (golden prefix, e.g. /data/w114/golden-search27f: <prefix>-x, <prefix>-c).
#   cfg     C's settings file (tile CD families attn_qg + cd-families), diff-checked
#   levels  rxrun K2.5 i8 levels (mkconf115.py: arm settings, seed offset): N1, then per arm <ARM>0, <ARM>1; N0 = w114's
#           K2.5 level (stamp 4f831984, seed 0) and its V2 3.00 mix mix-k25-1m (its 27f scores reused when the slice is 27f)
#   packs   rad-convert of each level's final.recipe (install-116; conv.lock + ram.lock exclusive, memguard), radmix of the
#           336 trunk weights into V2 3.00, census (336 exl3i K2.5)
#   hyg     rxq.pairs mixes <ARM><S> N<S> --arm <all trunk families> (PAIR OK required before any score)
#   score   radiance KLD on both parts (card0.lock + ram.lock shared)
#   verdict verdict115.py per arm
# Restartable (outputs .part -> rename, rxrun's DONE); stop: touch $B/STOP. Usage (gpubox):
#   setsid nohup bash /data/w115/code/v3/e3/settle/w115/b115.sh > /data/w115/b/run.out 2>&1 < /dev/null &
set -u
: ${RXM_WORK:=/root}
B=/data/w115/b; CODE=/data/w115/code; W=$CODE/v3/e3/settle/w115; PY=/data/venv-exl3/bin/python
MODEL=/data/workspace/models/qwen38-27b-bf16 V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad T=/data/swap27/tools I8=/data/install-116 H8=/data/swap27/home
N0LV=$(ls -d /data/w114/lv/q27v3-4f831984*-k2.5-i8 2>/dev/null | head -1) N0MIX=/data/w114/mix/mix-k25-1m.rad
FAMS=attn_qg,attn_k,attn_v,attn_output,ssm_qkv,ssm_z,ssm_out,ffn_gate,ffn_up,ffn_down
export ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RX=(env PYTHONPATH=$CODE/pipeline:/data/w115/exl3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src $PY)
mkdir -p $B/cfg $B/pack $B/mix $B/scores $B/logs; cd $B
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $B/result.txt; }
die() { log "B115 STOP: $*"; exit 1; }
stopped() { [ -f $B/STOP ] && die "STOP file"; return 0; }
memok() { local m i; for i in $(seq 90); do m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge $1 ] && return 0
  stopped; [ $i = 1 ] && log "$2: MemAvailable $m GB < $1 GB, waiting"; sleep 120; done; die "MemAvailable $m GB < $1 GB for 3 h ($2)"; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
exrun() { ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; "$@" ); }
TRUNK() { "${RX[@]}" -c "from rxq import mix; print(''.join(n + chr(10) for n in mix.trunk_names()), end='')"; }
ARMS=$(cat $B/arms 2>/dev/null) || die "no $B/arms"
SL=$(cat $B/slice 2>/dev/null) || die "no $B/slice"; GX=$SL-x; GC=$SL-c
[ -f $GX/kld.json ] && [ -f $GC/kld.json ] || die "slice $SL: missing parts"

settings() { case $1 in N) echo $CODE/pipeline/config/v3.toml;; R) echo /data/w115/a/cfg/R.toml;; C) echo $B/cfg/Cb.toml;; esac; }
lvroot() { echo $B/lv-$1$2; }
lvdir() { if [ "$1$2" = N0 ]; then echo $N0LV; else ls -d $(lvroot $1 $2)/q27v3-*-k2.5-i8 2>/dev/null | head -1; fi; }

cfg() {
  case " $ARMS " in *" C "*) ;; *) return 0;; esac
  [ -f $B/cfg/Cb.toml ] && return 0
  local F; F=$(cat $B/cd-families) || die "no cd-families"; local L="families = [\"attn_qg\""
  for f in ${F//,/ }; do L="$L, \"$f\""; done; L="$L]"
  sed -E "s/^families = \[\"attn_qg\"\]/$L/" $CODE/pipeline/config/v3.toml > $B/cfg/Cb.toml
  [ "$(diff $CODE/pipeline/config/v3.toml $B/cfg/Cb.toml | grep -c '^[<>]')" = 2 ] || die "cfg C: not exactly one line changed"
  log "cfg C: $(grep '^families' $B/cfg/Cb.toml)"
}

n0check() {
  [ -f $B/n0.ok ] && return 0
  [ -n "$N0LV" ] && [ -f $N0LV/DONE ] && [ -f $N0MIX ] || die "N0: w114's K2.5 level / mix missing ($N0LV, $N0MIX)"
  local so; so=$($PY -c "import json, glob; print(sorted({json.load(open(f))['seed_offset'] for f in glob.glob('$N0LV/specs/L*.json')}))")
  [ "$so" = "[0]" ] || die "N0: seed offsets $so"
  grep -q 4f8319840e59c5d0 $N0LV/specs/L00-g0.json || die "N0: stamp"
  ln -sf $N0MIX $B/mix/mix-N0.rad; touch $B/mix/census-N0.ok
  log "N0 = $N0LV + $N0MIX (seed 0, stamp 4f831984)"; touch $B/n0.ok
}

level() { local a=$1 s=$2 R; R=$(lvroot $a $s)
  [ -n "$(lvdir $a $s)" ] && [ -f $(lvdir $a $s)/DONE ] && return 0; stopped
  "${RX[@]}" $W/mkconf115.py $W/lv115-tpl.json $(settings $a) $R $s $B/cfg/lv-$a$s.json > $B/logs/conf-$a$s.log 2>&1 \
    || die "mkconf $a$s ($B/logs/conf-$a$s.log)"
  log "level $a$s: $(tail -1 $B/logs/conf-$a$s.log)"
  ( cd $B; env PYTHONPATH=$CODE/pipeline $PY -m rxrun.run $B/cfg/lv-$a$s.json >> $B/logs/rxrun-$a$s.out 2>&1 ) || die "rxrun $a$s"
  [ -f $(lvdir $a $s)/DONE ] || die "rxrun $a$s: not DONE (held?)"
  log "level $a$s done: $(grep -h '"seconds"' $(lvdir $a $s)/out/*/job.json | awk -F: '{s += $2} END {printf "%.0f job-s", s}')"
}

pack() { local t=$1 L P=$B/pack/$1.rad; L=$(lvdir ${t:0:1} ${t:1:1})
  [ -f $P ] && return 0; stopped
  local try; for try in 1 2; do [ -f $P ] && break; memok 24 "pack $t"
  exrun bash -c "cd /workspace; s0=\$(date +%s); rm -f $P.part
    RADIANCE_HOME=$H8 RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I8/bin/rad-convert models/qwen38-27b-bf16 --recipe $L/final.recipe \
      --set quantization_config.quant_method=exl3 -o $P.part > $B/logs/conv-$t.log 2>&1 && mv $P.part $P
    echo \"\$(date -u +%H:%M:%S) pack $t rc=\$? \$(( \$(date +%s) - s0 ))s\" >> $B/result.txt"
  [ -f $P ] || { rm -f $P.part; cp $B/logs/conv-$t.log $B/logs/conv-$t.try$try.log; log "pack $t try $try failed"; }; done
  [ -f $P ] || die "pack $t FAIL ($B/logs/conv-$t.log)"
}

mix() { local t=$1 M=$B/mix/mix-$1.rad
  [ -f $M ] && [ -f $B/mix/census-$t.ok ] && return 0; stopped; memok 8 "mix $t"; pack $t
  TRUNK | awk -v p=$B/pack/$t.rad '{print $1, p}' > $B/mix/$t.mix
  [ "$(wc -l < $B/mix/$t.mix)" = 336 ] || die "mixfile $t"
  exrun bash -c "rm -f $M.part; $T/radmix $M.part $V2 $B/mix/$t.mix > $B/logs/radmix-$t.log 2>&1 && mv $M.part $M" || die "radmix $t"
  $PY -I $T/census.py $M --mix $B/mix/$t.mix --base $V2 > $B/mix/census-$t.txt 2>&1 || die "census $t"
  grep -qE "exl3i  *2.5  *mul1i8 *f16  *336 " $B/mix/census-$t.txt || die "$t: not 336 exl3i K2.5"
  touch $B/mix/census-$t.ok; log "mix $t ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}

hyg() { local a=$1 s=$2 f=$B/mix/hyg-$1$2
  [ -f $f.ok ] && return 0
  nice -n 10 "${RX[@]}" -m rxq.pairs mixes $B/mix/mix-$a$s.rad $B/mix/mix-N$s.rad --arm $FAMS > $f.txt 2>&1
  local x; x=$(tail -1 $f.txt); log "hygiene $a$s vs N$s: ${x:0:300}"
  [[ "$x" == "PAIR OK"* ]] && touch $f.ok || die "hygiene $a$s"
}

score1() { local t=$1 p=$2 G o=$B/scores/$1-$2; G=$SL-$p
  [ -f $o.json.rows ] && return 0
  if [ $t = N0 ] && [ "$SL" = /data/w114/golden-search27f ] && [ -f /data/w114/scores/k25-1m-f-$p.json.rows ]; then
    ln -s /data/w114/scores/k25-1m-f-$p.json.rows $o.json.rows; ln -s /data/w114/scores/k25-1m-f-$p.json $o.json
    log "score N0-$p: w114's k25-1m on 27f reused"; return 0; fi
  stopped; memok 6 "score $t"
  cardrun bash -c "s0=\$(date +%s); RADIANCE_HOME=$H8 $I8/bin/radiance --model $B/mix/mix-$t.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $B/logs/score-$t-$p.log 2>&1; rc=\$?
    [ \$rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo \"\$(date -u +%H:%M:%S) score $t-$p rc=\$rc \$(( \$(date +%s) - s0 ))s\" >> $B/result.txt"
  [ -f $o.json.rows ] || die "score $t-$p FAIL ($B/logs/score-$t-$p.log)"
}

main() {
  cfg; n0check
  for a in $ARMS; do level $a 0; done
  level N 1
  for a in $ARMS; do level $a 1; done
  for t in N1 $(for a in $ARMS; do echo ${a}0 ${a}1; done); do mix $t; done
  for a in $ARMS; do hyg $a 0; hyg $a 1; done
  for t in N0 N1 $(for a in $ARMS; do echo ${a}0 ${a}1; done); do for p in x c; do score1 $t $p; done; done
  for a in $ARMS; do
    nice -n 10 $PY $W/verdict115.py $B $a $GX $GC > $B/verdict-$a.txt.part 2>&1 && mv $B/verdict-$a.txt.part $B/verdict-$a.txt \
      || die "verdict $a"; log "verdict $a: $(tail -1 $B/verdict-$a.txt)"; done
}
log "b115 start (pid $$): arms $ARMS, slice $SL"
main && log "B115 DONE"
