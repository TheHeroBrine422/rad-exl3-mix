#!/bin/bash
# a115b.sh [STEP ...] -- w115 #152-01 / #153-01 on box2 (../w115.md, amendment 1; a115.sh was the gpubox version).
#   cfg      arm settings files (R: + [quant.reldlq]; C_<F>: tile CD families attn_qg + F), diff-checked, stamps.json
#   units    N (L10 / L11 / L39, K2.5 / 4 / 1.5, seeds 0 / 1), R (L10 / L39), C_<F> (F's tensors of L10 / L11 / L39, K2.5 / 4,
#            seeds 0 / 1); LANES lanes (one per box2 card, HIP index), each unit under card<N>.lock then ram.lock shared;
#            the lanes take units in order from one list (claims under q.lock)
#   f2       w98's f2calc98.py over every unit (CPU, nice 10, 16 threads, ram.lock shared)
#   report   report115.py OUT c2-w114.json -> $A/report.txt
# Restartable: finished units have .done; a rerun redoes unfinished ones. Stop: touch $A/STOP (lanes stop between units).
# Usage (box2): setsid nohup bash /data/w115/scripts/a115b.sh > /data/w115/a/run.out 2>&1 < /dev/null &
set -u
source /data/w115/env.sh
A=/data/w115/a; CODE=/data/w115/code; S=/data/w115/scripts; OUT=$A/out; CFG=$A/cfg; LK=/data/q27/locks
MODEL=/data/q27/models/qwen38-27b-bf16 HESS=/data/q27/challenge/hessians/calib HOUT=/data/q27/hout/bn1920
STAMP=4f8319840e59c5d0 LANES=${LANES:-0 1 2 3}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
P=model.language_model.layers
FAMS="ssm_qkv ssm_z ffn_gate ffn_up ffn_down attn_v"
mkdir -p $OUT $CFG $A/logs; cd $A
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
die() { log "A115 STOP: $*"; exit 1; }
memok() { local m i; for i in $(seq 60); do m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge $1 ] && return 0
  [ $i = 1 ] && log "$2: MemAvailable $m GB < $1 GB, waiting"; sleep 60; done; die "MemAvailable $m GB < $1 GB for 1 h ($2)"; }

keys() { # keys LAYERS [FAMILY]: the checkpoint keys of those layers (all, or one family)
  $PY -c "
import sys; from rxq import qwen35
ls = [int(x) for x in sys.argv[1].split(',')]; f = sys.argv[2] if len(sys.argv) > 2 else None
ks = [k for l in ls for _, g in qwen35.layer_plan(l) for k in g if f is None or qwen35.family(k) == f]
print(','.join(ks))" "$@"; }

cfg() {
  [ -f $CFG/stamps.json ] && return 0
  local B=$CODE/pipeline/config/v3.toml
  { cat $B; printf '\n[quant.reldlq]\n# #152 (w115): re-LDLQ after the refit, one extra pass\nenable = true\npasses = 1\n'; } > $CFG/R.toml
  [ "$(diff $B $CFG/R.toml | grep -c '^>')" = 5 ] && [ "$(diff $B $CFG/R.toml | grep -c '^<')" = 0 ] || die "cfg R: diff"
  for F in $FAMS; do
    sed -E "s/^families = \[\"attn_qg\"\]/families = [\"attn_qg\", \"$F\"]/" $B > $CFG/C_$F.toml
    [ "$(diff $B $CFG/C_$F.toml | grep -c '^[<>]')" = 2 ] || die "cfg C_$F: not exactly one line changed"
  done
  $PY -c "
import json, sys; from rxq import config as C
d = sys.argv[1]; fams = sys.argv[3].split()
st = {'base': C.settings_stamp(C.load(sys.argv[2])), 'R': C.settings_stamp(C.load(d + '/R.toml'))}
st.update({f: C.settings_stamp(C.load(d + '/C_' + f + '.toml')) for f in fams})
assert st['base'] == '$STAMP', st
assert C.load(d + '/R.toml')['quant']['reldlq'] == {'enable': True, 'passes': 1}
json.dump(st, open(d + '/stamps.json', 'w'), indent=1); print(st)" $CFG $B "$FAMS" > $CFG/stamps.log 2>&1 || die "cfg stamps ($CFG/stamps.log)"
  log "cfg: $(cat $CFG/stamps.log)"
}

mklist() { # $A/units.txt: NAME KEYS K SEED CONFIG, in run order (N before R at each K x S: c3 needs both)
  [ -f $A/units.txt ] && return 0
  local B=$CODE/pipeline/config/v3.toml NK RK; NK=$(keys 10,11,39); RK=$(keys 10,39)
  { for K in 2.5 4 1.5; do for s in 0 1; do
      echo "N-K$K-s$s $NK $K $s $B"; echo "R-K$K-s$s $RK $K $s $CFG/R.toml"; done; done
    for F in $FAMS; do for K in 2.5 4; do for s in 0 1; do
      echo "C-$F-K$K-s$s $(keys 10,11,39 $F) $K $s $CFG/C_$F.toml"; done; done; done; } > $A/units.txt.part
  mv $A/units.txt.part $A/units.txt; log "units: $(wc -l < $A/units.txt)"
}

claim() { # next unit not done and not claimed this run -> stdout
  ( flock 9; while read name rest; do
      [ -f $OUT/$name/.done ] && continue; grep -qx "$name" $A/claims.txt 2>/dev/null && continue
      echo $name >> $A/claims.txt; echo "$name $rest"; break
    done < $A/units.txt ) 9>$A/q.lock
}

unit() { # unit CARD NAME KEYS K SEED CONFIG
  local c=$1 name=$2 ks=$3 K=$4 s=$5 cf=$6; local d=$OUT/$name
  memok 24 $name
  local t=$(date +%s) n; n=$(echo $ks | tr ',' '\n' | wc -l); rm -rf $d; mkdir -p $d
  ( exec 5>$LK/card$c.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7
    log "$name: card $c held after $(( $(date +%s) - t )) s ($n tensors)"
    ROCR_VISIBLE_DEVICES=$c nice -n 5 $PY -m rxq quant --keys $ks -K $K --seed-offset $s --ckpt $MODEL --hess $HESS \
      -o $d --device 0 --save-wq --hout $HOUT --config $cf >> $d.log 2>&1 )
  local rc=$?
  [ $rc = 0 ] && [ $(ls $d/wq/*.safetensors 2>/dev/null | wc -l) = $n ] && [ $(ls $d/t-*.safetensors | wc -l) = $n ] && touch $d/.done
  log "$name: card $c rc $rc, $(( $(date +%s) - t )) s wall (incl. lock wait)"
  [ -f $d/.done ]
}

lane() { local c=$1 u
  while :; do
    [ -f $A/STOP ] && { log "lane $c: STOP file"; return 1; }
    u=$(claim); [ -z "$u" ] && return 0
    unit $c $u || { log "lane $c: unit ${u%% *} failed (see $OUT/${u%% *}.log)"; return 1; }
  done
}

units() {
  mklist; rm -f $A/claims.txt
  local pids=() c
  for c in $LANES; do lane $c & pids+=($!); sleep 20; done
  local fail=0; for p in ${pids[@]}; do wait $p || fail=1; done
  [ $fail = 0 ] || die "a lane failed"
  local left; left=$(while read name _; do [ -f $OUT/$name/.done ] || echo $name; done < $A/units.txt)
  [ -z "$left" ] || die "units not done: $left"
  log "units: all $(wc -l < $A/units.txt) done"
}

f2() {
  memok 24 f2
  ( exec 7>$LK/ram.lock; flock -s 7
    THREADS=16 nice -n 10 $PY -I $CODE/v3/e3/settle/w98/f2calc98.py $OUT $MODEL $HESS $HOUT $CODE/pipeline ) >> $A/logs/f2.log 2>&1 \
    && log "f2: done ($(grep -c ' f2 ' $A/logs/f2.log) lines this and earlier runs)" || die "f2calc98 (logs/f2.log)"
}

report() {
  $PY $S/report115.py $OUT /data/w115/c2-w114.json > $A/report.txt.part 2>&1 && mv $A/report.txt.part $A/report.txt \
    && log "report: $(tail -1 $A/report.txt)" || die "report115 (report.txt.part)"
}

for st in ${*:-cfg units f2 report}; do
  case $st in cfg|mklist|units|f2|report) $st ;; *) die "unknown step $st" ;; esac
done
log "A115 DONE: ${*:-cfg units f2 report}"
