#!/bin/bash
# a115.sh [STEP ...] -- w115 #152-01 / #153-01 (../w115.md): real-quant f2 screens on gpubox card 0, after w114.
#   wait114  until /data/w114/result.txt ends in "D114 DONE" / "D114 STOP" and no process runs from /data/w114
#   cfg      arm settings files (R: + [quant.reldlq]; C_<F>: tile CD families attn_qg + F), diff-checked, stamps.json
#   units    N (L10 / L11 / L39, K1.5 / 2.5 / 4, seeds 0 / 1), R (L10 / L39, same K x S), C_<F> (F's tensors of L10 / L11
#            / L39, K2.5 / 4, seeds 0 / 1); one rxq quant per unit under card0.lock then ram.lock (shared)
#   f2       w98's f2calc98.py over every unit (CPU, nice 10, 8 threads, ram.lock shared)
#   report   report115.py -> $A/report.txt
# Restartable: finished units have .done (after the data); rerun resumes. Stop: touch $A/STOP.
# Usage (gpubox): setsid nohup bash /data/w115/code/v3/e3/settle/w115/a115.sh > /data/w115/a/run.out 2>&1 < /dev/null &
set -u
: ${RXM_WORK:=/root}
A=/data/w115/a; CODE=/data/w115/code; S=$CODE/v3/e3/settle/w115; OUT=$A/out; CFG=$A/cfg
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib HOUT=/data/swap27/hout-bn PY=/data/venv-exl3/bin/python
STAMP=4f8319840e59c5d0
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/w115/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:/data/w115/exl3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
P=model.language_model.layers
FAMS="ssm_qkv ssm_z ffn_gate ffn_up ffn_down attn_v"
mkdir -p $OUT $CFG $A/logs; cd $A
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
die() { log "A115 STOP: $*"; exit 1; }
memok() { local m i; for i in $(seq 60); do m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge $1 ] && return 0
  [ $i = 1 ] && log "$2: MemAvailable $m GB < $1 GB, waiting"; sleep 60; done; die "MemAvailable $m GB < $1 GB for 1 h ($2)"; }

wait114() {  # the finished state must hold for 3 checks 5 min apart (w114 restarted its driver right after a STOP once)
  local n=0 ok=0
  while :; do
    [ -f $A/STOP ] && die "STOP file (waiting for w114)"
    local last; last=$(grep -E "D114 (DONE|STOP)" /data/w114/result.txt 2>/dev/null | tail -1)
    if [ -n "$last" ] && [ "$(tail -1 /data/w114/result.txt)" = "$last" ] && ! pgrep -f "/data/w114/" > /dev/null; then
      ok=$((ok + 1)); [ $ok -ge 3 ] && { log "w114 finished: $last"; return 0; }
    else ok=0; fi
    [ $((n % 12)) = 0 ] && log "waiting for w114 (last line: $(tail -1 /data/w114/result.txt 2>/dev/null | cut -c1-120))"
    n=$((n + 1)); sleep 300
  done
}

keys() { # keys LAYERS [FAMILY]: the checkpoint keys of those layers (all, or one family)
  "${RX[@]}" -c "
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
  "${RX[@]}" -c "
import json, sys; from rxq import config as C
d = sys.argv[1]; fams = sys.argv[3].split()
st = {'base': C.settings_stamp(C.load(sys.argv[2])), 'R': C.settings_stamp(C.load(d + '/R.toml'))}
st.update({f: C.settings_stamp(C.load(d + '/C_' + f + '.toml')) for f in fams})
assert st['base'] == '$STAMP', st
assert C.load(d + '/R.toml')['quant']['reldlq'] == {'enable': True, 'passes': 1}
json.dump(st, open(d + '/stamps.json', 'w'), indent=1); print(st)" $CFG $B "$FAMS" > $CFG/stamps.log 2>&1 || die "cfg stamps ($CFG/stamps.log)"
  log "cfg: $(cat $CFG/stamps.log)"
}

unit() { # unit NAME KEYS K SEED CONFIG
  local name=$1 ks=$2 K=$3 s=$4 cf=$5; local d=$OUT/$name
  [ -f $d/.done ] && return 0; [ -f $A/STOP ] && die "STOP before $name"
  memok 10 $name
  local t=$(date +%s) n; n=$(echo $ks | tr ',' '\n' | wc -l); mkdir -p $d
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    log "$name: card 0 held after $(( $(date +%s) - t )) s ($n tensors)"
    /data/swap27/tools/memguard.sh 3000 nice -n 5 "${RX[@]}" -m rxq quant --keys $ks -K $K --seed-offset $s --ckpt $MODEL --hess $HESS \
      -o $d --device 0 --save-wq --hout $HOUT --config $cf >> $d.log 2>&1 )
  local rc=$?
  [ $rc = 0 ] && [ $(ls $d/wq/*.safetensors 2>/dev/null | wc -l) = $n ] && [ $(ls $d/t-*.safetensors | wc -l) = $n ] && touch $d/.done
  log "$name: rc $rc, $(( $(date +%s) - t )) s wall (incl. lock wait)"
  [ -f $d/.done ] || die "unit $name failed (see $d.log)"
}

units() {
  local B=$CODE/pipeline/config/v3.toml NK RK; NK=$(keys 10,11,39); RK=$(keys 10,39)
  for K in 2.5 4 1.5; do for s in 0 1; do
    unit N-K$K-s$s $NK $K $s $B; unit R-K$K-s$s $RK $K $s $CFG/R.toml; done; done
  for F in $FAMS; do for K in 2.5 4; do for s in 0 1; do
    unit C-$F-K$K-s$s $(keys 10,11,39 $F) $K $s $CFG/C_$F.toml; done; done; done
}

f2() {
  memok 12 f2
  ( exec 7>$RXM_WORK/ram.lock; flock -s 7
    THREADS=8 nice -n 10 "${RX[@]}" -I $CODE/v3/e3/settle/w98/f2calc98.py $OUT $MODEL $HESS $HOUT $CODE/pipeline ) >> $A/logs/f2.log 2>&1 \
    && log "f2: done ($(grep -c ' f2 ' $A/logs/f2.log) lines this and earlier runs)" || die "f2calc98 (logs/f2.log)"
}

report() {
  "${RX[@]}" $S/report115.py $OUT /data/w114/lv > $A/report.txt.part 2>&1 && mv $A/report.txt.part $A/report.txt \
    && log "report: $(tail -1 $A/report.txt)" || die "report115 (report.txt.part)"
}

for st in ${*:-wait114 cfg units f2 report}; do
  case $st in wait114|cfg|units|f2|report) $st ;; *) die "unknown step $st" ;; esac
done
log "A115 DONE: ${*:-wait114 cfg units f2 report}"
