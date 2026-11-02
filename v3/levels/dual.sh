#!/bin/bash
# dual.sh CARD K -- two ourq processes on one card vs one: does a second level fill the first's non-kernel gaps?
# Ladder settings, layers 3-4 (one alone, then two at once on different seeds' outputs). Out /data/qtf2/dual/
: ${RXM_WORK:=/root}
C=${1:-0} K=${2:-2.5} O=/data/qtf2/dual PY=/data/venv-exl3/bin/python
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
[ -n "$DUAL_NOLOCK" ] || { exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; }   # NOLOCK: the caller holds them
cd $RXM_WORK; export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
X="--device 0 --codebook i8 --qtf2 --frac-ext /data/exl3frac/src --sigma-in-sweep 0.025,0.005,0.001 --layers 3-4"
lt() { grep -a "^layer" $1 | awk '{s += $3} END {print s}'; }
rm -rf $O/one $O/a $O/b
s=$(date +%s); $PY -I ourq/ourq.py -K $K -o $O/one $X > $O/one.log 2>&1; w1=$(( $(date +%s) - s ))
log "one: wall ${w1}s, layers $(lt $O/one.log)s"
s=$(date +%s)
$PY -I ourq/ourq.py -K $K -o $O/a $X > $O/a.log 2>&1 &
$PY -I ourq/ourq.py -K $K -o $O/b $X > $O/b.log 2>&1 &
wait; w2=$(( $(date +%s) - s ))
log "two at once: wall ${w2}s, layers a $(lt $O/a.log)s b $(lt $O/b.log)s; throughput x$(python3 -c "print(round(2 * $w1 / $w2, 2))") (startup included in both)"
cmp -s $O/a/q-L03.safetensors $O/one/q-L03.safetensors && cmp -s $O/b/q-L04.safetensors $O/one/q-L04.safetensors && log "outputs identical to the single run" || log "OUTPUTS DIFFER"
