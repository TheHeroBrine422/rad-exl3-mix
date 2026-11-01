#!/bin/bash
# levels.sh CARD "K K .." -- the V3 level ladder (chain0.sh's level loop, settings.env) on one card, so both cards can
# work through disjoint level lists. Resumes a level layer by layer (ourq.py skips finished layers); log $W/levels-cCARD.log
: ${RXM_WORK:=/root}; export RXM_WORK
C=$1; LEVELS=$2
W=/data/v3levels PY=/data/venv-exl3/bin/python L=/data/v3levels/levels-c$C.log
cd $RXM_WORK
log() { echo "$(date -u +%H:%M) $*" | tee -a $L; }
export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
. $W/settings.env
log "card $C levels $LEVELS with: $OURQ_ARGS"
for K in $LEVELS; do
  [ -f $W/K$K/q-L63.safetensors ] && continue
  s=$(date +%s)
  ( exec 9>$RXM_WORK/card$C.lock; flock 9
    $PY -I ourq/ourq.py -K $K -o $W/K$K --device 0 --frac-ext /data/exl3frac/src $OURQ_ARGS 2>&1 | grep -v -e Warning -e "cpu = _conv" ) >> $W/K$K.log
  [ -f $W/K$K/q-L63.safetensors ] && log "K$K done in $(( ($(date +%s) - s) / 60 )) min, $(du -sh $W/K$K | cut -f1)" || log "K$K FAIL"
done
log "card $C levels done"
