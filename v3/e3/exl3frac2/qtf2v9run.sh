#!/bin/bash
# qtf2v9run.sh CARD -- build the v9 menu source (both variants, with / without QTF2_HIOPT, both codebooks) and run
# qtf2v9bench. Own dirs. Log /data/qtf2/v9/run.log
: ${RXM_WORK:=/root}
C=${1:-0} O=/data/qtf2/v9 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$O/build MAX_JOBS=6
( exec 7>$RXM_WORK/ram.lock; flock -s 7
  for v in i8 mul1; do for d in "" QTF2_HIOPT; do
    QTF2_DEFS="$d" $PY -c "import sys; sys.path.insert(0, '$O/exl3frac2'); import exl3frac2; exl3frac2.load('$SRC', '$v')" >> $O/build.log 2>&1 \
      || { log "build $v '$d' FAIL"; exit 1; }
  done; done; log "builds OK" ) || exit 1
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
rm -f $O/bench.jsonl
ROCR_VISIBLE_DEVICES=$C $PY -I $O/exl3frac2/qtf2v9bench.py $SRC $O/bench.jsonl > $O/bench.log 2> $O/bench.err
log "bench: $(tail -1 $O/bench.log)"
