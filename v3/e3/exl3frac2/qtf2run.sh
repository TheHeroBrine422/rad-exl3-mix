#!/bin/bash
# qtf2run.sh -- build exl3frac2 (mul1 + i8) and run its gate qtf2check.py on card 0. Build under the shared RAM lock
# (hipcc next to a head quant would OOM), check under card0.lock. Log /data/qtf2/run.log, results /data/qtf2/check.jsonl.
: ${RXM_WORK:=/root}; export RXM_WORK
O=/data/qtf2 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src Q=$RXM_WORK/ourq/exl3frac2
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build MAX_JOBS=4
cd $Q && python3 mkqtf2.py >> $O/run.log
( exec 7>$RXM_WORK/ram.lock; flock -s 7
  for v in mul1 i8; do
    $PY -c "import sys; sys.path.insert(0, '$Q'); import exl3frac2; exl3frac2.load('$SRC', '$v')" > $O/build-$v.log 2>&1 \
      && log "build $v OK" || { log "build $v FAIL (build-$v.log)"; exit 1; }
  done ) || exit 1
( exec 9>$RXM_WORK/card0.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7
  ROCR_VISIBLE_DEVICES=0 $PY -I $Q/qtf2check.py $SRC 1024 > $O/check.jsonl 2> $O/check.err )
log "check: $(tail -1 $O/check.jsonl)"
