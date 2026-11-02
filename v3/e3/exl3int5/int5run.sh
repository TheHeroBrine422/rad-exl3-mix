#!/bin/bash
# int5run.sh CARD -- integer kernel LDS-barrier gate: patch a copy of the fork's ext source, build exl3cb i8 new + QT_FULLSYNC,
# qtint5check. Own extension dir. Out /data/qtf2/int5/{check.jsonl,run.log}
: ${RXM_WORK:=/root}
C=${1:-1} O=/data/qtf2/int5 PY=/data/venv-exl3/bin/python
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
rm -rf $O/src $O/code; cp -r /data/exl3frac/src $O/src; mkdir -p $O/code; cp -r /data/qtf2/int5/exl3cb /data/qtf2/int5/exl3frac /data/qtf2/int5/exl3int5 $O/code/
python3 $O/code/exl3int5/patch_qt_ldsbar.py $O/src | tee -a $O/run.log || exit 1
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$O/build MAX_JOBS=6 EXL3_QT_OPTIMIZED=1
( exec 7>$RXM_WORK/ram.lock; flock -s 7
  for d in QT_FULLSYNC ""; do
    EXL3CB_DEFS="$d" $PY -c "import sys; sys.path.insert(0, '$O/code/exl3cb'); import exl3cb; exl3cb.load('$O/src', 'i8')" >> $O/build.log 2>&1 \
      || { log "build '$d' FAIL"; exit 1; }
  done; log "builds OK" ) || exit 1
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
ROCR_VISIBLE_DEVICES=$C $PY -I $O/code/exl3int5/qtint5check.py $O/src 1024 > $O/check.jsonl 2> $O/check.err
log "int5: $(tail -1 $O/check.jsonl)"
