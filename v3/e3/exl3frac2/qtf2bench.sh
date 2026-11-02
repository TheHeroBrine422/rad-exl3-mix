#!/bin/bash
# qtf2bench.sh CARD -- qtf2bench.py (kernel time split: real / NOBACK / NOHIST / both / integer kernel), 2048 tiles a rate.
# Builds under the shared RAM lock. Out /data/qtf2/bench.jsonl, log bench.log
: ${RXM_WORK:=/root}; export RXM_WORK
C=${1:-0} O=/data/qtf2 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src Q=$RXM_WORK/ourq/exl3frac2
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build MAX_JOBS=4 EXL3_QT_OPTIMIZED=1
exec 7>$RXM_WORK/ram.lock; flock -s 7
for d in "QTF2_NOBACK" "QTF2_NOHIST" "QTF2_NOBACK QTF2_NOHIST"; do
  QTF2_DEFS="$d" $PY -c "import sys; sys.path.insert(0, '$Q'); import exl3frac2; exl3frac2.load('$SRC', 'i8')" >> $O/bench-build.log 2>&1 \
    || { echo "build $d FAIL" >> $O/bench.log; exit 1; }
done
exec 5>$RXM_WORK/card$C.lock; flock 5
rm -f $O/bench.jsonl
ROCR_VISIBLE_DEVICES=$C $PY -I $Q/qtf2bench.py $SRC 2048 $O/bench.jsonl > $O/bench.log 2>&1
