#!/bin/bash
# qtf2hi.sh CARD -- would integer K5-7 be faster on qtf2? Test build (MASK 0 instances at KA 5-7, own dir), then
# qtf2int6check at K5-7 vs the integer kernel (identity + tiles/s). Out /data/qtf2/hi/{int6.jsonl,run.log}
: ${RXM_WORK:=/root}
C=${1:-1} O=/data/qtf2/hi PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
rm -rf $O/src; mkdir -p $O/src; cp -r /data/qtf2/v8/src/exl3frac /data/qtf2/v8/src/exl3cb $O/src/; cp -r /data/qtf2/hi/exl3frac2 $O/src/
cd $O/src/exl3frac2 && QTF2_INT_HI=1 python3 mkqtf2.py | tee -a $O/run.log
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$O/build MAX_JOBS=6 EXL3_QT_OPTIMIZED=1
while pgrep -f "^/bin/bash /data/qtf2/multik/multik.sh" > /dev/null || pgrep -f "^/bin/bash /data/qtf2/deploy8.sh" > /dev/null; do sleep 10; done
( exec 7>$RXM_WORK/ram.lock; flock -s 7
  for v in mul1 i8; do $PY -c "import sys; sys.path.insert(0, '$O/src/exl3frac2'); import exl3frac2; exl3frac2.load('$SRC', '$v')" >> $O/build.log 2>&1 \
    || { log "build $v FAIL"; exit 1; }; done; log "builds OK" ) || exit 1
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
INT6_KS=5,6,7 ROCR_VISIBLE_DEVICES=$C $PY -I $O/src/exl3frac2/qtf2int6check.py $SRC 1024 > $O/int6.jsonl 2> $O/int6.err
log "hi: $(tail -1 $O/int6.jsonl)"
