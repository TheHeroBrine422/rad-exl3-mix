#!/bin/bash
# race.sh CARD -- qtf2race over the v7 split's builds (ref first) and the production build (v6). Out /data/qtf2/race/
: ${RXM_WORK:=/root}
C=${1:-1} O=/data/qtf2/race PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; rm -f $O/*.pt $O/out.jsonl
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
export ROCR_VISIBLE_DEVICES=$C ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201
S=/data/qtf2/v7s/src/exl3frac2; cp /data/qtf2/race/qtf2race.py $S/
for d in "QTF2_BYTEHIST QTF2_FULLSYNC" "" "QTF2_FULLSYNC" "QTF2_BYTEHIST"; do
  TORCH_EXTENSIONS_DIR=/data/qtf2/v7s/build QTF2_DEFS="$d" $PY -I $S/qtf2race.py $SRC 5 $O/r 2>> $O/err.log | tee -a $O/out.jsonl
done
cp /data/qtf2/race/qtf2race.py $RXM_WORK/ourq/exl3frac2/
TORCH_EXTENSIONS_DIR=/data/exl3frac/build QTF2_DEFS="" $PY -I $RXM_WORK/ourq/exl3frac2/qtf2race.py $SRC 5 $O/r 2>> $O/err.log | sed 's/"default"/"production-v6"/' | tee -a $O/out.jsonl
