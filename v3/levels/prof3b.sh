#!/bin/bash
# prof3b.sh -- card 1, after occ1: qprof3 at K2.5 L3-4 with tiles/s by call size. Out /data/prof2/prof2-K2.5.{json,log}
: ${RXM_WORK:=/root}
while pgrep -f "^/bin/bash /data/qtf2/occ1.sh" > /dev/null; do sleep 5; done
exec 5>$RXM_WORK/card1.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
cd $RXM_WORK; export ROCR_VISIBLE_DEVICES=1 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
/data/venv-exl3/bin/python -I ourq/qprof3.py 2.5 3-4 /data/prof2/prof2-K2.5.json --codebook i8 --qtf2 --frac-ext /data/exl3frac/src \
  --sigma-in-sweep 0.025,0.005,0.001 > /data/prof2/prof2-K2.5.log 2>&1
