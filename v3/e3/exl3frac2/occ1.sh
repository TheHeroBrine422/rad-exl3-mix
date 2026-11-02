#!/bin/bash
# occ1.sh -- card 1: qtf2occ (throughput by tiles per call / launch batch), after prof2. Out /data/qtf2/occ.jsonl
: ${RXM_WORK:=/root}
exec 5>$RXM_WORK/card1.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
export ROCR_VISIBLE_DEVICES=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
rm -f /data/qtf2/occ.jsonl
/data/venv-exl3/bin/python -I $RXM_WORK/ourq/exl3frac2/qtf2occ.py /data/exl3frac/src /data/qtf2/occ.jsonl > /data/qtf2/occ.log 2>&1
