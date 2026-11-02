#!/bin/bash
# dumpvec.sh -- oracle test vectors on card 0 (1 min). Out /data/qtf2/exl3i-vec.npz
C=${CARD:-0}; export ROCR_VISIBLE_DEVICES=$C ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
exec 9>/root/card$C.lock; flock 9; exec 7>/root/ram.lock; flock -s 7
/data/venv-exl3/bin/python -I /root/ourq/exl3i/dumpvec.py /data/exl3frac/src /data/qtf2/exl3i-vec.npz > /data/qtf2/dumpvec.log 2>&1
