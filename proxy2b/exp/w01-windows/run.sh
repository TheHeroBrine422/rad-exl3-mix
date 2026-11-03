#!/bin/bash
# run.sh -- w01-windows (ranked.md A4): CPU only, seconds. The qad-window composition the YAQA collectors draw (2048 and
# 1024 tokens, the first 1,024 / all windows, and the what-if "qad docs of >= 256 / 512 tokens as their own sequences").
# Reads the calib golden's kld.json + gen.u8 and pools-v3.json; writes /data/yb/w01-windows/windows-*.{txt,json}.
X=/workspace/rad-exl3-mix/proxy2b/exp/w01-windows O=/data/yb/w01-windows GD=${GD:-/data/challenge/calib-meta}
POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
w() { tag=$1; shift; [ -f $O/$tag.json ] || nice -n 19 python3 -I $X/windows.py $GD $POOLS "$@" $O/$tag.json > $O/$tag.txt; }
w windows-L2048-n1024 2048 1024 0
w windows-L2048-n1350 2048 1350 0
w windows-L1024-n1024 1024 1024 0
w windows-L1024-n4096 1024 4096 0
W_SHORT=256 w windows-L2048-short256 2048 100000 0
W_SHORT=512 w windows-L2048-short512 2048 100000 0
grep -h -B0 "qad windows\|total variation" $O/windows-*.txt | tee $O/result.txt
