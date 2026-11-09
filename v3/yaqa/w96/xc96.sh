#!/bin/bash
# xc96.sh -- w96 on Thor: Fisher pass over night1105's xcorp golden xc8k-b1 (minus golden-search27b-x's 65 pieces).
# Idempotent (fisher96 keeps per-range parts). Out /data/work/w96/fisher-xc.npz.
set -u
W=/data/work/w96 C=/data/work/w96/code/v3/yaqa/w96 G=/data/backup/gold27/xc8k-b1
cd $W
[ -s $W/groups-xc.npy ] || python3 -I $C/mkgroups96.py xc $G $W/in/misc/golden-search27b-x.docs $W/groups-xc.npy || exit 1
[ -s $W/fisher-xc.npz ] || OMP_NUM_THREADS=1 nice -n 19 python3 -I $C/fisher96.py $G $W/groups-xc.npy $W/fisher-xc.npz 8
echo XC96 DONE
