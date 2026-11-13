#!/bin/bash
# tp.sh BUILD TAG FUSE CARD [TIMEOUT] -- test_twopass of BUILD/pb/exl3i (a plugb.sh build) on HIP card CARD under its
# lock, EXL3I_A8_FUSE=FUSE, unbuffered into res/tpTAG.txt (+ exit code / wall). A hang ends at TIMEOUT s (240). (w137)
B=$1; TAG=$2; F=$3; C=$4; T=${5:-240}; W=/data/q27/w137; R=$W/res/tp$TAG.txt
case $B in *b10*) L=/opt/rocm-10.1/lib ;; *) L=/data/q27/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib ;; esac
cd $W/$B/pb/exl3i || exit 2
t0=$(date +%s)
flock /data/q27/locks/card$C.lock env HIP_VISIBLE_DEVICES=$C LD_LIBRARY_PATH=$L EXL3I_A8_FUSE=$F \
  timeout -s KILL $T stdbuf -oL -eL ./test_twopass $PWD/radiance_home/kernels/*exl3i*.so > $R 2>&1
echo "EXIT $? WALL $(( $(date +%s) - t0 ))s" >> $R; tail -3 $R
