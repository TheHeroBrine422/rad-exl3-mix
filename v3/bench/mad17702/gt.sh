#!/bin/bash
# gt.sh CARD BUILD TEST [ARGS...] -- one device test binary of BUILD (/data/q27/w150/BUILD/pb/exl3i) on box2 card CARD
# (HIP index) under its lock, its plugin's exl3i.so as the first argument; output to res/BUILD-TEST[-TAG].txt (TAG env).
# (w150, #177-02)
C=$1; B=$2; T=$3; shift 3; W=/data/q27/w150; P=$W/$B/pb/exl3i; mkdir -p $W/res
export HIP_VISIBLE_DEVICES=$C LD_LIBRARY_PATH=/opt/rocm-10.1/lib
o=$W/res/$B-$T${TAG:+-$TAG}.txt
flock /data/q27/locks/card$C.lock timeout ${TO:-1800} $P/$T $P/radiance_home/kernels/exl3i.so "$@" > $o 2>&1
echo "rc $? $(tail -n 1 $o)"
