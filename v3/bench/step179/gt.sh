#!/bin/bash
# gt.sh CARD TC BUILD TEST [ARGS...] -- one exl3i device test binary of /data/q27/w152/b<TC><BUILD>/pb/exl3i on box2 card
# CARD (HIP index) under its lock (+ ram.lock shared), the build's exl3i.so first; TC 10: /opt/rocm-10.1's runtime, 714:
# the pip SDK 7.14.1's. Output res/gt/<TC><BUILD>-TEST[-TAG].txt. (w152; from w150's gt.sh)
C=$1 TC=$2 B=$3 T=$4; shift 4; W=/data/q27/w152; P=$W/b$TC$B/pb/exl3i; mkdir -p $W/res/gt
case $TC in
  10)  L=/opt/rocm-10.1/lib ;;
  714) S=/data/q27/venv/lib/python3.14/site-packages; L=$S/_rocm_sdk_core/lib:$S/_rocm_sdk_libraries/lib:$S/_rocm_sdk_devel/lib ;;
esac
export HIP_VISIBLE_DEVICES=$C LD_LIBRARY_PATH=$L
o=$W/res/gt/$TC$B-$T${TAG:+-$TAG}.txt
flock -s /data/q27/locks/ram.lock flock /data/q27/locks/card$C.lock timeout ${TO:-1800} $P/$T $P/radiance_home/kernels/exl3i.so "$@" > $o 2>&1
echo "rc $? $(tail -n 1 $o)"
