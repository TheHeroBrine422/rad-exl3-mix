#!/bin/bash
# gemmus.sh CARD BUILD TAG [bench_gemmus args] -- bench_gemmus of box2 build /data/q27/w160/b/BUILD on HIP card CARD under
# its lock, libr4d from rad10-e157b (IQR / FP8 rows, radiance 1.17.0); output res/gemmus-TAG.{txt,csv}. (#184, w160)
C=$1 B=$2 T=$3; shift 3; Q=/data/q27; W=$Q/w160; E=$W/b/$B/pb/exl3i
cd $W && mkdir -p res && ROCR_VISIBLE_DEVICES=$C flock $Q/locks/card$C.lock timeout 3600 $E/bench_gemmus $E/radiance_home/kernels/exl3i.so \
  $Q/rad10-e157b/home/kernels/libr4d.so --csv res/gemmus-$T.csv "$@" > res/gemmus-$T.txt 2>&1
echo "gemmus $T rc $? $(date -u +%T)" >> res/gemmus.log
