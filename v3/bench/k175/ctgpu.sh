#!/bin/bash
# ctgpu.sh BUILD CARD -- every gpu-labelled ctest of BUILD/pb/<plugin> on HIP card CARD under its lock (log
# res/ct-BUILD.txt). (w137)
B=$1; C=$2; W=/data/q27/w137; R=$W/res/ct-$B.txt
case $B in *b10*) L=/opt/rocm-10.1/lib ;; *) L=/data/q27/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib ;; esac
: > $R
for p in exl3i exl3r4d exl3dense qwen35_exl3 exl3quant; do
  flock /data/q27/locks/card$C.lock env HIP_VISIBLE_DEVICES=$C LD_LIBRARY_PATH=$L PATH=/data/q27/venv/bin:$PATH \
    ctest --test-dir $W/$B/pb/$p -L gpu --timeout 600 --output-on-failure >> $R 2>&1
  echo "$p: ctest exit $?" >> $R
done
grep -E "tests passed|ctest exit|Failed|Timeout" $R
