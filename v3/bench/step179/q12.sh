#!/bin/bash
# q12.sh -- card 1, after q11: test_aq of mres3 (= mres2 + the xb padding zeroed in test_res) on ROCm 10.1 and 7.14. (w152)
W=/data/q27/w152; cd $W
while pgrep -f "^bash tools/q11[.]sh" > /dev/null || [ ! -d b714mres3/home ]; do sleep 20; done
for tc in 10 714; do
  case $tc in 10) R4=/data/q27/rad10-perf/install/share/radiance/kernels/libr4d.so ;; 714) R4=/data/q27/rad/install/share/radiance/kernels/libr4d.so ;; esac
  echo "$(date -u +%T) q12 test_aq $tc mres3: $(bash tools/gt.sh 1 $tc mres3 test_aq $R4)" >> logs/q.log
done
