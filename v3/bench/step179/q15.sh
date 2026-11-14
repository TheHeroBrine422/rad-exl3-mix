#!/bin/bash
# q15.sh -- card 1: the rebase on qc-exl3-v3 b83698d (w155's exl3i_rerank; lead note 11-14 12:20 AM CST), build rb =
# 708739c: test_aq (+ libr4d) and w155's test_rerank on ROCm 10.1 and 7.14; KLD def on mad K2.5 (vs mbase's rows); gdec
# on mad K2.5 (vs the all-ordered build). (w152)
W=/data/q27/w152; cd $W; Q=/data/q27
while [ ! -d b714rb/home ]; do sleep 20; done
for tc in 10 714; do
  case $tc in 10) R4=$Q/rad10-perf/install/share/radiance/kernels/libr4d.so ;; 714) R4=$Q/rad/install/share/radiance/kernels/libr4d.so ;; esac
  echo "$(date -u +%T) q15 test_aq $tc rb: $(bash tools/gt.sh 1 $tc rb test_aq $R4)" >> logs/q.log
  echo "$(date -u +%T) q15 test_rerank $tc rb: $(bash tools/gt.sh 1 $tc rb test_rerank)" >> logs/q.log
done
MK=$Q/w150/packs/q27v3-635bac02-k2.5-mad-a8.rad
bash tools/kld.sh 1 $MK rb-def >> logs/kld.log 2>&1
bash tools/gdec.sh 1 $MK rb $W/b10rb/home
echo "$(date -u +%T) q15 DONE" >> logs/q.log
