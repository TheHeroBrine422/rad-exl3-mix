#!/bin/bash
# ct.sh CARD BUILD TEST [ARGS] -- one exl3i device test of box2 build /data/q27/w160/b/BUILD under the card lock; output
# res/ct-BUILD-TEST.txt, its last line + rc in res/ct.log. TEST all: every device test in turn. (#184, w160)
C=$1 B=$2 T=$3; shift 3; Q=/data/q27; W=$Q/w160; E=$W/b/$B/pb/exl3i; SO=$E/radiance_home/kernels/exl3i.so
cd $W && mkdir -p res
one() { local t=$1; shift; ROCR_VISIBLE_DEVICES=$C flock $Q/locks/card$C.lock timeout 3600 $E/$t $SO "$@" > res/ct-$B-$t.txt 2>&1
  echo "$B $t rc $? $(date -u +%T): $(tail -n 1 res/ct-$B-$t.txt)" >> res/ct.log; }
if [ "$T" = all ]; then for t in exl3i_gpu_test test_twopass test_aq test_r16 test_ks test_a4 test_rerank; do one $t; done
else one $T "$@"; fi
