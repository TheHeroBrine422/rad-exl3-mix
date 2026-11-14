#!/bin/bash
# gdbg155.sh [ARGS] -- (w155) rebuild exl3i in W/pb/exl3i from W/src and run test_rerank on HIP 7 under the card lock
# (small cases only with TEST_RERANK_SMALL=1). Output W/res/test_rerank-dbg.txt. The debugging loop of #180-02.
W=/data/q27/w155; R=/opt/rocm-10.1; export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
cmake --build $W/pb/exl3i -j 8 > $W/pb/exl3i-dbg.log 2>&1 || { echo BUILD FAIL; exit 1; }
( flock 5; flock -s 4; ROCR_VISIBLE_DEVICES=7 timeout 900 $W/pb/exl3i/test_rerank $(find $W/pb/exl3i -name exl3i.so | head -1) "$@" ) \
  5>/data/q27/locks/card7.lock 4>/data/q27/locks/ram.lock > $W/res/test_rerank-dbg.txt 2>&1
grep -E "FAIL|PASS" $W/res/test_rerank-dbg.txt | cut -c1-160
