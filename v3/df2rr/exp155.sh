#!/bin/bash
# exp155.sh TAG -- (w155) a kernel experiment: exl3i from W/src-exp (a tar of the worktree) built into W/pb-exp (W/home is
# not touched), then test_rerank --quick (the real-size heads at M 7 + the small cases: bits vs the host reference, time)
# on HIP 7 under the card lock. Output W/res/exp-TAG.txt.
W=/data/q27/w155; S=$W/src-exp; I=$W/inst; R=/opt/rocm-10.1; RR=/data/q27/rad10/rocm-repeatable
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
HIPC=(-DCMAKE_HIP_COMPILER=$RR/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R
      -DRAD_HIP_VOPD_REPEATABLE=ON)
[ -d $W/pb-exp/CMakeFiles ] || cmake -S $S/plugin/exl3i -B $W/pb-exp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" \
  -DRAD_GPU_TARGETS=gfx1201 -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" > $W/pb-exp.log 2>&1
nice -n 10 cmake --build $W/pb-exp -j 8 >> $W/pb-exp.log 2>&1 || { echo BUILD FAIL; exit 1; }
( flock 5; flock -s 4; ROCR_VISIBLE_DEVICES=7 timeout 900 $W/pb-exp/test_rerank $(find $W/pb-exp -name exl3i.so | head -1) --quick ) \
  5>/data/q27/locks/card7.lock 4>/data/q27/locks/ram.lock > $W/res/exp-$1.txt 2>&1
grep -E "us a launch|FAIL|PASS" $W/res/exp-$1.txt | sed 's/.*device/device/' | cut -c1-150
