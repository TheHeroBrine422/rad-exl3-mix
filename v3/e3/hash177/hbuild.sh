#!/bin/bash
# hbuild.sh NAME "DEFS" -- one exl3i plugin variant (+ its bench_twopass / device tests) of /data/q27/w139/src (radiance
# qc-exl3-v3-hash = qc-exl3-v3-perf 5c9956c + the hash bench switch) against rad10-perf's install, with rad10build.sh's
# compiler setup (ROCm 10.1, the VOPD-repeatable shadow clang, gfx1201); DEFS go into CMAKE_HIP_FLAGS (e.g.
# "-DEXL3I_BENCH_IQRH=2"). Output /data/q27/w139/pb/NAME; a home /data/q27/w139/home-NAME = rad10-perf/home with this
# exl3i.so (whole-model timing). Idempotent (NAME.ok). (w139, #177)
set -e; Q=/data/q27; W=$Q/rad10; P=$Q/rad10-perf; R=/opt/rocm-10.1; H=$Q/w139; N=$1; D=${2:-}
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$Q/venv/bin:$PATH
B=$H/pb/$N; mkdir -p $H/pb $H/logs
[ -f $B.ok ] && { echo "$N: built"; exit 0; }
C=$W/rocm-repeatable/lib/llvm/bin/clang++
( cmake -S $H/src/plugin/exl3i -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$P/install;$R" \
    -DRAD_GPU_TARGETS=gfx1201 -DRAD_HIPCC=$R/bin/hipcc -DCMAKE_HIP_COMPILER=$C -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R \
    "-DCMAKE_HIP_FLAGS=--rocm-path=$R $D" -DRAD_HIP_VOPD_REPEATABLE=ON -DEXL3I_BUILD_ID="w139-$N" \
  && nice -n 10 cmake --build $B -j ${J:-16} ) > $H/logs/build-$N.log 2>&1 || { echo "$N: BUILD FAIL ($H/logs/build-$N.log)"; exit 1; }
so=$(find $B -path '*radiance_home/*' -name exl3i.so | head -n 1)
rm -rf $H/home-$N; cp -a $P/home $H/home-$N; cp $so $H/home-$N/kernels/exl3i.so
touch $B.ok; echo "$N: built ($D)"
