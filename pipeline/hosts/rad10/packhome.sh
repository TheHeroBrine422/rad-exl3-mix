#!/bin/bash
# packhome.sh -- #170: pack homes for rad-convert while a3799b5's qwen35_exl3 cannot pack from the HF checkpoint ("name map
# for 'v.patch.weight': declared twice"; 8afc1fc before it declares the vision tower into every pack). qwen35_exl3 from
# 1aa18de (AQ; the last commit before the vision port, text-only declare, exl3i interface 3) built standalone, then
#   W/home-pack    = W/home (10.1)          with that qwen35_exl3.so (10.1 build: the 10.1 compiler setup)
#   W/ref714/home-pack = W/ref714/home (7.14) with that qwen35_exl3.so (7.14 build)
# Every other plugin stays a3799b5's (exl3quant / exl3i: R16 rows, encoding v3). Source W/src-1aa18de (git archive).
# Idempotent. Log W/logs/packhome.log. (w130)
set -e; W=/data/q27/rad10; Q=/data/q27; S=$W/src-1aa18de; R=/opt/rocm-10.1; P7=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel
exec >> $W/logs/packhome.log 2>&1; say() { echo "$(date -u +%T) $*"; }
[ -f $S/plugin/qwen35_exl3/CMakeLists.txt ] || { say "no $S"; exit 1; }
if [ ! -f $W/pb-pack10/ok ]; then
  ( export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$Q/venv/bin:$PATH
    cmake -S $S/plugin/qwen35_exl3 -B $W/pb-pack10 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$W/install;$R" \
      -DRAD_GPU_TARGETS=gfx1201 -DRAD_HIPCC=$R/bin/hipcc -DCMAKE_HIP_COMPILER=$W/rocm-repeatable/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R -DRAD_HIP_VOPD_REPEATABLE=ON && cmake --build $W/pb-pack10 -j 16 )
  touch $W/pb-pack10/ok; say "built 10.1"
fi
if [ ! -f $W/pb-pack7/ok ]; then
  ( export ROCM_PATH=$P7 PATH=$P7/bin:$Q/venv/bin:$PATH
    cmake -S $S/plugin/qwen35_exl3 -B $W/pb-pack7 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$Q/rad/install;$P7" \
      -DRAD_GPU_TARGETS=gfx1201 -DRAD_HIPCC=$P7/bin/hipcc && cmake --build $W/pb-pack7 -j 16 )
  touch $W/pb-pack7/ok; say "built 7.14"
fi
mk() { local from=$1 so=$2 to=$3; rm -rf $to.part; cp -a $from $to.part; cp $so $to.part/architectures/qwen35_exl3.so; rm -rf $to; mv $to.part $to; }
mk $W/home $(find $W/pb-pack10 -path '*radiance_home/*' -name qwen35_exl3.so) $W/home-pack
mk $W/ref714/home $(find $W/pb-pack7 -path '*radiance_home/*' -name qwen35_exl3.so) $W/ref714/home-pack
say "PACK HOMES READY"
