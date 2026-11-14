#!/bin/bash
# build153.sh -- (w153, #180, ds8) qwen35_exl3 from W/src (a tar of radiance-hero qc-exl3-v3-df2's plugin/ + tests/,
# COMMIT names it), built standalone against /data/q27/rad10-perf/install with rad10's compiler setup (as w147's
# build147.sh), host tests (every test not labelled gpu), then W/home = w147's home (qc-exl3-v3 179220f: today's
# qwen35_exl3 + exl3quant over rad10-perf/home) with this qwen35_exl3.so. CPU only (no card). Log W/logs/build153.log.
set -e; W=/data/q27/w153; S=$W/src; I=/data/q27/rad10-perf/install; R=/opt/rocm-10.1; RR=/data/q27/rad10/rocm-repeatable
BASE=/data/q27/w147/home
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
HIPC=(-DCMAKE_HIP_COMPILER=$RR/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R
      -DRAD_HIP_VOPD_REPEATABLE=ON)
mkdir -p $W/logs $W/pb; exec >> $W/logs/build153.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $(cat $S/COMMIT 2>/dev/null)"
p=qwen35_exl3
cmake -S $S/plugin/$p -B $W/pb/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
    -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" > $W/pb/$p.log 2>&1 && nice -n 10 cmake --build $W/pb/$p -j ${J:-8} >> $W/pb/$p.log 2>&1 \
    || { say "$p: BUILD FAIL ($W/pb/$p.log)"; exit 1; }
ctest --test-dir $W/pb/$p -LE gpu --output-on-failure >> $W/pb/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
say "$p: built, host tests $(grep -c 'Test #.*Passed' $W/pb/$p.log) passed"
rm -rf $W/home.part; cp -a $BASE $W/home.part
cp $(find $W/pb/$p -path '*architectures/*' -name "$p.so" | head -1) $W/home.part/architectures/
rm -rf $W/home; mv $W/home.part $W/home; say "HOME READY $W/home ($(sha256sum $W/home/architectures/$p.so | cut -c1-16))"
