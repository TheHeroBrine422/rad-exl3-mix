#!/bin/bash
# build147.sh PLUGIN... -- (w147, #178-02, box2 / ds8) the plugins the fused GDN a | b touches (qwen35_exl3: serves ssm_ab;
# exl3quant: imports it; exl3i / exl3dense: the serve home's kernels at the same commit), from W/src (git archive of
# radiance-hero qc-exl3-v3-ab147, COMMIT), built standalone against /data/q27/rad10-perf/install with rad10's compiler
# setup (as w144's build144.sh), host tests (every test not labelled gpu), then the home W/home = rad10-perf/home with these .so. CPU only (no card).
# Log W/logs/build147.log.
set -e; W=/data/q27/w147; S=$W/src; I=/data/q27/rad10-perf/install; R=/opt/rocm-10.1; RR=/data/q27/rad10/rocm-repeatable
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
HIPC=(-DCMAKE_HIP_COMPILER=$RR/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R
      -DRAD_HIP_VOPD_REPEATABLE=ON)
mkdir -p $W/logs $W/pb; exec >> $W/logs/build147.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $(cat $S/COMMIT 2>/dev/null) plugins $*"
for p in "$@"; do
  cmake -S $S/plugin/$p -B $W/pb/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" > $W/pb/$p.log 2>&1 && nice -n 10 cmake --build $W/pb/$p -j ${J:-8} >> $W/pb/$p.log 2>&1 \
      || { say "$p: BUILD FAIL ($W/pb/$p.log)"; exit 1; }
  ctest --test-dir $W/pb/$p -LE gpu --output-on-failure >> $W/pb/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $W/pb/$p.log) passed"
done
rm -rf $W/home.part; cp -a /data/q27/rad10-perf/home $W/home.part
for f in $(find $W/pb -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $W/home.part/$d ] && cp $f $W/home.part/$d/; done
rm -rf $W/home; mv $W/home.part $W/home; say "HOME READY $W/home"
