#!/bin/bash
# build153r.sh -- (w153, #180, ds8) after the rebase onto qc-exl3-v3 e054e71 (w150's mad): exl3i, exl3quant and
# qwen35_exl3 from W/src-r (a tar of radiance-hero qc-exl3-v3-df2's plugin/ + tests/ at the rebased commit, COMMIT),
# built standalone against /data/q27/rad10-perf/install with rad10's compiler setup (as build153.sh / w147's
# build147.sh), host tests (every test not labelled gpu), then W/home-r = rad10-perf/home with these .so. CPU only.
# Log W/logs/build153r.log.
set -e; W=/data/q27/w153; S=$W/src-r; I=/data/q27/rad10-perf/install; R=/opt/rocm-10.1; RR=/data/q27/rad10/rocm-repeatable
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
HIPC=(-DCMAKE_HIP_COMPILER=$RR/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R
      -DRAD_HIP_VOPD_REPEATABLE=ON)
mkdir -p $W/logs $W/pbr; exec >> $W/logs/build153r.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $(cat $S/COMMIT 2>/dev/null)"
for p in exl3i exl3quant qwen35_exl3; do
  cmake -S $S/plugin/$p -B $W/pbr/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" > $W/pbr/$p.log 2>&1 && nice -n 10 cmake --build $W/pbr/$p -j ${J:-8} >> $W/pbr/$p.log 2>&1 \
      || { say "$p: BUILD FAIL ($W/pbr/$p.log)"; exit 1; }
  ctest --test-dir $W/pbr/$p -LE gpu --output-on-failure >> $W/pbr/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $W/pbr/$p.log) passed"
done
rm -rf $W/home-r.part; cp -a /data/q27/rad10-perf/home $W/home-r.part
for f in $(find $W/pbr -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $W/home-r.part/$d ] && cp $f $W/home-r.part/$d/; done
rm -rf $W/home-r; mv $W/home-r.part $W/home-r
say "HOME READY $W/home-r ($(for p in exl3i qwen35_exl3; do sha256sum $(find $W/home-r -name $p.so) | cut -c1-16; done | tr '\n' ' '))"
