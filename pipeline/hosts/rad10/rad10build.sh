#!/bin/bash
# rad10build.sh [W] -- radiance (W/src: radiance-hero qc-exl3-v3-perf, git archive + COMMIT, plus core-42960d5..ca5491d.patch,
# i.e. rad-convert --no-error and --kld-progress, which the frozen 7.14 install has) with ROCm 10.1 the way
# radiance's Dockerfile builds it: device code by ROCm's clang from the shadow tree W/rocm-repeatable (the VOPD-patched
# libLLVM, llvm10-vopd.sh), --rocm-path=/opt/rocm-10.1, -DRAD_HIP_VOPD_REPEATABLE=ON; gfx1201, FFmpeg off (as the 7.14
# build), install W/install. Then our five plugins standalone against it with the same compiler setup, host tests, and
# the plugin home W/home (= install/share/radiance + their .so, as tools/plugbuild.sh). VOPD=0: plain 10.1 (stock clang,
# pairing off, the build's default) into W/{build,install,pb,home}-novopd. OUT=DIR SRC=DIR: another source / prefix
# (default W, W/src). FF=DIR: with media decoding, FFmpeg from ff10build.sh in DIR (RAD_WITH_FFMPEG=ON, its lib dir in the
# install RPATH). Idempotent; log OUT/logs/rad10build[-novopd].log. Never touches /data/q27/rad. (w130, #170)
set -e; W=${1:-/data/q27/rad10}; R=/opt/rocm-10.1; O=${OUT:-$W}; S=${SRC:-$W/src}; X=; [ "${VOPD:-1}" = 1 ] || X=-novopd
B=$O/build$X; I=$O/install$X; PB=$O/pb$X; H=$O/home$X
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
FFA=(-DRAD_WITH_FFMPEG=OFF)
if [ -n "${FF:-}" ]; then
  export PKG_CONFIG_PATH=$FF/inst/lib/pkgconfig
  FFA=(-DRAD_WITH_FFMPEG=ON -DPKG_CONFIG_EXECUTABLE=$FF/venv/bin/pkgconf "-DCMAKE_INSTALL_RPATH=$R/lib;$FF/inst/lib")
fi
mkdir -p $O/logs; exec >> $O/logs/rad10build$X.log 2>&1
say() { echo "$(date -u +%T) $*"; }
if [ "${VOPD:-1}" = 1 ]; then
  C=$W/rocm-repeatable/lib/llvm/bin/clang++; [ -f $W/rocm-repeatable/.ok ] || { say "no shadow tree: run llvm10-vopd.sh"; exit 1; }
  HIPC=(-DCMAKE_HIP_COMPILER=$C -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R -DRAD_HIP_VOPD_REPEATABLE=ON)
else
  HIPC=(-DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R)
fi
say "start $(head -n 1 $S/COMMIT) ${HIPC[*]} ${FFA[*]}"
if [ ! -f $I/.core.ok ]; then
  [ -f $B/CMakeCache.txt ] || cmake -S $S -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1201 \
     -DCMAKE_INSTALL_PREFIX=$I "${FFA[@]}" -DRAD_HIPCC=$R/bin/hipcc -DCMAKE_PREFIX_PATH=$R "${HIPC[@]}"
  nice -n 10 cmake --build $B -j ${J:-24}
  cmake --install $B; touch $I/.core.ok; say "CORE DONE"
fi
mkdir -p $PB
for p in exl3quant qwen35_exl3 exl3dense exl3r4d exl3i; do
  [ -f $S/plugin/$p/CMakeLists.txt ] || continue
  [ -f $PB/$p.ok ] && continue
  say "$p: build"
  cmake -S $S/plugin/$p -B $PB/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" > $PB/$p.log 2>&1 && nice -n 10 cmake --build $PB/$p -j ${J:-24} >> $PB/$p.log 2>&1 \
      || { say "$p: BUILD FAIL ($PB/$p.log)"; exit 1; }
  ctest --test-dir $PB/$p -L host --output-on-failure >> $PB/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $PB/$p.log) passed"; touch $PB/$p.ok
done
rm -rf $H.part; cp -a $I/share/radiance $H.part
for f in $(find $PB -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $H.part/$d ] && cp $f $H.part/$d/; done
rm -rf $H; mv $H.part $H; say "HOME READY $H"
