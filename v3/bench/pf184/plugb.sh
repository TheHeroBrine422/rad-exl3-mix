#!/bin/bash
# plugb.sh TAG [XFLAGS] [PLUGINS] -- our plugins (default exl3i) from box2 /data/q27/w160/src/TAG, standalone, ROCm 10.1's
# VOPD clang, against /data/q27/rad10-e157b/install (the reference build of qc-exl3-v3 120132a, this branch's base:
# same core); host tests run; /data/q27/w160/b/TAG/pb/<p> + /data/q27/w160/b/TAG/home (e157b's home with these
# plugins swapped in). Idempotent (pb/<p>.ok). Log b/TAG/plugb.log. (w137 plugb.sh / w150 plugb150.sh, #184 w160)
T=$1; XF=${2:-}; PL=${3:-exl3i}; Q=/data/q27; S=$Q/w160/src/$T; O=$Q/w160/b/$T; PB=$O/pb; H=$O/home
R=/opt/rocm-10.1; I=$Q/rad10-e157b/install; export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$Q/venv/bin:$PATH
C=$Q/rad10/rocm-repeatable/lib/llvm/bin/clang++
mkdir -p $PB; exec >> $O/plugb.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start SRC $S ($(cat $S/COMMIT)) XF '$XF' PL '$PL'"
for p in $PL; do
  [ -f $S/plugin/$p/CMakeLists.txt ] || continue
  [ -f $PB/$p.ok ] && continue
  say "$p: build"
  cmake -S $S/plugin/$p -B $PB/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc -DCMAKE_HIP_COMPILER=$C -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R "-DCMAKE_HIP_FLAGS=--rocm-path=$R $XF" \
      -DRAD_HIP_VOPD_REPEATABLE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > $PB/$p.log 2>&1 \
    && nice -n 10 cmake --build $PB/$p -j ${J:-24} >> $PB/$p.log 2>&1 || { say "$p: BUILD FAIL ($PB/$p.log)"; exit 1; }
  ctest --test-dir $PB/$p -L host --output-on-failure >> $PB/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $PB/$p.log) passed"; touch $PB/$p.ok
done
rm -rf $H.part; cp -a $Q/rad10-e157b/home $H.part
for f in $(find $PB -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $H.part/$d ] && cp $f $H.part/$d/; done
rm -rf $H; mv $H.part $H; say "HOME READY $H"
