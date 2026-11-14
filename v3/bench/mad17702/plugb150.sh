#!/bin/bash
# plugb150.sh SRC OUT [XFLAGS] [PLUGINS] -- our plugins (default exl3quant qwen35_exl3 exl3dense exl3r4d exl3i) from radiance
# tree SRC, standalone, ROCm 10.1's VOPD clang against /data/q27/rad10-perf/install (w137's plugb.sh TC 10, plus extra HIP
# flags XFLAGS); host tests run; OUT/pb/<p> + OUT/home (rad10-perf's share/radiance + these plugins). Idempotent
# (OUT/pb/<p>.ok). Log OUT/plugb.log. (w150, #177-02)
S=$1; O=$2; XF=${3:-}; PL=${4:-"exl3quant qwen35_exl3 exl3dense exl3r4d exl3i"}; Q=/data/q27; PB=$O/pb; H=$O/home
R=/opt/rocm-10.1; I=$Q/rad10-perf/install; export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$Q/venv/bin:$PATH
C=$Q/rad10/rocm-repeatable/lib/llvm/bin/clang++
mkdir -p $PB; exec >> $O/plugb.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start SRC $S XF '$XF' PL '$PL'"
for p in $PL; do
  [ -f $S/plugin/$p/CMakeLists.txt ] || continue
  [ -f $PB/$p.ok ] && continue
  say "$p: build"
  cmake -S $S/plugin/$p -B $PB/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc -DCMAKE_HIP_COMPILER=$C -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R "-DCMAKE_HIP_FLAGS=--rocm-path=$R $XF" \
      -DRAD_HIP_VOPD_REPEATABLE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ${BID:+-DEXL3I_BUILD_ID=$BID} > $PB/$p.log 2>&1 \
    && nice -n 19 cmake --build $PB/$p -j ${J:-16} >> $PB/$p.log 2>&1 || { say "$p: BUILD FAIL ($PB/$p.log)"; exit 1; }
  ctest --test-dir $PB/$p -L host --output-on-failure >> $PB/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $PB/$p.log) passed"; touch $PB/$p.ok
done
rm -rf $H.part; cp -a $I/share/radiance $H.part
for f in $(find $PB -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $H.part/$d ] && cp $f $H.part/$d/; done
rm -rf $H; mv $H.part $H; say "HOME READY $H"
