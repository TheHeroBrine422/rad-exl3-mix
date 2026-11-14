#!/bin/bash
# plugb.sh TC SRC OUT -- our five plugins from radiance tree SRC built standalone into OUT/pb (+ OUT/home), with
#   TC=10  : ROCm 10.1 the way rad10build.sh does (VOPD shadow clang /data/q27/rad10/rocm-repeatable) against the staged
#            perf core /data/q27/rad10-perf/install (home = its share/radiance + these plugins)
#   TC=714 : the pip ROCm SDK 7.14.1 against the frozen 7.14 core /data/q27/rad/install (as w130's plug714.sh)
# compile_commands.json exported (radisa). Host tests run. Idempotent (OUT/pb/<p>.ok). Log OUT/plugb.log. (w137, #175; w152 copy: PLUGS env selects plugins, default all five)
TC=$1; S=$2; O=$3; Q=/data/q27; PB=$O/pb; H=$O/home
case $TC in
  10)  R=/opt/rocm-10.1; I=$Q/rad10-perf/install; export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$Q/venv/bin:$PATH
       C=$Q/rad10/rocm-repeatable/lib/llvm/bin/clang++
       HIPC=(-DCMAKE_HIP_COMPILER=$C -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R -DRAD_HIP_VOPD_REPEATABLE=ON) ;;
  714) R=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel; I=$Q/rad/install; export ROCM_PATH=$R PATH=$R/bin:$Q/venv/bin:$PATH
       HIPC=() ;;
  *) echo "TC 10|714"; exit 2 ;;
esac
mkdir -p $PB; exec >> $O/plugb.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start TC $TC SRC $S"
for p in ${PLUGS:-exl3quant qwen35_exl3 exl3dense exl3r4d exl3i}; do
  [ -f $S/plugin/$p/CMakeLists.txt ] || continue
  [ -f $PB/$p.ok ] && continue
  say "$p: build"
  cmake -S $S/plugin/$p -B $PB/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > $PB/$p.log 2>&1 \
    && nice -n 19 cmake --build $PB/$p -j ${J:-8} >> $PB/$p.log 2>&1 || { say "$p: BUILD FAIL ($PB/$p.log)"; exit 1; }
  ctest --test-dir $PB/$p -L host --output-on-failure >> $PB/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $PB/$p.log) passed"; touch $PB/$p.ok
done
rm -rf $H.part; cp -a ${BASEHOME:-$I/share/radiance} $H.part
for f in $(find $PB -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $H.part/$d ] && cp $f $H.part/$d/; done
rm -rf $H; mv $H.part $H; say "HOME READY $H"
