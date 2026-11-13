#!/bin/bash
# plug714.sh -- the comparison build for the ROCm 10.1 switch (#170): our five plugins from the rad10 source tree
# (/data/q27/rad10/src, radiance-hero qc-exl3-v3-perf) built with the pip ROCm SDK 7.14.1 against the frozen 7.14 core
# /data/q27/rad/install (core files identical: the branch changes only plugin/), plugin home /data/q27/rad10/ref714/home.
# Same steps as tools/plugbuild.sh. Idempotent. Log /data/q27/rad10/logs/plug714.log. (w130)
Q=/data/q27; SRC=$Q/rad10/src; PREFIX=$Q/rad/install; A=$Q/rad10/ref714; B=$A/pb; H=$A/home
export ROCM_PATH=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel; export PATH=$ROCM_PATH/bin:$Q/venv/bin:$PATH
mkdir -p $B $Q/rad10/logs; exec >> $Q/rad10/logs/plug714.log 2>&1
log() { echo "$(date -u +%T) $*"; }
fail=0
for p in exl3quant qwen35_exl3 exl3dense exl3r4d exl3i; do
  [ -f $SRC/plugin/$p/CMakeLists.txt ] || continue
  [ -f $B/$p.ok ] && continue
  log "$p: build"
  cmake -S $SRC/plugin/$p -B $B/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX;$ROCM_PATH" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$ROCM_PATH/bin/hipcc > $B/$p.log 2>&1 && nice -n 10 cmake --build $B/$p -j ${J:-16} >> $B/$p.log 2>&1 || { log "$p: FAIL build ($B/$p.log)"; fail=1; continue; }
  if ctest --test-dir $B/$p -L host --output-on-failure >> $B/$p.log 2>&1 || ! grep -q "Test #" $B/$p.log; then
    log "$p: built, host tests $(grep -c 'Test #.*Passed' $B/$p.log) passed"; touch $B/$p.ok
  else log "$p: host tests FAIL ($B/$p.log)"; fail=1; fi
done
[ $fail = 0 ] || { log "BUILD FAIL"; exit 1; }
rm -rf $H.part; cp -a $PREFIX/share/radiance $H.part
for f in $(find $B -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $H.part/$d ] && cp $f $H.part/$d/; done
rm -rf $H; mv $H.part $H; log "HOME READY $H"
