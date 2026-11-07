#!/bin/bash
# build.sh -- #46-03 plugin home on gpubox: radiance 1.17.0 (/data/install-117, the ladder's install) + our five plugins
# from radiance-hero qc-exl3-v3 42960d5 (exl3i A4 / A8 rows on mul1i4, qwen35_exl3 per-weight act modes, exl3quant 0.3.0
# mul1i4 importer), built standalone into /data/a4cost/b/<plugin>, then copied into /data/a4cost/home. Host tests only
# (no GPU). gpubox RAM is tight: -j1, nice 19, a guard kills the build when MemAvailable < 3 GB. Idempotent.
A=/data/a4cost; SRC=$A/src; PREFIX=/data/install-117; B=$A/b; H=$A/home
export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/build.log; }
guard() { while kill -0 $1 2>/dev/null; do
    m=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    [ $m -lt 3000 ] && { log "MemAvailable $m MB < 3000: killing build"; pkill -P $1; kill $1; return 1; }
    sleep 2; done; }
mkdir -p $B; fail=0
for p in exl3quant qwen35_exl3 exl3dense exl3r4d exl3i; do
  [ -d $SRC/plugin/$p ] && [ -f $SRC/plugin/$p/CMakeLists.txt ] || continue
  [ -f $B/$p.ok ] && continue
  while [ $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) -lt 4000 ]; do sleep 60; done
  log "$p: build"
  ( cmake -S $SRC/plugin/$p -B $B/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PREFIX -DRAD_GPU_TARGETS=gfx1201 \
      > $B/$p.log 2>&1 && nice -n 19 cmake --build $B/$p -j 1 >> $B/$p.log 2>&1 ) & guard $!; wait $!; rc=$?
  [ $rc = 0 ] || { log "$p: FAIL build ($B/$p.log)"; fail=1; continue; }
  if ctest --test-dir $B/$p -L host --output-on-failure >> $B/$p.log 2>&1 || ! grep -q "Test #" $B/$p.log; then
    log "$p: built, host tests $(grep -c 'Test #.*Passed' $B/$p.log) passed"; touch $B/$p.ok
  else log "$p: host tests FAIL ($B/$p.log)"; fail=1; fi
done
[ $fail = 0 ] || { log "BUILD FAIL"; exit 1; }
rm -rf $H.part; cp -a $PREFIX/share/radiance $H.part
for f in $(find $B -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $H.part/$d ] && cp $f $H.part/$d/; done
ls -la $H.part/kernels $H.part/quantizers $H.part/architectures/qwen35_exl3.so >> $A/build.log
rm -rf $H; mv $H.part $H; log "HOME READY"
