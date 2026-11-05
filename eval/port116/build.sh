#!/bin/bash
# build.sh SRC PREFIX MODE JOBS -- radiance (from the fork tree SRC) installed to PREFIX, then our five plugins built
# standalone against it, then their tests. MODE host: host backend only (no ROCm; checks the port compiles and the
# host tests pass). MODE hip: device build for gfx1201 (gpubox). Logs one line a step; the build output goes to
# PREFIX.build/*.log. Idempotent: reruns reuse the build directories.
set -u
SRC=$1; PREFIX=$2; MODE=$3; J=${4:-2}
B=$PREFIX.build; mkdir -p $B
log() { echo "$(date -u +%H:%M:%S) $*"; }
case $MODE in
  host) CFG="-DRAD_WITH_HIP=OFF" ;;
  hip)  CFG="-DRAD_GPU_TARGETS=gfx1201" ;;
  *) echo "MODE host|hip"; exit 2 ;;
esac
log "radiance configure ($MODE)"
cmake -S $SRC -B $B/radiance -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX $CFG \
  > $B/radiance-cfg.log 2>&1 || { log "FAIL configure (tail $B/radiance-cfg.log)"; exit 1; }
log "radiance build -j$J"
cmake --build $B/radiance -j $J > $B/radiance-build.log 2>&1 || { log "FAIL build (tail $B/radiance-build.log)"; exit 1; }
cmake --install $B/radiance > $B/radiance-install.log 2>&1 || { log "FAIL install"; exit 1; }
log "radiance installed: $($PREFIX/bin/radiance --version 2>&1 | head -1)"
fail=0
for p in exl3dense exl3quant exl3r4d exl3i qwen35_exl3; do
  [ $MODE = host ] && [ $p = exl3r4d -o $p = exl3i ] && { log "$p: device only, skipped on host"; continue; }
  if cmake -S $SRC/plugin/$p -B $B/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PREFIX $CFG \
       > $B/$p.log 2>&1 && cmake --build $B/$p -j $J >> $B/$p.log 2>&1; then
    if ctest --test-dir $B/$p --output-on-failure >> $B/$p.log 2>&1; then log "$p: build + tests PASS ($(grep -c 'Test #' $B/$p.log) tests)"
    else log "$p: build ok, tests FAIL ($B/$p.log)"; fail=1; fi
  else log "$p: FAIL build ($B/$p.log)"; fail=1; fi
done
log "DONE fail=$fail"
