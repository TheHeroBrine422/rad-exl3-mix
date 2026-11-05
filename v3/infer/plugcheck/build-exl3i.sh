#!/bin/bash
# build-exl3i.sh ID -- build the exl3i kernel plugin and its tests in /root/rhv3 against radiance 1.15.1 (gpubox).
# ID: the source tree's commit (the build reports it). Log /root/rhv3/build-exl3i.log; last line BUILD OK / FAIL.
ID=${1:-unknown}
exec 7>/root/ram.lock; flock -s 7
cd /root/rhv3; export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
{ cmake -S plugin/exl3i -B build-exl3i -G Ninja -DCMAKE_PREFIX_PATH=/root/install-115 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DEXL3I_BUILD_ID="$ID" && time cmake --build build-exl3i -j6; } > build-exl3i.log 2>&1 \
  && echo "BUILD OK" >> build-exl3i.log || echo "BUILD FAIL" >> build-exl3i.log
