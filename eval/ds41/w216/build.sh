#!/bin/bash
# w216: configure + build the cleaned tree on box2 (restartable; ninja resumes). Log: /data/q27/w216/logs/build.log
cd /data/q27/w216/src
S=/opt/rocm
[ -f build/build.ninja ] || cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=$S -DCMAKE_HIP_COMPILER=$S/llvm/bin/amdclang-24 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j40 -- -k 0
echo BUILD_RC=$?
