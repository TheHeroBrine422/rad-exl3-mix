#!/bin/bash
# w213: configure + build the port tree on box2 (restartable; ninja resumes)
cd /data/q27/w213/src
S=/opt/rocm
[ -f build/build.ninja ] || cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=$S -DCMAKE_HIP_COMPILER=$S/llvm/bin/amdclang-24 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j32 -- -k 0
echo BUILD_RC=$?
