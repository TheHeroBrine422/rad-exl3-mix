#!/bin/bash
# w213: exl3i suite again with libstdc++ / libm on the HIP link (box2's amdclang-24 links HIP executables without them)
W=/data/q27/w213; SRC=$W/src; I=$W/install; L=$W/logs
exec {fd}> /data/q27/locks/card3.lock; flock $fd
export HIP_VISIBLE_DEVICES=3
cmake -S $SRC/plugin/exl3i -B $W/build-exl3i-b -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;/opt/rocm" -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang-24 -DCMAKE_EXE_LINKER_FLAGS="-lstdc++ -lm" -DRAD_PLUGIN_INSTALL_HOME=$W/home > $L/t2-conf-exl3i.log 2>&1; echo "conf RC=$?"
cmake --build $W/build-exl3i-b -j24 -- -k 0 > $L/t2-build-exl3i.log 2>&1; echo "build RC=$? $(grep -c '^FAILED' $L/t2-build-exl3i.log) failed"
ctest --test-dir $W/build-exl3i-b -j1 --no-tests=error --output-on-failure --timeout 1800 > $L/t2-ctest-exl3i.log 2>&1; echo "ctest RC=$? $(grep -E 'tests passed|tests failed' $L/t2-ctest-exl3i.log)"
timeout 900 $W/build-exl3i-b/test_moe $W/build-exl3i-b/radiance_home/kernels/exl3i.so > $L/t2-test-moe.log 2>&1; echo "test_moe RC=$? $(tail -n 2 $L/t2-test-moe.log | tr '\n' ' ')"
echo T2_DONE
