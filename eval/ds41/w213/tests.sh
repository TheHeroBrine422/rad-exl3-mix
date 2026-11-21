#!/bin/bash
# w213 step-1 build check on box2 (after build.sh): in-tree fast ctest, install, our plugins out of tree (exl3i suite incl.
# w207's MoE tests, exl3quant, qwen35_exl3) against the install, libquant exl3v3_tool selftest + a short tiles run.
# GPU parts under card3's lock. Logs in /data/q27/w213/logs/t-*.log; summary lines on stdout.
W=/data/q27/w213; SRC=$W/src; B=$SRC/build; I=$W/install; L=$W/logs
cd $W
ctest --test-dir $B -L '^fast$' -j8 --timeout 900 > $L/t-ctest-fast.log 2>&1; echo "ctest fast RC=$? $(grep -E 'tests passed|tests failed' $L/t-ctest-fast.log)"
cmake --install $B --prefix $I > $L/t-install.log 2>&1; echo "install RC=$?"
exec {fd}> /data/q27/locks/card3.lock; flock $fd
export HIP_VISIBLE_DEVICES=3
for d in exl3i exl3quant qwen35_exl3; do
  cmake -S $SRC/plugin/$d -B $W/build-$d -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;/opt/rocm" -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang-24 -DRAD_PLUGIN_INSTALL_HOME=$W/home > $L/t-conf-$d.log 2>&1; echo "$d conf RC=$?"
  cmake --build $W/build-$d -j24 > $L/t-build-$d.log 2>&1; echo "$d build RC=$?"
  ctest --test-dir $W/build-$d -j1 --no-tests=error --output-on-failure --timeout 1800 > $L/t-ctest-$d.log 2>&1; echo "$d ctest RC=$? $(grep -E 'tests passed|tests failed' $L/t-ctest-$d.log)"
done
[ -x $W/build-exl3i/exl3i_moe_host_test ] && { $W/build-exl3i/exl3i_moe_host_test > $L/t-moe-host.log 2>&1; echo "moe host RC=$?"; }
[ -x $W/build-exl3i/test_moe ] && { timeout 900 $W/build-exl3i/test_moe $W/build-exl3i/radiance_home/kernels/exl3i.so > $L/t-test-moe.log 2>&1; echo "test_moe RC=$? $(tail -n 2 $L/t-test-moe.log | tr '\n' ' ')"; }
ROCM_PATH=/opt/rocm HIPCC=/opt/rocm/bin/hipcc $SRC/tests/exl3v3/build.sh $W/build-exl3v3 > $L/t-exl3v3-build.log 2>&1; echo "exl3v3 build RC=$?"
$W/build-exl3v3/exl3v3_tool selftest > $L/t-exl3v3-selftest.log 2>&1; echo "exl3v3 selftest RC=$? $(tail -n 1 $L/t-exl3v3-selftest.log)"
timeout 900 $W/build-exl3v3/exl3v3_tool tiles --tiles 4096 --host 512 > $L/t-exl3v3-tiles.log 2>&1; echo "exl3v3 tiles RC=$? $(tail -n 1 $L/t-exl3v3-tiles.log)"
echo TESTS_DONE
