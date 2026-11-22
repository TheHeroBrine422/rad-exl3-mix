#!/bin/bash
# w216 checks on box2 (after build.sh): in-tree fast ctest, install, our plugins out of tree (exl3i incl. test_moe, exl3quant,
# qwen35_exl3; HIP links with -lstdc++ -lm), libquant exl3v3 selftest + tiles, the host SIMT-emulator tests (linked by hand),
# libr4d's moe_gemm_mxfp4 rows on a card (r4d_selftest --case, --mxcheck), and the symbol gate (no ggml / radiance-exl3).
# GPU parts under card3's lock (waits up to 20 min). Logs /data/q27/w216/logs/t-*.log; summary lines on stdout.
W=/data/q27/w216; SRC=$W/src; B=$SRC/build; I=$W/install; L=$W/logs
cd $W
rm -rf $I; cmake --install $B --prefix $I > $L/t-install.log 2>&1; echo "install RC=$?"
# the symbol gate: dynamic symbols and strings of every plugin and library the build made
for f in $B/radiance_home/kernels/*.so $B/radiance_home/architectures/*.so $B/radiance_home/quantizers/*.so $B/bin/radiance $B/bin/rad-convert; do
  [ -f "$f" ] || continue
  n=$(strings -a "$f" | grep -iE 'ggml|exl3r4d|exl3ref|radiance-exl3|weight_sidecar' | grep -vc 'ggml_rms_norm')
  echo "gate $(basename $f): $n"
done > $L/t-gate.log; echo "gate: $(awk '{s+=$3} END {print s}' $L/t-gate.log) hits over $(wc -l < $L/t-gate.log) files"
D=$B/tests/r4d_emu; OBJS=$(ls $D/*.o | grep -v '/test_'); mkdir -p $B/tests/emu-w216
for t in ds41mx ds41walk mhc moefuse dsa pick norm f32 attn; do
  /opt/rocm/llvm/bin/amdclang-24 $D/test_$t.o $OBJS -ldl -lstdc++ -lm -o $B/tests/emu-w216/r4d_emu_$t > $L/t-emu-link-$t.log 2>&1 || { echo "emu $t LINK_FAIL"; continue; }
  timeout 1200 $B/tests/emu-w216/r4d_emu_$t > $L/t-emu-$t.log 2>&1; echo "emu $t RC=$? $(tail -n 1 $L/t-emu-$t.log | cut -c1-150)"
done
ROCM_PATH=/opt/rocm HIPCC=/opt/rocm/bin/hipcc $SRC/tests/exl3v3/build.sh $W/build-exl3v3 > $L/t-exl3v3-build.log 2>&1; echo "exl3v3 build RC=$?"
exec {fd}> /data/q27/locks/card3.lock; flock -w 1200 $fd || { echo "card3 busy: GPU parts skipped"; echo TESTS_DONE; exit 0; }
export HIP_VISIBLE_DEVICES=3
# the fast suite has card tests (device_test, ...): under the lock too (run 2 ran it unlocked)
ctest --test-dir $B -L '^fast$' -j8 --timeout 900 > $L/t-ctest-fast.log 2>&1; echo "ctest fast RC=$? $(grep -E 'tests passed|tests failed' $L/t-ctest-fast.log)"
# exl3v3 tiles uses the card too (a 6 GiB search buffer): only under the lock (run 2 ran it unlocked and failed its
# hipMalloc on a card w214 held; rerun under the lock in tests2.sh)
$W/build-exl3v3/exl3v3_tool selftest > $L/t-exl3v3-selftest.log 2>&1; echo "exl3v3 selftest RC=$? $(tail -n 1 $L/t-exl3v3-selftest.log)"
timeout 900 $W/build-exl3v3/exl3v3_tool tiles --tiles 4096 --host 512 > $L/t-exl3v3-tiles.log 2>&1; echo "exl3v3 tiles RC=$? $(tail -n 1 $L/t-exl3v3-tiles.log)"
timeout 1800 $B/bin/r4d_selftest $B/radiance_home/kernels/libr4d.so --mxcheck > $L/t-r4d-mxcheck.log 2>&1; echo "r4d_selftest --mxcheck RC=$? $(grep -c '^MXREF.*: ok' $L/t-r4d-mxcheck.log) MXREF ok, $(grep -c '^MXREF.*FAIL' $L/t-r4d-mxcheck.log) MXREF FAIL; $(grep MXVERDICT $L/t-r4d-mxcheck.log)"
for d in exl3i exl3quant qwen35_exl3; do
  cmake -S $SRC/plugin/$d -B $W/build-$d -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;/opt/rocm" -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang-24 -DCMAKE_EXE_LINKER_FLAGS="-lstdc++ -lm" -DRAD_PLUGIN_INSTALL_HOME=$W/home > $L/t-conf-$d.log 2>&1; echo "$d conf RC=$?"
  cmake --build $W/build-$d -j24 -- -k 0 > $L/t-build-$d.log 2>&1; echo "$d build RC=$?"
  ctest --test-dir $W/build-$d -j1 --no-tests=error --output-on-failure --timeout 1800 > $L/t-ctest-$d.log 2>&1; echo "$d ctest RC=$? $(grep -E 'tests passed|tests failed' $L/t-ctest-$d.log)"
done
[ -x $W/build-exl3i/exl3i_moe_host_test ] && { $W/build-exl3i/exl3i_moe_host_test > $L/t-moe-host.log 2>&1; echo "moe host RC=$?"; }
[ -x $W/build-exl3i/test_moe ] && { timeout 900 $W/build-exl3i/test_moe $W/build-exl3i/radiance_home/kernels/exl3i.so > $L/t-test-moe.log 2>&1; echo "test_moe RC=$? $(tail -n 2 $L/t-test-moe.log | tr '\n' ' ')"; }
echo TESTS_DONE
