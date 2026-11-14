#!/bin/bash
# build155.sh -- (w155, #180-02, ds8, CPU only) exl3i + exl3quant + qwen35_exl3 from W/src (a tar of radiance-hero
# qc-exl3-v3-rr's plugin/ + tests/ + arch/common/rad_block_dflash2.h, COMMIT names it), built standalone against an
# overlay of /data/q27/rad10-perf/install whose include/radiance/arch/rad_block_dflash2.h is the branch's (the block
# gains op_rr_ext; the installed one is byte-identical to qc-exl3-v3 9f0a059's, checked below), with rad10's compiler
# setup (as w153's build153r.sh); host tests (every test not labelled gpu); then W/home = rad10-perf/home with these
# three .so. Log W/logs/build155.log.
set -e; W=/data/q27/w155; S=$W/src; I=/data/q27/rad10-perf/install; R=/opt/rocm-10.1; RR=/data/q27/rad10/rocm-repeatable
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
HIPC=(-DCMAKE_HIP_COMPILER=$RR/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R
      -DRAD_HIP_VOPD_REPEATABLE=ON)
mkdir -p $W/logs $W/pb; exec >> $W/logs/build155.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $(cat $S/COMMIT 2>/dev/null)"
# the overlay install: symlinks to the install, the block header the branch's
H=include/radiance/arch/rad_block_dflash2.h
[ "$(sha256sum < $I/$H | cut -c1-64)" = "$(cat $S/BASE_DFLASH2_SHA)" ] || { say "installed $H is not 9f0a059's"; exit 1; }
rm -rf $W/inst; cp -as $I $W/inst; rm $W/inst/$H; cp $S/arch/common/rad_block_dflash2.h $W/inst/$H
for p in exl3i exl3quant qwen35_exl3; do
  rm -rf $W/pb/$p
  cmake -S $S/plugin/$p -B $W/pb/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$W/inst;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" > $W/pb/$p.log 2>&1 && nice -n 10 cmake --build $W/pb/$p -j ${J:-8} >> $W/pb/$p.log 2>&1 \
      || { say "$p: BUILD FAIL ($W/pb/$p.log)"; exit 1; }
  ctest --test-dir $W/pb/$p -LE gpu --output-on-failure >> $W/pb/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $W/pb/$p.log) passed"
done
say "qwen35_exl3 RADIANCE_INCLUDE_DIR: $(grep -m1 RADIANCE_INCLUDE_DIR $W/pb/qwen35_exl3/CMakeCache.txt)"
rm -rf $W/home.part; cp -a /data/q27/rad10-perf/home $W/home.part
for f in $(find $W/pb -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $W/home.part/$d ] && cp $f $W/home.part/$d/; done
rm -rf $W/home; mv $W/home.part $W/home
say "HOME READY $W/home ($(for p in exl3i qwen35_exl3; do sha256sum $(find $W/home -name $p.so) | cut -c1-16; done | tr '\n' ' '))"
