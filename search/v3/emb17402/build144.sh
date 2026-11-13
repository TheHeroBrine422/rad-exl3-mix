#!/bin/bash
# build144.sh -- (w144, #174-02, box2) the three plugins embed_exl3i touches (exl3dense: the op; exl3quant: token_embd
# import; qwen35_exl3: serves it), from W/src (git archive of radiance-hero qc-exl3-v3-emb144, COMMIT), built standalone
# against /data/q27/rad10-perf/install (core 5c9956c = the branch's base) with rad10's compiler setup (ROCm 10.1, the
# VOPD-repeatable shadow clang; pipeline/hosts/rad10/rad10build.sh), host tests, then the serve home W/home =
# rad10-perf/home with these three .so. CPU only (no card). Log W/logs/build144.log.
set -e; W=/data/q27/w144; S=$W/src; I=/data/q27/rad10-perf/install; R=/opt/rocm-10.1; RR=/data/q27/rad10/rocm-repeatable
export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:/data/q27/venv/bin:$PATH
HIPC=(-DCMAKE_HIP_COMPILER=$RR/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R
      -DRAD_HIP_VOPD_REPEATABLE=ON)
mkdir -p $W/logs $W/pb; exec >> $W/logs/build144.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $(cat $S/COMMIT)"
for p in exl3dense exl3quant qwen35_exl3; do
  cmake -S $S/plugin/$p -B $W/pb/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$R" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$R/bin/hipcc "${HIPC[@]}" > $W/pb/$p.log 2>&1 && nice -n 10 cmake --build $W/pb/$p -j ${J:-12} >> $W/pb/$p.log 2>&1 \
      || { say "$p: BUILD FAIL ($W/pb/$p.log)"; exit 1; }
  ctest --test-dir $W/pb/$p -L host --output-on-failure >> $W/pb/$p.log 2>&1 || { say "$p: HOST TESTS FAIL"; exit 1; }
  say "$p: built, host tests $(grep -c 'Test #.*Passed' $W/pb/$p.log) passed"
done
rm -rf $W/home.part; cp -a /data/q27/rad10-perf/home $W/home.part
for f in $(find $W/pb -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $W/home.part/$d ] && cp $f $W/home.part/$d/; done
rm -rf $W/home; mv $W/home.part $W/home; say "HOME READY $W/home"
