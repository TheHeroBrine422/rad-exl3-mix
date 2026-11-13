#!/bin/bash
# llvm10-vopd.sh SRC [R] [W] -- radiance's Dockerfile `llvm` + `build` compiler steps, on the host: ROCm's libLLVM rebuilt
# from ROCm's llvm-project at the commit R's clang names, with SRC/docker/llvm-vopd-pairing.patch (repeatable VOPD
# pairing), configured as the Dockerfile does; then the shadow ROCm tree W/rocm-repeatable (every entry a link into R
# except lib/llvm/bin, copied, and the rebuilt libLLVM), whose clang++ the radiance build uses with
# -DRAD_HIP_VOPD_REPEATABLE=ON --rocm-path=R. R = /opt/rocm-10.1 (rocm10-get.sh), W = /data/q27/rad10.
# Restartable (the fetch and the build dir are kept until the shadow exists). Log W/logs/llvm.log. (w130, #170)
set -e; SRC=$1; R=${2:-/opt/rocm-10.1}; W=${3:-/data/q27/rad10}; D=$W/rocm-repeatable
L=$R/lib/llvm; S=$R/lib/rocm_sysdeps; LP=$W/llvm-project; LB=$W/build-llvm
export PATH=/data/q27/venv/bin:$PATH                                # cmake, ninja
mkdir -p $W/logs; exec >> $W/logs/llvm.log 2>&1
say() { echo "$(date -u +%T) $*"; }
[ -f $D/.ok ] && { say "shadow present: $D"; exit 0; }
rev=$($L/bin/clang --version | sed -n 's|.*ROCm/llvm-project.git \([0-9a-f]\{40\}\).*|\1|p')
[ -n "$rev" ] || { say "no ROCm/llvm-project commit in: $($L/bin/clang --version | head -1)"; exit 1; }
say "start: llvm-project $rev"
if [ ! -f $LP/.fetched ]; then
  rm -rf $LP; git init -q $LP; cd $LP; git remote add origin https://github.com/ROCm/llvm-project.git
  git sparse-checkout set --no-cone '/*' '!/*/' /cmake/ /libc/ /llvm/ /third-party/ \
    '!/llvm/test/' '!/llvm/unittests/' '!/llvm/docs/' '!/llvm/benchmarks/'
  git fetch -q --depth 1 --filter=blob:none origin "$rev"; git checkout -q FETCH_HEAD
  git apply $SRC/docker/llvm-vopd-pairing.patch; touch .fetched; say "fetched + patched"
fi
cd $LP
if [ ! -f $LB/build.ninja ]; then
  cmake -G Ninja -S llvm -B $LB -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;SPIRV;X86" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_DYLIB_COMPONENTS=all -DLLVM_ENABLE_RTTI=OFF -DLLVM_ENABLE_EH=OFF -DLLVM_ENABLE_FFI=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_CURL=OFF \
    -DLLVM_ENABLE_HTTPLIB=OFF -DLLVM_ENABLE_TELEMETRY=ON -DLLVM_ENABLE_PLUGINS=ON \
    -DLLVM_ENABLE_LIBPFM=OFF \
    -DLLVM_ENABLE_ZLIB=FORCE_ON -DZLIB_INCLUDE_DIR=$S/include -DZLIB_LIBRARY=$S/lib/librocm_sysdeps_z.so.1 \
    -DLLVM_ENABLE_ZSTD=FORCE_ON -Dzstd_INCLUDE_DIR=$S/include -Dzstd_LIBRARY=$S/lib/librocm_sysdeps_zstd.so.1 \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_DOCS=OFF -DLLVM_BUILD_TOOLS=OFF -DLLVM_USE_LINKER=lld -DLLVM_PARALLEL_LINK_JOBS=2 \
    -DCMAKE_C_COMPILER=$L/bin/clang -DCMAKE_CXX_COMPILER=$L/bin/clang++ > $W/logs/llvm-configure.log 2>&1 \
    || { tail -40 $W/logs/llvm-configure.log; exit 1; }
  say "configured"
fi
nice -n 10 ninja -C $LB ${J:+-j$J} LLVM > $W/logs/llvm-build.log 2>&1 || { grep -m20 -B2 -A8 'error' $W/logs/llvm-build.log; exit 1; }
so=$(basename $(readlink -f $L/lib/libLLVM.so)); [ -f $LB/lib/$so ] || { say "built no $so"; exit 1; }
say "built $so"
# the shadow tree, as the Dockerfile's build stage (R's root is a plain directory here, not alternatives links)
rm -rf $D; mkdir -p $D/lib/llvm/lib
for e in $R/* $R/.info; do n=${e##*/}; [ $n = lib ] || ln -s $e $D/$n; done
for e in $R/lib/*; do n=${e##*/}; [ $n = llvm ] || ln -s $e $D/lib/$n; done
for e in $R/lib/llvm/*; do n=${e##*/}; [ $n = bin ] || [ $n = lib ] || ln -s $e $D/lib/llvm/$n; done
cp -a $R/lib/llvm/bin $D/lib/llvm/
for e in $R/lib/llvm/lib/*; do n=${e##*/}
  if [ $n = $so ]; then cp $LB/lib/$so $D/lib/llvm/lib/
  elif [ -L $e ] && [ "$(readlink $e)" = $so ]; then ln -s $so $D/lib/llvm/lib/$n
  else ln -s $e $D/lib/llvm/lib/$n; fi
done
touch $D/.ok; say "SHADOW READY $D"
