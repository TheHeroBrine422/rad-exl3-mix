#!/bin/bash
# radbuild-perf.sh [TGZ] -- the staged radiance build of qc-exl3-v3-perf for box2 / ds8 (#164-02, w136), into
# OUT=/data/q27/rad10-perf (never /data/q27/rad or /data/q27/rad10, which hostd reads).
#   TGZ: `git archive --prefix=src/` of radiance-hero qc-exl3-v3-perf (5c9956c = 374503f + the merge of qc-exl3-v3, i.e.
#        rad-convert --no-error and --kld-progress, which rad10 carried as core-42960d5..ca5491d.patch); unpacked to
#        OUT/src unless OUT/src exists. Its tree equals rad10/r374/src (w130).
# Toolchain and recipe are w130's rad10 (pipeline/hosts/rad10/: /opt/rocm-10.1, VOPD-repeatable shadow clang
# /data/q27/rad10/rocm-repeatable, gfx1201, our five plugins + their host tests). FF=DIR (default /data/q27/rad10/ffmpeg
# when present; FF= for none): media decoding as rad10/r374.
#   OUT/install, OUT/home   serve (radiance, KLD); the branch head's qwen35_exl3 (vision tower + DFlash2)
#   OUT/home-pack           pack (rad-convert): the same home with qwen35_exl3 from 1aa18de (text-only declare). The head's
#                           plugin declares the bf16 vision tower at convert (639ad2f), so its packs carry model.visual.*
#                           (+0.92 GB on K4, w130); every other plugin (exl3quant / exl3i: R16 rows, encoding v3) is the
#                           head's. Text packs from home-pack are byte-identical to rad10/home-pack's.
# Idempotent (each step skips when its marker exists); log OUT/logs/radbuild-perf.log.
set -e; Q=/data/q27; W=$Q/rad10; O=${OUT:-$Q/rad10-perf}; R=/opt/rocm-10.1; RB=${RB:-$W/scripts/rad10build.sh}
FFD=${FF-$W/ffmpeg}; [ -d "$FFD/inst/lib" ] || FFD=
mkdir -p $O/logs; LOG=$O/logs/radbuild-perf.log; say() { echo "$(date -u +%T) $*" | tee -a $LOG; }
if [ ! -d $O/src ]; then
  [ -n "${1:-}" ] || { say "no $O/src and no TGZ"; exit 1; }
  tar xzf $1 -C $O; say "src from $1"
fi
[ -f $O/src/COMMIT ] || { say "write $O/src/COMMIT (the fork commit) first"; exit 1; }
# 1 core + plugins + OUT/home (rad10build.sh: OUT / SRC / FF)
OUT=$O SRC=$O/src FF=$FFD bash $RB $W; tail -n 1 $O/logs/rad10build.log | tee -a $LOG
grep -q "HOME READY $O/home" $O/logs/rad10build.log || { say "rad10build failed ($O/logs/rad10build.log)"; exit 1; }
# 2 pack home: qwen35_exl3 at 1aa18de, the 10.1 compiler setup, against OUT/install
S1=$W/src-1aa18de; PP=$O/pb-pack
[ -f $S1/plugin/qwen35_exl3/CMakeLists.txt ] || { say "no $S1 (git archive of 1aa18de)"; exit 1; }
if [ ! -f $PP/ok ]; then
  ( export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$Q/venv/bin:$PATH
    cmake -S $S1/plugin/qwen35_exl3 -B $PP -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$O/install;$R" \
      -DRAD_GPU_TARGETS=gfx1201 -DRAD_HIPCC=$R/bin/hipcc -DCMAKE_HIP_COMPILER=$W/rocm-repeatable/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_HIP_FLAGS=--rocm-path=$R -DRAD_HIP_VOPD_REPEATABLE=ON \
      && nice -n 10 cmake --build $PP -j ${J:-24} ) >> $LOG 2>&1 || { say "pack plugin build FAIL"; exit 1; }
  touch $PP/ok; say "pack qwen35_exl3 (1aa18de) built"
fi
so=$(find $PP -path '*radiance_home/*' -name qwen35_exl3.so | head -n 1)
rm -rf $O/home-pack.part; cp -a $O/home $O/home-pack.part; cp $so $O/home-pack.part/architectures/qwen35_exl3.so
rm -rf $O/home-pack; mv $O/home-pack.part $O/home-pack
( cd $O && find install home home-pack -type f -print0 | sort -z | xargs -0 sha256sum ) > $O/tree.sha256
say "RADBUILD-PERF READY $O (src $(head -n 1 $O/src/COMMIT); ff ${FFD:-off}; tree.sha256)"
