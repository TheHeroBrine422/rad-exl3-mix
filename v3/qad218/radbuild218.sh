#!/bin/bash
# radbuild218.sh (iceman, #218) -- radiance qc-exl3-v3 9a20d9be (the V3 release tree, git archive) + its exl3 plugins with the system
# ROCm 7.14 (/opt/rocm), gfx1201, into /data/w218/rad/install + plugin home /data/w218/rad/home (= box2 tools/radbuild.sh +
# plugbuild.sh). CPU only (nice 10, -j 6). For the one served read (score218.sh). Log /data/w218/logs/radbuild.log.
set -e; A=/data/w218/rad; S=$A/radiance; B=$A/build; I=$A/install; H=$A/home; PB=$A/pb
export ROCM_PATH=/opt/rocm; export PATH=$ROCM_PATH/bin:$PATH
exec >> /data/w218/logs/radbuild.log 2>&1; echo "$(date -u +%T) start"
[ -f $B/CMakeCache.txt ] || cmake -S $S -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1201 \
   -DCMAKE_INSTALL_PREFIX=$I -DRAD_WITH_FFMPEG=OFF -DRAD_HIPCC=$ROCM_PATH/bin/hipcc -DCMAKE_PREFIX_PATH=$ROCM_PATH
nice -n 10 cmake --build $B -j 6
cmake --install $B
mkdir -p $PB
for p in exl3quant qwen35_exl3 exl3i; do
  [ -f $S/plugin/$p/CMakeLists.txt ] || continue
  [ -f $PB/$p.ok ] && continue
  cmake -S $S/plugin/$p -B $PB/$p -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$I;$ROCM_PATH" -DRAD_GPU_TARGETS=gfx1201 \
      -DRAD_HIPCC=$ROCM_PATH/bin/hipcc && nice -n 10 cmake --build $PB/$p -j 6 && touch $PB/$p.ok
done
rm -rf $H.part; cp -a $I/share/radiance $H.part
for f in $(find $PB -path '*radiance_home/*' -name '*.so'); do d=$(basename $(dirname $f)); [ -d $H.part/$d ] && cp $f $H.part/$d/; done
rm -rf $H; mv $H.part $H; echo "$(date -u +%T) RADBUILD218 DONE"
