#!/bin/bash
# rcbuild.sh -- box2: rad-convert from radiance 42960d5 + rad-convert-no-error.patch, own build dir (w123); log logs/rcbuild.log
set -e; Q=/data/q27; W=$Q/w123; S=$W/radsrc; B=$W/radbuild
export ROCM_PATH=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel; export PATH=$ROCM_PATH/bin:$Q/venv/bin:$PATH
exec >> $W/logs/rcbuild.log 2>&1; echo "$(date -u +%T) start"
[ -d $S ] || { cp -a $Q/rad/radiance $S; cd $S && patch -p1 < $W/rad-convert-no-error.patch; }
[ -f $B/CMakeCache.txt ] || cmake -S $S -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1201 \
   -DCMAKE_INSTALL_PREFIX=$W/radinst -DRAD_WITH_FFMPEG=OFF -DRAD_HIPCC=$ROCM_PATH/bin/hipcc -DCMAKE_PREFIX_PATH=$ROCM_PATH
nice -n 15 cmake --build $B -j 12 --target rad-convert
echo "$(date -u +%T) RCBUILD DONE"
