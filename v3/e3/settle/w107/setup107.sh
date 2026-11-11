#!/bin/bash
# setup107.sh W100DIR W102DIR MYDIR BOX -- stage w107's CPU work dir on colo / Thor from w100's and w102's (read-only
# reuse: their code copy with the box-built libviterbi.so / libring102.so, row cache, w102's stored scales, H halves,
# bn1920 H_out). Expects MYDIR/stage/ (w107/ code + gates/s100.json) already copied. Idempotent.
set -eu
W100=$1 W102=$2 D=$3 BOX=$4
mkdir -p $D/gates $D/out $D/logs; cd $D
[ -d code ] || { cp -a $W100/code code.part; mv code.part code; }
rm -rf code/v3/e3/settle/w102; cp -a $W102/code/v3/e3/settle/w102 code/v3/e3/settle/w102
[ -f code/v3/e3/settle/w102/libring102.so ] || { echo "no libring102.so in $W102"; exit 1; }
rm -rf code/v3/e3/settle/w107; cp -a stage/w107 code/v3/e3/settle/w107
cp stage/s100.json gates/s100.json
[ -d cache ] || { cp -a $W100/cache cache.part; mv cache.part cache; }
[ -d scales ] || { cp -a $W102/scales scales.part; mv scales.part scales; }
[ -e hb ] || ln -s $(readlink -f $W100/hb) hb
[ -e hout-bn ] || ln -s $(readlink -f $W100/hout-bn) hout-bn
cp $W100/q87.py q87.py
python3 code/v3/e3/settle/w107/plan107.py $BOX > jobs-$BOX.txt
echo "setup ok: $(wc -l < jobs-$BOX.txt) jobs"; ls -la
