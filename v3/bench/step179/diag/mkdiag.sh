#!/bin/bash
# mkdiag.sh SRCNAME -- on box2: src-SRCNAME + exl3_diag.h included in qwen35_exl3.cpp -> src-diag-SRCNAME, qwen35_exl3
# built into b10diag-SRCNAME/home (timing diagnostic only). (w152)
N=$1; W=/data/q27/w152; cd $W && rm -rf src-diag-$N b10diag-$N && cp -a src-$N src-diag-$N && cp tools/exl3_diag.h src-diag-$N/plugin/qwen35_exl3/ &&
sed -i 's|^#include "exl3_dense_op.h"$|#include "exl3_dense_op.h"\n#include "exl3_diag.h"|' src-diag-$N/plugin/qwen35_exl3/qwen35_exl3.cpp &&
grep -n exl3_diag src-diag-$N/plugin/qwen35_exl3/qwen35_exl3.cpp &&
PLUGS=qwen35_exl3 BASEHOME=$W/b10base/home bash tools/plugb.sh 10 $W/src-diag-$N $W/b10diag-$N < /dev/null; tail -n 2 b10diag-$N/plugb.log
