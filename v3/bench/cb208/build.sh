#!/bin/bash
# build.sh [OUTDIR] [CAND ..] -- one binary a candidate (cb208_<Cand>), in parallel.
set -u
S=$(cd "$(dirname "$0")" && pwd); B=${1:-$S/bin}; shift || true
ALL="Raw Base Mad MadF Mad4 Mad4F Mul1I8 Int4Grid Qtip1Mad Qtip3Inst Cptc6Lds Cptc6Glb Cptc6Eq3 Cptc6FLds Cptc6FGlb Cptc5Lds Cptc5Glb Cptc5Reg St6Lds St6Glb St8Lds St8Glb St10Lds St10Glb St12Lds St12Glb CptcMin Lc1Lds Lc1Glb"
C=${*:-$ALL}; mkdir -p $B
printf '%s\n' $C | xargs -P ${JOBS:-6} -I{} sh -c "nice -n 10 ${HIPCC:-hipcc} -O3 -std=c++17 --offload-arch=gfx1201 -DCAND={} $S/cb208.hip -o $B/cb208_{} 2> $B/build_{}.log && echo built {} || { echo FAILED {}; tail -5 $B/build_{}.log; }"
