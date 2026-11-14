#!/bin/bash
# bc.sh PAR [NT] -- #177 synthetic search curve on colo (w139): bcurve.py over {mad, hash, mul1i8} x K 2 / 3 / 4 x arms
# (exact V16-V20, beam B18 / B19 / B20 x M, beam + segment refine), one job a (code, K, arm), PAR at once, nice 19,
# one thread each. Restartable (bcurve.py skips arms already in its OUT). Builds the libraries with -march=native.
# Run from its directory (beamL.c = w71's, beamS.c, bcurve.py). Out res/<code>-K<k>-<arm>.jsonl. (w139)
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p res logs
[ libbeamL.so -nt beamL.c ] || gcc -O3 -march=native -fopenmp -shared -fPIC beamL.c -o libbeamL.so
[ libbeamS.so -nt beamS.c ] || gcc -O3 -march=native -fopenmp -shared -fPIC beamS.c -o libbeamS.so
export OMP_NUM_THREADS=1 NT=${2:-256}
ARMS="V16 V17 V18 V19 V20 B18-1024 B18-4096 B18-16384 B18-65536 B19-4096 B19-16384 B19-65536 B20-1024 B20-4096 B20-16384 B20-65536 P20-16384-12"
for c in mad hash mul1i8; do for k in 2 3 4; do
  for a in $ARMS; do echo "$k $c $a"; done; done; done |
  xargs -P ${1:-12} -L 1 sh -c 'o=res/$1-K$0-$2.jsonl; [ -s $o ] && exit 0
    nice -n 19 python3 -I bcurve.py $0 $1 $2 $o $NT >> logs/$1-K$0.log 2>&1'   # gain: a deterministic V16 sweep a job
echo "BC DONE $(date -u +%T)" >> logs/bc.log
