#!/bin/bash
# c2.sh SAMPLES THREADS K... -- CPTC i.i.d. tables (A.2 protocol, fixed E2M5) + Keff vs mad / mul1. Run in /data/work/cptcv.
cd /data/work/cptcv; S=$1 T=$2; shift 2; mkdir -p logs
export OMP_WAIT_POLICY=passive OMP_NUM_THREADS=$T CPTC_LIBDIR=/data/work/cptcv/v3/e3 CPTC_E3DIR=/data/work/cptcv/v3/e3
for K in "$@"; do (
  [ -f cptc/tables/table-K$K.npy ] || python3 -u cptc/cptc.py table $K $S > logs/table-K$K.log 2>&1
  KEFF_NTRY=${KN:-16} python3 -u cptc/runs/keffcmp.py $K ${KB:-16384} > logs/keff-K$K.log 2>&1
  echo "$(date -u +%H:%M) K$K done: $(tail -1 logs/keff-K$K.log)" >> logs/c2.done ) &
done; wait; echo "C2 ALL DONE $(date -u +%H:%M)" >> logs/c2.done
