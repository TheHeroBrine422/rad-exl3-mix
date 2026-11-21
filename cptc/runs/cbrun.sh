#!/bin/bash
# cbrun.sh -- cbkeff.py over the w208 candidate set x K1-4 on colo, one single-threaded job each, restartable.
cd /data/work/cptcv; mkdir -p logs/cbk
export CPTC_VIT3=1 CPTC_LIBDIR=/data/work/cptcv/v3/e3 CPTC_E3DIR=/data/work/cptcv/v3/e3 OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive KEFF_NTRY=4
for K in 1 2 3 4; do for n in mad mul1_i8 int4grid mad4 cptcmin lc1 cptc5 cptc6u cptcst10; do echo "$n $K"; done; done |
  xargs -P ${P:-34} -L 1 sh -c 'f=logs/cbk/$0-K$1.log; grep -q "^{" $f 2>/dev/null && exit 0; python3 cptc/runs/cbkeff.py $0 $1 16384 > $f.part 2>&1 && mv $f.part $f; echo "$(date -u +%H:%M) $0 K$1 rc=$?" >> logs/cbk/done'
echo "CBK ALL DONE $(date -u +%H:%M)" >> logs/cbk/done
