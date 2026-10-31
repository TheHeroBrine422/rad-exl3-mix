#!/bin/bash
# vtcpu.sh -- the tiered-embedding preps on a CPU box (CPU box B): emb-ck.bf16 (embtable.py) copied and fake-quantised
# per plan, all in parallel (single-threaded numpy each). Output out/<tag>.bf16 + out/<tag>.done; vtpull.sh moves them.
cd "$(dirname "$0")"; mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f rank-guard-counts.npy ] || python3 -I vtguard.py rank-sq-counts.npy rank-guard-counts.npy
one() {  # RANK PFX PLAN
  local t=vt-emb${2:+-$2}-${3//[:,]/_}
  cp emb-ck.bf16 out/$t.bf16.part && nice python3 -I embtier.py out/$t.bf16.part 0 $1 "$3" --write > out/$t.log 2>&1 \
    && mv out/$t.bf16.part out/$t.bf16 && touch out/$t.done || echo FAIL > out/$t.done
}
for p in 100000:4,1 50000:4,1 10000:4,90000:3,1 100000:3,1 30000:4,2 30000:3,2 20000:4,30000:3,2; do one rank-sq-counts.npy "" $p & done
for p in 100000:4,2 50000:4,2 30000:4,2 100000:4,1 50000:4,1; do one rank-guard-counts.npy g $p & done
wait; echo VTCPUDONE > out/ALLDONE
