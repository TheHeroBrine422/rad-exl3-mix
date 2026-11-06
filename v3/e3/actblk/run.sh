#!/bin/bash
# actblk/run.sh [PAR] -- #103 on colo (README.md): trellis3b.py, ffn_down of #95 (L10, L30) plus L0 / L43 / L63
# (#57's down tensors, robustness only), codebook cubic0.02_i8, 128 rows, #95's settings (factors 0.85-1.5, CD=0,
# free start), K2 / 3 / 4, E3_ORDER natural,blk128 in one job (paired rows). Run from the copy whose parent holds
# e3.py, trellis.py, libviterbi.so. Idempotent (data/<job>.jsonl.part renamed when done; running jobs skipped). PAR (4)
# jobs, ~12 GB each. LAYERS overrides the layer list (robustness extension: LAYERS="2 58 61", high block-trace spread).
cd "$(dirname "$0")"; mkdir -p data ../logs
export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  E3_FACTORS=0.85,1.0,1.15,1.3,1.5 CD=0 E3_ORDER=natural,blk128
one() {  # one NAME LAYER K
  [ -s data/$1.jsonl ] && return
  pgrep -f "trellis3b.py .* data/$1.jsonl.part" > /dev/null && return   # running (another runner)
  rm -f data/$1.jsonl.part
  nice -n 10 python3 -W ignore trellis3b.py $2 ffn_down cubic0.02_i8 $3 128 data/$1.jsonl.part > ../logs/actblk-$1.log 2>&1 && mv data/$1.jsonl.part data/$1.jsonl
  echo "$(date -u +%H:%M) rc=$? $1"
}
export -f one
for L in ${LAYERS:-10 30 0 43 63}; do for K in 2 3 4; do echo "d$L-K$K $L $K"; done; done | xargs -P ${1:-4} -L 1 bash -c 'one "$@"' _
echo ACTBLK DONE
