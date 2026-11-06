#!/bin/bash
# test08.sh -- mkgold27.py against proxy2b/mkgold.py on Qwen3.5-0.8B (4070 box): a 5-document public test corpus,
# N = 1 (f32), 2 (f32), 2 (bf16), 3 (f32), 3 (bf16); each compared with mkgold.py's rows (cmpgold.py). GPU steps under
# the 4070's gpu.lock, one hold, VRAM_CAP_GB (default 7.2). W = work dir (default /data/yb/w18/t08).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
M=${M:-/data/models/qwen35-08b}; W=${W:-/data/yb/w18/t08}; export VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
rm -rf "$W"; mkdir -p "$W"
CUDA_VISIBLE_DEVICES= nice -n 19 python3 "$D/mktest.py" "$M" "$W/t.jsonl"
flock /data/yb/gpu.lock timeout 840 bash -c "
  set -e
  python3 $D/mkgold27.py $M $W/t.jsonl $W/n1 --scratch $W/s-n1 --stages 1
  python3 $R/proxy2b/mkgold.py $M $W/n1 $W/ref 0 1 2 3 4
  python3 $D/mkgold27.py $M $W/t.jsonl $W/n2 --scratch $W/s-n2 --stages 2
  python3 $D/mkgold27.py $M $W/t.jsonl $W/n2b --scratch $W/s-n2b --stages 2 --scratch-dtype bf16
  python3 $D/mkgold27.py $M $W/t.jsonl $W/n3 --scratch $W/s-n3 --stages 3
  python3 $D/mkgold27.py $M $W/t.jsonl $W/n3b --scratch $W/s-n3b --stages 3 --scratch-dtype bf16
"
for g in n1 n2 n2b n3 n3b; do echo "$g $(CUDA_VISIBLE_DEVICES= nice -n 19 python3 $D/cmpgold.py $W/ref $W/$g)"; done | tee "$W/cmp.txt"
