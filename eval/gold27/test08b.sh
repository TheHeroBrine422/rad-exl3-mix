#!/bin/bash
# test08b.sh -- after test08.sh: --attn-chunk (query-block attention, not bit-exact by design) and a resume from
# partial runs, on the same 0.8B corpus, each compared with mkgold.py's rows (W/ref). Under the 4070's gpu.lock.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
M=${M:-/data/models/qwen35-08b}; W=${W:-/data/yb/w18/t08}; export VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
rm -rf "$W"/ac "$W"/s-ac "$W"/kr "$W"/s-kr
flock /data/yb/gpu.lock timeout 600 bash -c "
  python3 $D/mkgold27.py $M $W/t.jsonl $W/ac --scratch $W/s-ac --stages 2 --scratch-dtype bf16 --attn-chunk 1024
  # resume: stage 0 only, then documents 0:2 through stages 1-2, then the whole command again
  python3 $D/mkgold27.py $M $W/t.jsonl $W/kr --scratch $W/s-kr --stages 3 --scratch-dtype bf16 --only-stages 0:1 --keep-scratch
  python3 $D/mkgold27.py $M $W/t.jsonl $W/kr --scratch $W/s-kr --stages 3 --scratch-dtype bf16 --docs 0:2 --keep-scratch
  python3 $D/mkgold27.py $M $W/t.jsonl $W/kr --scratch $W/s-kr --stages 3 --scratch-dtype bf16
"
for g in ac kr; do echo "$g $(CUDA_VISIBLE_DEVICES= nice -n 19 python3 $D/cmpgold.py $W/ref $W/$g)"; done | tee "$W/cmp-b.txt"
