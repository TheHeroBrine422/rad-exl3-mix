#!/bin/bash
# yaqaFact12.sh -- YAQA08.md amendment 15: #78-02 (cubic int8 seeds 4-5) and #78-03 (i8c + diag H_out vs diag H_out).
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 C=/data/yb/c18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact12.txt; }
while pgrep -f "^/bin/bash /workspace/rad-exl3-mix/proxy2b/[y]aqaFact(7|8|9|10|11).sh" > /dev/null; do sleep 60; done
q() { tag=$1; k=$2; cb=$3; shift 3
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k $cb $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
cd /tmp
for s in 4 5; do for k in 3 2; do q k${k}base_s$s $k i8 --seed-offset $s; q k${k}cub_s$s $k i8c --seed-offset $s; done; done
for s in 0 1 2; do sfx=$([ $s = 0 ] && echo "" || echo "_s$s"); for k in 3 2; do
  q k${k}edycub$sfx $k i8c --hout $C/hdiag-snap1024-dy --sigma-reg-out 0.01 --seed-offset $s
done; done
log FACT12 DONE
