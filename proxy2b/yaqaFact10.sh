#!/bin/bash
# yaqaFact10.sh -- #78 KL test: cubic int8 codebook (i8c) vs i8, 0.8B, base settings, seed-paired. Waits for 7-9.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact10.txt; }
while pgrep -f "^/bin/bash /workspace/rad-exl3-mix/proxy2b/[y]aqaFact[789].sh" > /dev/null; do sleep 60; done
q() { tag=$1; k=$2; cb=$3; shift 3
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k $cb $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
cd /tmp
for k in 3 2; do for s in 0 1 2 3; do sfx=$([ $s = 0 ] && echo "" || echo "_s$s")
  q k${k}base$sfx $k i8 --seed-offset $s
  q k${k}cub$sfx $k i8c --seed-offset $s
done; done
for s in 0 1; do sfx=$([ $s = 0 ] && echo "" || echo "_s$s")
  q k4base$sfx 4 i8 --seed-offset $s; q k4cub$sfx 4 i8c --seed-offset $s
done
log FACT10 DONE
