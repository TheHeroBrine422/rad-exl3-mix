#!/bin/bash
# yaqaFact11.sh -- #33-21 (YAQA08.md amendment 14): diagonal H_out with short qad docs (YAQA_SHORT=256) vs long
# windows only; 0.8B, K3 / K2, 3 seeds. Waits for yaqaFact7-10.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 C=/data/yb/c21
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact11.txt; }
while pgrep -f "^/bin/bash /workspace/rad-exl3-mix/proxy2b/[y]aqaFact(7|8|9|10).sh" > /dev/null; do sleep 60; done
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
cd /tmp
if [ ! -f $C/hdiag-snap1600-dy/model.language_model.layers.23.mlp.down_proj.safetensors ]; then
  W=/data/c08work; mkdir -p $W/challenge/golden; ln -sfn /data/challenge/calib-meta $W/challenge/golden/calib
  RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json YAQA_FP32=1 YAQA_DIAG=1 \
  YAQA_DIAG_EXTRA=1 YAQA_SHORT=256 YAQA_SEED=0 YAQA_SNAP=1600 YAQA_SAVE_EVERY=100 \
    python3 /workspace/rad-exl3-mix/v3/yaqa/collect.py $C 1600 2048 > $O/c21.log 2>&1
  grep -q "^DONE" $O/c21.log || { log "COLLECT FAIL"; exit 1; }
  log "collected: $(grep -h 'YAQA_SHORT\|^DONE' $O/c21.log | tr '\n' ' ')"
fi
for s in 0 1 2; do sfx=$([ $s = 0 ] && echo "" || echo "_s$s"); for k in 3 2; do
  q k${k}dymix$sfx $k --hout $C/hdiag-snap1600-dy --sigma-reg-out 0.01 --seed-offset $s
  q k${k}dylong$sfx $k --hout $C/hdiag-snap1600-dy-long --sigma-reg-out 0.01 --seed-offset $s
done; done
log FACT11 DONE
