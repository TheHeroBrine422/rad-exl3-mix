#!/bin/bash
# yaqaFact7.sh -- #33-18 (YAQA08.md amendment 11): estimator check (B / xw / dy) and dose 128-1024, 0.8B, 3 seeds.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 C=/data/yb/c18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact7.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
cd /tmp
if [ ! -f $C/hdiag-snap1024-dy/model.language_model.layers.23.mlp.down_proj.safetensors ]; then
  W=/data/c08work; mkdir -p $W/challenge/golden; ln -sfn /data/challenge/calib-meta $W/challenge/golden/calib
  RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json YAQA_FP32=1 YAQA_DIAG=1 \
  YAQA_DIAG_EXTRA=1 YAQA_SEED=0 YAQA_SNAP=128,256,512,1024 YAQA_SAVE_EVERY=64 \
    python3 /workspace/rad-exl3-mix/v3/yaqa/collect.py $C 1024 2048 > $O/c18.log 2>&1
  grep -q "^DONE" $O/c18.log || { log "COLLECT FAIL"; exit 1; }
  log "collected: $(grep '^DONE' $O/c18.log); halves: $(python3 -c "import json; h=json.load(open('$C/hdiag-snap1024/halves.json')); print(h['median'], h['p5'])")"
fi
declare -A SRC=([B]=hdiag-snap1024 [xw]=hdiag-snap1024-xw [dy]=hdiag-snap1024-dy [B512]=hdiag-snap512 [B256]=hdiag-snap256 [B128]=hdiag-snap128)
for s in 0 1 2; do for e in B xw dy B512 B256 B128; do for k in 3 2; do
  sfx=$([ $s = 0 ] && echo "" || echo "_s$s")
  q k${k}e$e$sfx $k --hout $C/${SRC[$e]} --sigma-reg-out 0.01 --seed-offset $s
done; done; done
log FACT7 DONE
