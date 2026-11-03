#!/bin/bash
# yaqaFact13.sh -- #33-22 (YAQA08.md amendment 16): Σdy² dose 1,350 / 2,048 / 2,700 (repeated windows, fresh labels)
# vs #33-18's edy (1,024), 0.8B, K3 / K2, 3 seeds.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 C=/data/yb/c22
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact13.txt; }
while pgrep -f "quantize/[o]urq.py|proxy2b/[e]valq.py|yaqa/[c]ollect.py" > /dev/null; do sleep 60; done
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
cd /tmp
if [ ! -f $C/hdiag-snap2700-dy/model.language_model.layers.23.mlp.down_proj.safetensors ]; then
  W=/data/c08work; mkdir -p $W/challenge/golden; ln -sfn /data/challenge/calib-meta $W/challenge/golden/calib
  RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json YAQA_FP32=1 YAQA_DIAG=1 \
  YAQA_DIAG_EXTRA=1 YAQA_SEED=0 YAQA_SNAP=1024,1350,2048,2700 YAQA_SAVE_EVERY=100 \
    python3 /workspace/rad-exl3-mix/v3/yaqa/collect.py $C 2700 2048 > $O/c22.log 2>&1
  grep -q "^DONE" $O/c22.log || { log "COLLECT FAIL"; exit 1; }
  log "collected: $(grep -h 'cycling\|^DONE' $O/c22.log | tr '\n' ' ')"
  python3 - <<'PY' | tee -a $O/result-fact13.txt
import glob
from safetensors import safe_open
a, b = "/data/yb/c18/hdiag-snap1024-dy", "/data/yb/c22/hdiag-snap1024-dy"; w = 0
for f in glob.glob(f"{a}/*.safetensors"):
    with safe_open(f, "pt") as x, safe_open(f.replace(a, b), "pt") as y: w = max(w, float((x.get_tensor("hout_diag") - y.get_tensor("hout_diag")).abs().max()))
print("c22 snap1024 vs c18 snap1024 (must be 0):", w)
PY
fi
for n in 1350 2048 2700; do for s in 0 1 2; do sfx=$([ $s = 0 ] && echo "" || echo "_s$s"); for k in 3 2; do
  q k${k}dy$n$sfx $k --hout $C/hdiag-snap$n-dy --sigma-reg-out 0.01 --seed-offset $s
done; done; done
log FACT13 DONE
