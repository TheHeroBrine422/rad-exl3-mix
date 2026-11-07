#!/bin/bash
# i8real.sh OUTDIR [LAYERS] -- w46 item 5 (reval.md): ourq.py --codebook i8 vs mul1 on real 0.8B tensors at K6.5 / 7 / 8,
# proxy error rel = sum(err) / sum(ref) per linear from --save-wq's rowerr.npz (tr(dW H dW^T) / tr(W H W^T) under the
# undamped calib H_in), to check w45's settle.py numbers (i8 / mul1 = x1.018 K6.5, x1.27 K7, x4.5 K8) outside the CPU
# proxy. The ladder's settings as proxy2b/q2b.sh (qtf2, scaled Hessian, sigma 0.025 / input group 0.005), layers
# LAYERS (default 6-7: one GDN layer, one attention layer). Runs on the 4070 box under `flock /data/yb/gpu.lock`
# for the whole run (shared with the lead's queues), VRAM_CAP_GB 7.2. Skips arms whose shards exist (resumable).
# Reads /data/p08 (template, Hessian link) and the cached kernels in /data/torch_ext; writes only OUTDIR.
set -u
O=$1; L=${2:-6-7}
mkdir -p $O /tmp/w46-run
SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
t0=$(date +%s)
exec 9>/data/yb/gpu.lock
flock 9
echo "lock $(date -u +%H:%M:%S) after $(( $(date +%s) - t0 ))s"
t1=$(date +%s)
for K in 6.5 7 8; do for CB in mul1 i8; do
  D=$O/k$K-$CB; lo=${L%-*}; hi=${L#*-}
  [ -f $D/q-L$(printf %02d $hi).safetensors ] && { echo "skip $D"; continue; }
  ( cd /tmp/w46-run && PYTHONPATH=/data/src/exllamav3 RXM_WORK=/data/p08 RXM_CKPT=/data/models/qwen35-08b \
    RXM_TEMPLATE=/data/p08/template.json TORCH_EXTENSIONS_DIR=/data/torch_ext VRAM_CAP_GB=7.2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 600 \
    python3 /workspace/rad-exl3-mix/quantize/ourq.py -K $K -o $D --device 0 --layers $L --codebook $CB --qtf2 \
    --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq ) > $O/k$K-$CB.log 2>&1
  echo "K$K $CB rc=$? $(( $(date +%s) - t1 ))s"
done; done
flock -u 9
echo "released $(date -u +%H:%M:%S), held $(( $(date +%s) - t1 ))s"
echo I8REAL DONE
