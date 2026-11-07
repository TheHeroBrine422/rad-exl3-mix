#!/bin/bash
# w46: test ourq-rowerr.patch on the 4070 (one layer, K7 mul1), under gpu.lock
set -u
P=/tmp/w46-i8real/pq; rm -rf $P; mkdir -p $P/quantize; ln -s /workspace/rad-exl3-mix/v3 $P/v3
cp /workspace/rad-exl3-mix/quantize/ourq.py $P/quantize/ && (cd $P && patch -p1 < /workspace/rad-exl3-mix/v3/e3/settle/reval/ourq-rowerr.patch)
exec 9>/data/yb/gpu.lock; flock 9; t1=$(date +%s); echo "lock $(date -u +%T)"
cd /tmp/w46-run && PYTHONPATH=/data/src/exllamav3 RXM_WORK=/data/p08 RXM_CKPT=/data/models/qwen35-08b RXM_TEMPLATE=/data/p08/template.json \
  TORCH_EXTENSIONS_DIR=/data/torch_ext PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 300 \
  python3 $P/quantize/ourq.py -K 7 -o /tmp/w46-i8real/pt/k7-mul1 --device 0 --layers 7-7 --codebook mul1 --qtf2 \
  --frac-ext /data/src/exllamav3/exllamav3/exllamav3_ext --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq > /tmp/w46-i8real/pt.log 2>&1
echo "rc=$? held $(( $(date +%s) - t1 ))s"; flock -u 9; echo PT DONE
