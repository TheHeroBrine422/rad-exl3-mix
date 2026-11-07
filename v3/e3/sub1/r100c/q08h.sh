#!/bin/bash
# q08h.sh K OUT [ourq args...] -- proxy2b/q2b.sh's 0.8B trunk quant (same settings: qtf2, scaled Hessian, damping
# 0.025, input groups 0.005 / 0.05 below K1, every trunk layer, --save-wq) with --codebook hash, from the w48 mirror
# (setup4070.sh: patched ourq.py, own TORCH_EXTENSIONS_DIR). Hash A / B: env EXL3HASH_A / EXL3HASH_B (default w27's
# A 0xd0474b, IQR's B). Uses /data/p08 as q2b.sh prepared it (template.json, Hessian link).
K=$1 O=$2; shift 2
W=/data/yb/w48; SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
[ -f /data/p08/template.json ] || { echo "no /data/p08/template.json (run q2b.sh 08 once)"; exit 1; }
SIGIN=$(python3 -c "print(0.05 if $K < 1 else 0.005)")
mkdir -p /tmp/w48-run && cd /tmp/w48-run
PYTHONPATH=/data/src/exllamav3 RXM_WORK=/data/p08 RXM_CKPT=/data/models/qwen35-08b RXM_TEMPLATE=/data/p08/template.json \
  TORCH_EXTENSIONS_DIR=$W/torch_ext TORCH_CUDA_ARCH_LIST=8.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 $W/code/quantize/ourq.py -K $K -o $O --device 0 --layers 0-23 --codebook hash --qtf2 \
  --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in $SIGIN --save-wq "$@"
