#!/bin/bash
# rq.sh K CODEBOOK OUT [realq.py options] -- [ourq args...] -- REAL-Q (realq.py) on the 0.8B with exactly q2b.sh's ourq
# settings (qtf2, scaled Hessian, damping 0.025, input groups 0.005 / 0.05 below K1, every trunk layer unless --layers is
# given again, --save-wq). FISHER (default /data/yb/d1109/fisher/fisher.pt) is passed as --fisher.
K=$1 CB=$2 O=$3; shift 3
RQ=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do RQ+=("$1"); shift; done; [ "$1" = "--" ] && shift
M=/data/models/qwen35-08b W=/data/p08; mkdir -p $W/challenge/hessians; ln -sfn /data/h08/calib $W/challenge/hessians/calib
python3 -c "import json; c=json.load(open('$M/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$W/template.json', 'w'))"
SIGIN=$(python3 -c "print(0.05 if $K < 1 else 0.005)")
SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
mkdir -p /tmp/rq-run && cd /tmp/rq-run   # not -I: PYTHONPATH carries the in-place exllamav3 build (as q2b.sh)
PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=$M RXM_TEMPLATE=$W/template.json \
  TORCH_EXTENSIONS_DIR=/data/torch_ext PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} \
  python3 /workspace/rad-exl3-mix/proxy2b/exp/d1109/realq.py --fisher ${FISHER:-/data/yb/d1109/fisher/fisher.pt} "${RQ[@]}" -- \
  -K $K -o $O --device 0 --layers 0-23 --codebook $CB --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 \
  --sigma-reg-in $SIGIN --save-wq "$@"
