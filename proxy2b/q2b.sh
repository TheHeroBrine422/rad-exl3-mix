#!/bin/bash
# q2b.sh MODEL K CODEBOOK OUT [ourq args...] -- ourq.py on a proxy model (4070 box) with the trunk ladder's settings
# (ladder2.sh: qtf2, scaled Hessian, damping 0.025, input groups 0.005 / 0.05 below K1), every trunk layer, the
# dequantized weights saved for evalq.py (--save-wq). MODEL: 2b | 08. Hessians /data/h<model>/calib.
# NOSH=1: without the scaled Hessian (calibration test 4).
M=$1 K=$2 CB=$3 O=$4; shift 4
declare -A CK=([2b]=/data/models/qwen35-2b [08]=/data/models/qwen35-08b)
W=/data/p$M; mkdir -p $W/challenge/hessians; ln -sfn /data/h$M/calib $W/challenge/hessians/calib
python3 -c "import json; c=json.load(open('${CK[$M]}/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$W/template.json', 'w'))"
SIGIN=$(python3 -c "print(0.05 if $K < 1 else 0.005)")
SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
mkdir -p /tmp/q2b-run && cd /tmp/q2b-run   # not -I: PYTHONPATH carries the in-place exllamav3 build; run from an empty dir
PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=${CK[$M]} RXM_TEMPLATE=$W/template.json \
  TORCH_EXTENSIONS_DIR=/data/torch_ext PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /workspace/rad-exl3-mix/quantize/ourq.py -K $K -o $O --device 0 --layers 0-23 --codebook $CB --qtf2 \
  --frac-ext $SRC $([ -z "$NOSH" ] && echo --scaled-hessian) --sigma-reg 0.025 --sigma-reg-in $SIGIN --save-wq "$@"
