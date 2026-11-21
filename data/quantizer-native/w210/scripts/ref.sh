#!/bin/bash
# w210 box2: the release pipeline (rad-exl3-mix pipeline/rxq over the exllamav3 fork) on layer-3 tensors, as the reference
# for the extras. ref.sh CARD TAG K KEYS(comma, short) [--set ...]
CARD=$1; TAG=$2; K=$3; KEYS=$4; shift 4
R=/data/q27/w210/ref; cd $R
export ROCM_HOME=/data/q27/venv/lib/python3.14/site-packages/_rocm_sdk_devel; export ROCM_PATH=$ROCM_HOME
export PATH=/data/q27/venv/bin:$ROCM_PATH/bin:$PATH PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$R/extbuild EXL3_QT_OPTIMIZED=1 OMP_NUM_THREADS=4 MAX_JOBS=8
export PYTHONPATH=$R/code/pipeline:$R/exl3 RXQ_RXM=$R/code RXQ_EXL3_SRC=$R/exl3src/exllamav3/exllamav3_ext ROCR_VISIBLE_DEVICES=$CARD PYTHONDONTWRITEBYTECODE=1
P=model.language_model.layers.3
KK=$(echo $KEYS | sed "s/[^,]*/$P.&/g")
[ -n "$WARM" ] && KK=model.language_model.layers.7.self_attn.k_proj,$KK   # a first tensor that takes the process's start-up (kernel loads)
mkdir -p /data/q27/w210/out/ref
[ -s /data/q27/w210/out/ref/$TAG.jsonl ] && exit 0
rm -rf $R/o/$TAG
flock /data/q27/locks/card$CARD.lock python -m rxq quant --keys $KK -K $K --codebook mad --ckpt /data/models/qwen38-27b-bf16 \
  --hess /data/q27/challenge/hessians/calib --hout /data/q27/w210/hout/bn1920 -o $R/o/$TAG "$@" > /data/q27/w210/out/ref/$TAG.log 2>&1
cat $R/o/$TAG/log.jsonl > /data/q27/w210/out/ref/$TAG.jsonl 2>/dev/null; cp $R/o/$TAG/tilecd.jsonl /data/q27/w210/out/ref/$TAG.tilecd.jsonl 2>/dev/null
rm -f $R/o/$TAG/*.safetensors
