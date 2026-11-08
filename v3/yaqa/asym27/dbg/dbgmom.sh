#!/bin/bash
# dbgmom.sh NAME TOK LAYERS [ENV=VAL ...] -- one sibmom27 debug run on card 0 (s1's level), OUT /data/asym27/dbg/NAME
# env as asym27.sh's card() after the w78 fix (no expandable_segments); extra VAR=VAL args override
# (e.g. PYTORCH_HIP_ALLOC_CONF=expandable_segments:True AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1)
N=$1 TOK=$2 LY=$3; shift 3
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
  TORCH_EXTENSIONS_DIR=/data/asym27/build \
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  POOLS=/root/challenge/pools-v3.json RXM_WORK=/root TRITON_CACHE_DIR=/data/asym27/triton
for kv in "$@"; do export "$kv"; done
O=/data/asym27/dbg/$N; rm -rf $O; mkdir -p /data/asym27/dbg
( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; cd /root
  echo "$(date -u +%T) start $N tok $TOK layers $LY alloc=[$PYTORCH_HIP_ALLOC_CONF] ser=[$AMD_SERIALIZE_KERNEL] $*"
  /data/swap27/tools/memguard.sh 3000 nice -n 5 /data/venv-exl3/bin/python -I /data/asym27/code/sibmom27.py \
    /data/workspace/models/qwen38-27b-bf16 /root/challenge/golden/calib $O --level /data/swap27/q-s1/K2.5 --codebook i8 \
    --tokens $TOK --wins moments --seed 0 --layers $LY --no-disk-check $SIBARGS
  rc=$?; echo "$(date -u +%T) end $N rc=$rc" ) > /data/asym27/dbg/$N.log 2>&1
rm -rf $O/work
