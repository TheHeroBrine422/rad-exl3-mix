#!/bin/bash
# headdbg.sh K -- the K4.5 / K5.5 head segfault (refit_scales, surfaced at quantize.py:1643 in a kernel launch):
# serialized kernels so the faulting kernel reports itself, VRAM sampled every 2 s. Log /data/yaqa/headslab2/dbg-K<K>.*
: ${RXM_WORK:=/root}; export RXM_WORK
H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python K=$1
exec 9>/root/card0.lock; flock 9; exec 7>/root/ram.lock; flock 7
cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
export AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=1
( while :; do echo "$(date -u +%T) $(rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'used' | head -1 | awk '{print $NF}')"; sleep 2; done ) > $H/dbg-K$K.vram &
vp=$!
rm -rf /tmp/hdbg-K$K
$PY -X faulthandler -I ourq/ourq.py -K $K -o /tmp/hdbg-K$K --only-head --device 0 --save-wq --head-perm $H/perm.npy \
  --frac-ext /data/exl3frac/src > $H/dbg-K$K.log 2>&1
echo "rc=$?" >> $H/dbg-K$K.log; kill $vp
