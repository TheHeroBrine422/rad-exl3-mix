#!/bin/bash
# rowerr-test.sh -- 4070 GPU, ~2 min, AFTER rowerr.patch is applied to a copy (or the tree): quantize layer 5 of the
# 0.8B at K3 with q2b.sh's settings twice, with the attached ourq.py (OLD, default: git show HEAD:quantize/ourq.py) and the
# patched one; the q-L05 shard tensors and the wq weights must be identical (the patch only touches rowerr.npz), and
# rowerrcheck.py must say WRONG for the old run and OK for the new one.
F=$(cd "$(dirname "$0")" && pwd) R=/workspace/rad-exl3-mix T=/data/yb/w04-blockgain/rowerr-test
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $T; git -C $R show HEAD:quantize/ourq.py > $T/ourq-old.py
for v in old new; do
  [ -f $T/$v/q-L05.safetensors ] && continue
  Q=$([ $v = old ] && echo $T/ourq-old.py || echo $R/quantize/ourq.py)
  mkdir -p /tmp/q2b-run && cd /tmp/q2b-run && W=/data/p08 && \
  PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=/data/models/qwen35-08b RXM_TEMPLATE=$W/template.json \
    TORCH_EXTENSIONS_DIR=/data/torch_ext python3 $Q -K 3 -o $T/$v --device 0 --layers 5-5 --codebook i8 --qtf2 \
    --frac-ext /data/src/exllamav3/exllamav3/exllamav3_ext --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq > $T/$v.log 2>&1
done
python3 - <<PY
import glob, os, torch
from safetensors.torch import load_file
a, b = load_file("$T/old/q-L05.safetensors"), load_file("$T/new/q-L05.safetensors")
same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
for f in glob.glob("$T/old/wq/*.safetensors"):
    same &= torch.equal(load_file(f)["weight"], load_file(f.replace("/old/", "/new/"))["weight"])
print("shards and wq identical:", same)
PY
for v in old new; do echo "$v: $(CUDA_VISIBLE_DEVICES= python3 $F/rowerrcheck.py $T/$v/wq | head -1)"; done
