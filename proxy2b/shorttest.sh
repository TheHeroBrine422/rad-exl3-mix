#!/bin/bash
# shorttest.sh -- collect.py YAQA_SHORT on the 0.8B (one GPU): (1) without it, bit-identical to diag08test's
# straight 40-seq run; (2) with YAQA_SHORT=256: 40 straight = 20 + resumed 20, and the -long / -dy-long files exist.
export VRAM_CAP_GB=${VRAM_CAP_GB:-6} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/data/c08work; export RXM_WORK=$W RXM_CKPT=/data/models/qwen38-27b-bf16 RXM_CKPT=/data/models/qwen35-08b
export YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json YAQA_FP32=1 YAQA_DIAG=1 YAQA_SEED=0
C=/workspace/rad-exl3-mix/v3/yaqa/collect.py O=/data/c08short; rm -rf $O; mkdir -p $O; cd /tmp
YAQA_SNAP=20,40 YAQA_SAVE_EVERY=1000 python3 $C $O/noshort 40 2048 > $O/noshort.log 2>&1
YAQA_SHORT=256 YAQA_DIAG_EXTRA=1 YAQA_SNAP=20,40 YAQA_SAVE_EVERY=1000 python3 $C $O/straight 40 2048 > $O/straight.log 2>&1
YAQA_SHORT=256 YAQA_DIAG_EXTRA=1 YAQA_SNAP=20 YAQA_SAVE_EVERY=20 python3 $C $O/resumed 20 2048 > $O/r1.log 2>&1
YAQA_SHORT=256 YAQA_DIAG_EXTRA=1 YAQA_SNAP=40 YAQA_SAVE_EVERY=1000 python3 $C $O/resumed 40 2048 > $O/r2.log 2>&1
grep -h "YAQA_SHORT\|resuming\|Error" $O/*.log | sort -u | head
python3 - <<'PY'
import glob, os
from safetensors import safe_open
def mx(a, b):
    w = 0; n = 0
    for f in glob.glob(f"{a}/*.safetensors"):
        with safe_open(f, "pt") as x, safe_open(f.replace(a, b), "pt") as y: w = max(w, float((x.get_tensor("hout_diag") - y.get_tensor("hout_diag")).abs().max())); n += 1
    return n, w
O = "/data/c08short"
print("no-SHORT vs the earlier reference (40):", mx(f"{O}/noshort/hdiag-snap40", "/data/c08test/straight/hdiag-snap40"))
for s in ("hdiag-snap40", "hdiag-snap40-dy", "hdiag-snap40-long", "hdiag-snap40-dy-long"):
    print(f"SHORT straight vs resumed, {s}:", mx(f"{O}/straight/{s}", f"{O}/resumed/{s}"))
for l in open(f"{O}/straight.log"):
    if l.startswith("seq "): pass
import json; print("long seqs in 40:", json.load(open(f"{O}/straight/hdiag-snap40/halves.json"))["seqs"], "(n_long in state)")
PY
echo SHORTTEST DONE
