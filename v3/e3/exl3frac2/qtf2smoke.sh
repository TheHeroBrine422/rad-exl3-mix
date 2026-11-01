#!/bin/bash
# qtf2smoke.sh -- end to end: ourq K2.5 (i8 codebook, damping sweep) on layers 10-11 (GDN + full attention) with the old
# fractional kernel vs --qtf2; every tensor must be byte-identical (the shards' stamps differ by design). Out /data/qtf2/smoke.
: ${RXM_WORK:=/root}; export RXM_WORK
O=/data/qtf2/smoke PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
X="-K 2.5 --layers 10-11 --device 0 --codebook i8 --frac-ext $SRC --sigma-in-sweep 0.025,0.005,0.001"
for arm in old qtf2; do
  F=""; [ $arm = qtf2 ] && F="--qtf2"
  ( exec 9>$RXM_WORK/card0.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7
    $PY -I ourq/ourq.py $X -o $O/$arm $F > $O/$arm.log 2>&1 ) || { echo "$arm FAIL" >> $O/result.txt; exit 1; }
done
$PY - $O <<'PY' >> $O/result.txt
import sys, torch
from safetensors.torch import load_file
o = sys.argv[1]; bad = n = 0
for l in (10, 11):
    a = load_file(f"{o}/old/q-L{l:02d}.safetensors"); b = load_file(f"{o}/qtf2/q-L{l:02d}.safetensors")
    assert a.keys() == b.keys()
    for k in a:
        n += 1
        if not torch.equal(a[k], b[k]): bad += 1; print("DIFF", k)
print(f"smoke: {n} tensors, {bad} differ -> {'PASS' if bad == 0 else 'FAIL'}")
PY
