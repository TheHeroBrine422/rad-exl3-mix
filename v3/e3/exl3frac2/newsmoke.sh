#!/bin/bash
# newsmoke.sh -- the new rates end to end through ourq (qtf2, i8 codebook, damping sweep): K0.5 (sub-1) and K1.125
# (1/8 step) on layers 10-11. Pass: both finish, every tensor written, proxy error between its neighbours'
# (old ladder K1 / K2 proxy.jsonl for K1.125; for K0.5 only: < 1, finite). Out /data/qtf2/newsmoke/result.txt
: ${RXM_WORK:=/root}; export RXM_WORK
O=/data/qtf2/newsmoke PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; cd $RXM_WORK
C=${CARD:-0}; export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
for K in 0.5 1.125; do
  ( exec 9>$RXM_WORK/card$C.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7
    $PY -I ourq/ourq.py -K $K --layers 10-11 --device 0 --codebook i8 --qtf2 --frac-ext $SRC \
      --sigma-in-sweep 0.025,0.005,0.001 -o $O/K$K > $O/K$K.log 2>&1 ) && echo "K$K ran" >> $O/result.txt || echo "K$K FAIL" >> $O/result.txt
done
python3 - $O <<'PY' >> $O/result.txt
import json, sys, math
o = sys.argv[1]
def px(d):
    r = {}
    for l in open(f"{d}/proxy.jsonl"):
        x = json.loads(l)
        if ".layers.10." in x["key"] or ".layers.11." in x["key"]: r[x["key"]] = x["proxy_err"]
    return r
a, b = px(f"{o}/K0.5"), px(f"{o}/K1.125")
lo, hi = px("/data/v3levels/K2"), {}
print(f"K0.5: {len(a)} tensors, proxy median {sorted(a.values())[len(a)//2]:.4f}, max {max(a.values()):.4f}, finite {all(math.isfinite(v) for v in a.values())}")
print(f"K1.125: {len(b)} tensors, proxy median {sorted(b.values())[len(b)//2]:.4f}; vs K2 ladder median ratio "
      f"{sorted(b[k] / lo[k] for k in b if k in lo)[len(b)//2]:.2f} (expect ~3.5-4: ~1.9 bits of gap at ~2x per bit)")
print(f"K0.5 / K1.125 median ratio {sorted(a[k] / b[k] for k in a if k in b)[len(a)//2]:.2f} (expect ~1.6-2)")
PY
