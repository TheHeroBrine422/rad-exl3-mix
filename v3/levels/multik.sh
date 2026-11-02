#!/bin/bash
# multik.sh CARD -- identity of ourq's multi-level pass: -K 2.25,2.5,3 (fractional + integer) vs one run per level,
# same code (v8 builds), ladder settings, layers 3-4; plus K2.5 vs the old code's shards. Out /data/qtf2/multik/
: ${RXM_WORK:=/root}
C=${1:-1} O=/data/qtf2/multik PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
rm -rf $O/ourq $O/s-* $O/m-*; mkdir -p $O/ourq; cp -r /data/qtf2/v8/src/exl3frac /data/qtf2/v8/src/exl3cb /data/qtf2/v8/src/exl3frac2 $O/ourq/
cp $O/ourq.py.new $O/ourq/ourq.py
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
cd $O; export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/qtf2/v8/build
X="--device 0 --codebook i8 --qtf2 --frac-ext $SRC --sigma-in-sweep 0.025,0.005,0.001 --layers 3-4"
lt() { grep -a "^layer" $1 | awk '{s += $NF == "MiB" ? $(NF-4) : 0} END {print s}'; }
ls_() { grep -a "^layer" $1 | sed 's/.*: \([0-9]*\)s .*/\1/' | awk '{s += $1} END {print s}'; }
tot=0
for K in 2.25 2.5 3; do $PY -I ourq/ourq.py -K $K -o $O/s-$K $X > $O/s-$K.log 2>&1; t=$(ls_ $O/s-$K.log); tot=$((tot + t)); log "single K$K: ${t}s"; done
$PY -I ourq/ourq.py -K 2.25,2.5,3 -o "$O/m-{K}" $X > $O/m.log 2>&1; log "multi 2.25,2.5,3 rc=$?: $(ls_ $O/m.log)s vs singles ${tot}s"
for K in 2.25 2.5 3; do
  $PY -I - $O/s-$K $O/m-$K "multi vs single K$K" <<'PY' | tee -a $O/run.log
import json, sys, torch
from safetensors import safe_open
a, b, name = sys.argv[1:4]; diff = n = 0
for f in ("q-L03.safetensors", "q-L04.safetensors"):
    with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
        assert set(x.keys()) == set(y.keys()), f
        for k in x.keys():
            n += 1; ta, tb = x.get_tensor(k), y.get_tensor(k)
            if ta.shape != tb.shape or not bool((ta.contiguous().view(-1).view(torch.uint8) == tb.contiguous().view(-1).view(torch.uint8)).all()): diff += 1
    sa = json.loads(open(f"{a}/{f}", "rb").read(8 + int.from_bytes(open(f"{a}/{f}", "rb").read(8), "little"))[8:])["__metadata__"]["ourq_stamp"]
    sb = json.loads(open(f"{b}/{f}", "rb").read(8 + int.from_bytes(open(f"{b}/{f}", "rb").read(8), "little"))[8:])["__metadata__"]["ourq_stamp"]
pa = [json.loads(l) for l in open(f"{a}/proxy.jsonl")]; pb = [json.loads(l) for l in open(f"{b}/proxy.jsonl")]
pd = sum(1 for p, q in zip(pa, pb) if (p["key"], p["proxy_err"], p["sigma_reg"]) != (q["key"], q["proxy_err"], q["sigma_reg"]))
print(f"{name}: {n} tensors, {diff} differ; proxy rows {len(pa)}/{len(pb)}, {pd} differ; stamps {'equal' if sa == sb else 'DIFFER'} -> {'PASS' if diff == 0 and pd == 0 and len(pa) == len(pb) and sa == sb else 'FAIL'}")
PY
done
