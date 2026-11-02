#!/bin/bash
# e2e5.sh -- end-to-end identity of the speed pass (v5 kernel + quantize.py dead-compute cut + ourq Hessian cache):
# K2.5 layers 3-4, ladder settings, vs the old code's shards (ref-K2.5, from qprof3 on the v4 code). Card 0, after the
# v5 gate passes. Out /data/qtf2/v5/e2e/, result line in /data/qtf2/v5/run.log
: ${RXM_WORK:=/root}
O=/data/qtf2/v5 PY=/data/venv-exl3/bin/python
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
while ! grep -q "check vs fork" $O/run.log 2>/dev/null; do sleep 10; done
grep -q '"result": "PASS"' $O/v5check.jsonl && grep -q "check vs fork: .* 0 differences" $O/run.log || { log "e2e: gate did not pass, not run"; exit 1; }
rm -rf $O/ourq $O/e2e; cp -r $RXM_WORK/ourq $O/ourq; rm -rf $O/ourq/exl3frac2; cp -r $O/src/exl3frac2 $O/ourq/exl3frac2
cp $O/ourq.py.new $O/ourq/ourq.py
cp /data/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py $O/quantize.py.orig
cp $O/quantize.py.new /data/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py
exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
cd $O; export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$O/build
s=$(date +%s)
$PY -I ourq/ourq.py -K 2.5 -o $O/e2e --layers 3-4 --device 0 --codebook i8 --qtf2 --frac-ext /data/exl3frac/src \
  --sigma-in-sweep 0.025,0.005,0.001 > $O/e2e.log 2>&1
log "e2e run rc=$? $(( $(date +%s) - s ))s (old code: plain 136.5s, incl. startup)"
$PY -I - $O/ref-K2.5 $O/e2e <<'PY' | tee -a $O/run.log
import json, sys
from safetensors import safe_open
a, b = sys.argv[1:3]; diff = n = 0
for f in ("q-L03.safetensors", "q-L04.safetensors"):
    with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
        assert set(x.keys()) == set(y.keys()), f
        for k in x.keys():
            n += 1; ta, tb = x.get_tensor(k), y.get_tensor(k)
            if ta.shape != tb.shape or not bool((ta.view(-1).view(__import__("torch").uint8) == tb.view(-1).view(__import__("torch").uint8)).all()): diff += 1; print("DIFF", k)
pa = [json.loads(l) for l in open(f"{a}/proxy.jsonl")]; pb = [json.loads(l) for l in open(f"{b}/proxy.jsonl")]
pd = sum(1 for p, q in zip(pa, pb) if (p["key"], p["proxy_err"], p["sigma_reg"]) != (q["key"], q["proxy_err"], q["sigma_reg"]))
print(f"e2e identity: {n} tensors, {diff} differ; proxy rows {len(pa)} vs {len(pb)}, {pd} differ -> {'PASS' if diff == 0 and pd == 0 and len(pa) == len(pb) else 'FAIL'}")
PY
