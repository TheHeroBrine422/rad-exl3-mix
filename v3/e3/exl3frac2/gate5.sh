#!/bin/bash
# gate5.sh CARD -- final ourq (scaled Hessian flag, drift interpolation, logging) deployed and checked on the final qtf2 build:
#  a) old settings (sweep, no -sh) K2.5 L3-4 byte-identical to the old code's shards (the edits change nothing by default)
#  b) final ladder settings (-sh, fixed damping 0.025 / input 0.005) K2.5, K1.25, i8 K6.75, mul1 K7.75, sub-1 K0.5 (input
#     0.05): run clean, finite proxies; K2.5 proxies vs the old-settings run (expect lower: -sh)
# Log /data/qtf2/gate5/run.log, "GATE5 PASS" only if all pass.
: ${RXM_WORK:=/root}
C=${1:-0} O=/data/qtf2/gate5 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; rm -rf $O/K*; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
fail() { log "FAIL: $*"; exit 1; }
cp $RXM_WORK/ourq/ourq.py $O/ourq.py.bak; cp $O/ourq.py.new $RXM_WORK/ourq/ourq.py; cp $O/plaus.py $RXM_WORK/ourq/plaus.py
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
cd $RXM_WORK; export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
L="--device 0 --qtf2 --frac-ext $SRC --layers 3-4"
lt() { grep -a "^layer" $1 | sed 's/.*: \([0-9]*\)s .*/\1/' | awk '{s += $1} END {print s}'; }
$PY -I ourq/ourq.py -K 2.5 -o $O/K2.5-old --codebook i8 $L --sigma-in-sweep 0.025,0.005,0.001 > $O/K2.5-old.log 2>&1 || fail "old-settings run"
$PY -I /data/qtf2/v10/cmp.py /data/qtf2/v5/ref-K2.5 $O/K2.5-old "old settings vs old code" | tee -a $O/run.log | grep -q PASS || fail "identity"
F="--scaled-hessian --sigma-reg 0.025"
for kc in "2.5 i8 0.005" "1.25 i8 0.005" "6.75 i8 0.005" "7.75 mul1 0.005" "0.5 i8 0.05"; do
  set -- $kc
  $PY -I ourq/ourq.py -K $1 -o $O/K$1-$2 --codebook $2 $L $F --sigma-reg-in $3 > $O/K$1-$2.log 2>&1 || fail "K$1 $2 run"
  python3 -c "
import json, math; r = [json.loads(l) for l in open('$O/K$1-$2/proxy.jsonl')]
assert len(r) == 13 and all(math.isfinite(x['proxy_err']) and x['proxy_err'] > 0 and x['scaled_hessian'] for x in r), 'proxies'
print('  K$1 $2: 13 tensors ok, drift', r[0]['drift'])" | tee -a $O/run.log || fail "K$1 $2 proxies"
  log "  K$1 $2 $(lt $O/K$1-$2.log)s for 2 layers"
done
python3 - $O/K2.5-old $O/K2.5-i8 <<'PY' | tee -a $O/run.log
import json, math, sys
a = {json.loads(l)["key"]: json.loads(l)["proxy_err"] for l in open(sys.argv[1] + "/proxy.jsonl")}
b = {json.loads(l)["key"]: json.loads(l)["proxy_err"] for l in open(sys.argv[2] + "/proxy.jsonl")}
r = {k.split(".")[-1]: b[k] / a[k] for k in a}
print("  K2.5 final / old settings (in-sample proxy):", ", ".join(f"{k} {v:.3f}" for k, v in r.items()),
      f"; geo {math.exp(sum(map(math.log, r.values())) / len(r)):.3f}")
PY
log "GATE5 PASS"
