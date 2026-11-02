#!/bin/bash
# speed2deploy.sh (v8) -- deploy speed pass 2 to the production paths (backups first) and gate it there (card 0):
#   qtf2 v6 + integer K1-4 on qtf2 (/root/ourq/exl3frac2), integer-kernel LDS barrier (/data/exl3frac/src), ourq Hessian
#   cache (/root/ourq/ourq.py), exl3cb test-build switch; quantize.py's dead-compute cut is already live (e2e5 PASS).
# Gate on the production builds: qtf2check (vs the fork's kernel), int6check, e2e K2.5 L3-4 vs the old code's shards,
# e2e K3 L3-4 QTF2_INT=1 vs QTF2_INT=0. Log /data/qtf2/deploy/run.log
: ${RXM_WORK:=/root}
D=/data/qtf2/deploy8 B=/data/qtf2/deploy8/backup-$(date -u +%Y%m%dT%H%M) PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $D $B; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $D/run.log; }
exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
cp -a $RXM_WORK/ourq/exl3frac2 $B/; cp -a $RXM_WORK/ourq/ourq.py $RXM_WORK/ourq/exl3cb $B/
cp -a $SRC/quant/quantize_tiles_optimized.cuh $B/
cp /data/qtf2/v8/exl3frac2/{qtf2.cuh,qtf2_bind.cu,qtf2_types.cuh,qtf2_table.inc,mkqtf2.py,qtf2check.py,qtf2v5check.py,qtf2int6check.py,qtf2race.py,exl3frac2.py} $RXM_WORK/ourq/exl3frac2/
cp /data/qtf2/v8/exl3frac2/qtf2_inst_a*.cu $RXM_WORK/ourq/exl3frac2/
grep -q qt_lds_barrier $SRC/quant/quantize_tiles_optimized.cuh && { log "integer kernel source still patched: abort"; exit 1; }
log "deployed (backup $B)"
cd $RXM_WORK; export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build MAX_JOBS=6
$PY -I ourq/exl3frac2/qtf2check.py $SRC 1024 > $D/check.jsonl 2> $D/check.err
log "check vs fork: $(python3 -c "
import json; r = [json.loads(l) for l in open('$D/check.jsonl')]; e = [x for x in r if x.get('check') == 'exact']
print(len(e), 'old-rate sets,', sum(x['idx_diff'] + x['val_diff'] for x in e), 'differences')")"
$PY -I ourq/exl3frac2/qtf2int6check.py $SRC 1024 > $D/int6.jsonl 2> $D/int6.err; log "int6: $(tail -1 $D/int6.jsonl)"
RR=$($PY -c "import sys; sys.path.insert(0, 'ourq/exl3frac2'); import exl3frac2; e = exl3frac2.load('$SRC', 'i8'); print(','.join(f'{a}:{m}' for a, m in e.qtf2_instances()))" 2>/dev/null)
RACE_RATES=$RR $PY -I ourq/exl3frac2/qtf2race.py $SRC 4 $D/race > $D/race.jsonl 2> $D/race.err
log "race: $(python3 -c "
import json; r = [json.loads(l) for l in open('$D/race.jsonl')]
print(len(r), 'instances x 4 runs,', sum(1 for x in r if any(x['within_build_idx_diff'])), 'nondeterministic')")"
X="--device 0 --codebook i8 --qtf2 --frac-ext $SRC --sigma-in-sweep 0.025,0.005,0.001 --layers 3-4"
cmp() { $PY -I - "$@" <<'PY'
import json, sys, torch
from safetensors import safe_open
a, b, name = sys.argv[1:4]; diff = n = 0
for f in ("q-L03.safetensors", "q-L04.safetensors"):
    with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
        assert set(x.keys()) == set(y.keys()), f
        for k in x.keys():
            n += 1; ta, tb = x.get_tensor(k), y.get_tensor(k)
            if ta.shape != tb.shape or not bool((ta.contiguous().view(-1).view(torch.uint8) == tb.contiguous().view(-1).view(torch.uint8)).all()): diff += 1
pa = [json.loads(l) for l in open(f"{a}/proxy.jsonl")]; pb = [json.loads(l) for l in open(f"{b}/proxy.jsonl")]
pd = sum(1 for p, q in zip(pa, pb) if (p["key"], p["proxy_err"], p["sigma_reg"]) != (q["key"], q["proxy_err"], q["sigma_reg"]))
print(f"{name}: {n} tensors, {diff} differ; proxy rows {len(pa)}/{len(pb)}, {pd} differ -> {'PASS' if diff == 0 and pd == 0 and len(pa) == len(pb) else 'FAIL'}")
PY
}
lt() { grep -a "^layer" $1 | awk '{s += $3} END {print s "s for 2 layers"}'; }
rm -rf $D/e2e-K2.5 $D/e2e-K3-int $D/e2e-K3-qtf2
$PY -I ourq/ourq.py -K 2.5 -o $D/e2e-K2.5 $X > $D/e2e-K2.5.log 2>&1; log "K2.5 $(lt $D/e2e-K2.5.log) (old code 126s)"
cmp /data/qtf2/v5/ref-K2.5 $D/e2e-K2.5 "e2e K2.5 vs old code" | tee -a $D/run.log
QTF2_INT=0 $PY -I ourq/ourq.py -K 3 -o $D/e2e-K3-int $X > $D/e2e-K3-int.log 2>&1; log "K3 integer kernel $(lt $D/e2e-K3-int.log)"
QTF2_INT=1 $PY -I ourq/ourq.py -K 3 -o $D/e2e-K3-qtf2 $X > $D/e2e-K3-qtf2.log 2>&1; log "K3 on qtf2 $(lt $D/e2e-K3-qtf2.log)"
cmp $D/e2e-K3-int $D/e2e-K3-qtf2 "e2e K3 qtf2 vs integer kernel" | tee -a $D/run.log
log "deploy gate done"
rm -f /data/qtf2/dual/run.log; DUAL_NOLOCK=1 /data/qtf2/dual.sh 0 2.5
