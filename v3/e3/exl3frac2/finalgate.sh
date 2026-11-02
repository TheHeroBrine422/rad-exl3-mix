#!/bin/bash
# finalgate.sh CARD -- the final qtf2 (v9 menu, 51 instances) before the trunk requant: build in /data/qtf2/v10, gate
# there, then deploy to /root/ourq (backup first) and run the end-to-end smoke tests on the production paths.
#  1. v5check (V6): every instance, both codebooks, vs the reference arithmetic build (byte history, full barriers,
#     strict-less scan) -- identical, plus tiles / s
#  2. qtf2check: old rates vs the fork's kernel -- identical
#  3. int6check K1-4, 6, 7: qtf2 MASK 0 vs the integer kernels -- identical
#  4. race: every instance x 4 runs -- deterministic
#  5. deploy; e2e K2.5 L3-4 vs the old code's shards; K3 qtf2 vs integer kernel; K6.5 i8 and mul1:K7.5 L3-4 run clean
# Log /data/qtf2/v10/run.log; "FINAL GATE PASS" only if all pass.
: ${RXM_WORK:=/root}
C=${1:-0} O=/data/qtf2/v10 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
fail() { log "FAIL: $*"; exit 1; }
rm -rf $O/src; mkdir -p $O/src; cp -r $RXM_WORK/ourq/exl3frac $RXM_WORK/ourq/exl3cb $O/src/; cp -r $O/exl3frac2 $O/src/
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$O/build MAX_JOBS=6 EXL3_QT_OPTIMIZED=1
( exec 7>$RXM_WORK/ram.lock; flock -s 7
  for vd in "mul1|" "mul1|QTF2_BYTEHIST QTF2_FULLSYNC QTF2_SCANMIN" "i8|" "i8|QTF2_BYTEHIST QTF2_FULLSYNC QTF2_SCANMIN"; do
    v=${vd%%|*} d=${vd#*|}
    QTF2_DEFS="$d" $PY -c "import sys; sys.path.insert(0, '$O/src/exl3frac2'); import exl3frac2; exl3frac2.load('$SRC', '$v')" >> $O/build.log 2>&1 \
      || exit 1
  done ) || fail "build (build.log)"
log "builds OK"
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
export ROCR_VISIBLE_DEVICES=$C
S=$O/src/exl3frac2
V6=1 $PY -I $S/qtf2v5check.py $SRC 1024 > $O/v5check.jsonl 2> $O/v5check.err
grep -q '"result": "PASS"' $O/v5check.jsonl || fail "v5check $(tail -1 $O/v5check.jsonl)"; log "1 all instances vs reference arithmetic: PASS"
$PY -I $S/qtf2check.py $SRC 1024 > $O/check.jsonl 2> $O/check.err
n=$(python3 -c "
import json; e = [json.loads(l) for l in open('$O/check.jsonl')]; e = [x for x in e if x.get('check') == 'exact']
print(sum(x['idx_diff'] + x['val_diff'] for x in e) if e else -1)"); [ "$n" = 0 ] || fail "vs fork: $n"; log "2 old rates vs fork: PASS"
INT6_KS=1,2,3,4,6,7 $PY -I $S/qtf2int6check.py $SRC 1024 > $O/int6.jsonl 2> $O/int6.err
grep -q '"fails": 0' $O/int6.jsonl || fail "int6 $(tail -1 $O/int6.jsonl)"; log "3 integer K1-4, 6, 7 vs integer kernels: PASS"
RR=$($PY -c "import sys; sys.path.insert(0, '$S'); import exl3frac2; e = exl3frac2.load('$SRC', 'i8'); print(','.join(f'{a}:{m}' for a, m in e.qtf2_instances()))" 2>/dev/null)
RACE_RATES=$RR $PY -I $S/qtf2race.py $SRC 4 $O/race > $O/race.jsonl 2> $O/race.err
bad=$(python3 -c "
import json; r = [json.loads(l) for l in open('$O/race.jsonl')]; print(len(r), sum(1 for x in r if any(x['within_build_idx_diff'])))")
[ "${bad#* }" = 0 ] && [ "${bad% *}" -ge 51 ] || fail "race: $bad"; log "4 race ($bad instances, nondeterministic): PASS"
python3 - $O/v5check.jsonl <<'PY' | tee -a $O/run.log
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if '"cb": "i8"' in l]
print("i8 tiles/s: " + ", ".join(f"{x['K']:g} {x['v5_tps'] // 1000}k" for x in sorted(r, key=lambda x: x['K'])))
PY
# 5. deploy + end to end
B=$O/backup-$(date -u +%Y%m%dT%H%M); mkdir -p $B; cp -a $RXM_WORK/ourq/exl3frac2 $RXM_WORK/ourq/ourq.py $B/
rm -f $RXM_WORK/ourq/exl3frac2/qtf2_inst_a*.cu; cp $S/*.cu $S/*.cuh $S/*.inc $S/*.py $RXM_WORK/ourq/exl3frac2/
cp /data/qtf2/multik/ourq.py.new $RXM_WORK/ourq/ourq.py
export TORCH_EXTENSIONS_DIR=/data/exl3frac/build
cd $RXM_WORK
X="--device 0 --qtf2 --frac-ext $SRC --sigma-in-sweep 0.025,0.005,0.001 --layers 3-4"
lt() { grep -a "^layer" $1 | sed 's/.*: \([0-9]*\)s .*/\1/' | awk '{s += $1} END {print s}'; }
E=$O/e2e; rm -rf $E; mkdir -p $E
$PY -I ourq/ourq.py -K 2.5 -o $E/K2.5 --codebook i8 $X > $E/K2.5.log 2>&1 || fail "e2e K2.5 run"
$PY -I /data/qtf2/v10/cmp.py /data/qtf2/v5/ref-K2.5 $E/K2.5 "K2.5 vs old code" | tee -a $O/run.log | grep -q PASS || fail "e2e K2.5"
log "  K2.5 $(lt $E/K2.5.log)s for 2 layers"
QTF2_INT=0 $PY -I ourq/ourq.py -K 3 -o $E/K3-int --codebook i8 $X > $E/K3-int.log 2>&1 || fail "e2e K3 int run"
$PY -I ourq/ourq.py -K 3 -o $E/K3 --codebook i8 $X > $E/K3.log 2>&1 || fail "e2e K3 run"
$PY -I /data/qtf2/v10/cmp.py $E/K3-int $E/K3 "K3 qtf2 vs integer kernel" | tee -a $O/run.log | grep -q PASS || fail "e2e K3"
for kc in "6.5 i8" "7.5 mul1" "0.5 i8"; do
  set -- $kc
  $PY -I ourq/ourq.py -K $1 -o $E/K$1-$2 --codebook $2 $X > $E/K$1-$2.log 2>&1 || fail "e2e K$1 $2 run"
  python3 -c "
import json, math; r = [json.loads(l) for l in open('$E/K$1-$2/proxy.jsonl')]
assert len(r) == 13 and all(math.isfinite(x['proxy_err']) and x['proxy_err'] > 0 for x in r)
print('  K$1 $2: 13 tensors, proxy median', sorted(x['proxy_err'] for x in r)[6])" | tee -a $O/run.log || fail "e2e K$1 $2 proxies"
  log "  K$1 $2 $(lt $E/K$1-$2.log)s for 2 layers"
done
log "5 deploy + end to end: PASS"
log "FINAL GATE PASS"
