#!/bin/bash
# qtf2v8run.sh CARD -- v8 gate (full barriers again, per-rate packing) in its own source copy and extension dir (a running ourq keeps the deployed build):
# qtf2v5check (v5 vs byte history, every instance, both codebooks) then qtf2check (vs the fork's kernel, old rates).
# Out /data/qtf2/v5/{v5check,check}.jsonl, log run.log
: ${RXM_WORK:=/root}; export RXM_WORK
C=${1:-0} O=/data/qtf2/v8 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
rm -rf $O/src; mkdir -p $O/src; cp -r $RXM_WORK/ourq/exl3frac $RXM_WORK/ourq/exl3cb $O/src/; cp -r /data/qtf2/v8/exl3frac2 $O/src/
export ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$O/build MAX_JOBS=6
( exec 7>$RXM_WORK/ram.lock; flock -s 7
  for vd in "mul1|" "mul1|QTF2_BYTEHIST QTF2_FULLSYNC QTF2_SCANMIN" "i8|" "i8|QTF2_BYTEHIST QTF2_FULLSYNC QTF2_SCANMIN"; do
    v=${vd%%|*} d=${vd#*|}
    QTF2_DEFS="$d" $PY -c "import sys; sys.path.insert(0, '$O/src/exl3frac2'); import exl3frac2; exl3frac2.load('$SRC', '$v')" >> $O/build.log 2>&1 \
      || { log "build $v $d FAIL"; exit 1; }
  done; log "builds OK" ) || exit 1
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
export ROCR_VISIBLE_DEVICES=$C
V6=1 $PY -I $O/src/exl3frac2/qtf2v5check.py $SRC 1024 > $O/v5check.jsonl 2> $O/v5check.err; log "v5check: $(tail -1 $O/v5check.jsonl)"
$PY -I $O/src/exl3frac2/qtf2check.py $SRC 1024 > $O/check.jsonl 2> $O/check.err
log "check vs fork: $(python3 -c "
import json; r = [json.loads(l) for l in open('$O/check.jsonl')]; e = [x for x in r if x.get('check') == 'exact']
print(len(e), 'old-rate sets,', sum(x['idx_diff'] + x['val_diff'] for x in e), 'differences')")"
EXL3_QT_OPTIMIZED=1 $PY -I $O/src/exl3frac2/qtf2int6check.py $SRC 1024 > $O/int6.jsonl 2> $O/int6.err; log "int6: $(tail -1 $O/int6.jsonl)"
# race / determinism at every instance (4 runs each, the i8 build): any run-to-run or vs-first-run difference = FAIL
RR=$($PY -c "import sys; sys.path.insert(0, '$O/src/exl3frac2'); import exl3frac2; e = exl3frac2.load('$SRC', 'i8'); print(','.join(f'{a}:{m}' for a, m in e.qtf2_instances()))" 2>/dev/null)
cp $O/src/exl3frac2/qtf2race.py /tmp/ 2>/dev/null
RACE_RATES=$RR $PY -I $O/src/exl3frac2/qtf2race.py $SRC 4 $O/race > $O/race.jsonl 2> $O/race.err
log "race: $(python3 -c "
import json; r = [json.loads(l) for l in open('$O/race.jsonl')]
bad = [x for x in r if any(x['within_build_idx_diff'])]
print(len(r), 'instances x 4 runs,', len(bad), 'nondeterministic', [(x['ka'], x['mask']) for x in bad][:6])")"
