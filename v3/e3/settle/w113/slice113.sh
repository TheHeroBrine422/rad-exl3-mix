#!/bin/bash
# slice113.sh -- golden-search27e (../w113.md #148-02 "Slice"): fresh two-part 27B search slice, gpubox, CPU only (ram.lock shared).
#   calib part: calib_slice 60000 tokens, cap 8192, seed 11148, search pool (pools-v3), excluding every document in
#               eval/pools/ledger.jsonl (working tree, with 27d-c; EXCL_C = a kld.json-style list made on the VM) and
#               golden-search27d-c -> golden_pick
#   xcorp part: pickx84 130 pieces of night1105 xc8k-b2-c1 (never scored), seed 11148, excluding the xcorp documents of
#               golden-search27b-x, 27c-x, 27d-x and hin27/mX
# Restartable: a finished part (kld.json) is skipped. w110's slice110.sh with the batch, seed and exclusions changed.
set -u
A=/data/w113/b; CODE=/data/w113/code; PY=/data/venv-exl3/bin/python; SEED=11148
GC=$A/golden-search27e-c GX=$A/golden-search27e-x EXCL_C=$A/excl/ledger-docs EXCL_X=$A/excl/x-prev.json
N=/data/night1105; B=xc8k-b2-c1
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
die() { log "SLICE113 STOP: $*"; exit 1; }
mkdir -p $A/excl
[ -f $EXCL_C/kld.json ] || die "missing $EXCL_C/kld.json (made on the VM from the ledger)"
if [ ! -f $EXCL_X ]; then
  $PY - $EXCL_X /data/w91/golden-search27b-x/kld.json /data/w92/golden-search27c-x/kld.json /data/w110/golden-search27d-x/kld.json <<'PY' || die "x exclusion list"
import json, sys
x = sorted({d["xdoc"] for f in sys.argv[2:] for d in json.load(open(f))["docs"]})
json.dump({"docs": [{"orig": i} for i in x]}, open(sys.argv[1], "w"))
print(len(x), "xcorp docs from earlier search slices")
PY
fi
if [ ! -f $GC/kld.json ]; then
  docs=$(EXCLUDE_GOLDEN=$EXCL_C,/data/w110/golden-search27d-c POOLS=$CODE/eval/pools/pools-v3.json \
         python3 -I $CODE/mix/calib_slice.py /data/challenge/corpus/calib.jsonl 60000 8192 $SEED search 2> $GC.slice.log) || die "calib_slice"
  echo "$docs" > $GC.docs; rm -rf $GC.part
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY -I $CODE/eval/golden_pick.py /data/challenge/golden/calib \
      /data/challenge/corpus/calib.jsonl $GC.part $GC.jsonl $docs ) > $GC.pick.log 2>&1 && mv $GC.part $GC || die "golden_pick (calib part)"
  log "slice calib part: $(cat $GC.slice.log)"
fi
if [ ! -f $GX/kld.json ]; then
  rm -rf $GX.part
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY -I $CODE/v3/e3/settle/w84/pickx84.py $N/gold/$B $N/corpora/chunks/$B.jsonl \
      $N/gold/$B/corpus.meta.jsonl 130 $SEED $GX.part $GX.jsonl --exclude-orig /data/hin27/mX/kld.json --exclude-orig $EXCL_X ) \
      2> $GX.pick.log && mv $GX.part $GX && mv $GX.part.docs $GX.docs || die "pickx84 (xcorp part)"
  log "slice xcorp part: $(head -1 $GX.pick.log)"
fi
for g in $GC $GX; do [ -f $g.summary ] || $PY -c "import json, numpy as np; m = json.load(open('$g/kld.json')); g = np.fromfile('$g/gen.u8', np.uint8)
print('$(basename $g)', len(m['docs']), 'docs', sum(d['tokens'] for d in m['docs']), 'tokens', m['rows'], 'rows', int(g.sum()), 'scored')" > $g.summary
  log "$(cat $g.summary)"; done
echo "GX=$GX" > $A/slice.env.part; echo "GC=$GC" >> $A/slice.env.part; mv $A/slice.env.part $A/slice.env
