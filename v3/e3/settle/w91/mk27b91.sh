#!/bin/bash
# mk27b91.sh -- w84/mk27b.sh with the output dir as $A (w91: /data/w84 was deleted; the xcorp part re-cut at w84's
# recorded pick, cutx91.py); golden-search27b (../w84.md amendment 1), gpubox CPU (no card): calib part cut from the published calib
# golden (calib_slice.py + golden_pick.py), xcorp part from night1105's xc8k-b1 golden. Idempotent.
A=${A:-/data/w91} W=${W84:-$A/code/v3/e3/settle/w84} T=/data/swap27/tools PY=/data/venv-exl3/bin/python
cd $A
if [ ! -f golden-search27b-c/kld.json ]; then
  docs=$(EXCLUDE_GOLDEN=/data/swap27/golden-search27 POOLS=/root/challenge/pools-v3.json python3 -I $T/calib_slice.py \
         /data/challenge/corpus/calib.jsonl 30000 8192 20261108 search 2> golden-search27b-c.slice.log)
  echo "$docs" > golden-search27b-c.docs
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY -I $T/golden_pick.py /data/challenge/golden/calib /data/challenge/corpus/calib.jsonl \
      golden-search27b-c.part golden-search27b-c.jsonl $docs ) && mv golden-search27b-c.part golden-search27b-c || { echo "MK27B FAIL calib"; exit 1; }
fi
if [ ! -f golden-search27b-x/kld.json ]; then   # w91: pickx84's meta file went with /data/w84: re-cut at the recorded pick
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY -I $A/code/v3/e3/settle/w91/cutx91.py /data/night1105/gold/xc8k-b1 \
      $A/code/v3/e3/settle/data/w84/golden-search27b-x-kld.json golden-search27b-x.part 2> golden-search27b-x.pick.log ) \
    && mv golden-search27b-x.part golden-search27b-x && mv golden-search27b-x.part.docs golden-search27b-x.docs || { echo "MK27B FAIL xcorp"; exit 1; }
fi
for g in golden-search27b-c golden-search27b-x; do
  $PY -c "import json, numpy as np; m = json.load(open('$g/kld.json')); g = np.fromfile('$g/gen.u8', np.uint8)
print('$g', len(m['docs']), 'docs', sum(d['tokens'] for d in m['docs']), 'tokens', m['rows'], 'rows', int(g.sum()), 'scored')"; done
echo "MK27B DONE"
