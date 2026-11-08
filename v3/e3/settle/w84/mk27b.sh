#!/bin/bash
# mk27b.sh -- golden-search27b (../w84.md amendment 1), gpubox CPU (no card): calib part cut from the published calib
# golden (calib_slice.py + golden_pick.py), xcorp part from night1105's xc8k-b1 golden (pickx84.py). Idempotent.
A=/data/w84 W=$A/code/v3/e3/settle/w84 T=/data/swap27/tools PY=/data/venv-exl3/bin/python
cd $A
if [ ! -f golden-search27b-c/kld.json ]; then
  docs=$(EXCLUDE_GOLDEN=/data/swap27/golden-search27 POOLS=/root/challenge/pools-v3.json python3 -I $T/calib_slice.py \
         /data/challenge/corpus/calib.jsonl 30000 8192 20261108 search 2> golden-search27b-c.slice.log)
  echo "$docs" > golden-search27b-c.docs
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY -I $T/golden_pick.py /data/challenge/golden/calib /data/challenge/corpus/calib.jsonl \
      golden-search27b-c.part golden-search27b-c.jsonl $docs ) && mv golden-search27b-c.part golden-search27b-c || { echo "MK27B FAIL calib"; exit 1; }
fi
if [ ! -f golden-search27b-x/kld.json ]; then
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY -I $W/pickx84.py /data/night1105/gold/xc8k-b1 /data/night1105/corpora/xc8k-b1.jsonl \
      $A/xc8k-b1.meta.jsonl 65 20261108 golden-search27b-x.part golden-search27b-x.jsonl --exclude-orig /data/hin27/mX/kld.json 2> golden-search27b-x.pick.log ) \
    && mv golden-search27b-x.part golden-search27b-x && mv golden-search27b-x.part.docs golden-search27b-x.docs || { echo "MK27B FAIL xcorp"; exit 1; }
fi
for g in golden-search27b-c golden-search27b-x; do
  $PY -c "import json, numpy as np; m = json.load(open('$g/kld.json')); g = np.fromfile('$g/gen.u8', np.uint8)
print('$g', len(m['docs']), 'docs', sum(d['tokens'] for d in m['docs']), 'tokens', m['rows'], 'rows', int(g.sum()), 'scored')"; done
echo "MK27B DONE"
