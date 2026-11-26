#!/bin/bash
# chain233b.sh (box2; setsid nohup) -- after chain233a: the dp100 recipe as ONE rank on cards 2,3, unsynced, 10 steps; py-spy samples the process during steps 4-7
# (all threads with idle frames, then GIL holders only) -> runs/pipefree/pyspy-{all,gil}.txt (collapsed stacks)
W=/data/q27/w233; C=$W/code/v3/qad233
until [ "$(grep -c "pipesync rc" $W/run233.log 2>/dev/null)" -ge 2 ]; do sleep 30; done      # the first pipesync line (9:12 PM) was a failed start
OUT=$W/runs/pipefree; mkdir -p $OUT
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256 $C/run233.sh 2,3 pipefree qad220.py --budgets 6e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 \
  --dynamo-limit 64 --cap 262144 --cap-keep /data/q27/queue/slices/jlong-ids.json --plan-ext /data/q27/queue/ext/ext-plan-226c.json --ext-root /data/q27/queue/ext \
  --lr-steps 3e-4:1.5e6,1e-4:5.6e6 --split-layers 32 --pipeline --state-card-gb 4 --attn-gqa --pipe-xfer host --snap-at none --max-steps 10 --state-budget-gb 8 --card-mem-gb 28.5 &
until [ "$(grep -c '"what": "main"' $OUT/train.jsonl 2>/dev/null)" -ge 3 ]; do sleep 10; kill -0 $! 2>/dev/null || exit 0; done
P=$(pgrep -f "^/data/q27/venv/bin/python /data/q27/w233/code/v3/qad220/qad220.py $OUT " | head -1)
py-spy record -p $P -r 40 -d 60 --idle --threads -f raw -o $OUT/pyspy-all.txt --nonblocking > $OUT/pyspy-all.log 2>&1
py-spy record -p $P -r 40 -d 60 --gil --threads -f raw -o $OUT/pyspy-gil.txt --nonblocking > $OUT/pyspy-gil.log 2>&1
wait
