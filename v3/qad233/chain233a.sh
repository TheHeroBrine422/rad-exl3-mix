#!/bin/bash
# chain233a.sh (box2; setsid nohup) -- w233's first GPU block: (1) prof233 on card 2 (one-card segment profile, cap 8192), (2) the dp100 recipe as ONE rank on
# cards 2,3 for 12 steps with W225_SYNC=1 (GPU time per pipeline phase). Each waits for its locks (run233.sh).
C=/data/q27/w233/code/v3/qad233
$C/run233.sh 2 prof1 prof233.py --budgets 6e6 --cap 8192 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256 W225_SYNC=1 $C/run233.sh 2,3 pipesync qad220.py --budgets 6e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --cap-keep /data/q27/queue/slices/jlong-ids.json \
  --plan-ext /data/q27/queue/ext/ext-plan-226c.json --ext-root /data/q27/queue/ext --lr-steps 3e-4:1.5e6,1e-4:5.6e6 --split-layers 32 --pipeline --state-card-gb 4 --attn-gqa \
  --pipe-xfer host --snap-at none --max-steps 12 --state-budget-gb 8 --card-mem-gb 28.5
