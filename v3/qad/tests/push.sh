#!/bin/bash
# push v3/qad + pipeline/rxq/{__init__,deq}.py to a host dir: push.sh SSH DEST
S=$1; DEST=$2
cd /workspace/rad-exl3-mix
tar czf - --exclude=__pycache__ --exclude=v3/qad/tests/push.sh v3/qad pipeline/rxq/deq.py pipeline/rxq/swj.py pipeline/rxq/mix.py pipeline/rxq/qwen35.py eval/pools/pools-v3.json eval/golden_pick.py mix/paired.py | $S "mkdir -p $DEST && tar xzf - -C $DEST"
