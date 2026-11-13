#!/bin/bash
# a8pull144.sh -- (w144, #174-02, VM) per-document sums of every finished A8 anchor (ds8: w135's a8docs.py over
# /data/q27/w144/a8/scores) pulled with w135's docs (the incumbents' w132 references and the Lloyd swaps) into
# rad-local/workers/w144/a8docs/, then calobs.py three times: vs the incumbent (fit rows a8-17402.obs.jsonl, labels
# emb17402a8-*), vs the paired Lloyd row (a8-17402-vs-lloyd.json: trellis tK against Lloyd at similar bytes) and vs
# the bf16 table (a8-17402-vs-bf16.json, amendment 2).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); W=/workspace/rad-local/workers/w144; R=/workspace/rad-local/workers/w131/run
SSH=/workspace/rad-local/bin/ssh-ds8.sh
$SSH 'cd /data/q27/w144/a8 && mkdir -p docs && args=""; for f in scores/*-a8-s171b.json.rows; do n=$(basename $f -s171b.json.rows)
  [ -f docs/$n-s171b.json ] || args="$args $n=$f"; done
  [ -z "$args" ] || /data/q27/venv/bin/python -I /data/q27/emb/code/a8docs.py /data/q27/w131/slices/golden-s171b docs $args' < /dev/null
mkdir -p $W/a8docs
$SSH 'cd /data/q27/w144/a8/docs && tar c .' < /dev/null | tar x -C $W/a8docs
$SSH 'cd /data/q27/emb/a8/docs && tar c .' < /dev/null | tar x -C $W/a8docs
python3 -I $HERE/../emb174/calobs.py $W/a8plan144.json $R/drv $R/registry.json $W/a8docs $W/a8-17402.obs.jsonl --prefix emb17402a8 \
  --json $W/a8-17402.json 2>&1 | grep -v "no result yet"
python3 -I $HERE/../emb174/calobs.py $W/a8plan144-vs.json $R/drv $R/registry.json $W/a8docs $W/a8-17402-vs.obs.jsonl --prefix emb17402vs \
  --json $W/a8-17402-vs-lloyd.json 2>&1 | grep -v "no result yet"
# amendment 2: the host candidates (t8m, i8row) and the incumbent's Lloyd l3 (ctl) against the bf16 table
python3 -I $HERE/../emb174/calobs.py $W/a8plan144-bf16.json $R/drv $R/registry.json $W/a8docs $W/a8-17402-bf16.obs.jsonl --prefix emb17402bf \
  --json $W/a8-17402-vs-bf16.json 2>&1 | grep -v "no result yet"
