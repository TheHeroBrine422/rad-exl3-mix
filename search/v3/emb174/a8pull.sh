#!/bin/bash
# a8pull.sh -- (w135, #174, VM) per-document sums of every finished A8 anchor (ds8: a8docs.py, swap + w132's reference)
# pulled into rad-local/workers/w135/a8docs/, then calobs.py -> a8-174.obs.jsonl / a8-174.json (labels emb174a8-*).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); W=/workspace/rad-local/workers/w135; R=/workspace/rad-local/workers/w131/run
SSH=/workspace/rad-local/bin/ssh-ds8.sh
$SSH 'cd /data/q27/emb && args=""; for f in a8/scores/*-a8-s171b.json.rows; do n=$(basename $f -s171b.json.rows); t=${n%-e*-a8}; t=${t%-bf16-a8}
  [ -f a8/docs/$n-s171b.json ] || args="$args $n=$f"; [ -f a8/docs/ref-$t-a8-s171b.json ] || args="$args ref-$t-a8=/data/q27/w132/scores/$t-a8-s171b.json.rows"; done
  [ -z "$args" ] || /data/q27/venv/bin/python -I code/a8docs.py /data/q27/w131/slices/golden-s171b a8/docs $args'
mkdir -p $W/a8docs; $SSH 'cd /data/q27/emb/a8/docs && tar c .' | tar x -C $W/a8docs
python3 -I $HERE/calobs.py $W/a8plan.json $R/drv $R/registry.json $W/a8docs $W/a8-174.obs.jsonl --prefix emb174a8 --json $W/a8-174.json 2>&1 | grep -v "no result yet"
