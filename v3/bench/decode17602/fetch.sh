#!/bin/bash
# fetch.sh -- #176-02 (w159): copy ds8's small results (row jsonl, reduced stamps, rocm-smi snapshots, bench / mk / run
# logs, recipes) into the repo's data/decode17602/ (VM side; server logs and stamp CSVs stay on ds8).
set -eu
D=$(cd "$(dirname "$0")/../../.." && pwd)/data/decode17602
rm -rf $D/res; mkdir -p $D/res $D/logs $D/recipes   # a clean mirror of ds8 res/ (top level; the card-7 rows are in c7/)
/workspace/rad-local/bin/ssh-ds8.sh 'cd /data/q27/w159 && tar c res/*.jsonl res/st-*.json res/smi-*.json res/bench-*.txt logs/run.log logs/mk.log recipes 2>/dev/null' \
  | tar x -C $D --no-same-owner
echo "$(ls $D/res/*.jsonl | wc -l) rows, $(ls $D/res/st-*.json 2>/dev/null | wc -l) stamp files"
