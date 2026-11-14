#!/bin/bash
# pull.sh -- box2 /data/q27/w160/res's small outputs (bench txt / csv, test outputs, KLD reports without the per-position
# rows, e2e jsonl + summaries) -> rad-exl3-mix data/pf184/. (#184, w160)
D=/workspace/rad-exl3-mix/data/pf184; mkdir -p $D
/workspace/rad-local/bin/ssh-box2.sh 'cd /data/q27/w160/res && tar czf - --exclude="*.rows" --exclude="serve-*.log" --exclude="kld/*.log" --exclude="old" . 2>/dev/null | base64 -w0' | base64 -d | tar xzf - -C $D
ls $D | head -50
