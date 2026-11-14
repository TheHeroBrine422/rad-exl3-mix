#!/bin/bash
# collect157.sh -- (w157, #174-03) box2's small outputs -> rad-exl3-mix data/emb17403/ (no rows files, no containers):
# test outputs, KLD summaries (.json), docsums, perf logs / jsonl, drafter texts / metrics / notes, quant / slim / mix
# logs, rel.jsonl, donor inspections.
D=/workspace/rad-exl3-mix/data/emb17403; B2=/workspace/rad-local/bin/ssh-box2.sh
mkdir -p $D
timeout 600 $B2 'cd /data/q27/w157 && tar czf - --exclude="*.rows" --exclude="*.rad" res logs rel.jsonl q-eighths.txt dtmp/*.log mix/*.mix mix/*.mixlog rerun/t-i8/K2.5.log 2>/dev/null' < /dev/null | tar xzf - -C $D
ls -R $D | head -50; du -sh $D
