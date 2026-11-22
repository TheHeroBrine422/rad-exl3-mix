#!/bin/bash
# collect217.sh -- (VM) copy #217's small raw results from box2 /data/q27/w217 into data/qad217/: scores/*.json (radiance --kld-out summaries; the per-row .rows files stay
# on box2), per-arm run logs (train.jsonl, fixed.jsonl, plan.json, run.log), run217.log / q217.log, the job lists.
D=/workspace/rad-exl3-mix/data/qad217; mkdir -p $D/scores $D/runs
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=120 -i /workspace/card_session_ed25519 -p 2241 root@127.0.0.1"
$SSH 'cd /data/q27/w217; tar cf - scores/*.json run217.log q217.log jobs217.txt q217-card*.txt pre217-card*.txt fin217-card*.txt $(for a in runs/*; do for f in train.jsonl fixed.jsonl plan.json run.log; do [ -e $a/$f ] && echo $a/$f; done; done) 2>/dev/null' | tar xf - -C $D
du -sh $D; ls $D/scores | wc -l
