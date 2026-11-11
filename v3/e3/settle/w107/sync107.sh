#!/bin/bash
# sync107.sh -- VM: pull w107's small outputs from Thor / colo / gpubox into data/w107 (rows only; idempotent)
D=$(cd $(dirname $0)/../data && pwd)/w107; B=/workspace/rad-local/bin
mkdir -p $D/out/thor $D/out/colo $D/real $D/swj $D/gates
timeout 120 $B/ssh-thor.sh 'cd /data/work/w107/out && tar c *.jsonl 2>/dev/null' | tar x -m -C $D/out/thor
timeout 120 $B/ssh-r640.sh 'cd /data/work/w13/w107/out && tar c *.jsonl 2>/dev/null' | tar x -m -C $D/out/colo
timeout 120 $B/ssh-gpubox.sh 'cd /data/w107 && for u in cur-K0.125 tab-K0.125 cur-K0.5 tab-K0.5; do [ -f q/$u/run/.done ] && cp q/$u/run/log.jsonl real-log-$u.jsonl; done; tar c result.txt real-*.jsonl swj/result.txt 2>/dev/null' | tar x -m -C $D/real
for f in $D/real/real-log-*.jsonl; do [ -f "$f" ] && mv $f $D/real/log-${f##*/real-log-}; done
[ -f $D/real/swj/result.txt ] && mv $D/real/swj/result.txt $D/swj/result.txt && rmdir $D/real/swj
ls $D/out/thor | wc -l; ls $D/out/colo | wc -l; ls $D/real
