#!/bin/bash
# smoke203.sh SIZE (box2, w203) -- the smoke test of vramvd/SIZE.rad: two servers on card 0 (under card0.lock, ram.lock
# shared, one at a time): drafter off (--num-speculative-tokens 0), then on (the default depth). Greedy: the 8 gen8 text
# prompts (128 tokens; tokens per decode step from the counters) and 2 image prompts (quad.png, 128 tokens).
# Out: smoke203/SIZE/{serve-*.log, startup-*.txt, graph-*.json, text-*.jsonl/.txt, img-*.jsonl, metrics-*.txt, cmp-*.txt, tie-*.txt}.
# BIN: the radiance binary (default: the FFmpeg build of the same source, ff203/build: images need it; plugins from rad21/home).
# Restartable: a mode with done-MODE is skipped. Stop by killing this script's own PIDs only.
set -u
SZ=$1; export SMOKE_MODEL=$SZ; Q=/data/q27; NEW=$Q/w201/rad21; BIN=${BIN:-$Q/release-v3/vramvd/ff203/build/bin/radiance}; M=$Q/release-v3/vramvd/$SZ.rad; RB=$Q/release-v3/smoke203
O=$RB/$SZ; T=$RB/tools; IMG=$RB/img; PORT=${PORT:-8203}; LK=$Q/locks; PY="/data/q27/venv/bin/python -I"
mkdir -p $O; [ -f $M ] || { echo "no $M"; exit 1; }
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/smoke.log; }

run() {   # run off|on: one server, its prompts, the stop
  local mode=$1 a=() i=0 SP
  [ "$mode" = off ] && a=(--num-speculative-tokens 0)
  [ -f $O/done-$mode ] && return 0
  say "serve $mode: start"
  ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$NEW/home $BIN --model $M --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 2 --mm-max-patches 4096 --host 127.0.0.1 --port $PORT "${a[@]}" > $O/serve-$mode.log 2>&1 &
  SP=$!
  while [ $i -lt 1500 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do
    sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
  if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
    grep -i "speculat\|rerank\|rescor\|dflash\|drafter\|vision\|tower\|media\|head" $O/serve-$mode.log > $O/startup-$mode.txt
    curl -s -m 20 http://127.0.0.1:$PORT/graph > $O/graph-$mode.json 2>/dev/null
    $PY $T/smoke_text.py $PORT $T/gen8.jsonl $O/text-$mode.jsonl 128 > $O/text-$mode.txt 2>&1
    $PY $T/genimg.py $PORT $T/img-smoke.jsonl $IMG $O/img-$mode.jsonl 128 > $O/img-$mode.txt 2>&1
    curl -s -m 20 http://127.0.0.1:$PORT/metrics > $O/metrics-$mode.txt
    touch $O/done-$mode; say "serve $mode: prompts done ($(tail -1 $O/text-$mode.txt))"
  else say "serve $mode: NO HEALTH ($(grep -m1 -i '^E \|error' $O/serve-$mode.log | cut -c1-200))"; fi
  kill -INT $SP 2>/dev/null; for j in $(seq 90); do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill $SP 2>/dev/null; wait $SP 2>/dev/null; say "serve $mode: stopped"
}
gpu() { ( flock 9; flock -s 8; run $1 ) 9> $LK/card0.lock 8> $LK/ram.lock; }

gpu off
gpu on
if [ -f $O/done-off ] && [ -f $O/done-on ]; then
  $PY $T/dcmp.py $O/text-off.jsonl $O/text-on.jsonl > $O/cmp-text.txt 2>&1
  $PY $T/dcmp.py $O/img-off.jsonl $O/img-on.jsonl > $O/cmp-img.txt 2>&1
  $PY $T/tiemargin.py $O/text-off.jsonl $O/text-on.jsonl > $O/tie-text.txt 2>&1
  $PY $T/tiemargin.py $O/img-off.jsonl $O/img-on.jsonl > $O/tie-img.txt 2>&1
  say "compare done"
fi
