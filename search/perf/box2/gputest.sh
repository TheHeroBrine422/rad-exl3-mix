#!/bin/bash
# gputest.sh -- after the move runs, before packtest2: scoring settings on golden h2 (51 docs) vs the default run
# s27-h2 (rows cmp + phase times), card 3 (w123). Then packtest2.sh.
cd /data/q27/w123
while pgrep -f "^/bin/bash ./moves.sh" > /dev/null; do sleep 20; done
B=/data/q27/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; G=/data/q27/w123/gold/h2
t() { local n=$1; shift; [ -f scores/st-$n.json.rows ] && return
  ./run.sh 3 st-$n $B $G "$@" | sed "s/^{/{\"test\": \"$n\", /" >> logs/settings.jsonl
  if cmp -s scores/st-$n.json.rows scores/s27-h2.json.rows; then echo "{\"test\": \"$n\", \"rows\": \"IDENTICAL\"}"; else echo "{\"test\": \"$n\", \"rows\": \"DIFFER\"}"; fi >> logs/settings.jsonl; }
t def
t mml8k --max-model-len 8192
t bt4096 --max-num-batched-tokens 4096
t bt8192 --max-num-batched-tokens 8192 --max-model-len 8192
t ns2 --max-num-seqs 2 --max-model-len 8192
t ns4 --max-num-seqs 4 --max-model-len 8192
echo GPUTEST DONE >> logs/settings.jsonl
exec ./packtest2.sh
