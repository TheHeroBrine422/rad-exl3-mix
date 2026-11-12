#!/bin/bash
# gputest2.sh -- gputest.sh continued (documents are 8192 tokens: max-model-len 8192 refuses them; 12288 instead),
# card 3 after the running bt4096 test; then packtest2.sh (w123)
cd /data/q27/w123
while pgrep -f "scores/st-bt4096" > /dev/null; do sleep 5; done
if cmp -s scores/st-bt4096.json.rows scores/s27-h2.json.rows; then echo "{\"test\": \"bt4096\", \"rows\": \"IDENTICAL\"}"; else echo "{\"test\": \"bt4096\", \"rows\": \"DIFFER\"}"; fi >> logs/settings.jsonl
B=/data/q27/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; G=/data/q27/w123/gold/h2
t() { local n=$1; shift; [ -f scores/st-$n.json.rows ] && return
  ./run.sh 3 st-$n $B $G "$@" | sed "s/^{/{\"test\": \"$n\", /" >> logs/settings.jsonl
  if cmp -s scores/st-$n.json.rows scores/s27-h2.json.rows; then echo "{\"test\": \"$n\", \"rows\": \"IDENTICAL\"}"; else echo "{\"test\": \"$n\", \"rows\": \"DIFFER\"}"; fi >> logs/settings.jsonl; }
t mml12k --max-model-len 12288
t ns2 --max-num-seqs 2 --max-model-len 12288
t ns4 --max-num-seqs 4 --max-model-len 12288
t bt8192 --max-num-batched-tokens 8192
echo GPUTEST DONE >> logs/settings.jsonl
exec ./packtest2.sh
