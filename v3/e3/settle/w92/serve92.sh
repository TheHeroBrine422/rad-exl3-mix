#!/bin/bash
# serve92.sh TAG MIX OUT PORT INSTALL HOME -- (e2e92.sh, under card0.lock + ram.lock shared) radiance's OpenAI server on
# 127.0.0.1:PORT with MIX, gen92.py's requests (generation for TAG fs0, prefill speed for both), the VRAM budget from the
# server log, then the server is stopped (its own PID only).
set -u
tag=$1 m=$2 O=$3 PORT=$4 I=$5 H=$6; PY=/data/venv-exl3/bin/python
ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H $I/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
  --max-num-seqs 1 --host-headroom-mib 8192 --host 127.0.0.1 --port $PORT --served-model-name w92 > $O/serve-$tag.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && break; kill -0 $SP 2>/dev/null || exit 1; sleep 5; done
curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null || exit 1
$PY -I $(dirname $0)/gen92.py $tag $PORT $O /data/w92/golden-search27c-x.jsonl || exit 1
grep -E "static weights|activation arena|kv cache  |vram_weights|vram_kv" $O/serve-$tag.log | head -5 > $O/vram-$tag.txt
