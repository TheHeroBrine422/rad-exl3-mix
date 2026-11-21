#!/bin/bash
# smoke-mik.sh TAG -- w212b: serve mikoskinen's EXL3 2.25 bpw container (same settings as score-mik.sh, no kld keys) on 127.0.0.1:8011, ask one greedy
# question, stop the server. Output /data/q27/w212/runs/TAG/{config.yaml,serve.log,answer.json}. Env as score-mik.sh (POOL, HEADROOM, TP, CARDS).
TAG=$1; W=/data/q27/w212; R=$W/runs/$TAG
mkdir -p $R
CARDS=${CARDS:-0 1 2 3}
for c in $CARDS; do exec {fd}> /data/q27/locks/card$c.lock; flock $fd; done
cat > $R/config.yaml <<YEOF
server:
  host: "127.0.0.1"
  port: 8011
cards: [$(echo $CARDS | sed 's/ /,/g')]
memory:
  gpu-headroom-mib: ${HEADROOM:-96}
models:
  ds41-mik225:
    model: $W/mik225/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
    tp: ${TP:-4}
    tp-wire: exact
    placement: expert_tiered
    host-pool-mib: ${POOL:-49152}
    ngram-placement: disk
    kv-cache-dtype: bf16
    max-model-len: 9216
    max-num-seqs: 1
    max-num-batched-tokens: 2048
    chat-template-kwargs: {"enable_thinking":false}
YEOF
export RADIANCE_HOME=/data/radiance-ds41/build/radiance_home
/data/radiance-ds41/build/bin/radiance --config $R/config.yaml > $R/serve.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 360); do
  kill -0 $PID 2>/dev/null || { echo "server exited early" > $R/answer.json; exit 1; }
  curl -sf http://127.0.0.1:8011/v1/models > /dev/null && break
  sleep 5
done
curl -s http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"ds41-mik225","messages":[{"role":"user","content":"What is the capital of France? Answer in one short paragraph, then write a Python function that reverses a string."}],"max_tokens":200,"temperature":0}' > $R/answer.json
kill $PID; wait $PID
