#!/bin/bash
# Env EXTRA_YAML: one more model key (e.g. "num-speculative-tokens: 0" for the speculation-off arm); PROMPT, MAXTOK.
# smoke-port.sh TAG -- w213 (the port; else w212b's smoke-mik.sh): serve mikoskinen's EXL3 2.25 bpw container (same settings as score-mik.sh, no kld keys) on 127.0.0.1:8011, ask one greedy
# question, stop the server. Output /data/q27/w212/runs/TAG/{config.yaml,serve.log,answer.json}. Env as score-mik.sh (POOL, HEADROOM, TP, CARDS).
TAG=$1; W=/data/q27/w212; R=/data/q27/w213/runs/$TAG; B=${BUILD:-/data/q27/w213/src/build}
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
${EXTRA_YAML:+    $EXTRA_YAML}
YEOF
export RADIANCE_HOME=$B/radiance_home
$B/bin/radiance --config $R/config.yaml > $R/serve.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 360); do
  kill -0 $PID 2>/dev/null || { echo "server exited early" > $R/answer.json; exit 1; }
  curl -sf http://127.0.0.1:8011/v1/models > /dev/null && break
  sleep 5
done
BODY=$(PROMPT="${PROMPT:-What is the capital of France? Answer in one short paragraph, then write a Python function that reverses a string.}" MAXTOK=${MAXTOK:-200} python3 -c 'import json,os; print(json.dumps({"model":"ds41-mik225","messages":[{"role":"user","content":os.environ["PROMPT"]}],"max_tokens":int(os.environ["MAXTOK"]),"temperature":0}))')
curl -s http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" > $R/answer.json
kill $PID; wait $PID
