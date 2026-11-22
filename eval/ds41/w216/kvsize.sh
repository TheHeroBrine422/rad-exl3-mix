#!/bin/bash
# kvsize.sh LEN... -- w216: the deepseek41 KV allocation a card at each --max-model-len, TP4, bf16 KV, on bakon's
# reference build (box2 /data/radiance-ds41, read-only) with mikoskinen's EXL3 2.25 container (w212b's settings: one
# sequence, no speculation). Starts the server, waits for the KV plan lines, stops it (kills its PID). Takes the four
# card locks in order. Output /data/q27/w216/kv/LEN/{config.yaml,serve.log}. Restartable: skips a length whose log
# already has the plan.
W=/data/q27/w212; O=/data/q27/w216/kv
for c in 0 1 2 3; do exec {fd}> /data/q27/locks/card$c.lock; flock -n $fd || { echo "card$c busy"; exit 1; }; done
export RADIANCE_HOME=/data/radiance-ds41/build/radiance_home
for LEN in "$@"; do
  R=$O/$LEN; mkdir -p $R
  grep -q "kv: the cache can grow" $R/serve.log 2>/dev/null && { echo "$LEN done"; continue; }
  cat > $R/config.yaml <<YEOF
cards: [0,1,2,3]
memory:
  gpu-headroom-mib: 96
models:
  ds41-mik225:
    model: $W/mik225/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
    tp: 4
    tp-wire: exact
    placement: expert_tiered
    host-pool-mib: 49152
    ngram-placement: disk
    kv-cache-dtype: bf16
    max-model-len: $LEN
    max-num-seqs: 1
    max-num-batched-tokens: 2048
    checkpoint-interval: 8192
    chat-template-kwargs: {"enable_thinking":false}
YEOF
  /data/radiance-ds41/build/bin/radiance --config $R/config.yaml > $R/serve.log 2>&1 &
  P=$!
  for i in $(seq 1 120); do
    sleep 5
    grep -q "kv: the cache can grow" $R/serve.log && break
    kill -0 $P 2>/dev/null || break
  done
  sleep 5
  kill $P 2>/dev/null; wait $P 2>/dev/null
  echo "$LEN: $(grep -c 'kv: the cache can grow' $R/serve.log) plan line(s)"
done
