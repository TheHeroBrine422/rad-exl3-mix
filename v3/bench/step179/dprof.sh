#!/bin/bash
# dprof.sh CARD HOME PACK OUTLOG [CTXFILE] -- decode per-op profile (lead's v3/bench/iqrcmp/dprof.sh with the home and
# pack as arguments, under card + ram (shared) locks): --profile-ops (dump every 64 steps), one prompt (short, or the
# CTXFILE text: e.g. the 8k prompt) + NTOK (default 320) greedy tokens; the lead's dprof.py takes last - first dump / steps. (w152, #179)
C=$1 H=$2 M=$3 L=$4 CF=${5:-}; Q=/data/q27; P=$((18250 + C)); I=$Q/rad10-perf/install
: > $L; ulimit -n 65536
(
flock 5; flock -s 6
ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H RADIANCE_PROFILE_EVERY=64 $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
  --max-model-len 16384 --max-num-seqs 1 --host 127.0.0.1 --port $P --profile-ops >> $L 2>&1 &
SP=$!
for i in $(seq 1 240); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:$P/health)" = 200 ] && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -n 5 $L; exit 1; }; sleep 5; done
if [ -n "$CF" ]; then
  python3 -I -c 'import json,sys; print(json.dumps({"model":"m","prompt":open(sys.argv[1]).read(),"max_tokens":int(sys.argv[2]),"temperature":0,"ignore_eos":True}))' $CF ${NTOK:-320} > $L.req
else
  echo '{"model":"m","prompt":"Write a long, detailed history of the printing press.","max_tokens":'${NTOK:-320}',"temperature":0,"ignore_eos":true}' > $L.req
fi
curl -s -m 900 http://127.0.0.1:$P/v1/completions -H "Content-Type: application/json" -d @$L.req -o $L.resp
sleep 2; kill -INT $SP; wait $SP
echo "dumps $(grep -c 'per-op timing, rank 0' $L)"
) 5>$Q/locks/card$C.lock 6>$Q/locks/ram.lock
