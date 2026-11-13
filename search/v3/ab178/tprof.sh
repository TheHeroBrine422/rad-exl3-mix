#!/bin/bash
# tprof.sh MIX.rad CARD OUTLOG PROMPT -- RULES-178 timing (w142, #178): the container served on 127.0.0.1 with --profile-ops
# (dumps every 4 steps), one request of PROMPT (~8k tokens, one prefill chunk) + 256 greedy tokens, then stopped. The
# first dump after the prefill step is the prefill table; the last minus that one is decode (tprof.py). Run under the
# card's lock. Profiled times are not an unprofiled run's (the engine says so): shares, not wall times.
set -u
M=$1 C=$2 L=$3 PR=$4 Q=/data/q27 P=18142
: > $L
ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$Q/rad10/home RADIANCE_PROFILE_EVERY=4 $Q/rad10/install/bin/radiance --model $M --tp 1 \
  --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --max-num-batched-tokens 8192 --host-headroom-mib 8192 \
  --host 127.0.0.1 --port $P --profile-ops >> $L 2>&1 &
SP=$!
for i in $(seq 1 120); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:$P/health)" = 200 ] && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -n 5 $L; exit 1; }
  sleep 5
done
/data/q27/venv/bin/python -c "import json,sys; print(json.dumps({'model':'m','prompt':open(sys.argv[1]).read(),'max_tokens':256,'temperature':0,'ignore_eos':True}))" $PR > $L.req
curl -s -m 900 http://127.0.0.1:$P/v1/completions -H "Content-Type: application/json" -d @$L.req -o $L.resp
echo "usage $(/data/q27/venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('usage'))" $L.resp)"
sleep 2; kill $SP; wait $SP
echo "dumps $(grep -c 'per-op timing, rank 0' $L)"
