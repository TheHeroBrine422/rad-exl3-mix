#!/bin/bash
# dprof150.sh MODEL.rad CARD OUTLOG [extra radiance args] -- decode per-op profile (lead, #177 takeover 11-14): served with
# --profile-ops (dump every 64 steps; HOME_DIR = the plugin home, w150 copy of the lead's iqrcmp/dprof.sh), one short prompt + 320 greedy tokens; dprof.py takes last − first dump / steps.
set -u
M=$1 C=$2 L=$3; shift 3; Q=/data/q27 P=$((18150 + C)); I=$Q/rad10-perf/install H=${HOME_DIR:-$Q/w137/g3/home}
: > $L; ulimit -n 65536
ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H RADIANCE_PROFILE_EVERY=64 $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
  --max-model-len 8192 --max-num-seqs 1 --host 127.0.0.1 --port $P --profile-ops "$@" >> $L 2>&1 &
SP=$!
for i in $(seq 1 240); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:$P/health)" = 200 ] && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -n 5 $L; exit 1; }; sleep 5; done
curl -s -m 900 http://127.0.0.1:$P/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"m","prompt":"Write a long, detailed history of the printing press.","max_tokens":320,"temperature":0,"ignore_eos":true}' -o $L.resp
sleep 2; kill -INT $SP; wait $SP
echo "dumps $(grep -c 'per-op timing, rank 0' $L)"
