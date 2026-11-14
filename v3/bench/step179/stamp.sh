#!/bin/bash
# stamp.sh CARD HOME PACK OUT [CTXFILE] -- RADIANCE_AQL_STAMP timing of a served request (every dispatch of every Nth
# recorded pass on the card, device timestamps, no sync between ops, so unlike --profile-ops the step is the real one):
# one prompt (short, or CTXFILE's text) + NTOK (default 320) greedy tokens, N = STAMP_N (default 8) -> OUT.csv (+ OUT.log).
# NTOK=1 STAMP_N=1 with the 8k prompt: the prefill's chunks. Card + ram (shared) locks. Env SRVENV: VAR=VALUE words; XARGS: more
# radiance arguments.
# WATCHS: see below. Skips when OUT.csv exists. (w152, #179)
C=$1 H=$2 M=$3 O=$4 CF=${5:-}; Q=/data/q27; P=$((18550 + C)); I=$Q/rad10-perf/install
[ -s $O.csv ] && exit 0; ulimit -n 65536; rm -f $O.csv.part
(
flock 5; flock -s 6
env ${SRVENV:-} ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H RADIANCE_AQL_STAMP=${STAMP_N:-8} RADIANCE_AQL_STAMP_OUT=$O.csv.part \
  $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host 127.0.0.1 --port $P \
  ${XARGS:-} > $O.log 2>&1 &
SP=$!
for i in $(seq 1 240); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:$P/health)" = 200 ] && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -n 5 $O.log; exit 1; }; sleep 5; done
# WATCHS=S: cardclock.py --watch (SMU clocks / power / throttlers ten times a second, the server's step counters)
# over S seconds from 6 s before the request -> OUT.clock.json / .txt
if [ -n "${WATCHS:-}" ]; then
  python3 -I $Q/rad10-perf/src/scripts/cardclock.py --watch --port $P --seconds $WATCHS --window 5 --json $O.clock.json \
    > $O.clock.txt 2>&1 & WP=$!; sleep 11
fi
if [ -n "$CF" ]; then
  python3 -I -c 'import json,sys; print(json.dumps({"model":"m","prompt":open(sys.argv[1]).read(),"max_tokens":int(sys.argv[2]),"temperature":0,"ignore_eos":True}))' $CF ${NTOK:-320} > $O.req
else
  echo '{"model":"m","prompt":"Write a long, detailed history of the printing press.","max_tokens":'${NTOK:-320}',"temperature":0,"ignore_eos":true}' > $O.req
fi
curl -s -m 900 http://127.0.0.1:$P/v1/completions -H "Content-Type: application/json" -d @$O.req -o $O.resp
[ -n "${WP:-}" ] && wait $WP
sleep 2; kill -INT $SP; wait $SP
[ -s $O.csv.part ] && mv $O.csv.part $O.csv; echo "$(basename $O): $(wc -l < $O.csv 2>/dev/null) lines"
) 5>$Q/locks/card$C.lock 6>$Q/locks/ram.lock
