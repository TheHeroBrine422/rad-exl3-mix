#!/bin/bash
# gdec.sh CARD PACKNAME|/PACK/PATH.rad TAG HOME -- decode numerics: one server (rad10-perf install, HOME), 8 fixed prompts (gdec.py's,
# written for this check), each 96 greedy tokens with top-20 logprobs, at max-num-seqs 1 (T 1 steps) -> res/gdec/PACK-TAG.jsonl.
# gdec.py A B compares two runs token by token and logprob by logprob. Card + ram (shared) locks. (w152, #179)
C=$1 PN=$2 TAG=$3 H=$4; Q=/data/q27; W=$Q/w152; P=$((18450 + C)); I=$Q/rad10-perf/install; M=$Q/w138/packs/$PN.rad
case $PN in /*) M=$PN; PN=$(basename $PN .rad) ;; esac   # a pack by path
O=$W/res/gdec/$PN-$TAG; mkdir -p $W/res/gdec; [ -s $O.jsonl ] && exit 0; ulimit -n 65536
(
flock 5; flock -s 6
ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
  --max-model-len 16384 --max-num-seqs 1 --host 127.0.0.1 --port $P > $O.log 2>&1 &
SP=$!
for i in $(seq 1 240); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:$P/health)" = 200 ] && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -n 5 $O.log; exit 1; }; sleep 5; done
python3 -I $W/tools/gdec.py run $P $O.jsonl.part && mv $O.jsonl.part $O.jsonl
kill -INT $SP; wait $SP
) 5>$Q/locks/card$C.lock 6>$Q/locks/ram.lock
